"""
SPA Relative Model — per-query-frame coordinate transformation for 4D M-RoPE.

When ``--relative`` is active the dataset returns ``image_xyz_relative``:
a list of N_images tensors each shaped ``(N_frames, llm_H, llm_W, 3)``.
For image k, ``image_xyz_relative[k][f]`` is every patch's 3-D scene coordinate
expressed in frame-f's camera coordinate system:

    P_in_frame_f = R_f^{-1} @ (P_world - t_f)

At attention time, Q from frame-f sees ALL K tokens' positions in frame-f
coordinates, making the spatial relation between any two patches
frame-of-reference–consistent.

Architecture additions
──────────────────────
SpaRelativeAttentionWrapper(nn.Module)
    Wraps a Qwen3.5 self-attention module.
    When position_embeddings_per_frame / token_frame_ids kwargs are present,
    runs N_frames extra attention passes and blends outputs by query-frame.

SpaRelativeTextModel(SpaTextModel)
    Reads _pf_cache set by SpaRelativeModel.get_rope_index() and computes
    N_frames position_embeddings; passes them as extra kwargs to decoder layers.

SpaRelativeModel(SpaModel)
    Overrides get_rope_index() to handle image_xyz_relative:
        1. standard position_ids (world-frame xyz)
        2. N_frames per-frame position_ids (one per reference camera)
        3. token_frame_ids  (batch, seq): which frame each vision token belongs to
    Stores (2) and (3) in self.language_model._pf_cache for SpaRelativeTextModel.

SpaRelativeForConditionalGeneration(SpaForConditionalGeneration)
    Top-level model. Routes image_xyz_relative into get_rope_index().
"""

import itertools
import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5DynamicCache,
)
from transformers.masking_utils import create_causal_mask

from .spa_emb import (
    SpaVisionModel,
    SpaTextRotaryEmbedding,
    SpaTextModel,
    SpaModel,
    SpaForConditionalGeneration,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention wrapper — per-frame position embedding blending
# ─────────────────────────────────────────────────────────────────────────────

class SpaRelativeAttentionWrapper(nn.Module):
    """
    Wraps a Qwen3.5 attention module to support per-frame positional embeddings.

    Standard mode (position_embeddings_per_frame is None):
        Delegates directly to the wrapped attention, zero overhead.

    Relative mode (position_embeddings_per_frame provided):
        1. Runs standard attention → output for text-query tokens.
        2. For each frame f (where frame-f vision tokens exist as queries):
               runs attention with position_embeddings_per_frame[f] applied to
               BOTH Q and K, then replaces output rows for frame-f Q tokens.
        This makes every Q from frame f see K's xyz in frame f's camera space.

    KV-cache is not supported in relative mode (past_key_values must be None).
    When past_key_values is set, falls back to standard attention silently.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        # ── extra kwargs injected by SpaRelativeTextModel ──────────────────
        position_embeddings_per_frame=None,   # list[N_frames] of (cos, sin)
        token_frame_ids=None,                 # (batch, seq_len): -1=text, ≥0=frame idx
        **kwargs,
    ):
        # Fall back to standard attention when not in relative mode or during inference
        if (position_embeddings_per_frame is None
                or token_frame_ids is None
                or past_key_values is not None):
            return self.attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # ── Standard pass (world-frame coords) ────────────────────────────
        out_std = self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
            **kwargs,
        )
        # Qwen3_5Attention.forward returns (attn_output, attn_weights)
        hidden_out = out_std[0]              # (bs, seq, hidden_dim)
        result = hidden_out.clone()

        N_frames = len(position_embeddings_per_frame)
        bs = hidden_states.shape[0]

        for f in range(N_frames):
            frame_mask = (token_frame_ids == f)   # (bs, seq)
            if not frame_mask.any():
                continue

            # Run attention with frame-f position embeddings for BOTH Q and K
            out_f = self.attn(
                hidden_states,
                position_embeddings=position_embeddings_per_frame[f],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_out_f = out_f[0]              # (bs, seq, hidden_dim)

            # Replace output rows where this frame's tokens are the query
            for b in range(bs):
                mask_b = frame_mask[b]           # (seq,)
                result[b, mask_b] = hidden_out_f[b, mask_b]

        return (result,) + out_std[1:]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text model — computes per-frame position embeddings and routes them
# ─────────────────────────────────────────────────────────────────────────────

class SpaRelativeTextModel(SpaTextModel):
    """
    Extends SpaTextModel to compute and propagate per-frame position embeddings.

    Reads ``_pf_cache`` (set by SpaRelativeModel.get_rope_index before the
    language_model.forward call) and computes N_frames sets of (cos, sin).
    These are forwarded to every decoder layer as extra kwargs so that
    SpaRelativeAttentionWrapper can perform the per-frame blending.
    """

    def __init__(self, config):
        super().__init__(config)
        # Will be set by SpaRelativeModel.get_rope_index()
        self._pf_cache = None   # (position_ids_per_frame, token_frame_ids) | None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        # ── Identical setup to SpaTextModel.forward() ─────────────────────
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = (past_key_values.get_seq_length()
                         if past_key_values is not None else 0)
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Accept 5D position_ids (seq, t, x, y, z)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                5, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                5, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] >= 4:
            text_position_ids = position_ids[0]   # seq dim → causal mask
            position_ids = position_ids[1:]        # (t, x, y, z) → RoPE
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # Standard position embeddings (world-frame coords)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ── Per-frame position embeddings (relative mode) ─────────────────
        extra_decoder_kwargs: dict = {}
        pf_cache = self._pf_cache
        if pf_cache is not None and past_key_values is None:
            pf_position_ids, token_frame_ids = pf_cache
            # Each pf_position_ids[f] has shape (5, batch, seq); skip dim-0 (seq)
            position_embeddings_per_frame = [
                self.rotary_emb(hidden_states, pids_f[1:])
                for pids_f in pf_position_ids
            ]
            extra_decoder_kwargs["position_embeddings_per_frame"] = (
                position_embeddings_per_frame
            )
            extra_decoder_kwargs["token_frame_ids"] = token_frame_ids
        # ─────────────────────────────────────────────────────────────────

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask
                if decoder_layer.layer_type == "linear_attention"
                else causal_mask
            )
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    use_reentrant=False,
                    **extra_decoder_kwargs,
                    **kwargs,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **extra_decoder_kwargs,
                    **kwargs,
                )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backbone — extended get_rope_index for relative coords
# ─────────────────────────────────────────────────────────────────────────────

class SpaRelativeModel(SpaModel):
    """
    Extends SpaModel.get_rope_index() to handle image_xyz_relative.

    For each forward pass the method:
      1. Builds standard position_ids (world-frame xyz, same as SpaModel).
      2. If image_xyz_relative provided, builds N_frames additional position_ids
         (one per reference camera frame) and token_frame_ids.
      3. Caches (2) in self.language_model._pf_cache so that
         SpaRelativeTextModel.forward() can compute per-frame position embeddings.
    """

    def __init__(self, config):
        # Use SpaRelativeTextModel instead of SpaTextModel
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = SpaVisionModel._from_config(config.vision_config)
        self.language_model = SpaRelativeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    # ── helper: which sequence positions belong to which image ────────────

    def _build_token_frame_ids(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        """
        Return (batch, seq_len) long tensor: -1 for text, k for tokens of image k.

        Uses only mm_token_type_ids (0=text, 1=image/2=video).  Consecutive runs
        of non-zero type are assigned to successive image indices 0, 1, 2, …
        (matching the order images appear in the sequence, same as get_rope_index).
        """
        bs, seq_len = input_ids.shape
        out = torch.full((bs, seq_len), -1, dtype=torch.long, device=input_ids.device)

        for b in range(bs):
            m = mm_token_type_ids[b]
            mask = attention_mask[b].bool() if attention_mask is not None else None
            if mask is not None:
                m_valid = m[mask]
            else:
                m_valid = m

            img_idx = 0
            for key, grp in itertools.groupby(enumerate(m_valid.tolist()),
                                              lambda x: x[1]):
                grp = list(grp)
                grp_start = grp[0][0]  # position in the (possibly masked) sequence
                grp_len = len(grp)
                if key != 0:           # image or video tokens
                    if mask is not None:
                        valid_pos = mask.nonzero(as_tuple=True)[0]
                        for i in range(grp_len):
                            orig = valid_pos[grp_start + i].item()
                            out[b, orig] = img_idx
                    else:
                        out[b, grp_start: grp_start + grp_len] = img_idx
                    img_idx += 1

        return out

    # ── overridden get_rope_index ─────────────────────────────────────────

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        image_xyz=None,
        image_xyz_relative=None,   # list[N_img] of (N_frames, H, W, 3)
        coord_scale: float = 100.0,
        polar: bool = False,
        **kwargs,
    ):
        """
        Extended get_rope_index supporting per-frame relative coordinates.

        Args:
            image_xyz_relative: list of N_images tensors, each (N_frames, H, W, 3).
                image_xyz_relative[k][f] = patch xyz of image k in frame-f camera space.
            All other args: same as SpaModel.get_rope_index().

        Side effect:
            Sets self.language_model._pf_cache = (position_ids_per_frame, token_frame_ids)
            when image_xyz_relative is provided; clears it otherwise.

        Returns:
            (position_ids, mrope_position_deltas) — standard shapes, world-frame.
        """
        # ── Determine world-frame xyz for standard position_ids ───────────
        if image_xyz is None and image_xyz_relative is not None:
            # frame 0 of image_xyz_relative = world-frame coords (pts3d already
            # in world/frame-0 coordinates, as confirmed by the data pipeline)
            image_xyz = [xyz_rel[0] for xyz_rel in image_xyz_relative]

        # ── Standard position_ids (world-frame) ───────────────────────────
        position_ids, mrope_deltas = super().get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            image_xyz=image_xyz,
            coord_scale=coord_scale,
            polar=polar,
            **kwargs,
        )

        # ── Per-frame position_ids (relative mode) ────────────────────────
        if image_xyz_relative is not None:
            N_frames = image_xyz_relative[0].shape[0]
            pf_position_ids = []
            for f in range(N_frames):
                xyz_f = [xyz_rel[f] for xyz_rel in image_xyz_relative]
                pids_f, _ = super().get_rope_index(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    image_xyz=xyz_f,
                    coord_scale=coord_scale,
                    polar=polar,
                    **kwargs,
                )
                pf_position_ids.append(pids_f)

            token_frame_ids = self._build_token_frame_ids(
                input_ids, mm_token_type_ids, attention_mask
            )
            self.language_model._pf_cache = (pf_position_ids, token_frame_ids)
        else:
            self.language_model._pf_cache = None

        return position_ids, mrope_deltas


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class SpaRelativeForConditionalGeneration(SpaForConditionalGeneration):
    """
    Top-level model with per-query-frame relative coordinate support.

    Usage mirrors SpaForConditionalGeneration; pass image_xyz_relative instead
    of (or in addition to) image_xyz:

        outputs = model(
            input_ids=...,
            pixel_values=...,
            image_grid_thw=...,
            image_xyz_relative=xyz_relative,   # list of (N_frames, H, W, 3)
        )

    Call ``patch_attention_layers(model)`` once after applying LoRA to replace
    all self_attn modules with SpaRelativeAttentionWrapper.
    """

    def __init__(self, config):
        # Call SpaForConditionalGeneration.__init__ (creates SpaModel), then
        # replace with SpaRelativeModel (same parameter names → weights load fine).
        super().__init__(config)
        self.model = SpaRelativeModel(config)

    def forward(
        self,
        *args,
        image_xyz: object = None,
        image_xyz_relative: object = None,
        coord_scale: float = 100.0,
        polar: bool = False,
        **kwargs,
    ):
        """
        Thin wrapper: injects image_xyz and image_xyz_relative into kwargs so
        that get_rope_index() (called inside the parent forward) receives them.
        """
        if image_xyz is not None:
            kwargs["image_xyz"] = image_xyz
        if image_xyz_relative is not None:
            kwargs["image_xyz_relative"] = image_xyz_relative
        if coord_scale != 100.0:
            kwargs["coord_scale"] = coord_scale
        if polar:
            kwargs["polar"] = polar
        # Call grandparent (Qwen3_5ForConditionalGeneration.forward) to avoid
        # SpaForConditionalGeneration re-injecting image_xyz into kwargs twice.
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )
        return Qwen3_5ForConditionalGeneration.forward(self, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Post-LoRA attention patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_attention_layers(model: nn.Module) -> None:
    """
    Replace every self_attn module in the language model's decoder layers with a
    SpaRelativeAttentionWrapper.

    Call this AFTER get_peft_model() so LoRA adapters inside q_proj/k_proj/etc.
    are already in place and get carried along inside the wrapper.

    Args:
        model: the top-level model (SpaRelativeForConditionalGeneration or
               AnswerRelativeModel wrapping it).
    """
    # Resolve language_model through common wrappers (AnswerRelativeModel, DDP,
    # PEFT/Lora wrappers, and nested top-level model containers).
    def _find_language_model(root: nn.Module) -> nn.Module | None:
        queue = [root]
        seen: set[int] = set()

        while queue:
            cur = queue.pop(0)
            cur_id = id(cur)
            if cur_id in seen:
                continue
            seen.add(cur_id)

            lm = getattr(cur, "language_model", None)
            if isinstance(lm, nn.Module) and hasattr(lm, "layers"):
                return lm

            for attr in ("spa_model", "module", "base_model", "model"):
                nxt = getattr(cur, attr, None)
                if isinstance(nxt, nn.Module) and id(nxt) not in seen:
                    queue.append(nxt)

        return None

    lm = _find_language_model(model)
    if lm is None:
        raise AttributeError(
            "Cannot find language_model inside the provided model. "
            "Expected SpaRelativeForConditionalGeneration or a wrapper around it."
        )

    n_wrapped = 0
    for layer in lm.layers:
        if (hasattr(layer, "self_attn")
                and not isinstance(layer.self_attn, SpaRelativeAttentionWrapper)):
            layer.self_attn = SpaRelativeAttentionWrapper(layer.self_attn)
            n_wrapped += 1

    if n_wrapped == 0:
        raise RuntimeError(
            "patch_attention_layers: no self_attn modules found. "
            "Make sure the model is a SpaRelativeForConditionalGeneration."
        )
    return n_wrapped
