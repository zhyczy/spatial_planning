"""
Spatial Attention LLM — Qwen3.5 with per-layer geometric attention bias.

Wires SpatialAttentionBias (spatial_attention_block.py) into every transformer
layer's self-attention.  Position embedding stays exactly as Qwen3.5 ships it
(3D M-RoPE, mrope_section [11, 11, 10] — UNCHANGED).  3-D scene information
enters the model only through the per-layer bias — never through position_ids.

Math (per layer, per head):

    softmax( QK^T / √d  +  causal_mask  +  B_spatial ) V

    B_spatial[h, i, j] = ( W₂ · GELU( W₁ · feat_ij ) )[h]      if i, j are vision tokens
                       = 0                                      otherwise

    feat_ij = (n_x, n_y, n_z, d)   ∈ ℝ⁴
              n  = (p_j − p_i) / ‖p_j − p_i‖     unit direction (scale-decoupled)
              d  = ‖p_j − p_i‖                   magnitude

    W₁ ∈ ℝ^{hidden × 4},   W₂ ∈ ℝ^{H × hidden}     (hidden default 128)

The bias is added to ``attention_mask`` before it is fed to the underlying
attention kernel — works with eager / SDPA / flash because all of them treat
``attention_mask`` as additive pre-softmax.

Decode-phase short-circuit
──────────────────────────
During autoregressive generation, the query is a single newly-generated text
token (no spatial position) and the KV cache holds the prefilled vision tokens.
The bias row for a text query is all zero by definition, so the wrapper does a
clean fall-through to vanilla attention — no extra cost during decode.
The "spatial-aware" features were baked into the KV cache during prefill.

Architecture
────────────
SpatialAttnWrapper(nn.Module)
    Wraps one Qwen3.5 self_attn module. Owns its own SpatialAttentionBias —
    a per-layer 2-layer MLP (Linear(4→hidden) → GELU → Linear(hidden→num_heads))
    that maps the per-pair edge feature (n_x, n_y, n_z, d) to a per-head
    bias. When (flat_xyz, vision_mask) are provided AND we are in prefill,
    computes B and adds it to attention_mask. Otherwise pass-through.

SpatialAttnVanillaTextModel(Qwen3_5TextModel)
    Reads ``_spatial_cache`` set by SpatialAttnVanillaModel before the forward
    call, forwards (flat_xyz, vision_mask) as extra kwargs to every decoder
    layer.  No RoPE changes — just adds the spatial-cache injection.

SpatialAttnVanillaModel(Qwen3_5Model)
    Overrides forward() to pop image_xyz, build (flat_xyz, vision_mask) inline
    from mm_token_type_ids, and stash them on language_model._spatial_cache.
    Use it by swapping into a stock Qwen3_5ForConditionalGeneration's .model
    after from_pretrained — see the swap recipe at the bottom of this file.

patch_attention_layers_spatial(model)
    Call AFTER get_peft_model() to wrap each self_attn with SpatialAttnWrapper.
"""

import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5DynamicCache,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5TextModel,
    Qwen3_5Model,
    Qwen3_5VisionModel,
)
from transformers.masking_utils import create_causal_mask

from .spatial_attention_block import SpatialAttentionBias


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention wrapper — adds geometric bias to attention scores
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttnWrapper(nn.Module):
    """
    Wraps a Qwen3.5 self_attn module to add a per-head geometric bias on
    vision-vision attention pairs.

    Three execution paths:
      1. No spatial info given       → pass-through to underlying attn.
      2. Decode (KV cache populated) → pass-through (text query, bias=0).
      3. Prefill with spatial info   → bias = SpatialAttentionBias(xyz, mask)
                                       added to attention_mask, then standard
                                       attention runs.

    Args:
        attn:       the original Qwen3_5Attention module.
        num_heads:  H — number of attention heads (sets the MLP's output
                    dimension so each head gets its own bias scalar per pair).
    """

    def __init__(self, attn: nn.Module, num_heads: int):
        super().__init__()
        self.attn = attn
        self.bias_module = SpatialAttentionBias(num_heads=num_heads, zero_init=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        # ── injected by SpatialAttnTextModel ───────────────────────────────
        flat_xyz=None,
        vision_mask=None,
        # ──────────────────────────────────────────────────────────────────
        **kwargs,
    ):
        # Decode short-circuit — KV cache already populated → query is text.
        is_decode = (
            past_key_values is not None
            and past_key_values.get_seq_length() > 0
        )

        if flat_xyz is None or vision_mask is None or is_decode:
            return self.attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # Prefill with spatial info — compute B and add to attention_mask.
        # bias: (B, H, L, L); zero outside vision-vision sub-block.
        bias = self.bias_module(flat_xyz, vision_mask)

        if attention_mask is not None:
            # attention_mask is typically (B, 1, L_q, L_k) with large negative
            # values (e.g. -1e4 / -65504 / -inf) at masked positions and 0
            # elsewhere.
            bias = bias.to(attention_mask.dtype)

            # Broadcast-add: (B, 1, L_q, L_k) + (B, H, L_q, L_k) → (B, H, L_q, L_k).
            new_mask = attention_mask + bias

            # ── Causal-mask safety: re-mask after the additive bias ──────────
            # Vision-vision pairs are unmasked (mask = 0) so bias is added in
            # full. Text→future-text pairs are masked with a large negative
            # value, and our bias is supposed to be 0 there (the vision-vision
            # sub-block excludes them). But if bias *ever* leaks a positive
            # value into a masked entry — bf16 noise, a runaway training step,
            # an off-by-one in vision_mask, etc. — the additive sum could
            # creep above zero and silently break causality / leak padding.
            # Threshold -1e4 is comfortably above any value that's "real
            # attention" (post-softmax-pre-bias scores are O(1)) and well
            # below any HF-style mask value (-1e4, -65504, -inf).
            is_masked = attention_mask < -1e4
            new_mask = torch.where(is_masked, attention_mask, new_mask)
        else:
            new_mask = bias

        # Dtype safety: PyTorch SDPA requires attn_mask.dtype == query.dtype.
        # Q comes from q_proj(hidden_states) so its dtype tracks hidden_states.
        # Our bias_module.mlp is fresh-created in fp32 (it's added AFTER the
        # backbone .to(bfloat16) cast), and the upstream attention_mask may
        # also be fp32 in some HF code paths. Cast new_mask explicitly here.
        new_mask = new_mask.to(hidden_states.dtype)

        return self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=new_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Post-LoRA attention patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_attention_layers_spatial(model: nn.Module) -> int:
    """
    Replace standard-attention self_attn modules in the language model with
    SpatialAttnWrapper. **Linear-attention layers are skipped** — their kernel
    does not consume the additive `attention_mask`, so any bias passed through
    wrapper would be silently dropped and its parameters would never receive
    gradients (causing DDP `find_unused_parameters` failures).

    Discrimination is by ``decoder_layer.layer_type``: layers tagged
    ``"linear_attention"`` are bypassed; everything else is wrapped.

    Call AFTER get_peft_model() so LoRA adapters inside q_proj/k_proj/v_proj/o_proj
    are already in place — they get carried along inside the wrapper.

    Returns:
        int — number of layers wrapped (== number of standard-attn layers).
    """

    def _find_language_model(root: nn.Module) -> nn.Module | None:
        queue = [root]
        seen: set[int] = set()
        while queue:
            cur = queue.pop(0)
            if id(cur) in seen:
                continue
            seen.add(id(cur))
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
            "Expected SpatialAttnVanillaForConditionalGeneration or a wrapper around it."
        )

    num_heads = lm.config.num_attention_heads

    n_wrapped = 0
    n_skipped_linear = 0
    for layer in lm.layers:
        # Qwen3_5DecoderLayer puts the token-mixer at different attribute
        # names depending on layer_type:
        #   layer_type == "linear_attention" → layer.linear_attn  (DeltaNet)
        #   layer_type == "full_attention"   → layer.self_attn    (SDPA)
        # We only patch full-attn layers; linear-attn ones don't even have
        # `self_attn` so checking layer_type FIRST keeps the skip-count log
        # accurate and the intent explicit.
        if getattr(layer, "layer_type", None) == "linear_attention":
            n_skipped_linear += 1
            continue
        if not hasattr(layer, "self_attn"):
            continue
        if isinstance(layer.self_attn, SpatialAttnWrapper):
            continue
        layer.self_attn = SpatialAttnWrapper(
            layer.self_attn, num_heads=num_heads,
        )
        n_wrapped += 1

    if n_wrapped == 0:
        raise RuntimeError(
            "patch_attention_layers_spatial: no self_attn modules wrapped."
        )
    if n_skipped_linear:
        import logging
        logging.getLogger(__name__).info(
            f"patch_attention_layers_spatial: wrapped {n_wrapped} standard-attn "
            f"layers, skipped {n_skipped_linear} linear-attn layers (their "
            f"kernels ignore the additive attention_mask)."
        )
    return n_wrapped


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model containers — Qwen3.5 backbone + spatial cache routing
# ─────────────────────────────────────────────────────────────────────────────
# Spatial info reaches the wrapper via _spatial_cache, populated inside
# Model.forward (no get_rope_index override — Qwen's get_rope_index runs
# unchanged and produces 3-row position_ids exactly as in stock Qwen3.5).
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAttnVanillaTextModel(Qwen3_5TextModel):
    """
    Vanilla Qwen3_5TextModel + spatial cache injection to layers.

    Identical to Qwen3_5TextModel.forward except for one extra step right
    before the decoder loop: read self._spatial_cache (set by the parent
    SpatialAttnVanillaModel.forward) and add (flat_xyz, vision_mask) to the
    kwargs that go into each layer's self_attn.
    """

    def __init__(self, config):
        super().__init__(config)
        self._spatial_cache = None  # (flat_xyz, vision_mask) | None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        # ── Replicate Qwen3_5TextModel.forward setup verbatim ─────────────
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

        # mrope: 4 = (text_seq, t, h, w) — Qwen original 3D M-RoPE
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                4, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
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

        # ── Prefix-mask hole: make V↔V attention bidirectional ───────────────
        # Stock `create_causal_mask` is strict lower-triangular: a vision query
        # at position i can attend to vision keys j only if j ≤ i. That's a
        # waste — vision tokens have no autoregressive structure among
        # themselves, and our SpatialAttentionBias was designed assuming a full
        # N×N geometric field. Without this hole, bias entries on the upper
        # triangle of the V×V sub-block get drowned by the -inf mask (and the
        # downstream torch.where in SpatialAttnWrapper actively pins them back
        # to -inf), so half the geometric signal would be silently discarded.
        #
        # Surgery: zero out causal_mask at every (i, j) where BOTH positions
        # are vision tokens. Text → text causality is untouched, V → T and
        # T → V cells stay as-is (still respect their original -inf if any),
        # and the -inf wall protecting future text tokens remains intact.
        # Decode short-circuit: only run this in prefill — during incremental
        # decode the query is a single text token, so V↔V bidirectionality
        # is irrelevant (and vision_mask wouldn't even cover the new query).
        if (self._spatial_cache is not None
                and causal_mask is not None
                and (past_key_values is None
                     or past_key_values.get_seq_length() == 0)):
            _, vision_mask = self._spatial_cache               # (B, L) bool
            # Build (B, 1, L, L) bool: True iff query i AND key j are vision.
            #   .unsqueeze(1).unsqueeze(3) → (B, 1, L, 1)  — query (row) axis
            #   .unsqueeze(1).unsqueeze(2) → (B, 1, 1, L)  — key   (col) axis
            is_vv_pair = (
                vision_mask.unsqueeze(1).unsqueeze(3)
                & vision_mask.unsqueeze(1).unsqueeze(2)
            )
            causal_mask = torch.where(
                is_vv_pair,
                torch.zeros((), dtype=causal_mask.dtype, device=causal_mask.device),
                causal_mask,
            )
        # ─────────────────────────────────────────────────────────────────────

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ── Spatial cache → extra self_attn kwargs (only diff from parent) ──
        extra_attn_kwargs: dict = {}
        cache = self._spatial_cache
        is_prefill = (
            past_key_values is None
            or past_key_values.get_seq_length() == 0
        )
        if cache is not None and is_prefill:
            flat_xyz, vision_mask = cache
            extra_attn_kwargs["flat_xyz"] = flat_xyz
            extra_attn_kwargs["vision_mask"] = vision_mask
        # ────────────────────────────────────────────────────────────────────

        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
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
                    **extra_attn_kwargs,
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
                    **extra_attn_kwargs,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class SpatialAttnVanillaModel(Qwen3_5Model):
    """
    Qwen3_5Model with image_xyz → _spatial_cache routing. RoPE untouched.

    Builds (flat_xyz, vision_mask) inline in forward (no get_rope_index
    override — Qwen's 3D M-RoPE handles position_ids itself).
    """

    def __init__(self, config):
        # Bypass Qwen3_5Model.__init__ to swap in our text model; replicate
        # the rest of its setup (visual + rope_deltas).
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = SpatialAttnVanillaTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def forward(
        self,
        *args,
        image_xyz=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        # Build _spatial_cache before the parent forward — the cache is
        # consumed by SpatialAttnVanillaTextModel.forward downstream.
        #
        # Input shape conventions for `image_xyz`:
        #   • B == 1 shorthand : list[Tensor]            — per-image xyz tensors
        #                        for the single sample.  Old default-collate
        #                        path produces this form.
        #   • B  ≥ 1 explicit  : list[list[Tensor]]      — outer length == B,
        #                        each inner list is per-image xyz tensors for
        #                        that sample.  Use this when collate_fn batches
        #                        multiple samples per device.
        # We auto-detect by peeking at image_xyz[0] and normalize to the
        # list[list[Tensor]] form. Different samples have different vision-
        # token counts → pad_sequence to (B, N_max, 3); samples with N_b<N_max
        # are zero-padded and the per-sample valid count comes from vision_mask.
        if image_xyz is not None and mm_token_type_ids is not None:
            if len(image_xyz) > 0 and isinstance(image_xyz[0], torch.Tensor):
                # Single-sample shorthand → wrap into outer list of length 1.
                image_xyz = [image_xyz]

            # Per-sample flatten: each sample's per-image xyz tensors → (N_b, 3)
            per_sample = [
                torch.cat([x.reshape(-1, 3) for x in sample], dim=0)
                for sample in image_xyz
            ]                                                       # list[(N_b, 3)]

            # Pad to a single tensor (B, N_max, 3). Padded entries are zero —
            # they get a non-trivial pairwise feature downstream but are never
            # scattered into the final bias (see SpatialAttentionBias.forward
            # which slices [:N_b, :N_b] per sample), so they cost FLOPs only
            # and contribute zero gradient.
            flat_xyz = torch.nn.utils.rnn.pad_sequence(
                per_sample, batch_first=True, padding_value=0.0,
            )                                                       # (B, N_max, 3)

            vision_mask = (mm_token_type_ids != 0)                  # (B, L)

            B = flat_xyz.shape[0]
            assert vision_mask.shape[0] == B, (
                f"Batch mismatch: vision_mask has {vision_mask.shape[0]} rows "
                f"but image_xyz provides {B} samples"
            )
            # Per-sample length sanity (.tolist() is one sync for all B rows).
            xyz_lens  = [t.shape[0] for t in per_sample]
            mask_lens = vision_mask.sum(dim=1).tolist()
            for b, (n_xyz, n_true) in enumerate(zip(xyz_lens, mask_lens)):
                assert n_xyz == n_true, (
                    f"Sample {b}: image_xyz has {n_xyz} positions but "
                    f"vision_mask has {n_true} True entries"
                )

            self.language_model._spatial_cache = (flat_xyz, vision_mask)
        else:
            self.language_model._spatial_cache = None

        return super().forward(
            *args,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )


# NOTE: there is no top-level SpatialAttn{...}ForConditionalGeneration class.
# To use this stack, load a stock Qwen3_5ForConditionalGeneration and swap its
# inner ``.model`` for SpatialAttnVanillaModel:
#
#     from transformers.models.qwen3_5.modeling_qwen3_5 import (
#         Qwen3_5ForConditionalGeneration,
#     )
#     spa = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, ...)
#     new_inner = SpatialAttnVanillaModel(spa.config)
#     new_inner.load_state_dict(spa.model.state_dict(), strict=True)
#     spa.model = new_inner
#     spa.tie_weights()                          # re-tie lm_head ↔ embed_tokens
#     patch_attention_layers_spatial(spa)        # install the per-layer bias
#
# See train_atten.py / build_model() for the full setup.
