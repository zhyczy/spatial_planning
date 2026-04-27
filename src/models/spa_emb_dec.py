"""
SPA Decoupled Position Embedding — extra XYZ RoPE in the pass-through region.

Background (see md/findings/qwen35_rope_decoupling.md):
    Qwen3.5 splits each head_dim=256 into:
      - rotary  64 dims (25%, partial_rotary_factor=0.25): position-carrying
      - pass-through 192 dims (75%): content-only, never rotated

This module ADDS a second RoPE (XYZ-only) on top of the original, by carving
66 dims out of the 192 pass-through region:

    head_dim layout (256 dims):
    ┌───────── original RoPE 64 ─────────┐┌──── new XYZ RoPE 66 ────┐┌─ pass-through 126 ─┐
    │ Qwen 3D M-RoPE [11, 11, 10] (UNCHG) ││ x: 22d  y: 22d  z: 22d  ││ untouched           │
    │ dims 0..63                          ││ dims 64..129            ││ dims 130..255       │
    └─────────────────────────────────────┘└─────────────────────────┘└─────────────────────┘

The new XYZ RoPE:
    - x / y / z share the SAME 11-band inv_freq spectrum (symmetric across axes)
    - rope_theta = 10000  (default; runtime coord_scale is configurable)
    - coord_scale = 100.0 default → matches the existing 4D / coordinate pipeline
    - With cs=100, θ=10000: per-axis wavelength range 0.063m .. 272m
          covers cm-precision to ~300m extents (≈ 4330× spectrum span)
    - For text tokens, xyz = (0, 0, 0) → cos=1, sin=0 → identity rotation
    - For image patch tokens, xyz = per-patch scene coords (same as image_xyz)

Architecture
────────────
SpaXYZRotaryEmbedding(nn.Module)
    Builds (xyz_cos, xyz_sin) of shape (B, S, xyz_dim=66) from per-token
    (B, S, 3) xyz tensor.  Uses split-half pairing within the 66-dim slice.

SpaDecAttentionWrapper(nn.Module)
    Wraps a Qwen3_5Attention.  Replicates its forward but inserts an extra
    rotation on dims [64..129] of Q and K using xyz_cos / xyz_sin.

SpaDecTextModel(Qwen3_5TextModel)
    - Reads self._xyz_pos (set by SpaDecModel before forward).
    - Computes xyz_position_embeddings via self.xyz_rotary_emb.
    - Forwards them as an extra kwarg to every decoder layer.

SpaDecModel(Qwen3_5Model)
    Computes per-token xyz tensor in forward() and stashes it on
    self.language_model._xyz_pos.  Original mrope_section (Qwen [11,11,10])
    is left untouched.

SpaDecForConditionalGeneration(Qwen3_5ForConditionalGeneration)
    Top-level wrapper.  Routes image_xyz into SpaDecModel.

patch_attention_layers_dec(model)
    Replaces every self_attn with SpaDecAttentionWrapper.  Call AFTER LoRA.
"""

import itertools
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5Model,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5DynamicCache,
    Qwen3_5ModelOutputWithPast,
    apply_rotary_pos_emb,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.masking_utils import create_causal_mask


# ─────────────────────────────────────────────────────────────────────────────
# 1. XYZ Rotary Embedding (sequential x|y|z layout, rope_theta=1000)
# ─────────────────────────────────────────────────────────────────────────────

class SpaXYZRotaryEmbedding(nn.Module):
    """
    Symmetric 3-axis RoPE for ``xyz_dim`` dims (default 66 = 33 bands × 2).

    x, y, z share the SAME frequency spectrum (``inv_freq_axis``), rather than
    carving 33 sequential bands across the three axes.  This means each axis
    spans the full freq range from short- to long-wavelength bands.

    Per-axis layout (n_per_axis = xyz_dim // 6 = 11 for xyz_dim=66):
        x bands: inv_freq_axis[0 .. n_per_axis-1]
        y bands: inv_freq_axis[0 .. n_per_axis-1]   # same as x
        z bands: inv_freq_axis[0 .. n_per_axis-1]   # same as x

    inv_freq formula (computed over the per-axis slice of size 2·n_per_axis):
        per_axis_dim = 2 * n_per_axis
        inv_freq_axis[k] = 1.0 / rope_theta ** (2k / per_axis_dim),
                           k = 0 .. n_per_axis - 1

    Output cos/sin is duplicated split-half (Qwen convention) so that
    apply_rotary_pos_emb's split-half pairing works within the 66-dim slice.
    """

    def __init__(
        self,
        xyz_dim:             int   = 66,
        rope_theta:          float = 10000.0,
        default_coord_scale: float = 100.0,
    ):
        super().__init__()
        if xyz_dim % 6 != 0:
            raise ValueError(
                f"xyz_dim must be divisible by 6 (got {xyz_dim}); "
                "x, y, z each need an even number of dims for split-half pairing."
            )
        self.xyz_dim             = xyz_dim                # 66
        self.n_bands             = xyz_dim // 2           # 33
        self.n_per_axis          = self.n_bands // 3      # 11
        self.rope_theta          = rope_theta
        self.default_coord_scale = default_coord_scale

        # Per-axis inv_freq: 11 distinct frequencies, shared by x / y / z so all
        # three axes see an identical, symmetric spectrum.
        #   per_axis_dim = 22  (the "head dim equivalent" for one axis)
        #   inv_freq_axis[k] = 1 / rope_theta ** (2k / 22),  k = 0..10
        per_axis_dim = 2 * self.n_per_axis
        inv_freq_axis = 1.0 / (
            rope_theta ** (
                torch.arange(0, per_axis_dim, 2, dtype=torch.float32) / per_axis_dim
            )
        )
        self.register_buffer("inv_freq_axis", inv_freq_axis, persistent=False)  # (11,)

    # NOTE: no @torch.no_grad() — train_alternate.py's decouple path needs the
    # gradient w.r.t. xyz to flow back to the rotation matrix R (and thence
    # rotation_enc).  Other call sites (train_correspondence.py) feed xyz that
    # has no grad anyway, so removing the decorator is a no-op for them.
    def forward(
        self,
        xyz:         torch.Tensor,
        coord_scale: Optional[float] = None,
        polar:       bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz:         (batch, seq_len, 3) — per-token (x, y, z) in scene coords.
            coord_scale: runtime scale override; falls back to default_coord_scale.
            polar:       if True, convert Cartesian (x, y, z) → log-spherical
                         (log r, θ=atan2(y,x), α=atan2(√(x²+y²), z)) before
                         scaling. Matches xyz_to_polar convention in the dataset.
                         Text-token patches (xyz=0) stay at (0, 0, 0) ⇒ identity
                         rotation, so the fallback is preserved.
        Returns:
            cos, sin: (batch, seq_len, xyz_dim) for SpaDecAttentionWrapper.
        """
        if coord_scale is None:
            coord_scale = self.default_coord_scale

        if polar:
            # Cartesian → log-spherical; zero-xyz rows (text tokens) stay zero.
            x = xyz[..., 0]
            y = xyz[..., 1]
            z = xyz[..., 2]
            r = torch.sqrt(x * x + y * y + z * z)
            zero_mask = r == 0
            r_safe = r.clamp(min=1e-8)
            log_r = torch.log(r_safe)
            theta = torch.atan2(y, x)                                      # [-π, π]
            alpha = torch.atan2(torch.sqrt(x * x + y * y), z)              # [0, π]
            log_r = torch.where(zero_mask, torch.zeros_like(log_r), log_r)
            theta = torch.where(zero_mask, torch.zeros_like(theta), theta)
            alpha = torch.where(zero_mask, torch.zeros_like(alpha), alpha)
            xyz = torch.stack([log_r, theta, alpha], dim=-1)

        xyz = xyz.float() * float(coord_scale)                            # (B, S, 3)

        # All three axes use the SAME inv_freq spectrum → symmetric.
        inv = self.inv_freq_axis[None, None, :]                           # (1, 1, n_per_axis)
        x_freqs = xyz[..., 0:1] * inv                                     # (B, S, n_per_axis)
        y_freqs = xyz[..., 1:2] * inv                                     # (B, S, n_per_axis)
        z_freqs = xyz[..., 2:3] * inv                                     # (B, S, n_per_axis)

        freqs = torch.cat([x_freqs, y_freqs, z_freqs], dim=-1)            # (B, S, 33)
        emb   = torch.cat([freqs, freqs], dim=-1)                         # (B, S, 66)
        return emb.cos(), emb.sin()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: apply rotation to dims [offset : offset + xyz_dim] of q/k
# ─────────────────────────────────────────────────────────────────────────────

def _apply_xyz_rotary(
    q:        torch.Tensor,
    k:        torch.Tensor,
    xyz_cos:  torch.Tensor,
    xyz_sin:  torch.Tensor,
    offset:   int,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to dims [offset : offset + xyz_dim] of q and k.
    Split-half pairing is local to the xyz slice (independent of the original
    rotary region at dims [0..offset]).

    Args:
        q, k:    (..., n_heads, seq_len, head_dim)
        xyz_cos, xyz_sin: (batch, seq_len, xyz_dim)
        offset:  starting dim index of the xyz slice (= rotary_dim)
    Returns:
        rotated q, k with same shape.
    """
    xyz_cos = xyz_cos.unsqueeze(unsqueeze_dim)                    # broadcast over heads
    xyz_sin = xyz_sin.unsqueeze(unsqueeze_dim)
    xyz_dim = xyz_cos.shape[-1]
    end = offset + xyz_dim

    q_xyz = q[..., offset:end]
    k_xyz = k[..., offset:end]

    q_rot = (q_xyz * xyz_cos) + (rotate_half(q_xyz) * xyz_sin)
    k_rot = (k_xyz * xyz_cos) + (rotate_half(k_xyz) * xyz_sin)

    q_out = torch.cat([q[..., :offset], q_rot, q[..., end:]], dim=-1)
    k_out = torch.cat([k[..., :offset], k_rot, k[..., end:]], dim=-1)
    return q_out, k_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention wrapper — adds XYZ rotation on top of the standard one
# ─────────────────────────────────────────────────────────────────────────────

class SpaDecAttentionWrapper(nn.Module):
    """
    Wraps Qwen3_5Attention to apply XYZ RoPE on dims [xyz_offset : xyz_offset + xyz_dim]
    AFTER the standard RoPE has been applied to dims [0 : xyz_offset].

    Falls back to vanilla attention when ``xyz_position_embeddings`` is None.
    """

    def __init__(self, attn: nn.Module, xyz_offset: int = 64):
        super().__init__()
        self.attn       = attn
        self.xyz_offset = xyz_offset

    def forward(
        self,
        hidden_states:        torch.Tensor,
        position_embeddings:  tuple,                           # (cos, sin) for original RoPE
        attention_mask:       Optional[torch.Tensor] = None,
        past_key_values:      Optional[object]      = None,
        cache_position:       Optional[torch.Tensor] = None,
        # extra kwargs injected by SpaDecTextModel ────────────────────────────
        xyz_position_embeddings: Optional[tuple] = None,       # (xyz_cos, xyz_sin)
        **kwargs,
    ):
        # Fall back to standard attention when no XYZ embeddings supplied
        if xyz_position_embeddings is None:
            return self.attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # ── Replicate Qwen3_5Attention.forward, with extra XYZ rotation ──────
        attn = self.attn
        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states, gate = torch.chunk(
            attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states   = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Standard RoPE on dims [0 : xyz_offset]
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # NEW: XYZ RoPE on dims [xyz_offset : xyz_offset + xyz_dim]
        xyz_cos, xyz_sin = xyz_position_embeddings
        query_states, key_states = _apply_xyz_rotary(
            query_states, key_states, xyz_cos, xyz_sin, offset=self.xyz_offset,
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            attn, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 4. Text model — computes xyz_position_embeddings, routes to decoder layers
# ─────────────────────────────────────────────────────────────────────────────

class SpaDecTextModel(Qwen3_5TextModel):
    """
    Qwen3_5TextModel + an extra XYZ RoPE channel.

    Reads self._xyz_pos (set by SpaDecModel.forward before this method runs)
    and forwards xyz_position_embeddings to every decoder layer.

    During decode (past_key_values is not None and seq_len doesn't match the
    cached _xyz_pos), new tokens default to xyz=(0,0,0) — identity rotation.
    """

    def __init__(self, config):
        super().__init__(config)
        # Defaults; runtime coord_scale flows through SpaDecModel → ._coord_scale.
        self.xyz_rotary_emb = SpaXYZRotaryEmbedding(
            xyz_dim             = 66,
            rope_theta          = 10000.0,
            default_coord_scale = 100.0,
        )
        self.xyz_offset:   int   = 64       # dims 64..129 → XYZ RoPE
        self._xyz_pos            = None     # set per forward by SpaDecModel
        self._coord_scale: float = 100.0    # set per forward by SpaDecModel
        self._polar:       bool  = False    # set per forward by SpaDecModel

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
        # ── Standard setup (mirrors Qwen3_5TextModel.forward) ────────────────
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

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3, position_ids.shape[0], -1
            )

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=None,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ── Resolve xyz_pos for THIS forward step ────────────────────────────
        bs, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        xyz_pos = self._xyz_pos
        if xyz_pos is None or xyz_pos.shape[0] != bs or xyz_pos.shape[1] != seq_len:
            # Decode step (or no xyz_pos set): default to zeros (identity rotation)
            xyz_pos = torch.zeros(
                bs, seq_len, 3,
                dtype=torch.float32, device=inputs_embeds.device,
            )

        xyz_position_embeddings = self.xyz_rotary_emb(
            xyz_pos, coord_scale=self._coord_scale, polar=self._polar,
        )
        # Cast to match hidden_states dtype (bfloat16 typically)
        xyz_position_embeddings = (
            xyz_position_embeddings[0].to(hidden_states.dtype),
            xyz_position_embeddings[1].to(hidden_states.dtype),
        )
        extra_decoder_kwargs = {"xyz_position_embeddings": xyz_position_embeddings}

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
# 5. Backbone — computes per-token xyz, stashes it on the text model
# ─────────────────────────────────────────────────────────────────────────────

class SpaDecModel(Qwen3_5Model):
    """
    Qwen3.5 backbone with the original 3D M-RoPE preserved AND an extra XYZ
    channel computed per token (text → (0,0,0); image patches → image_xyz).

    The original mrope_section ([11, 11, 10] from Qwen pretraining) is NOT
    modified — the first 64 head_dim units retain Qwen's pretrained behaviour.
    """

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual         = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = SpaDecTextModel._from_config(config.text_config)
        self.rope_deltas    = None
        self.post_init()

    # ── helper: build (B, S, 3) xyz tensor for this batch ────────────────────
    def _compute_xyz_pos(
        self,
        input_ids:         torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw:    Optional[torch.Tensor] = None,
        attention_mask:    Optional[torch.Tensor] = None,
        image_xyz:         Optional[list]         = None,
    ) -> torch.Tensor:
        """
        Returns (batch, seq_len, 3) float tensor.
            - text tokens        → (0, 0, 0)
            - image patch tokens → corresponding entry of image_xyz
        """
        bs, seq_len = input_ids.shape
        xyz_pos = torch.zeros(
            bs, seq_len, 3, dtype=torch.float32, device=input_ids.device,
        )
        if image_xyz is None or image_grid_thw is None:
            return xyz_pos

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        grid_iter = iter(image_grid_thw)
        xyz_iter  = iter(image_xyz)

        for batch_idx in range(bs):
            mm_types = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                valid_idx = attention_mask[batch_idx].bool().nonzero(as_tuple=True)[0]
                mm_types_valid = mm_types[valid_idx].tolist()
            else:
                valid_idx = torch.arange(seq_len, device=input_ids.device)
                mm_types_valid = mm_types.tolist()

            cursor = 0
            for key, grp in itertools.groupby(
                enumerate(mm_types_valid), lambda x: x[1]
            ):
                grp_list = list(grp)
                grp_len  = len(grp_list)
                if key != 0:                                           # image / video
                    try:
                        grid_thw   = next(grid_iter)
                        xyz_coords = next(xyz_iter)                    # (llm_H, llm_W, 3)
                    except StopIteration:
                        cursor += grp_len
                        continue

                    n_t      = grid_thw[0].item()
                    xyz_flat = xyz_coords.reshape(-1, 3).to(xyz_pos.device)
                    if n_t > 1:
                        xyz_flat = xyz_flat.repeat(n_t, 1)             # (n_t * H * W, 3)

                    n_to_copy  = min(grp_len, xyz_flat.shape[0])
                    target_idx = valid_idx[cursor : cursor + n_to_copy]
                    xyz_pos[batch_idx, target_idx] = xyz_flat[:n_to_copy]

                cursor += grp_len

        return xyz_pos

    # ── forward: stash xyz_pos on the text model, then run normally ──────────
    def forward(
        self,
        input_ids:         Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        image_grid_thw:    Optional[torch.Tensor] = None,
        attention_mask:    Optional[torch.Tensor] = None,
        image_xyz:         Optional[list]         = None,
        coord_scale:       float                  = 100.0,
        polar:             bool                   = False,
        **kwargs,
    ):
        # Compute per-token xyz BEFORE the parent forward (which calls
        # self.language_model.forward, which reads _xyz_pos / _coord_scale).
        # When mm_token_type_ids is None (generate() loop: HF doesn't forward
        # custom kwargs through prepare_inputs_for_generation), leave _xyz_pos
        # as-is so a caller-set prefill tensor is preserved; SpaDecTextModel
        # falls back to zeros when the cached shape doesn't match seq_len.
        if input_ids is not None and mm_token_type_ids is not None:
            self.language_model._xyz_pos = self._compute_xyz_pos(
                input_ids         = input_ids,
                mm_token_type_ids = mm_token_type_ids,
                image_grid_thw    = image_grid_thw,
                attention_mask    = attention_mask,
                image_xyz         = image_xyz,
            )
        self.language_model._coord_scale = float(coord_scale)
        self.language_model._polar       = bool(polar)

        return super().forward(
            input_ids         = input_ids,
            mm_token_type_ids = mm_token_type_ids,
            image_grid_thw    = image_grid_thw,
            attention_mask    = attention_mask,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class SpaDecForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """
    Top-level model with decoupled XYZ RoPE in the pass-through region.

    Usage:
        model = SpaDecForConditionalGeneration.from_pretrained(base_path, ...)
        patch_attention_layers_dec(model)        # call AFTER LoRA wrapping
        out = model(
            input_ids=..., pixel_values=..., image_grid_thw=...,
            image_xyz=[xyz_img0, xyz_img1, ...], # list of (llm_H, llm_W, 3)
        )
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = SpaDecModel(config)

    def forward(
        self,
        *args,
        image_xyz:   Optional[list] = None,
        coord_scale: float          = 100.0,
        polar:       bool           = False,
        **kwargs,
    ):
        if image_xyz is not None:
            kwargs["image_xyz"] = image_xyz
        kwargs["coord_scale"] = float(coord_scale)
        kwargs["polar"]       = bool(polar)
        return super().forward(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Post-LoRA attention patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_attention_layers_dec(
    model:      nn.Module,
    xyz_offset: int = 64,
) -> int:
    """
    Replace every self_attn module in the language model's decoder layers with
    a SpaDecAttentionWrapper.  Call AFTER get_peft_model() so LoRA adapters are
    already inside q_proj / k_proj / v_proj / o_proj.

    Returns the number of attention modules wrapped.
    """
    def _find_language_model(root: nn.Module) -> Optional[nn.Module]:
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
            "Cannot find language_model on the supplied model. "
            "Expected SpaDecForConditionalGeneration or a wrapper around it."
        )

    n_wrapped = 0
    for layer in lm.layers:
        if (hasattr(layer, "self_attn")
                and not isinstance(layer.self_attn, SpaDecAttentionWrapper)):
            layer.self_attn = SpaDecAttentionWrapper(
                layer.self_attn, xyz_offset=xyz_offset,
            )
            n_wrapped += 1

    if n_wrapped == 0:
        raise RuntimeError(
            "patch_attention_layers_dec: no self_attn modules wrapped."
        )
    return n_wrapped
