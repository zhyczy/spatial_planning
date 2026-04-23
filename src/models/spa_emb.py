"""
SPA Model — Qwen3.5 with customizable positional embeddings.

Subclass hierarchy:
  SpaVisionModel(Qwen3_5VisionModel)
      └─ rot_pos_emb()              ← ViT internal 2D RoPE (row, col coords per patch)
      └─ fast_pos_embed_interpolate() ← ViT absolute 2D pos embed (additive, interpolated)

  SpaModel(Qwen3_5Model)
      └─ get_vision_position_ids()  ← LLM-side visual token 4D coords (t, x, y, z)
      └─ get_rope_index()           ← full position_ids assembly for the LLM sequence

  SpaForConditionalGeneration(Qwen3_5ForConditionalGeneration)
      └─ top-level model, wires everything together

Coordinate scheme for visual tokens:
    t — start_position in the LLM sequence when this image is encountered
        (same semantics as the original T dimension in M-RoPE; shared across
        all patches of one image)
    x, y, z — per-patch 3D scene coordinates (where each patch projects in the
               scene), discretized to integers via round(coord * coord_scale)

    All patch tokens within one image share the same (t, x, y, z).
    Within-image spatial structure is handled by the ViT's internal 2D RoPE.

    For text tokens all four dims equal the sequential position index
    (same behaviour as original 3D M-RoPE for text).

NOTE on 4D M-RoPE application:
    position_ids now has shape (4, batch, seq_len).  To apply this correctly
    inside attention the model config's rope_scaling.mrope_section must be
    updated from 3 equal parts to 4 equal parts of head_dim // 2.
    Example — if head_dim=128: mrope_section=[16, 16, 16, 16] instead of
    the default [21, 21, 22] (or similar).  Set this before loading:
        config.text_config.rope_scaling["mrope_section"] = [16, 16, 16, 16]

To use:
    model = SpaForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-VL-7B-Instruct")
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5Model,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5DynamicCache,
    Qwen3_5ModelOutputWithPast,
)
from transformers.masking_utils import create_causal_mask
from transformers.utils.generic import maybe_autocast


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Vision Encoder (ViT) positional embeddings
# ─────────────────────────────────────────────────────────────────────────────

class SpaVisionModel(Qwen3_5VisionModel):
    """
    Qwen3.5 ViT with overridable positional embeddings.

    The ViT applies two positional encodings per forward pass:
        1. fast_pos_embed_interpolate()  → additive absolute 2D pos embed
        2. rot_pos_emb()                 → 2D RoPE applied inside each attention block

    Override either (or both) below.
    """

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D rotary position embeddings for ViT attention.

        For each patch token, computes (row, col) coordinates in full resolution,
        then looks up the rotary frequency table.

        Args:
            grid_thw: (num_images, 3) — (T, H, W) for each image/video after patch embed.
                      H and W are in units of patches (not pixels).

        Returns:
            embeddings: (total_tokens, head_dim)
                head_dim = hidden_size // num_heads
                The (cos, sin) are derived from this in forward() as:
                    emb = cat(embeddings, embeddings, dim=-1)
                    position_embeddings = (emb.cos(), emb.sin())

        Coordinate layout (original):
            For a single image with T=1, H=4, W=4, merge_size=2:
              merged_h=2, merged_w=2 (2x2 blocks of 2x2 patches)
              Full-resolution row coords: 0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3
              Full-resolution col coords: 0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Use 1D sequential coords instead of 2D (ablation)
          - Add temporal dimension to the ViT RoPE
          - Use a different coordinate ordering
        ──────────────────────────────────────────────────────────────────────
        """
        # ── Original implementation ──────────────────────────────────────────
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, head_dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Full-resolution (row, col) for every patch token inside each block
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)  # (h*w, 2)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)  # reuse spatial coords across frames

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # (total_tokens, 2, head_dim//2)
        embeddings = embeddings.flatten(1)  # (total_tokens, head_dim)
        return embeddings
        # ── End original ──────────────────────────────────────────────────────

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute additive absolute 2D positional embeddings for ViT patches.

        Uses bilinear interpolation to adapt the learned pos_embed table
        (fixed grid of num_grid_per_side x num_grid_per_side) to arbitrary input resolution.

        Args:
            grid_thw: (num_images, 3) — (T, H, W) in patch units.

        Returns:
            patch_pos_embeds: (total_tokens_after_merge, hidden_size)
                Added to hidden_states before ViT blocks:
                    hidden_states = hidden_states + patch_pos_embeds

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Replace with sinusoidal 2D absolute pos embed (no learned params)
          - Use different interpolation (nearest, bicubic)
          - Disable entirely (ablation: set to zeros)
        ──────────────────────────────────────────────────────────────────────
        """
        # ── Original implementation ──────────────────────────────────────────
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # Four corners for bilinear interpolation
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),       # top-left
                (base_h[None].T + w_idxs_ceil[None]).flatten(),        # top-right
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),  # bottom-left
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),   # bottom-right
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
        # ── End original ──────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Part 2a: 4D sequential M-RoPE (text model rotary embedding + language model)
# ─────────────────────────────────────────────────────────────────────────────

class SpaTextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """
    Extends Qwen3.5 M-RoPE from 3D to N-D using sequential (non-interleaved) frequency assignment.

    With mrope_section = [s0, s1, ..., s_{N-1}] (sum = head_dim // 2):
        dimensions laid out sequentially: [dim_0_freqs | dim_1_freqs | ...]

    For [8, 8, 8, 8] with N=4 and head_dim=64 (32 freq components):
        dim 0 (t):   indices 0-7    (8 freqs)
        dim 1 (x3d): indices 8-15   (8 freqs)
        dim 2 (y3d): indices 16-23  (8 freqs)
        dim 3 (z3d): indices 24-31  (8 freqs)

    Text tokens: all N dims = sequential pos → all 32 freqs encode the same pos
    → mathematically identical to original [11,11,10] 3D text RoPE.
    """

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (N, bs, seq_len) where N = len(mrope_section)
        num_dims = len(self.mrope_section)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(num_dims, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(num_dims, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (N, bs, 1, seq_len)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            # freqs: (N, bs, seq_len, head_dim//2)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        Build per-band cos/sin from N position-axis frequency stacks.

        Two layouts are supported, switched by `self.visual_interleave`:

        1) Sequential (default): t|x|y|z laid out contiguously, section-sized blocks.
             band:  [0 .. s0-1 | s0 .. s0+s1-1 | ...]
             dim:   [   0      |       1       | ...]
           This is the original Qwen layout.

        2) Visual-interleave (`self.visual_interleave = True`): t keeps its s0
           bands at the high-freq end; dims 1..N-1 (e.g. x, y, z) round-robin
           through the remaining bands so each spans the full freq range.
             band:  [0 .. s0-1 | s0 s0+1 s0+2  s0+3 s0+4 s0+5  ...]
             dim:   [   0      |  1    2    3    1    2    3   ...]
           x/y/z each get ≈ (total - s0) / (N-1) bands covering high→low freq.
           Makes x/y/z symmetric under a single global scale.

        Args:
            freqs:        (N, bs, seq_len, head_dim//2)
            mrope_section: list of N ints
        Returns:
            (bs, seq_len, head_dim//2)
        """
        num_dims = len(mrope_section)
        n_bands = freqs.shape[-1]

        if getattr(self, "visual_interleave", False) and num_dims >= 2:
            # Cache band→dim layout per device
            if not hasattr(self, "_interleave_layout_cache"):
                self._interleave_layout_cache = {}
            cache_key = (n_bands, int(mrope_section[0]), num_dims, str(freqs.device))
            if cache_key not in self._interleave_layout_cache:
                layout = torch.empty(n_bands, dtype=torch.long, device=freqs.device)
                n_t = int(mrope_section[0])
                layout[:n_t] = 0  # t at high-freq end
                # Round-robin dims 1..N-1 for the remainder
                rest = torch.arange(n_bands - n_t, device=freqs.device)
                layout[n_t:] = 1 + (rest % (num_dims - 1))
                self._interleave_layout_cache[cache_key] = layout
            layout = self._interleave_layout_cache[cache_key]

            band_idx = torch.arange(n_bands, device=freqs.device)
            # Advanced indexing: pick freqs[layout[k], :, :, k] for each k
            out = freqs[layout, :, :, band_idx]  # (n_bands, bs, seq_len)
            return out.permute(1, 2, 0).contiguous()

        # Sequential layout (original behavior)
        freqs_out_list = []
        offset = 0
        for dim in range(num_dims):
            length = mrope_section[dim]
            freqs_out_list.append(freqs[dim, ..., offset:offset + length])
            offset += length
        freqs_out = torch.cat(freqs_out_list, dim=-1)
        return freqs_out


class SpaTextModel(Qwen3_5TextModel):
    """
    Qwen3.5 text model patched to support 5D position_ids (seq, t, x, y, z).

    The only change from the parent is that the position_ids split condition
    accepts shape[0] >= 4 instead of exactly 4, allowing 5D input where:
        position_ids[0]  = sequential position  (→ causal mask)
        position_ids[1:] = (t, x3d, y3d, z3d)  (→ 4D sequential RoPE)
    """

    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = SpaTextRotaryEmbedding(config=config)

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Accept 5D (seq, t, x3d, y3d, z3d) in addition to original 4D
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(5, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(5, position_ids.shape[0], -1)

        # ← only change from parent: >= 4 instead of == 4
        if position_ids.ndim == 3 and position_ids.shape[0] >= 4:
            # Dim 0 (seq) is float32 in our pipeline; cast to long for the
            # causal-mask utility (which expects integer positions).
            text_position_ids = position_ids[0].long()
            position_ids = position_ids[1:]        # (t, x3d, y3d, z3d) → rotary (float)
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
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
# Part 2: LLM visual token positional embeddings (M-RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class SpaModel(Qwen3_5Model):
    """
    Qwen3.5 multimodal backbone with 4D (t, x, y, z) M-RoPE for visual tokens.

    The LLM uses a 4D variant of M-RoPE where each token has coordinates:
        - Text tokens:  t = x = y = z = sequential 1D index
        - Image tokens: t = image index, (x, y, z) = 3D viewpoint coordinates

    The two entry points to modify:
        get_vision_position_ids()  — computes (t, x, y, z) for a single image
        get_rope_index()           — assembles 4D position_ids for the full sequence
    """

    def __init__(self, config):
        # Replicate Qwen3_5Model.__init__ but use SpaVisionModel and SpaTextModel
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = SpaVisionModel._from_config(config.vision_config)
        self.language_model = SpaTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        """Run frozen vision encoder without building a computation graph."""
        with torch.no_grad():
            out = super().get_image_features(pixel_values, image_grid_thw, **kwargs)
        # Detach so autograd does not trace back through the ViT
        if isinstance(out.pooler_output, (list, tuple)):
            out.pooler_output = [t.detach() for t in out.pooler_output]
        else:
            out.pooler_output = out.pooler_output.detach()
        if out.last_hidden_state is not None:
            out.last_hidden_state = out.last_hidden_state.detach()
        return out

    @staticmethod
    def _coord_scale_vec(coord_scale, device=None):
        """
        Normalize coord_scale (scalar | tuple/list/tensor of 3) into a (3,) float tensor.
        Allows per-axis scaling of (x, y, z) — useful because mrope_section assigns
        x/y/z to freq bands with very different inv_freq ranges.
        """
        if isinstance(coord_scale, (int, float)):
            return torch.tensor([float(coord_scale)] * 3, device=device, dtype=torch.float32)
        if isinstance(coord_scale, (tuple, list)):
            assert len(coord_scale) == 3, f"coord_scale list must have 3 entries, got {len(coord_scale)}"
            return torch.tensor([float(s) for s in coord_scale], device=device, dtype=torch.float32)
        if isinstance(coord_scale, torch.Tensor):
            t = coord_scale.to(device=device, dtype=torch.float32).reshape(-1)
            if t.numel() == 1:
                return t.expand(3).contiguous()
            assert t.numel() == 3, f"coord_scale tensor must have 1 or 3 elements, got {t.numel()}"
            return t
        raise TypeError(f"unsupported coord_scale type: {type(coord_scale)}")

    def get_vision_position_ids(
        self,
        start_position: int,
        xyz_coords: torch.Tensor,
        grid_thw,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        coord_scale=100.0,
        polar: bool = False,
        device=None,
    ) -> torch.LongTensor:
        """
        Compute 4D (t, x, y, z) position indices for all visual tokens of one image.

        Called once per image inside get_rope_index().

        Each patch token gets its own (x, y, z) — the 3D scene coordinates of that
        patch's projection.  The t dimension is shared across the whole image.

        Args:
            start_position: int
                The current position counter in the LLM sequence (same as original T).
                All tokens of this image share this value as their t coordinate.
                After this call, get_rope_index() advances start_position by
                max(H, W) // spatial_merge_size (not by token count).
            xyz_coords: (llm_H, llm_W, 3) float tensor
                Per-patch 3D scene coordinates after spatial merge, in row-major order.
                Shape must match llm_grid_h × llm_grid_w derived from grid_thw.
            grid_thw: (3,) — (T, H, W) in patch units after ViT merge.
            temp_merge_size: temporal downscale factor (usually 1 for images).
            spatial_merge_size: spatial downscale factor (from vision_config).
            coord_scale: multiplier applied to float xyz before rounding to int.
                Default 100 maps ±10 m → ±1000, which is a reasonable RoPE range.
            polar: if True, convert Cartesian (x, y, z) → log-spherical
                (log ρ, θ, α) first. Convention matches dataset xyz_to_polar:
                    log ρ = log(||xyz||)             — scale-invariant radius
                    θ     = atan2(y, x)  ∈ [-π, π]   — azimuth (raw radians)
                    α     = atan2(√(x²+y²), z) ∈ [0, π] — inclination (raw radians)
                RoPE positions are assigned as (no [0,1] normalization, so θ
                and α share the same angular resolution under one scale):
                    log ρ * coord_scale,
                    θ     * coord_scale,
                    α     * coord_scale.
            device: torch device.

        Returns:
            vision_position_ids: (5, llm_grid_t * llm_grid_h * llm_grid_w)
                [0] = seq   — sequential position in sequence (for causal mask)
                [1] = t     — start_position (same for every token in this image)
                [2] = x/logρ — per-patch discretized x (Cartesian) or log ρ (spherical)
                [3] = y/θ   — per-patch discretized y (Cartesian) or θ=azimuth (spherical)
                [4] = z/α   — per-patch discretized z (Cartesian) or α=inclination (spherical)

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Use a learned projection from float xyz → integer position bucket
          - Normalize xyz to a fixed integer range regardless of scene scale
        ──────────────────────────────────────────────────────────────────────
        """
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        num_tokens = llm_grid_t * llm_grid_h * llm_grid_w

        # xyz_coords: (llm_H, llm_W, 3) → (llm_H * llm_W, 3)
        xyz_flat = xyz_coords.reshape(-1, 3).to(device)          # (llm_H*llm_W, 3)
        # Repeat across temporal frames if llm_grid_t > 1
        if llm_grid_t > 1:
            xyz_flat = xyz_flat.repeat(llm_grid_t, 1)            # (num_tokens, 3)

        scale_vec = self._coord_scale_vec(coord_scale, device=device)         # (3,) float
        if polar:
            # Cartesian → log-spherical: (x, y, z) → (log ρ, θ, α)
            # Convention matches dataset xyz_to_polar:
            #   log ρ — scale-invariant radius (RoPE pos-diff = log(ρ_i/ρ_j),
            #           invariant to global scene scaling ρ → k·ρ)
            #   θ     = atan2(y, x)          ∈ [-π, π]  — azimuth
            #   α     = atan2(√(x²+y²), z)   ∈ [0, π]   — inclination
            rho     = torch.norm(xyz_flat, dim=-1).clamp(min=1e-6)            # (N,)
            log_rho = torch.log(rho)                                           # (N,) ∈ ℝ
            theta   = torch.atan2(xyz_flat[:, 1], xyz_flat[:, 0])              # (N,) ∈ [-π, π]
            alpha   = torch.atan2(
                torch.sqrt(xyz_flat[:, 0] ** 2 + xyz_flat[:, 1] ** 2),
                xyz_flat[:, 2],
            )                                                                  # (N,) ∈ [0, π]
            # Per-axis scale interpreted as (scale_log_ρ, scale_θ, scale_α).
            # Raw radians × scale — no [0,1] normalization — so θ and α share
            # the same angular resolution (1 rad difference ⇒ same position
            # delta for both). Float32, no round/long: RoPE uses position ×
            # inv_freq, integer quantization would discard sub-unit precision.
            rho_f   = (log_rho * scale_vec[0]).float()                         # (N,)
            theta_f = (theta   * scale_vec[1]).float()                         # (N,) θ ∈ [-π, π]
            alpha_f = (alpha   * scale_vec[2]).float()                         # (N,) α ∈ [0, π]
            xyz_pos = torch.stack([rho_f, theta_f, alpha_f], dim=1)            # (N, 3)
        else:
            # Per-axis scale broadcasts (N, 3) * (3,) → (N, 3).
            # Cartesian: round to integer-valued float (1 cm grid at scale=100).
            # Only polar stays continuous — angle discretization at scale=100 is
            # too coarse (~3.6°) for far-range patches.
            xyz_pos = (xyz_flat * scale_vec).round().float()                   # (num_tokens, 3)

        pos_t = torch.full(
            (num_tokens,), float(start_position), dtype=torch.float32, device=device,
        )
        pos_x = xyz_pos[:, 0]
        pos_y = xyz_pos[:, 1]
        pos_z = xyz_pos[:, 2]

        # seq row filled in by get_rope_index() after this call (needs actual seq offset)
        pos_seq = pos_t.clone()  # placeholder; overwritten in get_rope_index
        return torch.stack([pos_seq, pos_t, pos_x, pos_y, pos_z], dim=0)  # (5, num_tokens) float32

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_xyz: torch.Tensor | None = None,
        coord_scale=100.0,
        polar: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble 4D position_ids (shape: (4, batch, seq_len)) for the full LLM sequence.

        Iterates through token groups (text / image) and assigns:
          - Text tokens:  t=x=y=z = current_pos, current_pos+1, ...
                          (1D sequential, same on all four dimensions)
          - Image tokens: calls get_vision_position_ids() with (start_position, xyz_coords)
                          → (t, x, y, z) where t = start_position, xyz = 3D viewpoint

        Args:
            input_ids: (batch, seq_len)
            mm_token_type_ids: (batch, seq_len) — 0=text, 1=image, 2=video
            image_grid_thw: (num_images, 3) — (T, H, W) patch grid per image
            video_grid_thw: (num_videos, 3) — not used in 4D scheme (reserved)
            attention_mask: (batch, seq_len) — 0 for padding tokens
            image_xyz: list of (llm_H_i, llm_W_i, 3) float tensors, one per image,
                giving the 3D scene coordinates of each patch after spatial merge.
                llm_H_i = grid_thw[i][1] // spatial_merge_size, similarly for W.
                If None, xyz defaults to zeros for every patch of every image.
            coord_scale: multiplier applied to float xyz before rounding to int.
                Default 100 maps ±10 m → ±1000.

        Returns:
            position_ids: (5, batch, seq_len)
                Dimension ordering: [seq-dim, t-dim, x-dim, y-dim, z-dim]
            mrope_position_deltas: (batch, 1)
                Delta between max position and sequence length, used during
                auto-regressive generation to offset next-token positions.

        NOTE: The model config must have rope_scaling.mrope_section set to
        four equal sections of head_dim // 2 for 4D M-RoPE to be applied
        correctly inside attention.  See module docstring for details.

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Advance current_pos differently after each image block
          - Share a fixed "slot" for all images (same t for every image)
          - Use relative xyz offsets (coords relative to first image position)
        ──────────────────────────────────────────────────────────────────────
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        # Prepare xyz iterator; fall back to None (zeros allocated per-image below)
        xyz_iter = iter(image_xyz) if image_xyz is not None else None

        mrope_position_deltas = []
        # position_ids is kept as float32 end-to-end. Dim 0 (seq) holds integer
        # values cast to float; the causal-mask call in SpaTextModel.forward
        # casts it back to long. Dims 1-4 (t, x, y, z) carry continuous RoPE
        # positions (no round/long), which avoids sub-unit precision loss —
        # especially important in --polar where one integer unit can be several
        # degrees of angle.
        position_ids = torch.zeros(
            5,                      # ← 5D: (seq, t, x, y, z)
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float32,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            # Group consecutive tokens by modality type
            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0      # RoPE position budget (t coordinate for images)
            actual_seq  = 0      # true sequential position (for causal mask, dim 0)
            llm_pos_ids_list = []

            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:  # text
                    text_len = end_idx - start_idx
                    # dim-0 (seq) uses actual_seq for correct causal mask;
                    # dims 1-4 (t,x,y,z) use current_pos (RoPE budget).
                    # Float32 so position_ids can carry continuous xyz too.
                    seq_row = torch.arange(
                        actual_seq, actual_seq + text_len,
                        device=input_ids.device, dtype=torch.float32,
                    )
                    rope_row = torch.arange(
                        current_pos, current_pos + text_len,
                        device=input_ids.device, dtype=torch.float32,
                    )
                    plain_pos = torch.stack([seq_row, rope_row, rope_row,
                                             rope_row, rope_row], dim=0)  # (5, text_len)
                    llm_pos_ids_list.append(plain_pos)
                    current_pos += text_len
                    actual_seq  += text_len

                else:  # image (1) or video (2)
                    grid_thw = next(grid_iters[modality_type])
                    llm_h = grid_thw[1].item() // spatial_merge_size
                    llm_w = grid_thw[2].item() // spatial_merge_size
                    if xyz_iter is not None:
                        xyz_coords = next(xyz_iter)          # (llm_H, llm_W, 3)
                    else:
                        xyz_coords = torch.zeros(
                            llm_h, llm_w, 3,
                            dtype=torch.float32, device=input_ids.device,
                        )

                    vision_position_ids = self.get_vision_position_ids(
                        start_position=current_pos,
                        xyz_coords=xyz_coords,
                        grid_thw=grid_thw,
                        temp_merge_size=1,
                        spatial_merge_size=spatial_merge_size,
                        coord_scale=coord_scale,
                        polar=polar,
                        device=input_ids.device,
                    )                                        # (5, num_tokens)

                    # Fix dim-0 (seq): sequential positions for causal mask (float32)
                    num_img_tokens = vision_position_ids.shape[1]
                    vision_position_ids[0] = (
                        torch.arange(
                            num_img_tokens, device=input_ids.device, dtype=torch.float32,
                        ) + actual_seq
                    )
                    llm_pos_ids_list.append(vision_position_ids)

                    # Advance RoPE budget by max(H,W)/merge (same as original)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
                    actual_seq  += num_img_tokens

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(5, -1)

            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)

            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class SpaForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """
    Top-level Qwen3.5 VL model with 4D (t, x, y, z) positional embeddings.

    Usage:
        model = SpaForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    To pass 3D coordinates during forward / generate:
        # image_xyz: list of (llm_H_i, llm_W_i, 3) per-patch scene coords
        #   e.g. two images each with llm grid 16x16:
        outputs = model(
            input_ids=...,
            pixel_values=...,
            image_grid_thw=...,
            image_xyz=[coords_img0, coords_img1],  # each (llm_H, llm_W, 3)
        )

    NOTE: Before training with 4D M-RoPE, update the config:
        model.config.text_config.rope_scaling["mrope_section"] = [16, 16, 16, 16]
        # (values must sum to head_dim // 2; adjust for your model size)
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace the backbone with our custom SpaModel
        self.model = SpaModel(config)

    def forward(self, *args, image_xyz: torch.Tensor | None = None,
                coord_scale=100.0,
                polar: bool = False, **kwargs):
        """
        Thin wrapper that injects image_xyz into get_rope_index() via kwargs.

        image_xyz: (num_images, 3) float tensor of 3D camera coordinates,
                   in the same order as images appear left-to-right in the batch.
        coord_scale: passed through to get_rope_index / get_vision_position_ids.
        polar: if True, convert Cartesian (x, y, z) → log-spherical
               (log ρ, θ, α) for the M-RoPE position embedding of vision tokens.
        """
        if image_xyz is not None:
            kwargs["image_xyz"] = image_xyz
        # Pass coord_scale only if user overrode default scalar 100.0.
        # Always pass if it's a non-scalar (per-axis tuple/list/tensor).
        _pass_scale = (
            not isinstance(coord_scale, (int, float)) or float(coord_scale) != 100.0
        )
        if _pass_scale:
            kwargs["coord_scale"] = coord_scale
        if polar:
            kwargs["polar"] = polar
        return super().forward(*args, **kwargs)
