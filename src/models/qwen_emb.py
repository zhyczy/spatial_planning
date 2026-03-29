"""
SPA Model — Qwen3.5 with customizable positional embeddings.

Subclass hierarchy:
  SpaVisionModel(Qwen3_5VisionModel)
      └─ rot_pos_emb()              ← ViT internal 2D RoPE (row, col coords per patch)
      └─ fast_pos_embed_interpolate() ← ViT absolute 2D pos embed (additive, interpolated)

  SpaModel(Qwen3_5Model)
      └─ get_vision_position_ids()  ← LLM-side visual token (T, H, W) coords
      └─ get_rope_index()           ← full position_ids assembly for the LLM sequence

  SpaForConditionalGeneration(Qwen3_5ForConditionalGeneration)
      └─ top-level model, wires everything together

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
    Qwen3_5VisionModel,
    Qwen3_5Model,
    Qwen3_5ForConditionalGeneration,
)


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
# Part 2: LLM visual token positional embeddings (M-RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class SpaModel(Qwen3_5Model):
    """
    Qwen3.5 multimodal backbone with overridable LLM-side visual position IDs.

    The LLM uses M-RoPE: each token has 3D position coords (temporal, height, width).
        - Text tokens:  T = H = W = sequential 1D index  (same value on all 3 dims)
        - Image tokens: T = constant, H = row index, W = col index

    The two entry points to modify:
        get_vision_position_ids()  — computes coords for a single image/video
        get_rope_index()           — assembles position_ids for the full sequence
    """

    def __init__(self, config):
        # Replicate Qwen3_5Model.__init__ exactly, but swap VisionModel for SpaVisionModel
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = SpaVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device=None,
    ) -> torch.LongTensor:
        """
        Compute (T, H, W) position indices for visual tokens of one image/video.

        Called once per image/video segment inside get_rope_index().

        Args:
            start_position: int
                The current position counter in the LLM sequence.
                All returned coords are offset by this value.
            grid_thw: (3,) — (T, H, W) in patch units after merge.
                e.g. a 448x448 image with patch_size=14, spatial_merge_size=2
                → H = W = 32, after merge → llm_grid_h = llm_grid_w = 16
            temp_merge_size: temporal downscale factor (usually 1 for images)
            spatial_merge_size: spatial downscale factor (from vision_config)
            time_interval: spacing between temporal positions (for video fps control)
            device: torch device

        Returns:
            vision_position_ids: (3, llm_grid_t * llm_grid_h * llm_grid_w)
                [0] = temporal coords  (all same = start_position for images)
                [1] = height coords    (0, 0, ..., 1, 1, ..., H-1)
                [2] = width coords     (0, 1, ..., W-1, 0, 1, ...)

        After this call, get_rope_index() advances start_position by:
            max(H, W) // spatial_merge_size   (not by token count!)

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Make temporal also vary (e.g., use frame index even for images)
          - Use absolute global coords instead of start_position-relative
          - Normalize coords to [0, 1] range * some scale
          - Use 1D sequential coords for all visual tokens (ablation)
        ──────────────────────────────────────────────────────────────────────
        """
        # ── Original implementation ──────────────────────────────────────────
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        image_seq_length = llm_grid_t * llm_grid_h * llm_grid_w

        # Width: cycles 0,1,...,W-1 repeated for every (t, h) combination
        position_width = torch.arange(
            start_position, start_position + llm_grid_w, device=device
        ).repeat(llm_grid_h * llm_grid_t)

        # Height: each value repeated W*T times
        position_height = torch.arange(
            start_position, start_position + llm_grid_h, device=device
        ).repeat_interleave(llm_grid_w * llm_grid_t)

        # Temporal: all tokens of this image share the same temporal position
        position_temporal = torch.full(
            (image_seq_length,), start_position, device=device, dtype=torch.long
        )
        position_temporal = position_temporal * time_interval

        vision_position_ids = torch.stack(
            [position_temporal, position_height, position_width], dim=0
        )
        return vision_position_ids
        # ── End original ──────────────────────────────────────────────────────

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble position_ids (shape: (3, batch, seq_len)) for the full LLM sequence.

        Iterates through token groups (text / image / video) and assigns:
          - Text tokens: T=H=W = current_pos, current_pos+1, ...
          - Image tokens: calls get_vision_position_ids()
          - Video tokens: calls get_vision_position_ids() with time_interval > 1

        Args:
            input_ids: (batch, seq_len)
            mm_token_type_ids: (batch, seq_len) — 0=text, 1=image, 2=video
            image_grid_thw: (num_images, 3)
            video_grid_thw: (num_videos, 3)
            attention_mask: (batch, seq_len) — 0 for padding

        Returns:
            position_ids: (3, batch, seq_len)
            mrope_position_deltas: (batch, 1)
                The delta between the max position and the sequence length.
                Used during generation to correctly offset next-token positions.

        ── MODIFY BELOW ──────────────────────────────────────────────────────
        Ideas:
          - Change how current_pos advances after each image
          - Interleave images' positions (images share the same "slot")
          - Assign fixed position IDs to all images regardless of order
        ──────────────────────────────────────────────────────────────────────
        """
        # ── Original implementation ──────────────────────────────────────────
        import itertools

        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
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
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []

            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:  # text
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len

                else:  # image (1) or video (2)
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        temp_merge_size=1,
                        spatial_merge_size=spatial_merge_size,
                        device=input_ids.device,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    # Advance position budget by max(H, W) / merge_size, not by token count
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)

            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
        # ── End original ──────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class SpaForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """
    Top-level Qwen3.5 VL model with customizable positional embeddings.

    Usage:
        model = SpaForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace the backbone with our custom SpaModel
        self.model = SpaModel(config)
