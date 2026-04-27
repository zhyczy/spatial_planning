"""
rotation_rope_llm.py

RotationRoPEModel: rotation-aware model with an end-to-end differentiable
RoPE path, so the predicted rotation R receives gradient from lm_loss /
coord_loss directly (no gt_rotation supervision needed).

Why a custom differentiable RoPE
--------------------------------
The stock path R → rotated_xyz → position_ids → RoPE is broken by two
barriers in spa_emb.py:

  (1) get_rope_index discretises with `.round().long()`  → long dtype
  (2) SpaTextRotaryEmbedding.forward is `@torch.no_grad()`

This model bypasses both WITHOUT editing spa_emb.py, by:

  A. Building a float (5, bs, seq) position_ids tensor here (image xyz
     entries are `xyz * coord_scale` with NO rounding / NO cast to long).
  B. Computing cos/sin from that float tensor with a local, gradient-
     enabled `DifferentiableMRoPE` module that mirrors
     SpaTextRotaryEmbedding's math exactly but without the no_grad
     decorator.
  C. Driving the SpaTextModel decoder layers manually, passing the
     precomputed `(cos, sin)` tuple so each attention layer uses our
     differentiable RoPE instead of calling `self.rotary_emb` again.

Gradient flow with this model
-----------------------------
  lm_loss / coord_loss
        ↓
  hidden_states
        ↓
  decoder_layer × N  (uses (cos, sin) from DifferentiableMRoPE)
        ↓
  (cos, sin)
        ↓ (float)
  position_ids[1:] = float xyz * coord_scale
        ↓
  rotated_xyz = R @ xyz_world
        ↓
  R
        ↓
  rotation_enc params

Therefore the camera rotation encoder receives gradient end-to-end from
lm_loss / coord_loss — no ground-truth rotation or geodesic loss needed.
"""

import itertools
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from transformers.utils.generic import maybe_autocast

from .coordinate_llm import DepthPredictionTransformer
from .correspondence_llm import rot6d_to_rotmat
from .spa_emb_dec import _apply_xyz_rotary


# ---------------------------------------------------------------------------
# Helpers (integer 4D position_ids + R @ xyz)
# ---------------------------------------------------------------------------

def _build_token_txyz_int(
    input_ids:          torch.Tensor,   # (1, seq_len)
    image_token_id:     int,
    image_xyz_list:     list,           # list[k]: (llm_H_k, llm_W_k, 3)
    image_grid_thw:     torch.Tensor,   # (num_images, 3)
    spatial_merge_size: int,
    coord_scale:        float,
) -> torch.Tensor:
    """Build (seq_len, 4) long position_ids matching MLLM get_rope_index convention.

    Position assignment (mirrors SpaModel.get_rope_index / get_vision_position_ids):

      Text tokens:   t = x = y = z = current_pos  (sequential, starts at 1)
                     current_pos advances by text_len per text block.

      Image tokens:  t = current_pos at start of image block  (same for all patches)
                     x = round(x_world * coord_scale)
                     y = round(y_world * coord_scale)
                     z = round(z_world * coord_scale)
                     current_pos advances by max(llm_h, llm_w) per image block.

    Position 0 is reserved for the cam token prepended in CameraTokenRotationEncoder.
    Text positions therefore start at 1.

    Returns:
        (seq_len, 4) long — [t, x_int, y_int, z_int] per token.
    """
    seq_len = input_ids.shape[1]
    device  = input_ids.device
    result  = torch.zeros(seq_len, 4, dtype=torch.long, device=device)

    ids = input_ids[0]

    current_pos = 1   # 0 is reserved for cam token
    k = 0             # image counter
    i = 0

    while i < seq_len:
        if ids[i] != image_token_id:
            # ── text block ───────────────────────────────────────────────────
            j = i
            while j < seq_len and ids[j] != image_token_id:
                j += 1
            text_len = j - i
            run = torch.arange(current_pos, current_pos + text_len,
                               dtype=torch.long, device=device)
            result[i:j, 0] = run   # t
            result[i:j, 1] = run   # x
            result[i:j, 2] = run   # y
            result[i:j, 3] = run   # z
            current_pos += text_len
            i = j
        else:
            # ── image block ──────────────────────────────────────────────────
            if k < len(image_grid_thw) and k < len(image_xyz_list):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // spatial_merge_size
                llm_w = int(thw_k[2]) // spatial_merge_size
                n_tok = llm_h * llm_w

                xyz_k   = image_xyz_list[k].reshape(-1, 3).to(
                    device=device, dtype=torch.float32
                )
                xyz_int = (xyz_k * coord_scale).round().long()  # (n_tok, 3)

                result[i:i + n_tok, 0] = current_pos      # t shared across all patches
                result[i:i + n_tok, 1] = xyz_int[:, 0]
                result[i:i + n_tok, 2] = xyz_int[:, 1]
                result[i:i + n_tok, 3] = xyz_int[:, 2]

                current_pos += max(llm_h, llm_w)          # same advancement as MLLM
                i += n_tok
                k += 1
            else:
                i += 1  # unknown image token — skip

    return result


def _apply_rotation_to_xyz(
    R:        torch.Tensor,   # (3, 3) float32
    xyz_list: list,           # list of (..., 3) tensors
) -> list:
    """Return [R @ xyz for xyz in xyz_list].

    Preserves original dtype of each tensor in the list.
    """
    result = []
    for xyz in xyz_list:
        orig_dtype = xyz.dtype
        xyz_flat   = xyz.reshape(-1, 3).float()
        rotated    = (R @ xyz_flat.T).T                    # (N, 3)
        result.append(rotated.reshape(xyz.shape).to(orig_dtype))
    return result


def _build_per_token_xyz(
    input_ids:          torch.Tensor,   # (1, seq_len)
    image_token_id:     int,
    image_xyz_list:     list,           # list[k]: (llm_H_k, llm_W_k, 3)
    image_grid_thw:     torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build (1, seq_len, 3) FLOAT per-token xyz tensor for the decouple path.

    Text tokens                → (0, 0, 0).
    Image patches (image_token) → corresponding entry of image_xyz_list[k]
                                  in flat (H * W) order, repeated across t
                                  frames (matches SpaDecModel._compute_xyz_pos).

    Built with torch.cat (not slice-assignment) so gradients w.r.t.
    image_xyz_list (e.g. R-rotated coords) flow through to the rotation
    matrix. Output dtype follows image_xyz_list[0] (float32 for R-rotated).
    """
    seq_len = input_ids.shape[1]
    device  = input_ids.device

    if (image_xyz_list is not None and len(image_xyz_list) > 0
            and image_xyz_list[0].is_floating_point()):
        ref_dtype = image_xyz_list[0].dtype
    else:
        ref_dtype = torch.float32

    ids = input_ids[0]
    parts: list[torch.Tensor] = []
    k = 0
    i = 0
    while i < seq_len:
        if ids[i] != image_token_id:
            j = i
            while j < seq_len and ids[j] != image_token_id:
                j += 1
            parts.append(
                torch.zeros(j - i, 3, dtype=ref_dtype, device=device)
            )
            i = j
            continue

        if (image_grid_thw is None
                or k >= len(image_grid_thw)
                or image_xyz_list is None
                or k >= len(image_xyz_list)):
            parts.append(torch.zeros(1, 3, dtype=ref_dtype, device=device))
            i += 1
            continue

        thw_k = image_grid_thw[k]
        n_t   = int(thw_k[0])
        llm_h = int(thw_k[1]) // spatial_merge_size
        llm_w = int(thw_k[2]) // spatial_merge_size
        n_tok = llm_h * llm_w * max(n_t, 1)

        xyz_k = image_xyz_list[k].reshape(-1, 3).to(
            device=device, dtype=ref_dtype,
        )
        if n_t > 1:
            xyz_k = xyz_k.repeat(n_t, 1)

        n_avail   = xyz_k.shape[0]
        n_to_copy = min(n_tok, n_avail)
        if n_to_copy < n_tok:
            pad = torch.zeros(
                n_tok - n_to_copy, 3, dtype=ref_dtype, device=device,
            )
            parts.append(torch.cat([xyz_k[:n_to_copy], pad], dim=0))
        else:
            parts.append(xyz_k[:n_to_copy])

        i += n_tok
        k += 1

    if not parts:
        return torch.zeros(1, seq_len, 3, dtype=ref_dtype, device=device)
    return torch.cat(parts, dim=0).unsqueeze(0)            # (1, seq_len, 3)


def _build_token_thw_int(
    input_ids:          torch.Tensor,   # (1, seq_len)
    image_token_id:     int,
    image_grid_thw:     torch.Tensor,   # (num_images, 3)
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build (seq_len, 3) long [t, h, w] matching Qwen 3D M-RoPE convention.

    Used by `CameraTokenRotationEncoder` in --decouple mode (rotation encoder
    mirrors the LLM's stock 3D M-RoPE in the rotary 64 dims).

    Position assignment (mirrors Qwen3_5Model.get_rope_index):
      Text tokens:  t = h = w = current_pos (sequential).  Position 0 reserved
                    for the cam token; text starts at 1.
      Image tokens: t = current_pos shared across all patches in the image.
                    h = current_pos + h_idx  (h_idx ∈ [0, llm_h))
                    w = current_pos + w_idx  (w_idx ∈ [0, llm_w))
                    current_pos advances by max(llm_h, llm_w) per image block.

    Returns:
        (seq_len, 3) long — [t, h, w] per token.
    """
    seq_len = input_ids.shape[1]
    device  = input_ids.device
    result  = torch.zeros(seq_len, 3, dtype=torch.long, device=device)

    ids = input_ids[0]
    current_pos = 1
    k = 0
    i = 0
    while i < seq_len:
        if ids[i] != image_token_id:
            j = i
            while j < seq_len and ids[j] != image_token_id:
                j += 1
            text_len = j - i
            run = torch.arange(current_pos, current_pos + text_len,
                               dtype=torch.long, device=device)
            result[i:j, 0] = run
            result[i:j, 1] = run
            result[i:j, 2] = run
            current_pos += text_len
            i = j
        else:
            if k < len(image_grid_thw):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // spatial_merge_size
                llm_w = int(thw_k[2]) // spatial_merge_size
                n_tok = llm_h * llm_w

                h_grid = torch.arange(
                    llm_h, dtype=torch.long, device=device
                ).unsqueeze(1).expand(-1, llm_w).reshape(-1)        # (n_tok,)
                w_grid = torch.arange(
                    llm_w, dtype=torch.long, device=device
                ).unsqueeze(0).expand(llm_h, -1).reshape(-1)        # (n_tok,)

                result[i:i + n_tok, 0] = current_pos
                result[i:i + n_tok, 1] = current_pos + h_grid
                result[i:i + n_tok, 2] = current_pos + w_grid

                current_pos += max(llm_h, llm_w)
                i += n_tok
                k += 1
            else:
                i += 1

    return result


# ---------------------------------------------------------------------------
# M-RoPE attention + encoder layer (used only by CameraTokenRotationEncoder)
# ---------------------------------------------------------------------------

class _MRoPEAttention(nn.Module):
    """Multi-head self-attention with M-RoPE on Q and K.

    Default (non-decouple): 4D M-RoPE on the rotary head_dim using
        position_ids = (4, 1, T) integer [t, x_int, y_int, z_int].

    Decouple mode: matches the LLM main-path SpaDec dual structure —
        • rotary 64 dims  → standard 3D M-RoPE on (t, h, w) integer positions
        • dims [xyz_offset : xyz_offset + xyz_dim]  →  XYZ RoPE on float xyz_pos
        rope_emb's mrope_section MUST be 3D ([11,11,10]) and an
        xyz_rotary_emb (e.g. SpaXYZRotaryEmbedding) must be supplied.

    Args:
        d_model:        total model width = nhead × mllm_head_dim
        nhead:          number of attention heads
        rope_emb:       SpaTextRotaryEmbedding (4D in non-decouple, 3D in decouple)
        dropout:        attention dropout probability
        decouple:       if True, also apply a separate XYZ RoPE on dims
                        [xyz_offset : xyz_offset + xyz_dim] of Q / K
        xyz_rotary_emb: SpaXYZRotaryEmbedding instance (required if decouple)
        xyz_offset:     starting dim for the XYZ RoPE slice (default 64)
    """

    def __init__(
        self,
        d_model:        int,
        nhead:          int,
        rope_emb:       nn.Module,
        dropout:        float = 0.0,
        decouple:       bool  = False,
        xyz_rotary_emb: nn.Module | None = None,
        xyz_offset:     int = 64,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.scale    = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.rope_emb     = rope_emb
        self.attn_dropout = dropout

        self.decouple   = bool(decouple)
        self.xyz_offset = int(xyz_offset)
        if self.decouple:
            assert xyz_rotary_emb is not None, (
                "_MRoPEAttention(decouple=True) requires xyz_rotary_emb"
            )
            self.xyz_rotary_emb = xyz_rotary_emb
        else:
            self.xyz_rotary_emb = None

    def forward(
        self,
        x:            torch.Tensor,                 # (1, T, d_model)
        position_ids: torch.Tensor,                 # (N, 1, T) long; N=4 (4D) or N=3 (decouple)
        xyz_pos:      torch.Tensor | None = None,   # (1, T, 3) float — required if decouple
        coord_scale:  float = 100.0,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        if self.decouple:
            assert xyz_pos is not None, (
                "_MRoPEAttention.forward: xyz_pos required in decouple mode"
            )
            xyz_cos, xyz_sin = self.xyz_rotary_emb(
                xyz_pos, coord_scale=coord_scale, polar=False,
            )                                                   # (1, T, xyz_dim)
            # _apply_xyz_rotary internally unsqueezes cos/sin to broadcast over
            # heads (default unsqueeze_dim=1). Pass raw (B, T, xyz_dim) tensors.
            q, k = _apply_xyz_rotary(
                q, k,
                xyz_cos.to(q.dtype), xyz_sin.to(q.dtype),
                offset=self.xyz_offset,
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class _MRoPEEncoderLayer(nn.Module):
    """Pre-norm encoder layer: M-RoPE attention + FFN.

    In decouple mode, the attention also runs an XYZ RoPE on dims
    [xyz_offset : xyz_offset + xyz_dim] using xyz_pos.
    """

    def __init__(
        self,
        d_model:         int,
        nhead:           int,
        dim_feedforward: int,
        rope_emb:        nn.Module,
        dropout:         float = 0.0,
        decouple:        bool  = False,
        xyz_rotary_emb:  nn.Module | None = None,
        xyz_offset:      int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _MRoPEAttention(
            d_model, nhead, rope_emb, dropout,
            decouple=decouple, xyz_rotary_emb=xyz_rotary_emb,
            xyz_offset=xyz_offset,
        )
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:            torch.Tensor,                 # (1, T, d_model)
        position_ids: torch.Tensor,                 # (N, 1, T) long
        xyz_pos:      torch.Tensor | None = None,   # (1, T, 3) float (decouple only)
        coord_scale:  float = 100.0,
    ) -> torch.Tensor:
        x = x + self.drop(
            self.attn(self.norm1(x), position_ids,
                      xyz_pos=xyz_pos, coord_scale=coord_scale)
        )
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# CameraTokenRotationEncoder — predicts canonical R via shallow 4D M-RoPE encoder
# ---------------------------------------------------------------------------

class CameraTokenRotationEncoder(nn.Module):
    """Shallow TransformerEncoder that predicts a canonical rotation.

    Architecture
    ~~~~~~~~~~~~
    1. Linear projection   hidden_dim → d_model  (per MLLM token)
    2. Prepend learnable cam token at position (0, 0, 0, 0)
    3. _MRoPEEncoderLayer × num_layers
       Each layer applies 4D M-RoPE (same SpaTextRotaryEmbedding as MLLM)
       inside its multi-head attention.  No additive PE.
    4. cam token output → Linear → 6-D → rot6d_to_rotmat → (3, 3)

    Strict four-level alignment with MLLM M-RoPE
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Mechanism:   rotary PE applied inside attention on Q and K.
    Parameters:  same rope_theta (1e7) and mrope_section via rope_emb.
    Coordinates: integer (t, x, y, z) = round(world_xyz * scale).
                 Cam token at (0, 0, 0, 0).
    Dimensions:  head_dim = mllm_head_dim (256); rotary_dim = 64 (first 64
                 of each 256-dim head, matching MLLM's partial_rotary_factor).
                 d_model = nhead × mllm_head_dim.

    Args:
        hidden_dim:       MLLM hidden size (e.g. 2560)
        mllm_head_dim:    MLLM explicit head_dim (config.head_dim, 256)
        rope_emb:         SpaTextRotaryEmbedding created from the SAME
                          config.text_config (after mrope_section override)
        nhead:            attention heads in this encoder
                          → d_model = nhead × mllm_head_dim
        dim_feedforward:  FFN width
        num_layers:       encoder depth (2 = "shallow")
        dropout:          dropout probability
    """

    def __init__(
        self,
        hidden_dim:      int,
        mllm_head_dim:   int,
        rope_emb:        nn.Module,
        nhead:           int   = 4,
        dim_feedforward: int   = 2048,
        num_layers:      int   = 2,
        dropout:         float = 0.0,
        decouple:        bool  = False,
        xyz_rotary_emb:  nn.Module | None = None,
        xyz_offset:      int = 64,
    ):
        super().__init__()
        assert mllm_head_dim > 0 and nhead > 0
        self.d_model       = nhead * mllm_head_dim   # e.g. 4 × 256 = 1024
        self.nhead         = nhead
        self.mllm_head_dim = mllm_head_dim

        self.decouple   = bool(decouple)
        self.xyz_offset = int(xyz_offset)
        if self.decouple:
            assert xyz_rotary_emb is not None, (
                "CameraTokenRotationEncoder(decouple=True) requires xyz_rotary_emb"
            )
            self.xyz_rotary_emb = xyz_rotary_emb
        else:
            self.xyz_rotary_emb = None

        self.cam_token = nn.Parameter(torch.empty(1, self.d_model))
        nn.init.normal_(self.cam_token, std=0.02)

        self.input_proj = nn.Linear(hidden_dim, self.d_model)

        self.layers = nn.ModuleList([
            _MRoPEEncoderLayer(
                d_model         = self.d_model,
                nhead           = nhead,
                dim_feedforward = dim_feedforward,
                rope_emb        = rope_emb,
                dropout         = dropout,
                decouple        = self.decouple,
                xyz_rotary_emb  = self.xyz_rotary_emb,
                xyz_offset      = self.xyz_offset,
            )
            for _ in range(num_layers)
        ])

        # 6-D rotation head (Gram-Schmidt → SO(3))
        # Init so that step-0 output r6d = [1,0,0, 0,1,0] → R = I.
        self.rot_head = nn.Linear(self.d_model, 6)
        nn.init.zeros_(self.rot_head.weight)
        with torch.no_grad():
            self.rot_head.bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0.])
            )

    def forward(
        self,
        hidden_states:   torch.Tensor,                 # (1, seq_len, hidden_dim)
        token_txyz_int:  torch.Tensor | None = None,   # (seq_len, 4) — non-decouple
        token_thw_int:   torch.Tensor | None = None,   # (seq_len, 3) — decouple
        xyz_pos:         torch.Tensor | None = None,   # (1, seq_len, 3) — decouple
        coord_scale:     float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args (non-decouple): pass token_txyz_int (4D `[t, x_int, y_int, z_int]`).
        Args (decouple):     pass token_thw_int (3D `[t, h, w]`) + xyz_pos
                             ((1, seq, 3) float, gradient-OK if needed).
        Returns:
            R:        (3, 3) float32 rotation matrix (SO(3))
            cam_feat: (d_model,) cam-token output in encoder dtype
        """
        device = hidden_states.device

        x = self.input_proj(hidden_states[0])

        cam = self.cam_token.to(dtype=x.dtype, device=device)   # (1, d_model)
        x   = torch.cat([cam, x], dim=0)                        # (seq_len+1, d_model)
        x   = x.unsqueeze(0)

        if self.decouple:
            assert token_thw_int is not None and xyz_pos is not None, (
                "rotation_enc(decouple=True): need token_thw_int + xyz_pos"
            )
            # 3D position_ids with cam-token row at (0,0,0).
            cam_pos  = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
            mllm_pos = token_thw_int.long().T.unsqueeze(1)              # (3, 1, seq)
            position_ids = torch.cat([cam_pos, mllm_pos], dim=2)        # (3, 1, seq+1)

            # xyz_pos with cam-token row = 0 → identity rotation in XYZ RoPE.
            cam_xyz = torch.zeros(
                1, 1, 3, dtype=xyz_pos.dtype, device=device,
            )
            xyz_pos_full = torch.cat([cam_xyz, xyz_pos], dim=1)         # (1, seq+1, 3)

            for layer in self.layers:
                x = layer(
                    x, position_ids,
                    xyz_pos=xyz_pos_full, coord_scale=coord_scale,
                )
        else:
            assert token_txyz_int is not None, (
                "rotation_enc(non-decouple): need token_txyz_int"
            )
            cam_pos  = torch.zeros(4, 1, 1, dtype=torch.long, device=device)
            mllm_pos = token_txyz_int.long().T.unsqueeze(1)             # (4, 1, seq)
            position_ids = torch.cat([cam_pos, mllm_pos], dim=2)        # (4, 1, seq+1)

            for layer in self.layers:
                x = layer(x, position_ids)

        cam_feat = x[0, 0]                                       # (d_model,) bf16
        r6d      = self.rot_head(cam_feat).float()               # (6,) float32
        R        = rot6d_to_rotmat(r6d.unsqueeze(0)).squeeze(0)  # (3, 3)
        return R, cam_feat


# ---------------------------------------------------------------------------
# Differentiable 4D M-RoPE (same math as SpaTextRotaryEmbedding, no no_grad)
# ---------------------------------------------------------------------------

class DifferentiableMRoPE(nn.Module):
    """N-D M-RoPE that accepts float position_ids and preserves gradient.

    Mirrors `SpaTextRotaryEmbedding.forward` byte-for-byte except:
      - no `@torch.no_grad()` wrapper
      - position_ids is expected to be float (gradients flow through it)

    Args:
        inv_freq:      (rotary_dim // 2,) — shared with MLLM rotary_emb
        mrope_section: list[int] — per-dim frequency splits, same as MLLM
        attention_scaling: float — same as MLLM
    """

    def __init__(
        self,
        inv_freq:          torch.Tensor,
        mrope_section:     List[int],
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        # inv_freq is a fixed buffer (not trainable)
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section     = list(mrope_section)
        self.attention_scaling = float(attention_scaling)

    @classmethod
    def from_spa_rotary(cls, spa_rotary: nn.Module) -> "DifferentiableMRoPE":
        """Build from an existing SpaTextRotaryEmbedding instance."""
        return cls(
            inv_freq          = spa_rotary.inv_freq,
            mrope_section     = spa_rotary.mrope_section,
            attention_scaling = getattr(spa_rotary, "attention_scaling", 1.0),
        )

    def forward(
        self,
        x:            torch.Tensor,   # (bs, seq, hidden) — used only for dtype/device
        position_ids: torch.Tensor,   # (N, bs, seq) float, N = len(mrope_section)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_dims = len(self.mrope_section)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                num_dims, position_ids.shape[0], -1
            )

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(num_dims, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (N, bs, 1, seq)

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)                                  # (N, bs, seq, rd//2)
            freqs = self._apply_sequential_mrope(freqs, self.mrope_section)
            emb   = torch.cat((freqs, freqs), dim=-1)          # (bs, seq, rd)
            cos   = emb.cos() * self.attention_scaling
            sin   = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def _apply_sequential_mrope(
        freqs:         torch.Tensor,
        mrope_section: List[int],
    ) -> torch.Tensor:
        """Sequential N-D M-RoPE (no interleaving)."""
        num_dims       = len(mrope_section)
        freqs_out_list = []
        offset         = 0
        for dim in range(num_dims):
            length = mrope_section[dim]
            freqs_out_list.append(freqs[dim, ..., offset:offset + length])
            offset += length
        return torch.cat(freqs_out_list, dim=-1)


# ---------------------------------------------------------------------------
# Float 5D position_ids builder  (differentiable version of get_rope_index)
# ---------------------------------------------------------------------------

def _build_float_position_ids(
    input_ids:          torch.Tensor,        # (bs, seq)
    attention_mask:     torch.Tensor | None, # (bs, seq)
    image_token_id:     int,
    image_xyz:          List[torch.Tensor],  # list[k]: (llm_H, llm_W, 3)  (grad OK)
    image_grid_thw:     torch.Tensor,        # (num_images, 3)
    spatial_merge_size: int,
    coord_scale:        float,
) -> torch.Tensor:
    """Construct (5, bs, seq) float position_ids mirroring SpaModel.get_rope_index.

    Text tokens: t = x = y = z = current_pos, current_pos+1, ...
    Image tokens: t = current_pos (shared),
                  x/y/z = xyz_world * coord_scale  (NO rounding, gradient-preserving)
    Sequence dim [0] is sequential for causal mask construction.

    Only the Cartesian path is implemented — this model is used from the
    RotationRoPE training pipeline; coord_head reads vision-token hidden
    states directly, no auxiliary text token is needed.
    """
    batch_size, seq_len = input_ids.shape
    device              = input_ids.device

    # We will collect per-batch (5, seq) floats, then pad into one tensor.
    out = torch.zeros(
        5, batch_size, seq_len, dtype=torch.float32, device=device
    )

    grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None
    xyz_iter  = iter(image_xyz)      if image_xyz     is not None else None

    for batch_idx in range(batch_size):
        ids   = input_ids[batch_idx]
        mask  = (attention_mask[batch_idx].bool()
                 if attention_mask is not None
                 else torch.ones_like(ids, dtype=torch.bool))
        kept  = ids[mask]
        is_img = (kept == image_token_id).to(torch.int32)

        # Group by modality: 0 = text, 1 = image
        groups: list[tuple[int, int, int]] = []
        for key, grp in itertools.groupby(
            enumerate(is_img.tolist()), lambda it: it[1]
        ):
            grp = list(grp)
            groups.append((key, grp[0][0], grp[-1][0] + 1))

        current_pos = 0                                     # RoPE budget
        actual_seq  = 0                                     # causal seq dim
        rows: list[torch.Tensor] = []

        for modality, start_idx, end_idx in groups:
            n = end_idx - start_idx
            if modality == 0:
                # text block
                seq_row  = torch.arange(
                    actual_seq, actual_seq + n,
                    dtype=torch.float32, device=device,
                )
                rope_row = torch.arange(
                    current_pos, current_pos + n,
                    dtype=torch.float32, device=device,
                )
                plain_pos = torch.stack(
                    [seq_row, rope_row, rope_row, rope_row, rope_row],
                    dim=0,
                )                                             # (5, n)
                rows.append(plain_pos)
                current_pos += n
                actual_seq  += n
            else:
                # image block — pull next grid + xyz
                if grid_iter is None or xyz_iter is None:
                    raise RuntimeError(
                        "_build_float_position_ids: image token encountered but "
                        "image_grid_thw or image_xyz is None"
                    )
                grid_thw = next(grid_iter)
                xyz      = next(xyz_iter)                     # (llm_H, llm_W, 3) float
                llm_h    = int(grid_thw[1]) // spatial_merge_size
                llm_w    = int(grid_thw[2]) // spatial_merge_size
                n_tok    = llm_h * llm_w

                if n_tok != n:
                    # Fall back: just assume contiguous image tokens span `n`.
                    n_tok = n

                xyz_flat   = xyz.reshape(-1, 3).to(device=device, dtype=torch.float32)
                xyz_scaled = xyz_flat * float(coord_scale)    # (n_tok, 3) float, grad OK

                seq_row = torch.arange(
                    actual_seq, actual_seq + n_tok,
                    dtype=torch.float32, device=device,
                )
                t_row = torch.full(
                    (n_tok,), float(current_pos),
                    dtype=torch.float32, device=device,
                )
                img_pos = torch.stack(
                    [
                        seq_row,
                        t_row,
                        xyz_scaled[:, 0],
                        xyz_scaled[:, 1],
                        xyz_scaled[:, 2],
                    ],
                    dim=0,
                )                                             # (5, n_tok)
                rows.append(img_pos)

                current_pos += max(llm_h, llm_w)
                actual_seq  += n_tok

        merged = torch.cat(rows, dim=1)                       # (5, kept_len)
        out[:, batch_idx, mask] = merged.to(device=out.device)

    return out                                                # (5, bs, seq) float


# ---------------------------------------------------------------------------
# RotationRoPEModel
# ---------------------------------------------------------------------------

class RotationRoPEModel(nn.Module):
    """Rotation-aware model with end-to-end differentiable RoPE path.

    Step 4/5 uses a hand-rolled transformer loop that takes a gradient-
    preserving RoPE (cos, sin) tuple computed from float position_ids,
    so the rotation encoder is trained purely through lm_loss /
    coord_loss (back-propagated through the differentiable RoPE).
    No gt_rotation / rot_loss supervision is used.
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        rotation_enc:       CameraTokenRotationEncoder,
        coord_head:         DepthPredictionTransformer,
        image_token_id:     int,
        spatial_merge_size: int,
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
        decouple:           bool  = False,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.rotation_enc       = rotation_enc
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.decouple           = bool(decouple)

        # Differentiable 4D M-RoPE (only used by the non-decouple path: the
        # decouple path runs Qwen's stock 3D M-RoPE in the rotary 64 dims and
        # gets gradient flow through the separate XYZ RoPE in pass-through).
        if not self.decouple:
            inner       = self._unwrap()
            spa_inner   = inner.model                            # SpaModel
            spa_rotary  = spa_inner.language_model.rotary_emb    # SpaTextRotaryEmbedding
            self.diff_rope = DifferentiableMRoPE.from_spa_rotary(spa_rotary)
        else:
            self.diff_rope = None

    # ------------------------------------------------------------------

    def _unwrap(self) -> nn.Module:
        """Strip PEFT wrappers to reach SpaForConditionalGeneration."""
        m = self.spa_model
        seen = {id(m)}
        while hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
            if id(m) in seen:
                break
            seen.add(id(m))
        return m

    # ------------------------------------------------------------------

    def _call_rotation_enc(
        self,
        inputs_embeds:  torch.Tensor,
        input_ids:      torch.Tensor,
        image_xyz:      list,
        image_grid_thw: torch.Tensor,
        coord_scale:    float,
    ):
        """Build positional inputs for rotation_enc and call it.

        Branches on `self.decouple`:
          • non-decouple → 4D `_build_token_txyz_int` (matches LLM 4D M-RoPE)
          • decouple     → 3D `_build_token_thw_int` + per-token xyz_pos
                           (matches LLM dual stock 3D M-RoPE + XYZ RoPE)
        """
        if self.decouple:
            token_thw_int = _build_token_thw_int(
                input_ids, self.image_token_id,
                image_grid_thw, self.spatial_merge_size,
            )
            xyz_pos = _build_per_token_xyz(
                input_ids, self.image_token_id,
                image_xyz, image_grid_thw, self.spatial_merge_size,
            )
            return self.rotation_enc(
                inputs_embeds,
                token_thw_int = token_thw_int,
                xyz_pos       = xyz_pos,
                coord_scale   = float(coord_scale),
            )

        token_txyz_int = _build_token_txyz_int(
            input_ids, self.image_token_id,
            image_xyz, image_grid_thw, self.spatial_merge_size,
            coord_scale,
        )
        return self.rotation_enc(
            inputs_embeds,
            token_txyz_int = token_txyz_int,
        )

    # ------------------------------------------------------------------

    def _run_text_model_manual(
        self,
        text_model:         nn.Module,        # SpaTextModel
        inputs_embeds:      torch.Tensor,     # (bs, seq, hidden)
        attention_mask:     torch.Tensor,     # (bs, seq)
        position_ids_float: torch.Tensor,     # (5, bs, seq) float
    ) -> torch.Tensor:
        """Replicate SpaTextModel.forward but with differentiable RoPE.

        Returns last_hidden_state (bs, seq, hidden).
        """
        bs, seq_len, _ = inputs_embeds.shape
        device         = inputs_embeds.device

        cache_position = torch.arange(seq_len, device=device)

        # dim 0 of position_ids_float is the sequential row → needs to be long
        # for the causal mask construction.
        text_position_ids_long = position_ids_float[0].long()

        causal_mask = create_causal_mask(
            config         = text_model.config,
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            cache_position = cache_position,
            past_key_values= None,
            position_ids   = text_position_ids_long,
        )
        linear_attn_mask = text_model._update_linear_attn_mask(
            attention_mask, cache_position,
        )

        # Differentiable RoPE — accepts FLOAT (t, x, y, z) rows (dims 1-4).
        rope_pos = position_ids_float[1:]                    # (4, bs, seq)
        cos, sin = self.diff_rope(inputs_embeds, rope_pos)
        position_embeddings = (cos, sin)

        hidden_states = inputs_embeds
        for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
            layer_mask = (
                linear_attn_mask if decoder_layer.layer_type == "linear_attention"
                else causal_mask
            )
            if getattr(text_model, "gradient_checkpointing", False) and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings = position_embeddings,
                    attention_mask      = layer_mask,
                    position_ids        = rope_pos,
                    past_key_values     = None,
                    use_cache           = False,
                    cache_position      = cache_position,
                    use_reentrant       = False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings = position_embeddings,
                    attention_mask      = layer_mask,
                    position_ids        = rope_pos,
                    past_key_values     = None,
                    use_cache           = False,
                    cache_position      = cache_position,
                )

        hidden_states = text_model.norm(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------

    def _run_text_model_decouple(
        self,
        text_model:        nn.Module,                  # SpaDecTextModel
        spa_inner:         nn.Module,                  # SpaDecModel
        inputs_embeds:     torch.Tensor,               # (bs, seq, hidden)
        input_ids:         torch.Tensor,               # (bs, seq)
        attention_mask:    torch.Tensor,               # (bs, seq)
        image_grid_thw:    torch.Tensor | None,
        rotated_image_xyz: list | None,                # list[k]: (H, W, 3)
        coord_scale:       float,
    ) -> torch.Tensor:
        """Decouple-mode LLM forward (Qwen stock 3D M-RoPE + separate XYZ RoPE).

        Mirrors train_correspondence.py's --decouple architecture:
          • Rotary 64 dims  → Qwen original 3D M-RoPE on (t, h, w) integer
                              position_ids (built via Qwen3_5Model.get_rope_index).
          • Pass-through    → SpaXYZRotaryEmbedding on per-token (R-rotated) xyz,
            dims 64..129     routed to SpaDecAttentionWrapper via
                              SpaDecTextModel._xyz_pos.

        Gradient w.r.t. R flows through the XYZ RoPE (SpaXYZRotaryEmbedding's
        @torch.no_grad has been removed for train_alternate.py's purposes).
        """
        # mm_token_type_ids: 1 = image, 0 = text (no video).
        mm_token_type_ids = (input_ids == self.image_token_id).int()

        # Standard Qwen 3D M-RoPE position_ids — (3, bs, seq) integer.
        position_ids, _ = spa_inner.get_rope_index(
            input_ids         = input_ids,
            mm_token_type_ids = mm_token_type_ids,
            image_grid_thw    = image_grid_thw,
            video_grid_thw    = None,
            attention_mask    = attention_mask,
        )

        # Per-token xyz (gradient-preserving): text → (0,0,0); image → R @ xyz.
        xyz_pos = _build_per_token_xyz(
            input_ids,
            self.image_token_id,
            rotated_image_xyz if rotated_image_xyz is not None else [],
            image_grid_thw,
            self.spatial_merge_size,
        )

        # Stash on SpaDecTextModel — its forward picks these up to build
        # xyz_position_embeddings and dispatches to SpaDecAttentionWrapper.
        text_model._xyz_pos     = xyz_pos
        text_model._coord_scale = float(coord_scale)
        text_model._polar       = False

        out = text_model(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            position_ids   = position_ids,
            use_cache      = False,
        )
        return out.last_hidden_state

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,              # (1, seq_len)
        attention_mask:   torch.Tensor,              # (1, seq_len)
        pixel_values:     torch.Tensor | None,
        image_grid_thw:   torch.Tensor | None,
        image_xyz:        list | None = None,        # list[k]: (llm_H, llm_W, 3)
        image_xyz_hires:  list | None = None,        # list[k]: (llm_H*up, llm_W*up, 3)
        labels:           torch.Tensor | None = None,
        coord_scale:      float = 100.0,
        use_rotation_enc: bool  = True,
        use_coord_loss:   bool  = True,
        use_relative:     bool  = False,
        detach_coord_hidden: bool = False,
        **kwargs,
    ):
        """Single-pass forward with differentiable RoPE.

        Returns:
            R:         (3, 3) predicted rotation (or None)
            loss:      combined scalar loss (or None)
            loss_dict: dict of per-loss floats
        """
        _ldict: dict = {}

        inner     = self._unwrap()        # SpaForConditionalGeneration
        spa_inner = inner.model           # SpaModel
        lm_head   = inner.lm_head
        text_model = spa_inner.language_model

        # ── Step 1: merged inputs_embeds (text embed + vision encoder scatter) ──
        inputs_embeds = spa_inner.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = spa_inner.get_image_features(
                pixel_values, image_grid_thw, return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(list(image_embeds), dim=0)
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1) \
                                                            .expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ── Step 2: rotation encoder on merged inputs_embeds (detached) ──
        # When use_rotation_enc is False (e.g. warm-up epoch), skip the
        # encoder entirely: R stays None → rotated_xyz = image_xyz (identity).
        R = None
        cam_feat = None
        if use_rotation_enc and image_xyz is not None and image_grid_thw is not None:
            R, cam_feat = self._call_rotation_enc(
                inputs_embeds      = inputs_embeds.detach(),
                input_ids          = input_ids,
                image_xyz          = image_xyz,
                image_grid_thw     = image_grid_thw,
                coord_scale        = coord_scale,
            )                                              # (3, 3) float32, (d_model,)
            _ldict["R_trace"] = R.trace().item()

        # ── Step 3: rotated xyz (FLOAT, gradient-preserving) ──
        if R is not None and image_xyz is not None:
            rotated_xyz = _apply_rotation_to_xyz(R, image_xyz)
        else:
            rotated_xyz = image_xyz

        # ── Step 4-5: LLM forward (mode-dependent) ────────────────────────
        if self.decouple:
            hidden2 = self._run_text_model_decouple(
                text_model        = text_model,
                spa_inner         = spa_inner,
                inputs_embeds     = inputs_embeds,
                input_ids         = input_ids,
                attention_mask    = attention_mask,
                image_grid_thw    = image_grid_thw,
                rotated_image_xyz = rotated_xyz,
                coord_scale       = coord_scale,
            )                                              # (bs, seq, hidden)
        else:
            position_ids_float = _build_float_position_ids(
                input_ids          = input_ids,
                attention_mask     = attention_mask,
                image_token_id     = self.image_token_id,
                image_xyz          = rotated_xyz,
                image_grid_thw     = image_grid_thw,
                spatial_merge_size = self.spatial_merge_size,
                coord_scale        = coord_scale,
            )                                              # (5, bs, seq) float
            hidden2 = self._run_text_model_manual(
                text_model          = text_model,
                inputs_embeds       = inputs_embeds,
                attention_mask      = attention_mask,
                position_ids_float  = position_ids_float,
            )                                              # (bs, seq, hidden)
        logits2 = lm_head(hidden2)                         # (bs, seq, vocab)

        # ── LM loss ───────────────────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits2[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits2.device)
            mask         = shift_labels[0] != -100
            lm_loss = F.cross_entropy(
                shift_logits[0, mask], shift_labels[0, mask],
            )
            _ldict["lm_loss"] = lm_loss.item()
        del logits2

        # ── Coordinate loss in rotated frame ─────────────────────────────
        # When use_coord_loss is False (--no_coord), skip the coord head and
        # its L1 loss entirely: coord_head receives no gradient, and AdamW's
        # `p.grad is None: continue` skip keeps it untouched.
        coord_loss   = None
        coord_gt_src = image_xyz_hires if image_xyz_hires is not None else image_xyz

        if use_coord_loss and coord_gt_src is not None and image_grid_thw is not None:
            # --relative: GT is the original (un-rotated) input coordinates;
            # cam_feat (detached) is passed to coord_head so it can learn the
            # inverse mapping from rotated RoPE features → original frame.
            # Default: GT is R @ xyz (rotated frame), no cam conditioning.
            if use_relative:
                coord_gt = coord_gt_src
                _cam_cond = cam_feat.detach() if cam_feat is not None else None
            else:
                if R is not None:
                    coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src)
                else:
                    coord_gt = coord_gt_src
                _cam_cond = None

            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True,
            )[0]
            sms   = self.spatial_merge_size
            start = 0
            per_img: list[torch.Tensor] = []

            for k in range(min(len(coord_gt), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w
                if start + n_tok > len(vis_pos):
                    break

                coord_h_k = hidden2[0, vis_pos[start: start + n_tok]]
                # --alternate Phase B: detach LLM hidden to prevent coord_loss
                # gradient from flowing through (cos, sin) → R → rotation_enc.
                if detach_coord_hidden:
                    coord_h_k = coord_h_k.detach()
                pred_k    = self.coord_head(coord_h_k, llm_h, llm_w,
                                            cam_feat=_cam_cond)

                gt_k = coord_gt[k].to(pred_k.device, dtype=pred_k.dtype)
                per_img.append(F.l1_loss(pred_k, gt_k))
                start += n_tok

            if per_img:
                coord_loss = torch.stack(per_img).mean()
                _ldict["coord_loss"] = coord_loss.item()

        # ── Combine losses ────────────────────────────────────────────────
        loss = None
        if lm_loss is not None:
            loss = self.answer_weight * lm_loss
        if coord_loss is not None:
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return R, loss, (_ldict if _ldict else None)

    # ------------------------------------------------------------------
    # Decomposed forward (used by train_rl.py — lets the caller cache
    # inputs_embeds across multiple LLM forwards with different R values
    # and decide per-call whether to enable autograd on the LLM path).
    # ------------------------------------------------------------------

    def encode_inputs(
        self,
        input_ids:      torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Step 1 only: text embed + vision feature scatter → inputs_embeds."""
        inner         = self._unwrap()
        spa_inner     = inner.model
        inputs_embeds = spa_inner.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = spa_inner.get_image_features(
                pixel_values, image_grid_thw, return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(list(image_embeds), dim=0)
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1) \
                                                            .expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def compute_losses_from_R(
        self,
        R:                  torch.Tensor | None,
        inputs_embeds:      torch.Tensor,
        input_ids:          torch.Tensor,
        attention_mask:     torch.Tensor,
        image_xyz:          list | None,
        image_xyz_hires:    list | None,
        image_grid_thw:     torch.Tensor | None,
        labels:             torch.Tensor | None,
        coord_scale:        float = 100.0,
        use_coord_loss:     bool  = True,
        use_relative:       bool  = False,
        detach_coord_hidden: bool = False,
        cam_feat:           torch.Tensor | None = None,
        compute_reward:     bool = False,
    ):
        """Steps 3-5 + losses: rotate xyz, build float position_ids, run LLM, losses.

        Unlike `forward`, this method does NOT run the rotation encoder — R is
        supplied directly. When called under torch.no_grad() it acts as a
        cheap reward-collection pass; called with grad it back-propagates
        lm_loss through R to whatever produced R (head_res path).

        Args:
            R: (3, 3) rotation matrix (or None → identity). The caller decides
               whether R has grad (for head_res update) or not (for reward
               collection).
            compute_reward: if True, add `_ldict["acc"]` (0/1 top-1 accuracy on
               the first non-masked answer token — used as binary reward).

        Returns:
            (lm_loss, coord_loss, _ldict).  Each loss may be None.
        """
        _ldict: dict = {}

        inner      = self._unwrap()
        spa_inner  = inner.model
        lm_head    = inner.lm_head
        text_model = spa_inner.language_model

        # Step 3: rotated xyz ------------------------------------------------
        if R is not None and image_xyz is not None:
            rotated_xyz = _apply_rotation_to_xyz(R, image_xyz)
        else:
            rotated_xyz = image_xyz

        # Step 4-5: LLM forward (mode-dependent) ----------------------------
        if self.decouple:
            hidden2 = self._run_text_model_decouple(
                text_model        = text_model,
                spa_inner         = spa_inner,
                inputs_embeds     = inputs_embeds,
                input_ids         = input_ids,
                attention_mask    = attention_mask,
                image_grid_thw    = image_grid_thw,
                rotated_image_xyz = rotated_xyz,
                coord_scale       = coord_scale,
            )
        else:
            position_ids_float = _build_float_position_ids(
                input_ids          = input_ids,
                attention_mask     = attention_mask,
                image_token_id     = self.image_token_id,
                image_xyz          = rotated_xyz,
                image_grid_thw     = image_grid_thw,
                spatial_merge_size = self.spatial_merge_size,
                coord_scale        = coord_scale,
            )
            hidden2 = self._run_text_model_manual(
                text_model         = text_model,
                inputs_embeds      = inputs_embeds,
                attention_mask     = attention_mask,
                position_ids_float = position_ids_float,
            )
        logits2 = lm_head(hidden2)

        # LM loss + optional binary accuracy (reward) -----------------------
        lm_loss = None
        if labels is not None:
            shift_logits = logits2[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits2.device)
            mask         = shift_labels[0] != -100
            if mask.any():
                lm_loss = F.cross_entropy(
                    shift_logits[0, mask], shift_labels[0, mask],
                )
                _ldict["lm_loss"] = lm_loss.item()
                if compute_reward:
                    first_pos = mask.nonzero(as_tuple=True)[0][0].item()
                    pred = shift_logits[0, first_pos].argmax(-1).item()
                    gt   = int(shift_labels[0, first_pos].item())
                    _ldict["acc"] = 1.0 if pred == gt else 0.0
        del logits2

        # Coord loss --------------------------------------------------------
        coord_loss   = None
        coord_gt_src = image_xyz_hires if image_xyz_hires is not None else image_xyz

        if use_coord_loss and coord_gt_src is not None and image_grid_thw is not None:
            if use_relative:
                coord_gt  = coord_gt_src
                _cam_cond = cam_feat.detach() if cam_feat is not None else None
            else:
                if R is not None:
                    coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src)
                else:
                    coord_gt = coord_gt_src
                _cam_cond = None

            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True,
            )[0]
            sms   = self.spatial_merge_size
            start = 0
            per_img: list[torch.Tensor] = []
            for k in range(min(len(coord_gt), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w
                if start + n_tok > len(vis_pos):
                    break
                coord_h_k = hidden2[0, vis_pos[start: start + n_tok]]
                if detach_coord_hidden:
                    coord_h_k = coord_h_k.detach()
                pred_k = self.coord_head(coord_h_k, llm_h, llm_w,
                                         cam_feat=_cam_cond)
                gt_k   = coord_gt[k].to(pred_k.device, dtype=pred_k.dtype)
                per_img.append(F.l1_loss(pred_k, gt_k))
                start += n_tok
            if per_img:
                coord_loss = torch.stack(per_img).mean()
                _ldict["coord_loss"] = coord_loss.item()

        return lm_loss, coord_loss, _ldict


# ---------------------------------------------------------------------------
# Chiral cube rotation group + so(3) exponential map (used by RL encoder)
# ---------------------------------------------------------------------------

def _build_chiral_cube_group() -> torch.Tensor:
    """24 proper rotations of the cube (orientation-preserving symmetries).

    Constructed as signed permutation matrices with det = +1:
    for each permutation σ of (0,1,2) and each sign assignment
    (s0, s1, s2) ∈ {-1, +1}³, build M with M[i, σ(i)] = s_i.  6 × 8 = 48
    matrices total; 24 have det = +1 (proper rotations).

    Returns:
        (24, 3, 3) float32 tensor.
    """
    mats: list[torch.Tensor] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            M = torch.zeros(3, 3, dtype=torch.float32)
            for i, j in enumerate(perm):
                M[i, j] = float(signs[i])
            if torch.det(M) > 0.5:
                mats.append(M)
    # Sort descending by trace so identity (trace=3, angle=0) is index 0.
    # Angle θ = acos((trace - 1) / 2), so higher trace = smaller angle;
    # R_bins[0] = I, and neighbours are the lowest-angle cube rotations.
    mats.sort(key=lambda M: -(M[0, 0] + M[1, 1] + M[2, 2]).item())
    R_bins = torch.stack(mats, dim=0)
    assert R_bins.shape == (24, 3, 3), f"unexpected shape {R_bins.shape}"
    assert torch.allclose(R_bins[0], torch.eye(3)), "R_bins[0] must be identity"
    return R_bins


def _build_yaw_group(n_bins: int = 24) -> torch.Tensor:
    """n_bins evenly-spaced yaw rotations Rot_z(k · 2π / n_bins).

    Policy only decides camera yaw (heading around +z) — pitch and roll
    are fixed to 0 so the action space is effectively SO(2). R_bins[0] = I.

    Returns:
        (n_bins, 3, 3) float32 tensor.
    """
    angles = torch.arange(n_bins, dtype=torch.float32) * (2.0 * math.pi / n_bins)
    c = torch.cos(angles)
    s = torch.sin(angles)
    R = torch.zeros(n_bins, 3, 3, dtype=torch.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    assert torch.allclose(R[0], torch.eye(3)), "R_bins[0] must be identity"
    return R


def _hat(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix of (3,) vector v (differentiable in v)."""
    vx, vy, vz = v[0], v[1], v[2]
    zero = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([zero,  -vz,   vy]),
        torch.stack([vz,    zero,  -vx]),
        torch.stack([-vy,   vx,    zero]),
    ])


def _rodrigues(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map so(3) → SO(3) via Rodrigues formula.

    R = I + (sinθ / θ) · hat(δ) + ((1 - cosθ) / θ²) · hat(δ)²
      where δ = axis_angle and θ = ||δ||.  Uses the normalized form
      (I + sinθ · K + (1 - cosθ) · K²) with K = hat(δ/θ), ε-safe.

    Args:
        axis_angle: (3,) float tensor; magnitude = rotation angle (radians),
                    direction = rotation axis.  Fully differentiable.

    Returns:
        (3, 3) rotation matrix.
    """
    theta = torch.linalg.vector_norm(axis_angle) + eps
    K     = _hat(axis_angle / theta)
    I     = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# CameraTokenRotationEncoderRL — 24-anchor + optional 3-DOF residual head
# ---------------------------------------------------------------------------

class CameraTokenRotationEncoderRL(nn.Module):
    """Shallow M-RoPE encoder predicting (yaw_anchor_logits, per-anchor yaw residual).

    Yaw-only policy
    ~~~~~~~~~~~~~~~
    The policy only decides camera yaw (heading around +z) — pitch and roll
    are fixed to 0 so the action space collapses from SO(3) to SO(2).
    R_bins spans 24 evenly-spaced yaw bins (every 15°), and each anchor's
    residual is a single scalar Δyaw ∈ [-residual_clamp, residual_clamp].

    Output heads
    ~~~~~~~~~~~~
    head_cls: Linear(d_model, 24)   → yaw-anchor logits (softmax = π(a|s))
    head_res: Linear(d_model, 24)   → (24,) scalar yaw residual per anchor,
                                       zero-init, active only in "hybrid" mode.

    Final rotation (for anchor k):
        R_final(k) = Rot_z(yaw_k + residual_clamp · tanh(residual_all[k]))

    With zero-init head_cls → uniform π₀ at step 0; entropy bonus in the
    RL objective prevents dead anchors.  With zero-init head_res → Δyaw = 0,
    so R_final(k) = R_bins[k] at step 0 and residual only learns corrections
    on anchors the policy has already identified as high-reward.

    Args:
        hidden_dim:      MLLM hidden size.
        mllm_head_dim:   MLLM head_dim (must match rope_emb).
        rope_emb:        SpaTextRotaryEmbedding shared with MLLM.
        nhead:           attention heads in this encoder (d_model = nhead × head_dim).
        dim_feedforward: FFN width.
        num_layers:      encoder depth.
        dropout:         MUST be 0.0 for RL training — the 24 no_grad reward
                         pass and the grad pass must be deterministic on the
                         same inputs so advantages are comparable.
        action_space:    "discrete" (head_cls only, head_res = None) or
                         "hybrid"   (head_cls + head_res).
        residual_clamp:  max Δyaw magnitude (rad) after tanh.
                         Default π/24 ≈ 7.5° = half-width of a 15° yaw bin,
                         so adjacent anchors do not overlap.
    """

    def __init__(
        self,
        hidden_dim:      int,
        mllm_head_dim:   int,
        rope_emb:        nn.Module,
        nhead:           int   = 4,
        dim_feedforward: int   = 2048,
        num_layers:      int   = 2,
        dropout:         float = 0.0,
        action_space:    str   = "hybrid",
        residual_clamp:  float = math.pi / 24,
        decouple:        bool  = False,
        xyz_rotary_emb:  nn.Module | None = None,
        xyz_offset:      int   = 64,
    ):
        super().__init__()
        assert action_space in ("discrete", "hybrid"), \
            f"action_space must be 'discrete' or 'hybrid', got {action_space}"
        assert mllm_head_dim > 0 and nhead > 0
        self.d_model        = nhead * mllm_head_dim
        self.nhead          = nhead
        self.mllm_head_dim  = mllm_head_dim
        self.action_space   = action_space
        self.residual_clamp = float(residual_clamp)

        self.decouple   = bool(decouple)
        self.xyz_offset = int(xyz_offset)
        if self.decouple:
            assert xyz_rotary_emb is not None, (
                "CameraTokenRotationEncoderRL(decouple=True) requires xyz_rotary_emb"
            )
            self.xyz_rotary_emb = xyz_rotary_emb
        else:
            self.xyz_rotary_emb = None

        self.cam_token = nn.Parameter(torch.empty(1, self.d_model))
        nn.init.normal_(self.cam_token, std=0.02)

        self.input_proj = nn.Linear(hidden_dim, self.d_model)

        self.layers = nn.ModuleList([
            _MRoPEEncoderLayer(
                d_model         = self.d_model,
                nhead           = nhead,
                dim_feedforward = dim_feedforward,
                rope_emb        = rope_emb,
                dropout         = dropout,
                decouple        = self.decouple,
                xyz_rotary_emb  = self.xyz_rotary_emb,
                xyz_offset      = self.xyz_offset,
            )
            for _ in range(num_layers)
        ])

        # Anchor classifier — zero-init → uniform π at step 0.
        self.head_cls = nn.Linear(self.d_model, 24)
        nn.init.zeros_(self.head_cls.weight)
        nn.init.zeros_(self.head_cls.bias)

        # Per-anchor 1-DoF yaw-residual head — zero-init → Δyaw = 0 at step 0.
        if action_space == "hybrid":
            self.head_res = nn.Linear(self.d_model, 24)
            nn.init.zeros_(self.head_res.weight)
            nn.init.zeros_(self.head_res.bias)
        else:
            self.head_res = None

        # Fixed 24-element yaw-bin group (persistent=False so we don't
        # pollute state_dict with a constant).
        R_bins = _build_yaw_group(24)
        self.register_buffer("R_bins", R_bins, persistent=False)

    def forward(
        self,
        hidden_states:   torch.Tensor,                 # (1, seq_len, hidden_dim)
        token_txyz_int:  torch.Tensor | None = None,   # (seq_len, 4) — non-decouple
        token_thw_int:   torch.Tensor | None = None,   # (seq_len, 3) — decouple
        xyz_pos:         torch.Tensor | None = None,   # (1, seq_len, 3) — decouple
        coord_scale:     float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Args (non-decouple): pass token_txyz_int (4D `[t, x_int, y_int, z_int]`).
        Args (decouple):     pass token_thw_int (3D `[t, h, w]`) + xyz_pos
                             ((1, seq, 3) float, gradient-OK if needed).

        Returns:
            logits:       (24,) float32 — yaw-anchor classifier logits.
            residual_all: (24,) float32 raw Δyaw output (pre-clamp),
                          or None if action_space=="discrete".
            cam_feat:     (d_model,) cam-token feature in encoder dtype.
        """
        device = hidden_states.device

        x = self.input_proj(hidden_states[0])
        cam = self.cam_token.to(dtype=x.dtype, device=device)
        x   = torch.cat([cam, x], dim=0)
        x   = x.unsqueeze(0)

        if self.decouple:
            assert token_thw_int is not None and xyz_pos is not None, (
                "rotation_enc(decouple=True): need token_thw_int + xyz_pos"
            )
            cam_pos  = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
            mllm_pos = token_thw_int.long().T.unsqueeze(1)              # (3, 1, seq)
            position_ids = torch.cat([cam_pos, mllm_pos], dim=2)        # (3, 1, seq+1)

            cam_xyz = torch.zeros(
                1, 1, 3, dtype=xyz_pos.dtype, device=device,
            )
            xyz_pos_full = torch.cat([cam_xyz, xyz_pos], dim=1)         # (1, seq+1, 3)

            for layer in self.layers:
                x = layer(
                    x, position_ids,
                    xyz_pos=xyz_pos_full, coord_scale=coord_scale,
                )
        else:
            assert token_txyz_int is not None, (
                "rotation_enc(non-decouple): need token_txyz_int"
            )
            cam_pos  = torch.zeros(4, 1, 1, dtype=torch.long, device=device)
            mllm_pos = token_txyz_int.long().T.unsqueeze(1)
            position_ids = torch.cat([cam_pos, mllm_pos], dim=2)

            for layer in self.layers:
                x = layer(x, position_ids)

        cam_feat = x[0, 0]
        logits   = self.head_cls(cam_feat).float()              # (24,)
        if self.head_res is not None:
            residual_all = self.head_res(cam_feat).float()       # (24,) scalar Δyaw per anchor
        else:
            residual_all = None
        return logits, residual_all, cam_feat

    def compose_R(
        self,
        k:            int,
        residual_all: torch.Tensor | None,
    ) -> torch.Tensor:
        """R_final = Rot_z(yaw_k + residual_clamp · tanh(residual_all[k])).

        In discrete mode (or residual_all None) returns R_bins[k] directly.
        Return dtype is float32 regardless of the model's bf16 cast, matching
        the dtype convention established by the 6-D rotation head path.
        """
        R_k = self.R_bins[k].float()
        if residual_all is None or self.head_res is None:
            return R_k
        dyaw_raw = residual_all[k]                               # scalar
        dyaw     = self.residual_clamp * torch.tanh(dyaw_raw)
        axis_ang = torch.stack([
            torch.zeros_like(dyaw), torch.zeros_like(dyaw), dyaw,
        ])                                                        # (3,)
        R_delta  = _rodrigues(axis_ang)                           # (3, 3)
        return R_k @ R_delta
