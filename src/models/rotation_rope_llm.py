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


# ---------------------------------------------------------------------------
# M-RoPE attention + encoder layer (used only by CameraTokenRotationEncoder)
# ---------------------------------------------------------------------------

class _MRoPEAttention(nn.Module):
    """Multi-head self-attention with 4D M-RoPE on Q and K.

    Uses the MLLM's SpaTextRotaryEmbedding (same inv_freq, same mrope_section)
    and the MLLM's head_dim so the rotary frequency mapping is identical.

    Args:
        d_model:   total model width = nhead × mllm_head_dim
        nhead:     number of attention heads
        rope_emb:  SpaTextRotaryEmbedding initialised from the SAME
                   config.text_config as the MLLM (fixed buffers, no grad)
        dropout:   attention dropout probability
    """

    def __init__(
        self,
        d_model:  int,
        nhead:    int,
        rope_emb: nn.Module,
        dropout:  float = 0.0,
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

    def forward(
        self,
        x:            torch.Tensor,   # (1, T, d_model)
        position_ids: torch.Tensor,   # (4, 1, T) long — [t, x_int, y_int, z_int]
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class _MRoPEEncoderLayer(nn.Module):
    """Pre-norm encoder layer: M-RoPE attention + FFN."""

    def __init__(
        self,
        d_model:         int,
        nhead:           int,
        dim_feedforward: int,
        rope_emb:        nn.Module,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _MRoPEAttention(d_model, nhead, rope_emb, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:            torch.Tensor,   # (1, T, d_model)
        position_ids: torch.Tensor,   # (4, 1, T) long
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), position_ids))
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
    ):
        super().__init__()
        assert mllm_head_dim > 0 and nhead > 0
        self.d_model       = nhead * mllm_head_dim   # e.g. 4 × 256 = 1024
        self.nhead         = nhead
        self.mllm_head_dim = mllm_head_dim

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
        hidden_states:   torch.Tensor,   # (1, seq_len, hidden_dim)
        token_txyz_int:  torch.Tensor,   # (seq_len, 4) long [t, x_int, y_int, z_int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:   MLLM token hidden states from pass 1 (detached)
            token_txyz_int:  integer 4D positions matching MLLM convention.
                             Produced by _build_token_txyz_int().
        Returns:
            R:        (3, 3) float32 rotation matrix (SO(3))
            cam_feat: (d_model,) cam-token output in encoder dtype
        """
        device = hidden_states.device

        x = self.input_proj(hidden_states[0])

        cam = self.cam_token.to(dtype=x.dtype, device=device)   # (1, d_model)
        x   = torch.cat([cam, x], dim=0)                        # (seq_len+1, d_model)

        cam_pos  = torch.zeros(4, 1, 1, dtype=torch.long, device=device)
        mllm_pos = token_txyz_int.long().T.unsqueeze(1)          # (4, 1, seq_len)
        position_ids = torch.cat([cam_pos, mllm_pos], dim=2)     # (4, 1, seq_len+1)

        x = x.unsqueeze(0)
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

    Only the Cartesian / no-coord-token path is implemented — this model is
    used from the RotationRoPE training pipeline which does not insert
    <coord> text tokens.
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
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.rotation_enc       = rotation_enc
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight

        # Build our differentiable RoPE from the MLLM's own rotary_emb so
        # that inv_freq / mrope_section / attention_scaling match byte-for-byte.
        inner       = self._unwrap()
        spa_inner   = inner.model                            # SpaModel
        spa_rotary  = spa_inner.language_model.rotary_emb    # SpaTextRotaryEmbedding
        self.diff_rope = DifferentiableMRoPE.from_spa_rotary(spa_rotary)

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
            token_txyz_int = _build_token_txyz_int(
                input_ids, self.image_token_id,
                image_xyz, image_grid_thw, self.spatial_merge_size,
                coord_scale,
            )                                              # (seq_len, 4) long
            R, cam_feat = self.rotation_enc(
                inputs_embeds.detach(),
                token_txyz_int,
            )                                              # (3, 3) float32, (d_model,)
            _ldict["R_trace"] = R.trace().item()

        # ── Step 3: rotated xyz (FLOAT, gradient-preserving) ──
        if R is not None and image_xyz is not None:
            rotated_xyz = _apply_rotation_to_xyz(R, image_xyz)
        else:
            rotated_xyz = image_xyz

        # ── Step 4: build FLOAT 5D position_ids (no discretization) ──
        position_ids_float = _build_float_position_ids(
            input_ids          = input_ids,
            attention_mask     = attention_mask,
            image_token_id     = self.image_token_id,
            image_xyz          = rotated_xyz,
            image_grid_thw     = image_grid_thw,
            spatial_merge_size = self.spatial_merge_size,
            coord_scale        = coord_scale,
        )                                                  # (5, bs, seq) float

        # ── Step 5: manual text-model pass with differentiable RoPE ──
        hidden2 = self._run_text_model_manual(
            text_model          = text_model,
            inputs_embeds       = inputs_embeds,
            attention_mask      = attention_mask,
            position_ids_float  = position_ids_float,
        )                                                  # (bs, seq, hidden)
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

        # Step 4: float 5D position_ids --------------------------------------
        position_ids_float = _build_float_position_ids(
            input_ids          = input_ids,
            attention_mask     = attention_mask,
            image_token_id     = self.image_token_id,
            image_xyz          = rotated_xyz,
            image_grid_thw     = image_grid_thw,
            spatial_merge_size = self.spatial_merge_size,
            coord_scale        = coord_scale,
        )

        # Step 5: manual LLM forward with differentiable RoPE ---------------
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
    """Shallow M-RoPE encoder predicting (anchor_logits, per-anchor residual).

    Output heads
    ~~~~~~~~~~~~
    head_cls: Linear(d_model, 24)   → anchor logits (softmax = π(a|s))
    head_res: Linear(d_model, 72)   → (24, 3) axis-angle residual per anchor,
                                       zero-init, active only in "hybrid" mode.

    Final rotation (for anchor k):
        R_final(k) = R_bins[k] @ exp(hat(residual_clamp · tanh(residual_all[k])))

    With zero-init head_cls → uniform π₀ at step 0; entropy bonus in the
    RL objective prevents dead anchors.  With zero-init head_res → ΔR = I,
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
        residual_clamp:  max axis-angle magnitude (rad) per DoF after tanh.
                         Default π/6 ≈ 30° — chosen so adjacent cube anchors
                         (separated by 60°–90° geodesic) do not overlap.
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
        residual_clamp:  float = math.pi / 6,
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
            )
            for _ in range(num_layers)
        ])

        # Anchor classifier — zero-init → uniform π at step 0.
        self.head_cls = nn.Linear(self.d_model, 24)
        nn.init.zeros_(self.head_cls.weight)
        nn.init.zeros_(self.head_cls.bias)

        # Per-anchor 3-DoF residual head — zero-init → ΔR = I at step 0.
        if action_space == "hybrid":
            self.head_res = nn.Linear(self.d_model, 24 * 3)
            nn.init.zeros_(self.head_res.weight)
            nn.init.zeros_(self.head_res.bias)
        else:
            self.head_res = None

        # Fixed 24-element chiral cube rotation group (persistent=False so
        # we don't pollute state_dict with a constant).
        R_bins = _build_chiral_cube_group()
        self.register_buffer("R_bins", R_bins, persistent=False)

    def forward(
        self,
        hidden_states:  torch.Tensor,   # (1, seq_len, hidden_dim)
        token_txyz_int: torch.Tensor,   # (seq_len, 4) long
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Returns:
            logits:       (24,) float32 — anchor classifier logits.
            residual_all: (24, 3) float32 raw axis-angle output (pre-clamp),
                          or None if action_space=="discrete".
            cam_feat:     (d_model,) cam-token feature in encoder dtype.
        """
        device = hidden_states.device

        x = self.input_proj(hidden_states[0])
        cam = self.cam_token.to(dtype=x.dtype, device=device)
        x   = torch.cat([cam, x], dim=0)

        cam_pos  = torch.zeros(4, 1, 1, dtype=torch.long, device=device)
        mllm_pos = token_txyz_int.long().T.unsqueeze(1)
        position_ids = torch.cat([cam_pos, mllm_pos], dim=2)

        x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids)

        cam_feat = x[0, 0]
        logits   = self.head_cls(cam_feat).float()              # (24,)
        if self.head_res is not None:
            residual_all = self.head_res(cam_feat).float().view(24, 3)
        else:
            residual_all = None
        return logits, residual_all, cam_feat

    def compose_R(
        self,
        k:            int,
        residual_all: torch.Tensor | None,
    ) -> torch.Tensor:
        """R_final = R_bins[k] @ exp(hat(residual_clamp · tanh(residual_all[k]))).

        In discrete mode (or residual_all None) returns R_bins[k] directly.
        Return dtype is float32 regardless of the model's bf16 cast, matching
        the dtype convention established by the 6-D rotation head path.
        """
        R_k = self.R_bins[k].float()
        if residual_all is None or self.head_res is None:
            return R_k
        delta_raw = residual_all[k]                              # (3,)
        delta     = self.residual_clamp * torch.tanh(delta_raw)
        R_delta   = _rodrigues(delta)                            # (3, 3)
        return R_k @ R_delta
