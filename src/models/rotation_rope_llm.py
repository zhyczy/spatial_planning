"""
rotation_rope_llm.py

RotationRoPEModel: differentiable-RoPE variant of RotationModel.

Difference from RotationModel
-----------------------------
In the original RotationModel, the predicted rotation R only receives
gradient from `rot_loss` (geodesic against gt_rotation), because the
path R → rotated_xyz → position_ids → RoPE is broken by two barriers
in spa_emb.py:

  (1) get_rope_index discretises with `.round().long()`  → long dtype
  (2) SpaTextRotaryEmbedding.forward is `@torch.no_grad()`

This model fixes both WITHOUT editing spa_emb.py, by:

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
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from transformers.utils.generic import maybe_autocast

from .coordinate_llm import DepthPredictionTransformer
from .rotation_llm import (
    CameraTokenRotationEncoder,
    _apply_rotation_to_xyz,
    _build_token_txyz_int,
)


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

    Same high-level layout as RotationModel, but Step 4/5 uses a
    hand-rolled transformer loop that takes a gradient-preserving RoPE
    (cos, sin) tuple computed from float position_ids.

    The gt_rotation / rot_loss branch is intentionally dropped — in this
    model the rotation encoder is trained only through lm_loss /
    coord_loss (back-propagated through the differentiable RoPE).
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
        if use_rotation_enc and image_xyz is not None and image_grid_thw is not None:
            token_txyz_int = _build_token_txyz_int(
                input_ids, self.image_token_id,
                image_xyz, image_grid_thw, self.spatial_merge_size,
                coord_scale,
            )                                              # (seq_len, 4) long
            R = self.rotation_enc(
                inputs_embeds.detach(),
                token_txyz_int,
            )                                              # (3, 3) float32
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
        coord_loss   = None
        coord_gt_src = image_xyz_hires if image_xyz_hires is not None else image_xyz

        if coord_gt_src is not None and image_grid_thw is not None:
            if R is not None:
                coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src)
            else:
                coord_gt = coord_gt_src

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
                pred_k    = self.coord_head(coord_h_k, llm_h, llm_w)

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
