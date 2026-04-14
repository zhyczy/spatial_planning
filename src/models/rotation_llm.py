"""
rotation_llm.py

RotationModel: single-pass rotation-aware coordinate prediction.

Overview
--------
Step 1   Build merged input embeddings (text embed + vision encoder +
         masked_scatter) WITHOUT running any transformer layer.
         Vision encoder is wrapped in no_grad inside SpaModel.get_image_features.

Step 2   Feed the merged input embeddings to CameraTokenRotationEncoder
         → predicts canonical rotation R (3, 3).

Step 3   Rotate xyz:  xyz_rot = R @ xyz_world.

Step 4   Single MLLM transformer pass with:
             - inputs_embeds = merged text + visual features (reused from Step 1)
             - position_ids  = built via SpaModel.get_rope_index(image_xyz=xyz_rot)
         → last_hidden_state → lm_head → logits
                            → coord_head → per-patch xyz in rotated frame

Rotation encoder (CameraTokenRotationEncoder)
         Shallow TransformerEncoder over [cam_token | merged_inputs_embeds].
         4D M-RoPE identical to MLLM:
           1. Mechanism  — rotary PE applied inside each attention layer on Q and K.
           2. Parameters — same SpaTextRotaryEmbedding (rope_theta, mrope_section).
           3. Coordinates — mirrors MLLM get_rope_index: text sequential from 1,
                           image t=start_pos shared, xyz=round(world*scale).
                           Cam token at (0, 0, 0, 0).
           4. Dimensions  — head_dim = mllm_head_dim = 256; rotary_dim = 64.

Losses
------
  lm_loss    Causal cross-entropy on answer tokens.
  rot_loss   geodesic(R, R_gt)  (requires gt_rotation in batch).
  coord_loss L1 between coord head output and rotated GT.

Training note
-------------
There is no separate no-grad pass any more. The merged inputs_embeds is
used both for (a) the rotation encoder (detached, so rot_loss only updates
CameraTokenRotationEncoder params) and (b) the single language_model pass
(with grad, so LoRA + coord_head + lm_head get updated).
Gradient does NOT flow from coord/lm losses back into R because
get_rope_index discretises rotated_xyz with .round().long().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from src.loss.cam_loss import geodesic_loss
from .correspondence_llm import rot6d_to_rotmat
from .coordinate_llm import DepthPredictionTransformer
from .spa_emb import SpaTextRotaryEmbedding


# ---------------------------------------------------------------------------
# Helpers
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
# M-RoPE attention + encoder layer
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

        # Shared RoPE (fixed inv_freq buffer — not a trainable parameter)
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
        # q, k, v: (1, nhead, T, head_dim)

        # cos/sin from SpaTextRotaryEmbedding: (1, T, rotary_dim=64)
        # apply_rotary_pos_emb rotates the first `rotary_dim` dims of each head
        # (same as MLLM: first 64 of 256 per head).
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
# CameraTokenRotationEncoder
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

        # Learnable cam token
        self.cam_token = nn.Parameter(torch.empty(1, self.d_model))
        nn.init.normal_(self.cam_token, std=0.02)

        # Project MLLM hidden states → d_model
        self.input_proj = nn.Linear(hidden_dim, self.d_model)

        # Encoder layers — each uses the shared SpaTextRotaryEmbedding
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

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states:   torch.Tensor,   # (1, seq_len, hidden_dim)
        token_txyz_int:  torch.Tensor,   # (seq_len, 4) long [t, x_int, y_int, z_int]
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:   MLLM token hidden states from pass 1 (detached)
            token_txyz_int:  integer 4D positions matching MLLM convention.
                             Produced by _build_token_txyz_int() in RotationModel.
        Returns:
            R: (3, 3) float32 rotation matrix (SO(3))
        """
        device = hidden_states.device

        # Project MLLM tokens  →  (seq_len, d_model)
        x = self.input_proj(hidden_states[0])

        # Cam token at position (0, 0, 0, 0)  —  global anchor
        cam = self.cam_token.to(dtype=x.dtype, device=device)   # (1, d_model)
        x   = torch.cat([cam, x], dim=0)                        # (seq_len+1, d_model)

        # Build 4D position_ids for ALL tokens:
        #   [cam_token | mllm_tokens]  →  (4, 1, seq_len+1)
        cam_pos  = torch.zeros(4, 1, 1, dtype=torch.long, device=device)
        mllm_pos = token_txyz_int.long().T.unsqueeze(1)          # (4, 1, seq_len)
        position_ids = torch.cat([cam_pos, mllm_pos], dim=2)     # (4, 1, seq_len+1)

        # Encoder  →  (1, seq_len+1, d_model)
        x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, position_ids)

        # Predict rotation from cam token output (index 0).
        # Keep cam_feat in the encoder's dtype (bf16) so it matches rot_head
        # weights; cast to float32 only for the Gram-Schmidt step, which
        # needs higher precision to stay on SO(3).
        cam_feat = x[0, 0]                                       # (d_model,) bf16
        r6d      = self.rot_head(cam_feat).float()               # (6,) float32
        R        = rot6d_to_rotmat(r6d.unsqueeze(0)).squeeze(0)  # (3, 3)
        return R


# ---------------------------------------------------------------------------
# RotationModel
# ---------------------------------------------------------------------------

class RotationModel(nn.Module):
    """Single-pass rotation-aware coordinate prediction model.

    Step 1  Build merged inputs_embeds (text embed + vision encoder + scatter)
            without running any transformer layer.
    Step 2  CameraTokenRotationEncoder(inputs_embeds.detach()) → R.
    Step 3  rotated_xyz = R @ xyz_world.
    Step 4  Single language_model forward with inputs_embeds + rotated_xyz RoPE
            → hidden_states → lm_head / coord_head.

    Args:
        spa_model:           SpaForConditionalGeneration + LoRA
        rotation_enc:        CameraTokenRotationEncoder
        coord_head:          DepthPredictionTransformer
        image_token_id:      token id of <|image_pad|>
        spatial_merge_size:  vision spatial merge factor
        answer_weight:       weight for LM loss
        coord_weight:        weight for coordinate loss
        rot_weight:          weight for rotation geodesic loss
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
        rot_weight:         float = 1.0,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.rotation_enc       = rotation_enc
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.rot_weight         = rot_weight

    # ------------------------------------------------------------------

    def _unwrap(self) -> nn.Module:
        """Return the underlying SpaForConditionalGeneration (strip PEFT wrappers)."""
        m = self.spa_model
        while hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
            if m is self.spa_model:
                break
        return m

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,              # (1, seq_len)
        attention_mask:   torch.Tensor,              # (1, seq_len)
        pixel_values:     torch.Tensor | None,
        image_grid_thw:   torch.Tensor | None,
        image_xyz:        list | None = None,        # list[k]: (llm_H, llm_W, 3)
        image_xyz_hires:  list | None = None,        # list[k]: (llm_H*up, llm_W*up, 3)
        gt_rotation:      torch.Tensor | None = None,  # (3, 3) world-to-cam R_gt
        labels:           torch.Tensor | None = None,
        coord_scale:      float = 100.0,
        **kwargs,
    ):
        """
        Returns:
            R:         (3, 3) predicted rotation (or None)
            loss:      combined scalar loss (or None)
            loss_dict: dict of per-loss floats
        """
        _ldict: dict = {}

        inner     = self._unwrap()        # SpaForConditionalGeneration
        spa_inner = inner.model           # SpaModel
        lm_head   = inner.lm_head

        # ── Step 1: merged inputs_embeds (text embed + vision encoder scatter) ──
        inputs_embeds = spa_inner.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = spa_inner.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(list(image_embeds), dim=0)
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1) \
                                                            .expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ── Step 2: Rotation encoder on merged inputs_embeds (detached) ──
        R = None
        if image_xyz is not None and image_grid_thw is not None:
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

        # ── Step 3: rotated xyz ──
        if R is not None and image_xyz is not None:
            rotated_xyz = _apply_rotation_to_xyz(R, image_xyz)
        else:
            rotated_xyz = image_xyz

        # ── Step 4: position_ids from rotated xyz via SpaModel.get_rope_index ──
        mm_token_type_ids = (input_ids == self.image_token_id).to(torch.int32)
        position_ids, _deltas = spa_inner.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            image_xyz=rotated_xyz,
            coord_scale=coord_scale,
        )                                                  # (5, batch, seq_len)

        # ── Step 5: single transformer pass via language_model ──
        lm_out = spa_inner.language_model(
            input_ids      = None,
            inputs_embeds  = inputs_embeds,
            position_ids   = position_ids,
            attention_mask = attention_mask,
            return_dict    = True,
        )
        hidden2 = lm_out.last_hidden_state                 # (1, seq_len, hidden_dim)
        logits2 = lm_head(hidden2)                         # (1, seq_len, vocab)

        # ── LM loss (pass-2 logits) ───────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits2[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits2.device)
            mask         = shift_labels[0] != -100
            lm_loss = F.cross_entropy(
                shift_logits[0, mask], shift_labels[0, mask]
            )
            _ldict["lm_loss"] = lm_loss.item()
        del logits2

        # ── Rotation geodesic loss ────────────────────────────────────────────
        rot_loss = None
        if R is not None and gt_rotation is not None:
            R_gt     = gt_rotation.to(R.device, dtype=R.dtype)
            rot_loss = geodesic_loss(R.unsqueeze(0), R_gt.unsqueeze(0))
            _ldict["rot_loss"] = rot_loss.item()

        # ── Coordinate loss in rotated frame ─────────────────────────────────
        coord_loss   = None
        coord_gt_src = image_xyz_hires if image_xyz_hires is not None else image_xyz

        if coord_gt_src is not None and image_grid_thw is not None:
            if R is not None:
                coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src)
            else:
                coord_gt = coord_gt_src

            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True
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

        # ── Combine losses ────────────────────────────────────────────────────
        loss = None
        if lm_loss is not None:
            loss = self.answer_weight * lm_loss
        if rot_loss is not None:
            loss = (loss + self.rot_weight * rot_loss) if loss is not None \
                   else (self.rot_weight * rot_loss)
        if coord_loss is not None:
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return R, loss, (_ldict if _ldict else None)
