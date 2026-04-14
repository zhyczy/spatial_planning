"""
rotation_llm.py

RotationModel: Two-pass approach for learning a canonical coordinate frame.

Overview
--------
Pass 1   Run MLLM with the original world-frame XYZ position embeddings.
         The merged text+visual input embeddings (i.e. the tensor fed into
         transformer layer 0) are captured via a layer-0 pre-hook (detached /
         under no_grad so no extra LLM backward is triggered).

Rotation encoder (CameraTokenRotationEncoder)
         Shallow TransformerEncoder that takes *all* MLLM input embeddings
         (merged text tokens + projected visual tokens, before any transformer
         layer) plus one learnable cam token as input.

         Position encoding is IDENTICAL to the MLLM backbone, aligned on
         four levels:
           1. Mechanism  — 4D M-RoPE applied inside each attention layer
                           on Q and K.  No additive absolute PE.
           2. Parameters — same SpaTextRotaryEmbedding (rope_theta=1e7,
                           mrope_section=[2,xyz,xyz,xyz]) initialised from
                           the identical config.text_config as the MLLM.
           3. Coordinates — mirrors MLLM get_rope_index:
                           text tokens get sequential (pos,pos,pos,pos);
                           image tokens get (t=start_pos, xyz=round(world*scale)).
                           Cam token sits at (0,0,0,0); text starts at pos=1.
           4. Dimensions  — head_dim = config.head_dim = 256 (MLLM's
                           explicit head_dim); rotary_dim = 64 (first 64 of
                           256 per head).  d_model = nhead × 256 so the
                           RoPE frequency mapping is byte-for-byte identical.

         The cam token output is projected to 6-D → rot6d_to_rotmat → R.

Pass 2   MLLM forward with *rotated* XYZ:
             xyz_rotated = R @ (xyz_world - cam_pos_0)
         This expresses each 3-D position in a canonical coordinate frame
         aligned with the first frame's camera orientation.

Coordinate head
         Decodes pass-2 hidden states → sub-pixel (x, y, z) predictions
         in the rotated frame.
         Ground truth is also rotated:
             gt_rotated = R.detach() @ (xyz_hires - cam_pos_0)

Losses
------
  lm_loss    Causal cross-entropy on answer tokens (pass-2 logits).
  rot_loss   geodesic(R, R_gt)  where R_gt = world-to-camera rotation
             of frame 0  (optional; requires gt_rotation in the batch).
  coord_loss L1 between coord head output and rotated GT.

Training note
-------------
Pass 1 runs under torch.no_grad(), so the LLM weights are only
trained through pass 2 (coord + LM losses).  The rotation encoder
parameters receive gradients from:
  * rot_loss  (direct)
  * coord_loss → coord_head → h2 → spa_model(pass2) → rotated_xyz → R
"""

import re

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
    origin:   torch.Tensor,   # (3,) subtract before rotating
) -> list:
    """Return [R @ (xyz - origin) for xyz in xyz_list].

    Preserves original dtype of each tensor in the list.
    """
    result   = []
    origin_f = origin.float()
    for xyz in xyz_list:
        orig_dtype = xyz.dtype
        xyz_flat   = xyz.reshape(-1, 3).float()
        centered   = xyz_flat - origin_f.to(xyz_flat.device)
        rotated    = (R @ centered.T).T                    # (N, 3)
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
    Coordinates: integer (t, x, y, z) = round((world_xyz - origin) * scale).
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

        # Learnable cam token (sits at world origin after centring)
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
        self.rot_head = nn.Linear(self.d_model, 6)

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

        # Predict rotation from cam token output (index 0)
        cam_feat = x[0, 0].float()                               # (d_model,)
        r6d      = self.rot_head(cam_feat)                       # (6,)
        R        = rot6d_to_rotmat(r6d.unsqueeze(0)).squeeze(0)  # (3, 3)
        return R


# ---------------------------------------------------------------------------
# RotationModel
# ---------------------------------------------------------------------------

class RotationModel(nn.Module):
    """Two-pass rotation-aware coordinate prediction model.

    Pass 1  MLLM forward with original world-frame XYZ position embeddings.
            Merged text+visual input embeddings captured via layer-0 pre-hook
            (pass 1 runs under torch.no_grad() → no gradient through LLM).

    Rotation encoder
            CameraTokenRotationEncoder: all MLLM input embeddings + cam token,
            4D M-RoPE identical to MLLM → R.

    Pass 2  MLLM forward with rotated XYZ = R @ (xyz - cam_pos_0).

    Coordinate head
            Decodes pass-2 hidden states → xyz in rotated frame.
            GT: R.detach() @ (xyz_hires - cam_pos_0).

    Args:
        spa_model:           SpaForConditionalGeneration + LoRA
        rotation_enc:        CameraTokenRotationEncoder
        coord_head:          DepthPredictionTransformer
        image_token_id:      token id of <|image_pad|>
        spatial_merge_size:  vision spatial merge factor
        skip_layers:         layer indices for coord head hidden states
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
        skip_layers:        tuple[int, ...] = (-1,),
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
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.rot_weight         = rot_weight

        # Counters / buffers used by forward hooks.
        self._pass_counter      = 0
        self._input_embeds_1:  torch.Tensor | None = None   # pass-1 layer-0 input
        self._lm_head_input_2: torch.Tensor | None = None   # pass-2 lm_head input

        # Hook 1: first transformer layer → merged text+visual input embeddings
        # args[0] at layer-0 == inputs_embeds (shape: 1, seq_len, hidden_size)
        for name, mod in self.spa_model.named_modules():
            if re.search(r'\.layers\.0$', name):
                mod.register_forward_pre_hook(self._capture_input_embeds)
                break

        # Hook 2: lm_head → last-layer hidden states (used by coord head, pass 2)
        for name, mod in self.spa_model.named_modules():
            if name.endswith("lm_head"):
                mod.register_forward_pre_hook(self._capture_lm_input)
                break

    # ------------------------------------------------------------------

    def _capture_input_embeds(self, module, args):
        """Layer-0 pre-hook: capture merged text+visual input embeddings (pass 1)."""
        if self._pass_counter == 1:
            self._input_embeds_1 = args[0]   # (1, seq_len, hidden_size), detached via no_grad

    def _capture_lm_input(self, module, args):
        """lm_head pre-hook: capture last-layer hidden states (pass 2 only)."""
        if self._pass_counter == 2:
            self._lm_head_input_2 = args[0]

    def _get_hidden(self, outputs, pass_num: int) -> torch.Tensor:
        if pass_num == 1:
            return self._input_embeds_1      # text+visual embeddings → rotation encoder
        # pass 2: coord head uses last-layer hidden states
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        if only_last:
            return self._lm_head_input_2
        return outputs.hidden_states[self.skip_layers[0]]

    def _spa_forward(
        self,
        input_ids, attention_mask, pixel_values, image_grid_thw,
        image_xyz, coord_scale: float, pass_num: int,
    ):
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        self._pass_counter = pass_num
        return self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = not only_last,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,              # (1, seq_len)
        attention_mask:   torch.Tensor,              # (1, seq_len)
        pixel_values:     torch.Tensor | None,
        image_grid_thw:   torch.Tensor | None,
        image_xyz:        list | None = None,        # list[k]: (llm_H, llm_W, 3)
        image_xyz_hires:  list | None = None,        # list[k]: (llm_H*up, llm_W*up, 3)
        cam_pos_frame0:   torch.Tensor | None = None,  # (3,) first-frame cam pos
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

        origin = (cam_pos_frame0.float() if cam_pos_frame0 is not None
                  else torch.zeros(3, device=input_ids.device))

        # ── Pass 1: original XYZ position embedding (no gradient) ────────────
        with torch.no_grad():
            out1    = self._spa_forward(
                input_ids, attention_mask, pixel_values, image_grid_thw,
                image_xyz, coord_scale, pass_num=1,
            )
            hidden1 = self._get_hidden(out1, 1)   # (1, seq_len, hidden_dim)
        del out1

        # ── Rotation encoder: predict R from pass-1 hidden states ─────────────
        R = None
        if image_xyz is not None and image_grid_thw is not None:
            # Build integer 4D positions matching MLLM get_rope_index convention.
            # Text: t=x=y=z=sequential (starts at 1, 0 reserved for cam token).
            # Image: t=current_pos, xyz=round(world*scale).
            token_txyz_int = _build_token_txyz_int(
                input_ids, self.image_token_id,
                image_xyz, image_grid_thw, self.spatial_merge_size,
                coord_scale,
            )                                              # (seq_len, 4) long

            R = self.rotation_enc(
                hidden1,           # already detached (from no_grad context)
                token_txyz_int,
            )                                              # (3, 3) float32
            _ldict["R_trace"] = R.trace().item()

        # ── Build rotated XYZ for pass 2 ─────────────────────────────────────
        if R is not None and image_xyz is not None:
            rotated_xyz = _apply_rotation_to_xyz(R, image_xyz, origin)
        else:
            rotated_xyz = image_xyz

        # ── Pass 2: rotated XYZ position embedding (full gradient) ───────────
        out2    = self._spa_forward(
            input_ids, attention_mask, pixel_values, image_grid_thw,
            rotated_xyz, coord_scale, pass_num=2,
        )
        hidden2 = self._get_hidden(out2, 2)
        logits2 = out2.logits
        del out2

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
                coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src, origin)
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
