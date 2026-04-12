import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss.cam_loss import geodesic_loss, cycle_consistency_loss
from .correspondence_llm import rot6d_to_rotmat, PoseRegressionHead


class DepthPredictionTransformer(nn.Module):
    """Two-layer transformer for per-patch 3D coordinate prediction.

    Architecture:
      1. Linear projection:  hidden_dim → d_model
      2. Add 2D sinusoidal positional encoding (variable grid size)
      3. Two TransformerEncoderLayer (pre-norm, batch_first)
      4. Linear projection:  d_model → 3 * upscale²
      5. PixelShuffle(upscale) → (h*upscale, w*upscale, 3)

    Sinusoidal 2D PE is regenerated on-the-fly for each (h, w), so the head
    handles variable-resolution images without any learned position parameters.
    """

    def __init__(
        self,
        hidden_dim:      int   = 2560,
        d_model:         int   = 512,
        nhead:           int   = 8,
        dim_feedforward: int   = 2048,
        upscale_factor:  int   = 4,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.d_model        = d_model

        self.input_proj  = nn.Linear(hidden_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj  = nn.Linear(d_model, 3 * (upscale_factor ** 2))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def _sinusoidal_2d_pe(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> torch.Tensor:
        """2D sinusoidal positional encoding.  Returns (h*w, d_model)."""
        half = self.d_model // 2          # row enc and col enc each use half dims

        # Shared frequency bands
        dim_idx = torch.arange(half, device=device, dtype=torch.float32)
        div     = torch.pow(10000.0, 2 * (dim_idx // 2) / half)  # (half,)

        row_idx = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)  # (h,1)
        col_idx = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1)  # (w,1)

        row_enc = torch.zeros(h, half, device=device)
        row_enc[:, 0::2] = torch.sin(row_idx / div[0::2])
        row_enc[:, 1::2] = torch.cos(row_idx / div[1::2])

        col_enc = torch.zeros(w, half, device=device)
        col_enc[:, 0::2] = torch.sin(col_idx / div[0::2])
        col_enc[:, 1::2] = torch.cos(col_idx / div[1::2])

        # Broadcast and concat: (h, w, d_model)
        row_enc = row_enc.unsqueeze(1).expand(h, w, half)
        col_enc = col_enc.unsqueeze(0).expand(h, w, half)
        pe = torch.cat([row_enc, col_enc], dim=-1).reshape(h * w, self.d_model)
        return pe.to(dtype)

    def forward(
        self,
        hidden: torch.Tensor,   # (h*w, hidden_dim)
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (h*w, hidden_dim) — vision token hidden states
            h:      LLM patch grid height
            w:      LLM patch grid width
        Returns:
            (h*upscale, w*upscale, 3) — predicted xyz at sub-pixel resolution
        """
        x  = self.input_proj(hidden)                                    # (h*w, d_model)
        x  = x + self._sinusoidal_2d_pe(h, w, x.device, x.dtype)       # add 2D PE
        x  = self.transformer(x.unsqueeze(0)).squeeze(0)                # (h*w, d_model)
        x  = self.output_proj(x)                                        # (h*w, 3*up²)
        x  = x.view(1, h, w, -1).permute(0, 3, 1, 2)                   # (1, 3*up², h, w)
        x  = self.pixel_shuffle(x)                                      # (1, 3, h*up, w*up)
        return x[0].permute(1, 2, 0)                                    # (h*up, w*up, 3)


class CoordinatePlusModel(nn.Module):
    """
    SpaForConditionalGeneration (+ LoRA) with three supervision heads:

      1. PoseRegressionHead    → rot + trans + cycle loss   (at <pose> tokens)
      2. LM cross-entropy      → answer prediction loss     (at answer tokens)
      3. DepthPredictionTransformer → xyz coordinate loss
           - vision tokens (<|image_pad|>) per LLM patch per image
           - each token's hidden state decoded directly to (x, y, z) via MLP

    Args:
        spa_model:          backbone (SpaForConditionalGeneration + LoRA)
        pose_head:          PoseRegressionHead
        coord_head:         DepthPredictionTransformer
        pose_token_id:      token id of <pose>
        image_token_id:     token id of <|image_pad|>
        spatial_merge_size: spatial merge factor used by the vision encoder (2)
        skip_layers:        layer indices for skip-connection concat → pose head
        answer_weight:      scale factor for LM loss
        coord_weight:       scale factor for coordinate loss
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        pose_head:          PoseRegressionHead,
        coord_head:         DepthPredictionTransformer,
        pose_token_id:      int,
        image_token_id:     int,
        spatial_merge_size: int,
        skip_layers:        tuple[int, ...] = (-1,),
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
        polar:              bool  = False,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.pose_head          = pose_head
        self.coord_head         = coord_head
        self.pose_token_id      = pose_token_id
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.polar              = polar

        # Hook on lm_head to capture last hidden state without
        # output_hidden_states=True (which defeats gradient checkpointing).
        self._lm_head_input: torch.Tensor | None = None
        for name, mod in self.spa_model.named_modules():
            if name.endswith("lm_head"):
                mod.register_forward_pre_hook(self._capture_lm_input)
                break

    def _capture_lm_input(self, module, args):
        """Pre-hook: capture the input to lm_head (= post-norm last hidden state)."""
        self._lm_head_input = args[0]

    def forward(
        self,
        input_ids:      torch.Tensor,           # (1, seq_len)
        attention_mask: torch.Tensor,           # (1, seq_len)
        pixel_values:   torch.Tensor | None,    # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor | None,    # (num_images, 3)  [T, H, W]
        gt_transforms:  torch.Tensor | None = None,  # (K, 4, 4)
        image_xyz:      list | None = None,     # list[i] = (llm_H_i, llm_W_i, 3) for RoPE
        image_xyz_hires: list | None = None,    # list[i] = (llm_H_i*up, llm_W_i*up, 3) for coord loss
        coord_scale:    float = 100.0,
        cycle_weight:   float = 0.0,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        # ── backbone (single forward pass) ────────────────────────────────────
        # When skip_layers==[-1], avoid output_hidden_states=True so that
        # gradient checkpointing can actually discard intermediate activations.
        # The lm_head pre-hook captures the last hidden state for us.
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = not only_last,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # ── hidden states for pose & coord heads ─────────────────────────────
        if only_last:
            # Captured by lm_head pre-hook (post-norm last hidden state)
            hidden_pose  = self._lm_head_input
            hidden_coord = self._lm_head_input
        else:
            if len(self.skip_layers) == 1:
                hidden_pose  = outputs.hidden_states[self.skip_layers[0]]
                hidden_coord = outputs.hidden_states[self.skip_layers[0]]
            else:
                hidden_pose = torch.cat(
                    [outputs.hidden_states[i] for i in self.skip_layers], dim=-1
                )
                hidden_coord = outputs.hidden_states[self.skip_layers[-1]]
        logits = outputs.logits
        del outputs

        # ── pose prediction ───────────────────────────────────────────────────
        pose_positions = (input_ids[0] == self.pose_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(pose_positions) == 0:
            return None, None, None

        latents  = hidden_pose[0, pose_positions]    # (K, input_dim)
        preds_9d = self.pose_head(latents)            # (K, 9)

        pose_loss = None
        _ldict: dict = {}
        if gt_transforms is not None:
            K  = preds_9d.shape[0]
            gt = gt_transforms[:K].to(preds_9d.device, dtype=preds_9d.dtype)
            R_gt   = gt[:, :3, :3]
            t_gt   = gt[:, :3,  3]
            R_pred = rot6d_to_rotmat(preds_9d[:, :6])
            t_pred = preds_9d[:, 6:]
            rot_loss   = geodesic_loss(R_pred, R_gt)
            trans_loss = F.smooth_l1_loss(t_pred, t_gt, beta=0.5)
            pose_loss  = rot_loss + trans_loss
            _ldict = {"rot_loss": rot_loss.item(), "trans_loss": trans_loss.item()}
            if cycle_weight > 0.0:
                N = round((1 + math.sqrt(1 + 4 * K)) / 2)
                if N * (N - 1) == K:
                    c_r, c_t  = cycle_consistency_loss(R_pred, t_pred, N)
                    c_loss    = cycle_weight * (c_r + c_t)
                    pose_loss = pose_loss + c_loss
                    _ldict["cycle_loss"]   = c_loss.item()
                    _ldict["cycle_r_loss"] = c_r.item()
                    _ldict["cycle_t_loss"] = c_t.item()
            _ldict["pose_loss"] = pose_loss.item()

        # ── LM answer-prediction loss ─────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]                   # (1, seq_len-1, V)
            shift_labels = labels[:, 1:].to(logits.device)     # (1, seq_len-1)

            # 只取有效 label 位置的 logits，避免整个 (seq_len, V) 留在显存里
            mask = shift_labels[0] != -100                     # (seq_len-1,)
            shift_logits = shift_logits[0, mask]               # (N_valid, V)
            shift_labels = shift_labels[0, mask]               # (N_valid,)
            lm_loss = F.cross_entropy(shift_logits, shift_labels)

        # ── per-patch 3D coordinate prediction (PixelShuffle sub-pixel) ────────
        # One <coord> token per LLM patch; Linear+PixelShuffle decodes each to
        # (upscale, upscale) sub-pixels of (x,y,z).  GT is image_xyz_hires.
        coord_loss = None
        coord_gt = image_xyz_hires if image_xyz_hires is not None else image_xyz
        if coord_gt is not None and image_grid_thw is not None:
            # Use vision token positions directly (no <coord> tokens needed)
            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True
            )[0]

            sms   = self.spatial_merge_size
            start = 0
            per_img_losses: list[torch.Tensor] = []

            for k in range(min(len(coord_gt), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w

                if start + n_tok > len(vis_pos):
                    break

                coord_h_k = hidden_coord[0, vis_pos[start : start + n_tok]]
                pred_k = self.coord_head(coord_h_k, llm_h, llm_w)  # (llm_h*up, llm_w*up, 3)

                gt_k = coord_gt[k].to(pred_k.device, dtype=pred_k.dtype)
                if self.polar:
                    # (r, θ, α): weight angles by 1/π so a π-radian error equals
                    # a 1-unit r error in contribution.
                    w = pred_k.new_tensor([1.0, 1.0 / math.pi, 1.0 / math.pi])
                    per_img_losses.append((pred_k - gt_k).abs().mul(w).mean())
                else:
                    per_img_losses.append(F.l1_loss(pred_k, gt_k))

                start += n_tok

            if per_img_losses:
                coord_loss = torch.stack(per_img_losses).mean()

        # ── combine losses ────────────────────────────────────────────────────
        loss = None
        if pose_loss is not None:
            loss = pose_loss
        if lm_loss is not None:
            _ldict["lm_loss"] = lm_loss.item()
            loss = (loss + self.answer_weight * lm_loss) if loss is not None \
                   else (self.answer_weight * lm_loss)
        if coord_loss is not None:
            _ldict["coord_loss"] = coord_loss.item()
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return preds_9d, loss, (_ldict if _ldict else None)


class CoordinateModel(nn.Module):
    """
    Ablation of CoordinatePlusModel: pose regression removed entirely.

    Only two supervision signals:
      1. LM cross-entropy      → answer prediction loss    (at answer tokens)
      2. DepthPredictionTransformer → xyz coordinate loss    (at vision tokens)

    No <pose> tokens needed; no PoseRegressionHead; no gt_transforms.
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        coord_head:         DepthPredictionTransformer,
        image_token_id:     int,
        spatial_merge_size: int,
        skip_layers:        tuple[int, ...] = (-1,),
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
        polar:              bool  = False,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.coord_head         = coord_head
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight
        self.polar              = polar

        # Pre-hook on lm_head to capture last hidden state without
        # output_hidden_states=True (preserves gradient checkpointing savings).
        # Only used when skip_layers == [-1].
        self._lm_head_input: torch.Tensor | None = None
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        if only_last:
            for name, mod in self.spa_model.named_modules():
                if name.endswith("lm_head"):
                    mod.register_forward_pre_hook(self._capture_lm_input)
                    break

    def _capture_lm_input(self, module, args):
        self._lm_head_input = args[0]

    def forward(
        self,
        input_ids:       torch.Tensor,           # (1, seq_len)
        attention_mask:  torch.Tensor,           # (1, seq_len)
        pixel_values:    torch.Tensor | None,    # (total_patches, C, H, W)
        image_grid_thw:  torch.Tensor | None,    # (num_images, 3)
        gt_transforms:   torch.Tensor | None = None,   # ignored (ablation)
        image_xyz:       list | None = None,
        image_xyz_hires: list | None = None,
        coord_scale:     float = 100.0,
        cycle_weight:    float = 0.0,            # ignored (ablation)
        labels:          torch.Tensor | None = None,
        **kwargs,
    ):
        # ── backbone ─────────────────────────────────────────────────────────
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = not only_last,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # ── hidden states for coord head ───────────────────────────────────
        if only_last:
            # Captured by lm_head pre-hook (post-norm last hidden state)
            hidden_coord = self._lm_head_input
        else:
            hidden_coord = outputs.hidden_states[self.skip_layers[0]]

        logits = outputs.logits
        del outputs

        _ldict: dict = {}

        # ── LM answer-prediction loss ────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits.device)
            mask         = shift_labels[0] != -100
            lm_loss = F.cross_entropy(
                shift_logits[0, mask], shift_labels[0, mask]
            )

        # ── per-patch 3D coordinate prediction ──────────────────────────────
        coord_loss = None
        coord_gt   = image_xyz_hires if image_xyz_hires is not None else image_xyz
        if coord_gt is not None and image_grid_thw is not None:
            # Use vision token positions directly (no <coord> tokens needed)
            vis_pos = (input_ids[0] == self.image_token_id).nonzero(
                as_tuple=True
            )[0]

            sms   = self.spatial_merge_size
            start = 0
            per_img_losses: list[torch.Tensor] = []

            for k in range(min(len(coord_gt), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w

                if start + n_tok > len(vis_pos):
                    break

                coord_h_k = hidden_coord[0, vis_pos[start : start + n_tok]]
                pred_k    = self.coord_head(coord_h_k, llm_h, llm_w)

                gt_k = coord_gt[k].to(pred_k.device, dtype=pred_k.dtype)
                if self.polar:
                    w = pred_k.new_tensor([1.0, 1.0 / math.pi, 1.0 / math.pi])
                    per_img_losses.append((pred_k - gt_k).abs().mul(w).mean())
                else:
                    per_img_losses.append(F.l1_loss(pred_k, gt_k))
                start += n_tok

            if per_img_losses:
                coord_loss = torch.stack(per_img_losses).mean()

        # ── combine losses ────────────────────────────────────────────────────
        loss = None
        if lm_loss is not None:
            _ldict["lm_loss"] = lm_loss.item()
            loss = self.answer_weight * lm_loss
        if coord_loss is not None:
            _ldict["coord_loss"] = coord_loss.item()
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return None, loss, (_ldict if _ldict else None)
