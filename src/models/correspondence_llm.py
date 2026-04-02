import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss.cam_loss import geodesic_loss, cycle_consistency_loss

def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D continuous rotation → 3×3 rotation matrix via Gram-Schmidt.
    (..., 6)  →  (..., 3, 3)

    The first two columns of R are encoded in r6d[..., :3] and r6d[..., 3:6].
    Reference: Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.
    """
    # Cast to float32: bfloat16 lacks precision for Gram-Schmidt (eps=1e-6 ≈ 0
    # in bf16, causing 0/0 NaN when a1 ∥ a2 at random initialisation).
    orig_dtype = r6d.dtype
    r6d = r6d.float()
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as columns: R = [b1 | b2 | b3], restore original dtype
    return torch.stack([b1, b2, b3], dim=-1).to(orig_dtype)   # (..., 3, 3)


class PoseRegressionHead(nn.Module):
    """
    MLP regression head: input_dim → 9D pose vector.
    Output layout: first 6 dims = 6D rotation, last 3 dims = translation.

    input_dim = hidden_size * len(skip_layers), e.g.:
      skip_layers=[-1]     → input_dim = 2560  (last layer only)
      skip_layers=[-4,-1]  → input_dim = 5120  (concat of two layers)
    """

    def __init__(self, input_dim: int = 2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 9),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (input_dim,) or (N, input_dim) → (9,) or (N, 9)"""
        return self.mlp(latent)


class SpaCorrespondenceModel(nn.Module):
    """
    SpaForConditionalGeneration (+ LoRA) wrapped with a PoseRegressionHead.

    Forward:
      1. Runs the backbone with output_hidden_states=True.
      2. Finds all <pose> token positions in input_ids.
      3. Concatenates hidden states from skip_layers at each <pose> position
         (skip connection from intermediate layers preserves geometric detail).
      4. Passes each latent through PoseRegressionHead → 9D prediction.
      5. Optionally computes geodesic + translation + cycle-consistency loss.

    skip_layers: list of layer indices into outputs.hidden_states, e.g.
      [-1]      → last layer only  (input_dim = hidden_size)
      [-4, -1]  → concat 4th-from-last + last  (input_dim = 2 * hidden_size)

    Returns:
      preds_9d : (K, 9) tensor  —  K = number of <pose> tokens found
      loss     : scalar or None
    """

    def __init__(
        self,
        spa_model:      nn.Module,
        pose_head:      PoseRegressionHead,
        pose_token_id:  int,
        skip_layers:    tuple[int, ...] = (-1,),
    ):
        super().__init__()
        self.spa_model     = spa_model
        self.pose_head     = pose_head
        self.pose_token_id = pose_token_id
        self.skip_layers   = list(skip_layers)

    def forward(
        self,
        input_ids:      torch.Tensor,                    # (1, seq_len)
        attention_mask: torch.Tensor,                    # (1, seq_len)
        pixel_values:   torch.Tensor | None,             # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor | None,             # (num_images, 3)
        gt_transforms:  torch.Tensor | None = None,      # (K, 4, 4)
        image_xyz:      list | None = None,              # list of (llm_H, llm_W, 3)
        coord_scale:    float = 100.0,
        cycle_weight:   float = 0.0,
        **kwargs,
    ):
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = True,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # Skip connection: concatenate hidden states from selected layers
        # outputs.hidden_states: tuple[num_layers+1] of (1, seq_len, hidden_dim)
        if len(self.skip_layers) == 1:
            hidden = outputs.hidden_states[self.skip_layers[0]]    # (1, seq, H)
        else:
            hidden = torch.cat(
                [outputs.hidden_states[i] for i in self.skip_layers], dim=-1
            )                                                        # (1, seq, H*n)

        # Locate <pose> tokens (batch item 0)
        pose_positions = (input_ids[0] == self.pose_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(pose_positions) == 0:
            return None, None, None

        # Predict pose at each <pose> token
        latents  = hidden[0, pose_positions]       # (K, input_dim)
        preds_9d = self.pose_head(latents)          # (K, 9)

        if gt_transforms is None:
            return preds_9d, None, None

        K  = preds_9d.shape[0]
        gt = gt_transforms[:K].to(preds_9d.device, dtype=preds_9d.dtype)  # (K, 4, 4)

        R_gt   = gt[:, :3, :3]                       # (K, 3, 3)
        t_gt   = gt[:, :3,  3]                       # (K, 3)
        R_pred = rot6d_to_rotmat(preds_9d[:, :6])    # (K, 3, 3)
        t_pred = preds_9d[:, 6:]                     # (K, 3)

        rot_loss   = geodesic_loss(R_pred, R_gt)
        trans_loss = F.smooth_l1_loss(t_pred, t_gt, beta=0.5)
        loss = rot_loss + trans_loss

        _ldict: dict = {"rot_loss": rot_loss.item(), "trans_loss": trans_loss.item()}

        # Cycle consistency (rotation + translation)
        if cycle_weight > 0.0:
            N = round((1 + math.sqrt(1 + 4 * K)) / 2)
            if N * (N - 1) == K:                     # valid A(N,2) count
                c_r, c_t = cycle_consistency_loss(R_pred, t_pred, N)
                c_loss = cycle_weight * (c_r + c_t)
                loss   = loss + c_loss
                _ldict["cycle_loss"]   = c_loss.item()
                _ldict["cycle_r_loss"] = c_r.item()
                _ldict["cycle_t_loss"] = c_t.item()

        _ldict["pose_loss"] = loss.item()
        return preds_9d, loss, _ldict


class CorrespondencePlusModel(SpaCorrespondenceModel):
    """
    SpaCorrespondenceModel + auxiliary LM answer-prediction loss.

    When ``labels`` (1, seq_len) are provided the model additionally
    computes causal cross-entropy on the answer tokens and adds it
    (scaled by ``answer_weight``) to the pose regression loss.
    """

    def __init__(
        self,
        spa_model:      nn.Module,
        pose_head:      PoseRegressionHead,
        pose_token_id:  int,
        skip_layers:    tuple[int, ...] = (-1,),
        answer_weight:  float = 1.0,
    ):
        super().__init__(spa_model, pose_head, pose_token_id, skip_layers)
        self.answer_weight = answer_weight

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        gt_transforms:  torch.Tensor | None = None,
        image_xyz:      list | None = None,
        coord_scale:    float = 100.0,
        cycle_weight:   float = 0.0,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = True,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # ── pose prediction ───────────────────────────────────────────────────
        if len(self.skip_layers) == 1:
            hidden = outputs.hidden_states[self.skip_layers[0]]
        else:
            hidden = torch.cat(
                [outputs.hidden_states[i] for i in self.skip_layers], dim=-1
            )

        pose_positions = (input_ids[0] == self.pose_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(pose_positions) == 0:
            return None, None, None

        latents  = hidden[0, pose_positions]
        preds_9d = self.pose_head(latents)

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
            logits = outputs.logits                          # (1, seq_len, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # ── combine ───────────────────────────────────────────────────────────
        # _ldict already holds rot_loss / trans_loss / cycle_loss / pose_loss
        if pose_loss is not None and lm_loss is not None:
            loss = pose_loss + self.answer_weight * lm_loss
            _ldict["lm_loss"] = lm_loss.item()
        elif pose_loss is not None:
            loss = pose_loss
        elif lm_loss is not None:
            loss = self.answer_weight * lm_loss
            _ldict["lm_loss"] = lm_loss.item()
        else:
            loss = None

        return preds_9d, loss, (_ldict if _ldict else None)
