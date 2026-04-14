"""
camera_movement_rl.py

Camera pose estimation via Learnable Cam Token + Shallow Transformer Encoder.

Four-step forward propagation:

  Step 1: Frozen MLLM Feature Extraction
    Input: image I, instruction P → frozen MLLM → text tokens T = [t_1, t_2, ..., t_n]

  Step 2: Sequence Concatenation with Learnable Cam Token
    Introduce learnable C_init, concatenate: X = [C_init, t_1, t_2, ..., t_n]

  Step 3: Shallow Transformer Encoder with Self-Attention
    X' = TransformerEncoder(X)
    C_out = X'[0]  (extract first position: cam token after attention)

  Step 4: Rotation Matrix Prediction via 6D Continuous Rotation
    C_out → MLP → 6D vector → Gram-Schmidt Orthogonalization → R (3×3)

Key design:
  - MLLM completely frozen: protected from catastrophic forgetting
  - Learnable cam token: acts as active query, aggregates geometric information
  - Shallow encoder: minimal trainable parameters, efficient feature fusion
  - 6D rotation representation: ensures SO(3) constraint via Gram-Schmidt
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor
from peft import PeftConfig

from src.models.correspondence_llm import rot6d_to_rotmat
from src.loss.cam_loss import geodesic_loss

log = logging.getLogger(__name__)


class CamTokenRotationHead(nn.Module):
    """
    Regression head: 6D rotation vector prediction from cam token.

    Input: (hidden_dim,) or (batch_size, hidden_dim)
    Output: (6,) or (batch_size, 6)  —  6D rotation (first two columns of rotation matrix)
    """

    def __init__(self, hidden_dim: int = 2560, internal_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, 6),
        )

    def forward(self, cam_feature: torch.Tensor) -> torch.Tensor:
        """
        cam_feature: (..., hidden_dim) → (..., 6)
        """
        return self.mlp(cam_feature)


class CameraMovementRL(nn.Module):
    """
    Camera pose estimation model with frozen MLLM backbone.

    Architecture:
      1. Frozen SpaForConditionalGeneration (MLLM + LoRA in inference)
      2. Learnable Cam Token (per-sample initialization)
      3. Shallow Transformer Encoder (attention fusion)
      4. 6D Rotation Head (predicts rotation via Gram-Schmidt)

    Args:
        spa_model:          frozen SpaForConditionalGeneration
        image_token_id:     token id of <|image_pad|>
        hidden_dim:         hidden size of LLM (typically 2560)
        d_encoder:          dimension of transformer encoder (default 512)
        encoder_nhead:      number of attention heads (default 8)
        encoder_depth:      number of transformer encoder layers (default 2)
        dropout:            dropout rate (default 0.05)
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        image_token_id:     int,
        hidden_dim:         int = 2560,
        d_encoder:          int = 512,
        encoder_nhead:      int = 8,
        encoder_depth:      int = 2,
        dropout:            float = 0.05,
    ):
        super().__init__()
        self.spa_model      = spa_model
        self.image_token_id = image_token_id
        self.hidden_dim     = hidden_dim
        self.d_encoder      = d_encoder

        # ── Step 1: Freeze all MLLM parameters ────────────────────────────────
        for param in self.spa_model.parameters():
            param.requires_grad_(False)

        # ── Step 2: Learnable Cam Token ───────────────────────────────────────
        # Initialize cam token as a learnable embedding (1, hidden_dim)
        self.cam_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ── Step 3: Shallow Transformer Encoder ───────────────────────────────
        # Project from MLLM hidden_dim to encoder d_encoder
        self.input_proj = nn.Linear(hidden_dim, d_encoder)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_encoder,
            nhead          = encoder_nhead,
            dim_feedforward= d_encoder * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,  # pre-norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
        )

        # Project back from encoder d_encoder to hidden_dim for regression head
        self.output_proj = nn.Linear(d_encoder, hidden_dim)

        # ── Step 4: 6D Rotation Head ──────────────────────────────────────────
        self.rotation_head = CamTokenRotationHead(hidden_dim, internal_dim=1024)

    def forward(
        self,
        input_ids:      torch.Tensor,          # (batch_size, seq_len)
        attention_mask: torch.Tensor,          # (batch_size, seq_len)
        pixel_values:   torch.Tensor | None,   # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor | None,   # (num_images, 3)  [T, H, W]
        gt_transforms:  torch.Tensor | None = None,  # (batch_size, 4, 4)
        image_xyz:      list | None = None,    # unused in camera movement
        coord_scale:    float = 100.0,
        **kwargs,
    ):
        """
        Forward pass: MLLM extraction → cam token fusion → encoder → rotation prediction.

        Returns:
            preds_6d:    (batch_size, 6)  —  6D rotation prediction
            R_pred:      (batch_size, 3, 3)  —  3×3 rotation matrix
            loss_dict:   dict with rotation loss (if gt_transforms provided)
        """
        batch_size = input_ids.shape[0]
        device     = input_ids.device

        # ── Step 1: Frozen MLLM Forward Pass ──────────────────────────────────
        with torch.no_grad():
            outputs = self.spa_model(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                pixel_values         = pixel_values,
                image_grid_thw       = image_grid_thw,
                output_hidden_states = False,  # only last hidden state needed
                return_dict          = True,
                image_xyz            = image_xyz,
                coord_scale          = coord_scale,
                **kwargs,
            )
            # Extract last hidden state: (batch_size, seq_len, hidden_dim)
            lm_hidden = outputs.last_hidden_state

        # ── Step 2: Concatenate Learnable Cam Token ───────────────────────────
        # cam_token: (1, 1, hidden_dim) → broadcast to (batch_size, 1, hidden_dim)
        cam_tok_batch = self.cam_token.expand(batch_size, -1, -1)  # (bs, 1, hidden_dim)

        # Concatenate cam token with MLLM output tokens
        # X = [C_init, t_1, t_2, ..., t_n]
        X = torch.cat([cam_tok_batch, lm_hidden], dim=1)  # (bs, 1 + seq_len, hidden_dim)

        # Extend attention mask: prepend 1 (True) for cam token position
        cam_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        X_attention_mask = torch.cat([cam_mask, attention_mask], dim=1)  # (bs, 1 + seq_len)

        # ── Step 3: Shallow Transformer Encoder ───────────────────────────────
        # Project to encoder dimension
        X_proj = self.input_proj(X)  # (bs, 1 + seq_len, d_encoder)

        # Self-attention: cam token (X[0]) attends to all tokens including itself
        X_encoded = self.transformer_encoder(
            X_proj,
            src_key_padding_mask=(X_attention_mask == 0),  # True for padding
        )  # (bs, 1 + seq_len, d_encoder)

        # Project back to hidden_dim
        X_out = self.output_proj(X_encoded)  # (bs, 1 + seq_len, hidden_dim)

        # ── Step 4: Extract Cam Token & Predict Rotation ──────────────────────
        # C_out = X'[0]  (first position: enriched cam token)
        cam_out = X_out[:, 0, :]  # (batch_size, hidden_dim)

        # Predict 6D rotation
        preds_6d = self.rotation_head(cam_out)  # (batch_size, 6)

        # Convert 6D to 3×3 rotation matrix via Gram-Schmidt
        R_pred = rot6d_to_rotmat(preds_6d)  # (batch_size, 3, 3)

        # ── Compute Loss (if ground truth provided) ───────────────────────────
        loss_dict = {}
        rotation_loss = None

        if gt_transforms is not None:
            # gt_transforms: (batch_size, 4, 4) or similar
            # Extract rotation matrices: (batch_size, 3, 3)
            K = min(R_pred.shape[0], gt_transforms.shape[0])
            R_gt = gt_transforms[:K, :3, :3].to(R_pred.device, dtype=R_pred.dtype)
            R_pred_k = R_pred[:K]

            # Geodesic loss: rotation distance on SO(3)
            rotation_loss = geodesic_loss(R_pred_k, R_gt)
            loss_dict["rotation_loss"] = rotation_loss.item()

        return {
            "preds_6d": preds_6d,
            "R_pred": R_pred,
            "rotation_loss": rotation_loss,
            "loss_dict": loss_dict,
        }

    def get_trainable_params(self):
        """Return list of trainable parameters (excluding frozen MLLM)."""
        return [
            p for p in self.parameters()
            if p.requires_grad and p is not self.spa_model
        ]

    def get_trainable_param_count(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())

    def print_trainable_parameters(self):
        """Print trainable parameter count and percentage."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_param_count()
        pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {total_params:,} || "
            f"trainable%: {pct:.2f}%"
        )


# ── Model Builder Functions ───────────────────────────────────────────────────


def build_model_from_checkpoint(
    adapter_path: str,
    base_model_path: str | None = None,
    hidden_dim: int = 2560,
    d_encoder: int = 512,
    encoder_nhead: int = 8,
    encoder_depth: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[CameraMovementRL, int]:
    """
    Load frozen MLLM from LoRA checkpoint and build CameraMovementRL.

    Args:
        adapter_path:    path to LoRA checkpoint (e.g., train_records/.../step_550)
        base_model_path: base model path (auto-detected from adapter config if None)
        hidden_dim:      LLM hidden size (default 2560 for Qwen3.5-4B)
        d_encoder:       transformer encoder dimension
        encoder_nhead:   number of attention heads
        encoder_depth:   number of transformer encoder layers
        dtype:           torch dtype (default bfloat16)

    Returns:
        model:           CameraMovementRL instance
        image_token_id:  token id of <|image_pad|>
    """
    # ── Load LoRA config to get base model path if not provided ────────────
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    if base_model_path is None:
        base_model_path = adapter_config.base_model_name_or_path
        if base_model_path is None:
            raise ValueError(
                f"Could not determine base_model_path from {adapter_path}. "
                "Please provide base_model_path explicitly."
            )

    log.info(f"Loading base model from: {base_model_path}")
    log.info(f"Loading LoRA adapter from: {adapter_path}")

    # ── Load base model config ──────────────────────────────────────────────
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    # ── Load processor for image_token_id ───────────────────────────────────
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    log.info(f"<|image_pad|> token id = {image_token_id}")

    # ── Load base SpaForConditionalGeneration ──────────────────────────────
    from src.models import SpaForConditionalGeneration
    from peft import PeftModel

    spa_model = SpaForConditionalGeneration.from_pretrained(
        base_model_path,
        config              = config,
        torch_dtype         = dtype,
        attn_implementation = "sdpa",
    )
    log.info("Base SpaForConditionalGeneration loaded")

    # ── Load LoRA adapter into base model ──────────────────────────────────
    spa_model = PeftModel.from_pretrained(spa_model, adapter_path)
    log.info("LoRA adapter loaded and merged")

    # ── Freeze all MLLM parameters ─────────────────────────────────────────
    for param in spa_model.parameters():
        param.requires_grad_(False)
    log.info("All MLLM parameters frozen")

    # ── Create CameraMovementRL ────────────────────────────────────────────
    model = CameraMovementRL(
        spa_model           = spa_model,
        image_token_id      = image_token_id,
        hidden_dim          = hidden_dim,
        d_encoder           = d_encoder,
        encoder_nhead       = encoder_nhead,
        encoder_depth       = encoder_depth,
    )
    model.to(dtype)
    log.info("CameraMovementRL model created")
    model.print_trainable_parameters()

    return model, image_token_id
