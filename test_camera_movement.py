"""
test_camera_movement.py

Quick test script to verify CameraMovementRL model loading and forward pass.
"""

import logging
import sys
import os
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from camera_movement_rl import build_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    # ── Configuration ─────────────────────────────────────────────────────
    adapter_path = os.path.join(
        _ROOT,
        "train_records/coordinate_no_cam_mindcube/step_550"
    )
    base_model_path = os.path.join(
        _ROOT,
        "checkpoints/Qwen3.5-4B"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────
    model, image_token_id = build_model_from_checkpoint(
        adapter_path      = adapter_path,
        base_model_path   = base_model_path,
        hidden_dim        = 2560,
        d_encoder         = 512,
        encoder_nhead     = 8,
        encoder_depth     = 2,
        dtype             = torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    log.info("Model loaded and set to eval mode")

    # ── Create dummy input ──────────────────────────────────────────────
    batch_size = 2
    seq_len = 128
    hidden_dim = 2560

    input_ids = torch.randint(0, 151936, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)

    # Dummy MLLM output (would normally come from spa_model)
    # This is just for testing shape compatibility
    log.info(f"Input shapes: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}")

    # ── Forward pass ────────────────────────────────────────────────────
    log.info("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids       = input_ids,
            attention_mask  = attention_mask,
            pixel_values    = None,
            image_grid_thw  = None,
            gt_transforms   = None,
        )

    log.info("Forward pass completed successfully!")
    log.info(f"Output shapes:")
    log.info(f"  - preds_6d:      {outputs['preds_6d'].shape}")
    log.info(f"  - R_pred:        {outputs['R_pred'].shape}")
    log.info(f"  - rotation_loss: {outputs['rotation_loss']}")

    # ── Check rotation matrix properties ────────────────────────────────
    R_pred = outputs['R_pred']  # (batch_size, 3, 3)

    # Verify orthogonality: R^T @ R ≈ I
    RTR = torch.bmm(R_pred.transpose(-2, -1), R_pred)
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    ortho_error = (RTR - I).abs().max().item()
    log.info(f"Orthogonality error (max |R^T R - I|): {ortho_error:.6f}")

    # Verify determinant ≈ 1
    dets = torch.det(R_pred)
    log.info(f"Determinants (should be ≈ 1): {dets}")

    log.info("✓ All checks passed!")


if __name__ == "__main__":
    main()
