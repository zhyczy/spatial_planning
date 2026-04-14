"""
train_camera_movement.py

Training script for CameraMovementRL: camera pose estimation with frozen MLLM.

This script:
  1. Loads frozen MLLM from a pretrained LoRA checkpoint
  2. Freezes all MLLM parameters
  3. Trains only the learnable cam token + shallow encoder + rotation head

Usage:
  python train_camera_movement.py \\
      --adapter_path train_records/coordinate_no_cam_mindcube/step_550 \\
      --output_dir checkpoints/camera_movement_rl \\
      --num_epochs 10 \\
      --learning_rate 1e-4
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from camera_movement_rl import build_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DDP Helpers ───────────────────────────────────────────────────────────

local_rank: int = 0
world_size: int = 1


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def collate_fn(batch):
    """Batch collation for single-sample batches."""
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


def main(args: argparse.Namespace) -> None:
    """Main training loop."""
    global local_rank, world_size

    # ── DDP Initialization ────────────────────────────────────────────────
    _env_rank = os.environ.get("LOCAL_RANK")
    if _env_rank is not None:
        local_rank = int(_env_rank)
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        log.info(f"DDP: local_rank={local_rank}  world_size={world_size}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Single-GPU / CPU mode")

    # ── Create output directory ───────────────────────────────────────────
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log.info(f"Output directory: {args.output_dir}")

    # ── Load Model ────────────────────────────────────────────────────────
    log.info("Loading model from checkpoint...")
    model, image_token_id = build_model_from_checkpoint(
        adapter_path     = args.adapter_path,
        base_model_path  = args.base_model_path,
        hidden_dim       = args.hidden_dim,
        d_encoder        = args.d_encoder,
        encoder_nhead    = args.encoder_nhead,
        encoder_depth    = args.encoder_depth,
        dtype            = torch.bfloat16,
    )
    model = model.to(device)
    rank0_print(f"Model loaded. Device memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GiB")

    # ── DDP Wrapper ───────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        _model = model.module
    else:
        _model = model

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Only optimize trainable parameters (cam token, encoder, rotation head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr       = args.learning_rate,
        weight_decay = args.weight_decay,
    )
    rank0_print(f"Optimizer created with {len(trainable_params):,} trainable parameters")

    # ── Scheduler (optional) ──────────────────────────────────────────────
    if args.warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor  = 0.1,
            total_iters   = args.warmup_steps,
        )
    else:
        scheduler = None

    # ── Placeholder: Dataset Loading ──────────────────────────────────────
    # TODO: Implement dataset loading for camera poses
    # For now, we just show the structure
    log.info(f"Would load dataset from: {args.data_dir}")
    log.info("(Dataset implementation needed)")

    # ── Placeholder: Training Loop ────────────────────────────────────────
    log.info("Training setup complete. Waiting for dataset implementation.")
    rank0_print(
        f"\nTraining Configuration:\n"
        f"  - Epochs:         {args.num_epochs}\n"
        f"  - Learning rate:  {args.learning_rate}\n"
        f"  - Warmup steps:   {args.warmup_steps}\n"
        f"  - Output dir:     {args.output_dir}\n"
    )

    # Example training skeleton (placeholder):
    """
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids       = batch['input_ids'],
                attention_mask  = batch['attention_mask'],
                pixel_values    = batch.get('pixel_values'),
                image_grid_thw  = batch.get('image_grid_thw'),
                gt_transforms   = batch.get('gt_transforms'),
            )

            # Loss computation
            rotation_loss = outputs['rotation_loss']
            if rotation_loss is not None:
                rotation_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                train_loss += rotation_loss.item()
                num_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = train_loss / num_batches
                rank0_print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx+1}: loss={avg_loss:.4f}")

        # Save checkpoint
        if local_rank == 0:
            checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'model': _model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, checkpoint_path)
            log.info(f"Checkpoint saved to {checkpoint_path}")
    """


def parse_args():
    parser = argparse.ArgumentParser(description="Train CameraMovementRL")

    # Model paths
    parser.add_argument(
        "--adapter_path",
        default="train_records/coordinate_no_cam_mindcube/step_550",
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Base model path (auto-detect if None)",
    )

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=2560,
                        help="MLLM hidden dimension")
    parser.add_argument("--d_encoder", type=int, default=512,
                        help="Transformer encoder dimension")
    parser.add_argument("--encoder_nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--encoder_depth", type=int, default=2,
                        help="Number of transformer encoder layers")

    # Training
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)

    # Data
    parser.add_argument("--data_dir", default="datasets/",
                        help="Data directory (placeholder)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently 1 only)")
    parser.add_argument("--num_workers", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", default="checkpoints/camera_movement_rl")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
