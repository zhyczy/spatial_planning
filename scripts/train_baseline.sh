#!/usr/bin/env bash
# =============================================================================
# train_baseline.sh
#
# 3DRS-style LoRA fine-tuning of Qwen3.5-VL on SPAR spatial QA.
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_baseline.sh [num_gpus] [num_samples] [spatial_model]
#
#   num_gpus      — number of GPUs to use (default: all available)
#   num_samples   — truncate dataset to this many entries (default: all)
#   spatial_model — "vggt" | "mapanything" | "none" (default: vggt)
#
# Examples:
#   bash scripts/train_baseline.sh                    # all GPUs, full dataset, vggt
#   bash scripts/train_baseline.sh 2                  # 2 GPUs, full dataset, vggt
#   bash scripts/train_baseline.sh 2 100              # 2 GPUs, 100 samples, vggt
#   bash scripts/train_baseline.sh 1 6                # single GPU, 6 samples, vggt
#   bash scripts/train_baseline.sh 2 100 mapanything  # 2 GPUs, 100 samples, mapanything
#   bash scripts/train_baseline.sh 2 "" none          # 2 GPUs, full dataset, no spatial
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

# num_gpus: positional arg 1 (default: auto-detect)
if [ -n "${1:-}" ]; then
    NPROC="$1"
    CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
else
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC="$N_AVAIL"
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NPROC - 1)))
fi

# num_samples: positional arg 2 (default: empty = all)
MAX_SAMPLES="${2:-}"

# spatial_model: positional arg 3 (default: vggt)
SPATIAL_MODEL="${3:-vggt}"

# =============================================================================
# Paths
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_10k.json"

VGGT_PATH="$SPATIAL_DIR/checkpoints/VGGT-1B"
MAPANYTHING_PATH="$SPATIAL_DIR/checkpoints/map-anything-weights"

if [ "$SPATIAL_MODEL" = "vggt" ]; then
    SPATIAL_MODEL_PATH="$VGGT_PATH"
elif [ "$SPATIAL_MODEL" = "mapanything" ]; then
    SPATIAL_MODEL_PATH="$MAPANYTHING_PATH"
else
    SPATIAL_MODEL_PATH=""
fi

RUN_NAME="3drs_${SPATIAL_MODEL}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS=3
LR=2e-4
LORA_RANK=64
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4
SAVE_STEPS=500

WANDB_PROJECT="SPI"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="3drs_${SPATIAL_MODEL}_lora_r${LORA_RANK}_ep${EPOCHS}"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE      = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] SPATIAL_MODEL        = $SPATIAL_MODEL"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Build optional flags
# =============================================================================

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

SPATIAL_PATH_FLAG=""
if [ -n "$SPATIAL_MODEL_PATH" ]; then
    SPATIAL_PATH_FLAG="--spatial_model_path $SPATIAL_MODEL_PATH"
fi

# =============================================================================
# Run via torchrun (DDP)
# =============================================================================

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    29501 \
    baseline/3drs.py \
    --model_path     "$MODEL_PATH"      \
    --json_path      "$JSON_PATH"       \
    --output_dir     "$OUTPUT_DIR"      \
    --epochs         "$EPOCHS"          \
    --lr             "$LR"              \
    --lora_rank      "$LORA_RANK"       \
    --max_images     "$MAX_IMAGES"      \
    --grad_accum     "$GRAD_ACCUM"      \
    --num_workers    "$NUM_WORKERS"     \
    --save_steps     "$SAVE_STEPS"      \
    --spatial_model  "$SPATIAL_MODEL"   \
    --wandb_project  "$WANDB_PROJECT"   \
    --wandb_entity   "$WANDB_ENTITY"    \
    --wandb_run_name "$WANDB_RUN_NAME"  \
    $SPATIAL_PATH_FLAG \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
