#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration for relative camera pose
# prediction.  Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_correspondence.sh [num_gpus] [num_samples] [--plus]
#
#   num_gpus    — number of GPUs to use (default: all available)
#   num_samples — truncate dataset to this many entries (default: all)
#   --plus      — enable correspondence_plus mode (adds LM answer-prediction loss)
#
# Examples:
#   bash scripts/train_correspondence.sh               # all GPUs, full dataset
#   bash scripts/train_correspondence.sh 2             # 2 GPUs, full dataset
#   bash scripts/train_correspondence.sh 2 100         # 2 GPUs, 100 samples
#   bash scripts/train_correspondence.sh 1 6           # single GPU, 6 samples
#   bash scripts/train_correspondence.sh 2 100 --plus  # plus mode, 100 samples
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

# =============================================================================
# Hyperparameters
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_10k.json"
POS3D_DIR="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/3D_pos"

# --plus flag: positional arg 3 (default: disabled)
PLUS_FLAG=""
if [ "${3:-}" = "--plus" ] || [ "${3:-}" = "plus" ]; then
    PLUS_FLAG="--plus"
fi

RUN_NAME="correspondence${PLUS_FLAG:+_plus}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# EPOCHS=3
EPOCHS=5
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4

# SKIP_LAYERS="-8 -4 -1"
SKIP_LAYERS="-1"
CYCLE_WEIGHT=0.1
SAVE_STEPS=500

WANDB_PROJECT="SPI"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="lora_r${LORA_RANK}_ep${EPOCHS}_cycle${CYCLE_WEIGHT}${PLUS_FLAG:+_plus}"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE      = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] PLUS mode            = ${PLUS_FLAG:-disabled}"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Build optional --max_samples flag
# =============================================================================

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

# =============================================================================
# Run via torchrun (DDP)
# =============================================================================

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    29500 \
    train_correspondence.py \
    --model_path     "$MODEL_PATH"    \
    --json_path      "$JSON_PATH"     \
    --output_dir     "$OUTPUT_DIR"    \
    --pos3d_dir      "$POS3D_DIR"     \
    --epochs         "$EPOCHS"        \
    --lr             "$LR"            \
    --lora_rank      "$LORA_RANK"     \
    --max_images     "$MAX_IMAGES"    \
    --grad_accum     "$GRAD_ACCUM"    \
    --num_workers    "$NUM_WORKERS"   \
    --skip_layers    $SKIP_LAYERS     \
    --cycle_weight   "$CYCLE_WEIGHT"  \
    --save_steps     "$SAVE_STEPS"    \
    --wandb_project  "$WANDB_PROJECT" \
    --wandb_entity   "$WANDB_ENTITY"  \
    --wandb_run_name "$WANDB_RUN_NAME" \
    $MAX_SAMPLES_FLAG \
    $PLUS_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
