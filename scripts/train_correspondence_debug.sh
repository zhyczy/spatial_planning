#!/usr/bin/env bash
# =============================================================================
# train_correspondence_debug.sh
#
# Quick smoke-test: 6 training samples, 2 epochs, tiny grad-accum.
# Output and log saved to train_records/correspondence_debug/.
#
# Usage:
#   bash scripts/train_correspondence_debug.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# ── activate conda environment ────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spi

# =============================================================================
# Hyperparameters (debug-scale)
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_debug6.json"
POS3D_DIR="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/3D_pos"

RUN_NAME="correspondence_debug"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"
LOG_FILE="$OUTPUT_DIR/train.log"

EPOCHS=2
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=2          # smaller so optimizer steps happen within 6 samples
NUM_WORKERS=2

SKIP_LAYERS="-8 -4 -1"
CYCLE_WEIGHT=0.1
SAVE_STEPS=5          # save every 5 steps so we get at least one checkpoint

WANDB_PROJECT="SPI"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="$RUN_NAME"

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
fi

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] Output dir : $OUTPUT_DIR"
echo "[INFO] Log file   : $LOG_FILE"
echo "[INFO] Starting   : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Run (tee output to log file)
# =============================================================================

python train_correspondence.py \
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
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Log saved to $LOG_FILE"
