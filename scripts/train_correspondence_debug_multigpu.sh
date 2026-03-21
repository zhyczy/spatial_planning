#!/usr/bin/env bash
# =============================================================================
# train_correspondence_debug_multigpu.sh
#
# Multi-GPU smoke-test: 6 training samples, 2 epochs, tiny grad-accum.
# Runs on N GPUs via torchrun --nproc_per_node N.
# Output and log saved to train_records/correspondence_debug_multigpu/.
#
# Usage:
#   bash scripts/train_correspondence_debug_multigpu.sh [num_gpus]
#   e.g. bash scripts/train_correspondence_debug_multigpu.sh 2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# GPU count from argument (default: 2)
# =============================================================================

NPROC="${1:-2}"

# Build CUDA_VISIBLE_DEVICES string: 0,1,...,N-1
CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))

# =============================================================================
# Hyperparameters (debug-scale)
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_debug6.json"
POS3D_DIR="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/3D_pos"

RUN_NAME="correspondence_debug_multigpu"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

EPOCHS=2
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=2
NUM_WORKERS=0

SKIP_LAYERS="-1"
CYCLE_WEIGHT=0.1
SAVE_STEPS=5

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
echo "[INFO] NPROC_PER_NODE   = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] Output dir       : $OUTPUT_DIR"
echo "[INFO] Starting         : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Run via torchrun (multi-GPU DDP)
# =============================================================================

conda run --no-capture-output -n spc torchrun \
    --nproc_per_node "$NPROC" \
    --master_port    29502 \
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
    --wandb_project  ""

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
