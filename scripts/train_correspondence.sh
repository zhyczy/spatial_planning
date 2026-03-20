#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration for relative camera pose
# prediction using all A(N,2) ordered image pairs per scene.
#
# Usage:
#   bash scripts/train_correspondence.sh               # default (single node)
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_correspondence.sh
# =============================================================================

set -euo pipefail

# ── locate repo root (script lives in spatial_planning/scripts/) ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# ── activate conda environment ────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spi

# =============================================================================
# Hyperparameters
# =============================================================================

# -- paths --------------------------------------------------------------------
MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_10k.json"
OUTPUT_DIR="$SPATIAL_DIR/checkpoints/spa_correspondence"
POS3D_DIR="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/3D_pos"

# -- training -----------------------------------------------------------------
EPOCHS=10
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4          # max views per scene (A(4,2)=12 pairs)
GRAD_ACCUM=8          # effective batch = 8 scenes
NUM_WORKERS=4

# -- model architecture -------------------------------------------------------
# Concatenate hidden states from these LLM layer indices before PoseHead.
# Three-layer skip: early geometric + mid semantic + final
SKIP_LAYERS="-8 -4 -1"

# -- losses -------------------------------------------------------------------
CYCLE_WEIGHT=0.1      # rotation cycle-consistency loss weight (0 = disable)

# -- checkpointing ------------------------------------------------------------
SAVE_STEPS=500

# -- WandB (set WANDB_PROJECT="" to disable) ----------------------------------
WANDB_PROJECT="SPI"
WANDB_RUN_NAME="lora_r${LORA_RANK}_ep${EPOCHS}_cycle${CYCLE_WEIGHT}"
WANDB_ENTITY="actmrv"  # optional: your WandB username or team name

# =============================================================================
# GPU setup
# =============================================================================

# Default: use all available GPUs; override with CUDA_VISIBLE_DEVICES env var
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# =============================================================================
# Run
# =============================================================================

echo "[INFO] Starting training — $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Output dir : $OUTPUT_DIR"
echo "[INFO] WandB      : ${WANDB_PROJECT:-disabled}"

python train_correspondence.py \
    --model_path    "$MODEL_PATH"    \
    --json_path     "$JSON_PATH"     \
    --output_dir    "$OUTPUT_DIR"    \
    --pos3d_dir     "$POS3D_DIR"     \
    --epochs        "$EPOCHS"        \
    --lr            "$LR"            \
    --lora_rank     "$LORA_RANK"     \
    --max_images    "$MAX_IMAGES"    \
    --grad_accum    "$GRAD_ACCUM"    \
    --num_workers   "$NUM_WORKERS"   \
    --skip_layers   $SKIP_LAYERS     \
    --cycle_weight  "$CYCLE_WEIGHT"  \
    --save_steps    "$SAVE_STEPS"    \
    --wandb_project  "$WANDB_PROJECT"  \
    --wandb_entity   "$WANDB_ENTITY"   \
    --wandb_run_name "$WANDB_RUN_NAME"

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
