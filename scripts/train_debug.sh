#!/usr/bin/env bash
# =============================================================================
# train_debug.sh
#
# Smoke-test for train_vanilla.py (--ablation vanilla).
# Supports single or multi-GPU via torchrun --nproc_per_node N.
# Output saved to train_debug/.
#
# Usage:
#   bash scripts/train_debug.sh [num_gpus] [max_samples] [eval_steps]
#   e.g. bash scripts/train_debug.sh           # 1 GPU, all 10k samples, eval every 2 steps
#        bash scripts/train_debug.sh 2         # 2 GPUs, all 10k samples
#        bash scripts/train_debug.sh 2 50      # 2 GPUs, 50 samples
#        bash scripts/train_debug.sh 2 50 5    # eval every 5 optimizer steps
#        bash scripts/train_debug.sh 2 0 0     # all 10k samples, no eval
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

NPROC="${1:-1}"           # number of GPUs
MAX_SAMPLES="${2:-0}"     # training samples from train_10k.json (0 = all 10k)
EVAL_STEPS="${3:-2}"      # evaluate every N optimizer steps (0 = disable)
EVAL_MAX_NEW_TOKENS="${4:-32}"  # MCQ answers are ≤3 tokens; 32 is generous
EVAL_MAX_SAMPLES="${5:-0}"     # max eval samples per dataset (0 = full dataset)

# Build CUDA_VISIBLE_DEVICES string: 0,1,...,N-1
CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))

# =============================================================================
# Hyperparameters (debug-scale)
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"

JSON_PATH="$SPATIAL_DIR/datasets/train/SPAR_7M/spar/train_10k.json"

OUTPUT_DIR="$SPATIAL_DIR/train_debug"

EPOCHS=2
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=2          # small so optimizer steps fire quickly for debug
NUM_WORKERS=0
SAVE_STEPS=5          # save every 5 steps to get at least one checkpoint

WANDB_PROJECT="SPI"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="vanilla_debug_lora_r${LORA_RANK}_ep${EPOCHS}"

# =============================================================================
# Setup
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
export CUDA_LAUNCH_BLOCKING=1     # synchronous kernel launch → accurate stack traces
echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES} (0 = all 10k samples)"
echo "[INFO] EVAL_STEPS           = ${EVAL_STEPS} (0 = disabled)"
echo "[INFO] EVAL_MAX_NEW_TOKENS  = ${EVAL_MAX_NEW_TOKENS}"
echo "[INFO] EVAL_MAX_SAMPLES     = ${EVAL_MAX_SAMPLES} (0 = full dataset)"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Run via torchrun (single or multi-GPU DDP)
# =============================================================================

/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun \
    --nproc_per_node "$NPROC" \
    --master_port    29502 \
    train_vanilla.py \
    --model_path     "$MODEL_PATH"    \
    --json_path      "$JSON_PATH"     \
    --output_dir     "$OUTPUT_DIR"    \
    --ablation       vanilla          \
    --epochs         "$EPOCHS"        \
    --lr             "$LR"            \
    --lora_rank      "$LORA_RANK"     \
    --max_images     "$MAX_IMAGES"    \
    --grad_accum     "$GRAD_ACCUM"    \
    --num_workers    "$NUM_WORKERS"   \
    --max_samples    "$MAX_SAMPLES"   \
    --save_steps     "$SAVE_STEPS"    \
    --eval_steps            "$EVAL_STEPS"           \
    --eval_max_new_tokens   "$EVAL_MAX_NEW_TOKENS"  \
    $([ "$EVAL_MAX_SAMPLES" -gt 0 ] 2>/dev/null && echo "--eval_max_samples $EVAL_MAX_SAMPLES") \
    --wandb_project  "$WANDB_PROJECT" \
    --wandb_entity   "$WANDB_ENTITY"  \
    --wandb_run_name "$WANDB_RUN_NAME"

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
