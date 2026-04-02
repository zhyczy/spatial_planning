#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration for relative camera pose
# prediction on MindCube data.  Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_correspondence.sh [num_gpus] [num_samples] [--plus] [--ablation MODE]
#
#   num_gpus    — number of GPUs to use (default: all available)
#   num_samples — truncate dataset to this many entries (default: all)
#   --plus      — enable plus mode (adds LM answer-prediction loss)
#   --ablation MODE — ablation study: no_cam | vanilla
#
# Examples:
#   bash scripts/train_correspondence.sh               # all GPUs, full dataset
#   bash scripts/train_correspondence.sh 2             # 2 GPUs, full dataset
#   bash scripts/train_correspondence.sh 2 100         # 2 GPUs, 100 samples
#   bash scripts/train_correspondence.sh 1 6           # single GPU, 6 samples
#   bash scripts/train_correspondence.sh 2 100 --plus  # plus mode
#   bash scripts/train_correspondence.sh 2 100 --ablation no_cam   # ablation: no cam pred
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

# Parse positional args and flags from any position
NPROC=""
MAX_SAMPLES=""
PLUS_FLAG=""
ABLATION_FLAG=""
_positional=0
while [ $# -gt 0 ]; do
    case "$1" in
        --plus|plus)
            PLUS_FLAG="--plus"; shift ;;
        --ablation)
            ABLATION_FLAG="--ablation $2"; shift 2 ;;
        *)
            if [ $_positional -eq 0 ]; then
                NPROC="$1"
            elif [ $_positional -eq 1 ]; then
                MAX_SAMPLES="$1"
            fi
            _positional=$((_positional + 1))
            shift ;;
    esac
done

# num_gpus: default auto-detect
if [ -n "$NPROC" ]; then
    CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
else
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC="$N_AVAIL"
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NPROC - 1)))
fi

# =============================================================================
# Hyperparameters
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
JSON_PATH="$SPATIAL_DIR/datasets/train/MindCube/MindCube_train.jsonl"
MINDCUBE_RESULTS_DIR="$SPATIAL_DIR/datasets/train/MindCube/3d_results"

# Derive run name from mode
_ablation_name=""
if [ -n "$ABLATION_FLAG" ]; then
    _ablation_name="_$(echo "$ABLATION_FLAG" | awk '{print $2}')"
fi
RUN_NAME="correspondence_mindcube${PLUS_FLAG:+_plus}${_ablation_name}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

EPOCHS=3
# EPOCHS=5
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4

SKIP_LAYERS="-1"
CYCLE_WEIGHT=0.1
SAVE_STEPS=500
EVAL_STEPS=10

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="mindcube_lora_r${LORA_RANK}_ep${EPOCHS}_cycle${CYCLE_WEIGHT}${PLUS_FLAG:+_plus}${_ablation_name}"

# =============================================================================
# Setup
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EVAL_STEPS           = $EVAL_STEPS"
echo "[INFO] PLUS mode            = ${PLUS_FLAG:-disabled}"
echo "[INFO] ABLATION             = ${ABLATION_FLAG:-disabled}"
echo "[INFO] JSON_PATH            = $JSON_PATH"
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
    --model_path             "$MODEL_PATH"          \
    --json_path              "$JSON_PATH"           \
    --mindcube_results_dir   "$MINDCUBE_RESULTS_DIR" \
    --output_dir             "$OUTPUT_DIR"          \
    --epochs                 "$EPOCHS"              \
    --lr                     "$LR"                  \
    --lora_rank              "$LORA_RANK"           \
    --max_images             "$MAX_IMAGES"          \
    --grad_accum             "$GRAD_ACCUM"          \
    --num_workers            "$NUM_WORKERS"         \
    --skip_layers            $SKIP_LAYERS           \
    --cycle_weight           "$CYCLE_WEIGHT"        \
    --save_steps             "$SAVE_STEPS"          \
    --eval_steps             "$EVAL_STEPS"          \
    --wandb_project          "$WANDB_PROJECT"       \
    --wandb_entity           "$WANDB_ENTITY"        \
    --wandb_run_name         "$WANDB_RUN_NAME"      \
    $MAX_SAMPLES_FLAG \
    $PLUS_FLAG \
    $ABLATION_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
