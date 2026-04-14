#!/usr/bin/env bash
# =============================================================================
# train_rotation.sh
#
# Two-pass rotation-aware coordinate prediction training.
# LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) with:
#   - CameraTokenRotationEncoder  → predicts canonical rotation R
#   - DepthPredictionTransformer  → coordinate head in rotated frame
#
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_rotation.sh [num_gpus] [--max_samples N] \
#       [--rot_weight W] [--coord_weight W] [--coord_scale S]
#
#   num_gpus        — first positional arg, number of GPUs (default: all)
#   --max_samples N — truncate dataset to N entries (default: all)
#   --rot_weight  W — weight for geodesic rotation loss (default: 1.0)
#   --coord_weight W— weight for coordinate L1 loss (default: 1.0)
#   --coord_scale S — XYZ discretization multiplier, must match MLLM (default: 100.0)
#
# Examples:
#   bash scripts/train_rotation.sh                        # all GPUs
#   bash scripts/train_rotation.sh 2                      # 2 GPUs
#   bash scripts/train_rotation.sh 1 --max_samples 64    # debug run
#   bash scripts/train_rotation.sh 4 --rot_weight 0.5    # lower rot loss weight
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

NPROC=""
MAX_SAMPLES=""
ROT_WEIGHT=""
COORD_WEIGHT=""
COORD_SCALE=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --rot_weight)
            ROT_WEIGHT="$2"; shift 2 ;;
        --coord_weight)
            COORD_WEIGHT="$2"; shift 2 ;;
        --coord_scale)
            COORD_SCALE="$2"; shift 2 ;;
        *)
            if [ $_positional -eq 0 ]; then
                NPROC="$1"
            fi
            _positional=$((_positional + 1))
            shift ;;
    esac
done

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

EPOCHS=6
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4

SAVE_STEPS=200
EVAL_STEPS=100

# Rotation encoder architecture
# d_model = ROT_NHEAD × mllm_head_dim (e.g. 4 × 256 = 1024 for Qwen3.5-4B)
ROT_NHEAD=4
ROT_DIM_FEEDFORWARD=2048
ROT_NUM_LAYERS=2

# Default loss weights (can be overridden via CLI)
_ROT_WEIGHT="${ROT_WEIGHT:-1.0}"
_COORD_WEIGHT="${COORD_WEIGHT:-1.0}"
_COORD_SCALE="${COORD_SCALE:-100.0}"

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Run name / output dir
# =============================================================================

RUN_NAME="rotation_mindcube"
WANDB_RUN_NAME="rot_mindcube_r${LORA_RANK}_ep${EPOCHS}_rw${_ROT_WEIGHT}_cw${_COORD_WEIGHT}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# =============================================================================
# Setup
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] rot_weight           = $_ROT_WEIGHT"
echo "[INFO] coord_weight         = $_COORD_WEIGHT"
echo "[INFO] coord_scale          = $_COORD_SCALE"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Optional flags
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
    --master_port    29501 \
    train_rotation.py \
    --model_path             "$MODEL_PATH"             \
    --json_path              "$JSON_PATH"              \
    --mindcube_results_dir   "$MINDCUBE_RESULTS_DIR"   \
    --output_dir             "$OUTPUT_DIR"             \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
    --lora_rank              "$LORA_RANK"              \
    --max_images             "$MAX_IMAGES"             \
    --grad_accum             "$GRAD_ACCUM"             \
    --num_workers            "$NUM_WORKERS"            \
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --rot_weight             "$_ROT_WEIGHT"            \
    --coord_weight           "$_COORD_WEIGHT"          \
    --coord_scale            "$_COORD_SCALE"           \
    --rot_nhead              "$ROT_NHEAD"              \
    --rot_dim_feedforward    "$ROT_DIM_FEEDFORWARD"    \
    --rot_num_layers         "$ROT_NUM_LAYERS"         \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
