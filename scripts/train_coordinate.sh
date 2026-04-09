#!/usr/bin/env bash
# =============================================================================
# train_coordinate.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration with supervision signals:
#   - Full:     pose regression + LM answer + per-patch 3D coordinate
#   - Ablation: LM answer + per-patch 3D coordinate only (--no_cam)
#
# Usage:
#   bash scripts/train_coordinate.sh [num_gpus] [--no_cam] [--polar] [--skip_layers LAYER] [--max_samples N]
#
#   num_gpus            — first positional arg, number of GPUs (default: all)
#   --no_cam            — ablation: remove pose head, use CoordinateModel
#   --polar             — convert coord GT from (x,y,z) to spherical (r,θ,α)
#   --skip_layers LAYER — layer for pose/coord heads (default: -1 = last layer)
#                         -1 = Layer 32 (post-norm), -2 = Layer 31, etc.
#   --max_samples N     — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_coordinate.sh                      # all GPUs, full model, Layer 32
#   bash scripts/train_coordinate.sh 2                    # 2 GPUs, full model, Layer 32
#   bash scripts/train_coordinate.sh 6 --no_cam           # 6 GPUs, ablation
#   bash scripts/train_coordinate.sh 6 --no_cam --polar   # no_cam + spherical coord GT
#   bash scripts/train_coordinate.sh 1 --skip_layers -2   # single GPU, use Layer 31
#   bash scripts/train_coordinate.sh 1 --max_samples 6    # single GPU, 6 samples
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
NO_CAM_ARG=""
POLAR_FLAG=""
SKIP_LAYERS_ARG=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no_cam)
            NO_CAM_ARG="no_cam"; shift ;;
        --polar)
            POLAR_FLAG="--polar"; shift ;;
        --skip_layers)
            SKIP_LAYERS_ARG="$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
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

ANSWER_WEIGHT=1.0
COORD_WEIGHT=1.0
COORD_UPSCALE=4
SAVE_STEPS=50
EVAL_STEPS=50

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Mode-specific settings
# =============================================================================

# Helper: convert skip_layers value to layer name
layer_name() {
    local skip_val="$1"
    if [ "$skip_val" = "-1" ]; then
        echo "Layer 32 (post-norm, last)"
    elif [ "$skip_val" = "-2" ]; then
        echo "Layer 31 (penultimate)"
    else
        echo "Layer $(( 32 + skip_val ))"
    fi
}

SKIP_LAYERS="${SKIP_LAYERS_ARG:--1}"  # default to -1 if not specified
SKIP_LAYERS_DISPLAY="$(layer_name "$SKIP_LAYERS")"
SKIP_LAYERS_FLAG="--skip_layers ${SKIP_LAYERS}"

_polar_suffix="${POLAR_FLAG:+_polar}"

if [ "$NO_CAM_ARG" = "no_cam" ]; then
    RUN_NAME="coordinate_no_cam_mindcube${_polar_suffix}"
    WANDB_RUN_NAME="coord_no_cam_mindcube_r${LORA_RANK}_ep${EPOCHS}_coord${COORD_WEIGHT}${_polar_suffix}"
    NO_CAM_FLAG="--no_cam"
    CYCLE_FLAG=""
else
    CYCLE_WEIGHT=0.1
    RUN_NAME="coordinate_mindcube${_polar_suffix}"
    WANDB_RUN_NAME="coord_mindcube_r${LORA_RANK}_ep${EPOCHS}_cycle${CYCLE_WEIGHT}_coord${COORD_WEIGHT}${_polar_suffix}"
    NO_CAM_FLAG=""
    CYCLE_FLAG="--cycle_weight $CYCLE_WEIGHT"
fi

OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# =============================================================================
# Setup
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EVAL_STEPS           = $EVAL_STEPS"
echo "[INFO] Mode                 = ${NO_CAM_ARG:-full (pose+coord+lm)}"
echo "[INFO] Polar coord GT       = ${POLAR_FLAG:-disabled}"
echo "[INFO] Pose/Coord Heads at  = $SKIP_LAYERS_DISPLAY"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Build optional flags
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
    train_coordinate.py \
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
    --answer_weight          "$ANSWER_WEIGHT"          \
    --coord_weight           "$COORD_WEIGHT"           \
    --coord_upscale          "$COORD_UPSCALE"          \
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $SKIP_LAYERS_FLAG                                  \
    $CYCLE_FLAG                                        \
    $NO_CAM_FLAG                                       \
    $POLAR_FLAG                                        \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
