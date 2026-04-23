#!/usr/bin/env bash
# =============================================================================
# train_coordinate.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration with:
#   - LM answer loss
#   - Per-patch 3D coordinate loss (coord head reads vision-token hidden states)
#
# Usage:
#   bash scripts/train_coordinate.sh [num_gpus] [--polar] [--skip_layers LAYER] [--max_samples N] [--coord_scale_xyz SX SY SZ]
#
#   num_gpus            — first positional arg, number of GPUs (default: all)
#   --polar             — use log-spherical (log r, θ=azimuth, α=inclination)
#                         for both coord-loss target and visual-token 4D M-RoPE
#   --skip_layers LAYER — layer for coord head (default: -1 = last layer)
#                         -1 = Layer 32 (post-norm), -2 = Layer 31, etc.
#   --max_samples N     — truncate dataset to N entries (default: all)
#   --coord_scale_xyz SX SY SZ — per-axis RoPE scales for (x, y, z), overrides
#                         the default scalar coord_scale=100. Useful because
#                         mrope_section [2,10,10,10] gives x/y/z very different
#                         inv_freq ranges; typical values e.g. 14 1250 170000.
#   --interleave_vision — switch visual M-RoPE layout to interleaved: t keeps
#                         bands 0..s0-1 (high freq), then x/y/z round-robin
#                         through the remaining bands so each spans the full
#                         freq range. Makes x/y/z symmetric under a single
#                         scalar scale (no need for per-axis scaling).
#   --full              — force partial_rotary_factor=1.0 so every head_dim
#                         dimension gets RoPE (vs. default 0.25). Rebuilds
#                         mrope_section to sum=head_dim//2 (32 -> 128 for
#                         head_dim=256). Breaks the pretrained content/position
#                         split; only LoRA can adapt. Expect degraded LM loss
#                         initially.
#
# Examples:
#   bash scripts/train_coordinate.sh                      # all GPUs, Layer 32
#   bash scripts/train_coordinate.sh 2                    # 2 GPUs, Layer 32
#   bash scripts/train_coordinate.sh 6 --polar            # 6 GPUs, log-spherical coord GT
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
POLAR_FLAG=""
SKIP_LAYERS_ARG=""
COORD_SCALE_XYZ_ARG=""
INTERLEAVE_FLAG=""
FULL_FLAG=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --polar)
            POLAR_FLAG="--polar"; shift ;;
        --skip_layers)
            SKIP_LAYERS_ARG="$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --coord_scale_xyz)
            COORD_SCALE_XYZ_ARG="--coord_scale_xyz $2 $3 $4"; shift 4 ;;
        --interleave_vision)
            INTERLEAVE_FLAG="--interleave_vision"; shift ;;
        --full)
            FULL_FLAG="--full"; shift ;;
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

RUN_NAME="coordinate_mindcube${_polar_suffix}"
WANDB_RUN_NAME="coord_mindcube_r${LORA_RANK}_ep${EPOCHS}_coord${COORD_WEIGHT}${_polar_suffix}"

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
echo "[INFO] Polar coord GT       = ${POLAR_FLAG:-disabled}"
echo "[INFO] Coord scale xyz      = ${COORD_SCALE_XYZ_ARG:-default scalar 100}"
echo "[INFO] Interleave vision    = ${INTERLEAVE_FLAG:-disabled (sequential)}"
echo "[INFO] Full rotary          = ${FULL_FLAG:-disabled (partial=0.25)}"
echo "[INFO] Coord Head at        = $SKIP_LAYERS_DISPLAY"
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
    $POLAR_FLAG                                        \
    $COORD_SCALE_XYZ_ARG                               \
    $INTERLEAVE_FLAG                                   \
    $FULL_FLAG                                         \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
