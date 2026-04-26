#!/usr/bin/env bash
# =============================================================================
# train_coordinate.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration with:
#   - LM answer loss
#   - Per-patch 3D coordinate loss (coord head reads vision-token hidden states)
#
# Usage:
#   bash scripts/train_coordinate.sh [num_gpus] [--polar] [--decouple] [--skip_layers LAYER] [--coord_weight W] [--lora_rank R] [--max_samples N]
#
#   num_gpus            — first positional arg, number of GPUs (default: all)
#   --polar             — decouple architecture with log-spherical XYZ RoPE:
#                         Qwen original 3D M-RoPE [11,11,10] UNCHANGED in rotary
#                         64 dims + new XYZ RoPE (66 dims, rope_theta=1000) in
#                         pass-through dims 64..129 taking (log r, θ, α).
#                         Coord-loss GT also converted to (log r, θ, α) via
#                         MindCube_Train_Dataset_Coord_Polar. Mirrors
#                         train_correspondence.py --polar. Mutually exclusive
#                         with --decouple.
#   --decouple          — decoupled position embedding: Qwen original 3D M-RoPE
#                         [11,11,10] in rotary 64 dims (UNCHANGED) + new XYZ
#                         RoPE (66 dims, Cartesian, rope_theta=10000) in
#                         pass-through dims 64..129. Mirrors
#                         train_correspondence.py --decouple. Mutually
#                         exclusive with --polar.
#   --skip_layers LAYER — layer for coord head (default: -1 = last layer)
#                         -1 = Layer 32 (post-norm), -2 = Layer 31, etc.
#                         Non-default values get stamped into RUN_NAME (_sl<LAYER>).
#   --coord_weight W    — override coord-loss weight (default: 1.0). Non-default
#                         values get stamped into RUN_NAME (_cw<W>).
#   --lora_rank R       — override LoRA rank (default: 16). Non-default values
#                         get stamped into RUN_NAME (_r<R>).
#   --xyz_rope_dim N    — total head_dim units for the XYZ RoPE under
#                         --decouple / --polar (each axis x/y/z gets N/6 freq
#                         bands). Must be a positive multiple of 6 ≤ 192.
#                         Default 66 (= 11 bands per axis). Non-default values
#                         get stamped into RUN_NAME (_xrd<N>). No effect
#                         without --decouple / --polar.
#   --max_samples N     — truncate dataset to N entries (default: all)
#   --interleave_vision — switch visual M-RoPE layout to interleaved: t keeps
#                         bands 0..s0-1 (high freq), then x/y/z round-robin
#                         through the remaining bands so each spans the full
#                         freq range. No effect with --decouple.
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
INTERLEAVE_FLAG=""
DECOUPLE_FLAG=""
COORD_WEIGHT_ARG=""
LORA_RANK_ARG=""
XYZ_ROPE_DIM=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --polar)
            POLAR_FLAG="--polar"; shift ;;
        --decouple)
            DECOUPLE_FLAG="--decouple"; shift ;;
        --skip_layers)
            SKIP_LAYERS_ARG="$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --interleave_vision)
            INTERLEAVE_FLAG="--interleave_vision"; shift ;;
        --coord_weight)
            COORD_WEIGHT_ARG="$2"; shift 2 ;;
        --lora_rank)
            LORA_RANK_ARG="$2"; shift 2 ;;
        --xyz_rope_dim)
            XYZ_ROPE_DIM="$2"; shift 2 ;;
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

# Apply CLI overrides for hyperparams
[ -n "$COORD_WEIGHT_ARG" ] && COORD_WEIGHT="$COORD_WEIGHT_ARG"
[ -n "$LORA_RANK_ARG" ]    && LORA_RANK="$LORA_RANK_ARG"

_polar_suffix="${POLAR_FLAG:+_polar}"
_decouple_suffix="${DECOUPLE_FLAG:+_decouple}"

# Hyperparam suffixes — only stamped when deviating from defaults, so legacy
# runs keep their short names.
_r_suffix=""
[ "$LORA_RANK"    != "16"  ] && _r_suffix="_r${LORA_RANK}"
_cw_suffix=""
[ "$COORD_WEIGHT" != "1.0" ] && _cw_suffix="_cw${COORD_WEIGHT}"
_sl_suffix=""
[ "$SKIP_LAYERS"  != "-1"  ] && _sl_suffix="_sl${SKIP_LAYERS}"

# Stamp xyz_rope_dim into RUN_NAME only when overridden and the dim is in effect
# (--decouple or --polar). Default 66 → no suffix to keep legacy run names stable.
_xrd_suffix=""
if [ -n "$XYZ_ROPE_DIM" ] && [ "$XYZ_ROPE_DIM" != "66" ] \
   && { [ -n "$DECOUPLE_FLAG" ] || [ -n "$POLAR_FLAG" ]; }; then
    _xrd_suffix="_xrd${XYZ_ROPE_DIM}"
fi

RUN_NAME="coordinate_mindcube${_polar_suffix}${_decouple_suffix}${_r_suffix}${_cw_suffix}${_sl_suffix}${_xrd_suffix}"
WANDB_RUN_NAME="coord_mindcube_r${LORA_RANK}_ep${EPOCHS}_coord${COORD_WEIGHT}${_polar_suffix}${_decouple_suffix}${_sl_suffix}${_xrd_suffix}"

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
echo "[INFO] Decouple XYZ RoPE    = ${DECOUPLE_FLAG:-disabled}"
if [ -n "$DECOUPLE_FLAG" ] || [ -n "$POLAR_FLAG" ]; then
    echo "[INFO] xyz_rope_dim         = ${XYZ_ROPE_DIM:-66 (default)}"
else
    echo "[INFO] xyz_rope_dim         = N/A (no --decouple / --polar)"
fi
echo "[INFO] Interleave vision    = ${INTERLEAVE_FLAG:-disabled (sequential)}"
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

XYZ_ROPE_DIM_FLAG=""
if [ -n "$XYZ_ROPE_DIM" ]; then
    XYZ_ROPE_DIM_FLAG="--xyz_rope_dim $XYZ_ROPE_DIM"
fi

# =============================================================================
# Run via torchrun (DDP)
# =============================================================================

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    "${MASTER_PORT:-29502}" \
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
    $DECOUPLE_FLAG                                     \
    $INTERLEAVE_FLAG                                   \
    $XYZ_ROPE_DIM_FLAG                                 \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
