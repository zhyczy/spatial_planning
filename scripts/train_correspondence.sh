#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration (LM answer loss only)
# on MindCube data.  Multi-GPU via torchrun (DDP).
#
# Uses AnswerOnlyModel with 4D M-RoPE (use_xyz=True) by default.
# Pass --vanilla to use original Qwen 3D M-RoPE instead.
#
# Usage:
#   bash scripts/train_correspondence.sh [num_gpus] [--polar] [--vanilla] [--decouple] [--max_samples N] [--datasets mindcube|sat|...]
#
#   num_gpus            — first positional arg, number of GPUs (default: all)
#   --polar             — use the decouple architecture (Qwen 3D M-RoPE in
#                         rotary 64 + new XYZ RoPE in pass-through 64..129)
#                         BUT feed log-spherical (log r, θ=atan2(y,x),
#                         α=atan2(√(x²+y²),z)) into the XYZ RoPE. Mutually
#                         exclusive with --vanilla and --decouple.
#   --vanilla           — use original Qwen 3D M-RoPE (no image_xyz);
#                         disables --polar / --decouple
#   --decouple          — Qwen 3D M-RoPE [11,11,10] in rotary 64 (UNCHANGED)
#                         + new XYZ RoPE in pass-through dims 64..129 with
#                         **Cartesian** xyz. For log-spherical input, use
#                         --polar instead. Mutually exclusive with --vanilla
#                         / --polar.
#   --xyz_rope_dim N    — total head_dim units for the XYZ RoPE under
#                         --decouple / --polar (each axis x/y/z gets N/6 freq
#                         bands). Must be a positive multiple of 6 ≤ 192.
#                         Default 66 (= 11 bands per axis). Non-default values
#                         get stamped into RUN_NAME (_xrd<N>). No effect
#                         without --decouple / --polar.
#   --max_samples N     — truncate dataset to N entries (default: all; applied per-source)
#   --datasets LIST     — space-separated list of training datasets to concatenate.
#                         Choices: mindcube, sat. Default: mindcube only.
#                         Example: --datasets mindcube sat → ConcatDataset of
#                         MindCube_train.jsonl + SAT/train_36k.json. Each source
#                         uses its own (json_path, results_dir); paths default
#                         to the standard locations under datasets/train/{Name}/.
#                         RUN_NAME suffix: _ds<name1>+<name2>+... when not "mindcube" alone.
#
# Examples:
#   bash scripts/train_correspondence.sh                                  # all GPUs, 4D M-RoPE (Cartesian), MindCube only
#   bash scripts/train_correspondence.sh 2                                # 2 GPUs, 4D M-RoPE
#   bash scripts/train_correspondence.sh 2 --polar                        # decouple + log-spherical XYZ RoPE
#   bash scripts/train_correspondence.sh 2 --decouple                     # decouple + Cartesian XYZ RoPE
#   bash scripts/train_correspondence.sh 2 --vanilla                      # vanilla 3D M-RoPE
#   bash scripts/train_correspondence.sh 2 --datasets mindcube sat        # MindCube + SAT combined
#   bash scripts/train_correspondence.sh 2 --polar --datasets mindcube sat # combined + polar
#   bash scripts/train_correspondence.sh 1 --max_samples 6                # single GPU, 6 samples
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
VANILLA_FLAG=""
POLAR_FLAG=""
DECOUPLE_FLAG=""
XYZ_ROPE_DIM=""
DATASETS=()
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --vanilla)
            VANILLA_FLAG="--vanilla"; shift ;;
        --polar)
            POLAR_FLAG="--polar"; shift ;;
        --decouple)
            DECOUPLE_FLAG="--decouple"; shift ;;
        --xyz_rope_dim)
            XYZ_ROPE_DIM="$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --datasets)
            shift
            # Consume all following non-flag tokens as dataset names
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                DATASETS+=("$1"); shift
            done ;;
        *)
            if [ $_positional -eq 0 ]; then
                NPROC="$1"
            fi
            _positional=$((_positional + 1))
            shift ;;
    esac
done

# Default: mindcube only (matches the .py default)
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("mindcube")
fi

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

SAVE_STEPS=50
EVAL_STEPS=50

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Mode-specific settings
# =============================================================================

_vanilla_suffix="${VANILLA_FLAG:+_vanilla}"
_polar_suffix="${POLAR_FLAG:+_polar}"
_decouple_suffix="${DECOUPLE_FLAG:+_decouple}"

# Stamp xyz_rope_dim into RUN_NAME only when overridden and the dim is in effect
# (--decouple or --polar). Default 66 → no suffix to keep legacy run names stable.
_xrd_suffix=""
if [ -n "$XYZ_ROPE_DIM" ] && [ "$XYZ_ROPE_DIM" != "66" ] \
   && { [ -n "$DECOUPLE_FLAG" ] || [ -n "$POLAR_FLAG" ]; }; then
    _xrd_suffix="_xrd${XYZ_ROPE_DIM}"
fi

# Stamp dataset selection into RUN_NAME when more than one source or not the
# default (mindcube alone). Joined with '+' (e.g. _ds-mindcube+sat).
_ds_suffix=""
if [ "${#DATASETS[@]}" -gt 1 ] || [ "${DATASETS[0]}" != "mindcube" ]; then
    _ds_joined=$(IFS=+; echo "${DATASETS[*]}")
    _ds_suffix="_ds-${_ds_joined}"
fi

RUN_NAME="correspondence${_ds_suffix:-_mindcube}${_polar_suffix}${_decouple_suffix}${_vanilla_suffix}${_xrd_suffix}"
WANDB_RUN_NAME="corr${_ds_suffix:-_mindcube}_r${LORA_RANK}_ep${EPOCHS}${_polar_suffix}${_decouple_suffix}${_vanilla_suffix}${_xrd_suffix}"

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
echo "[INFO] Mode                 = ${VANILLA_FLAG:+vanilla (3D M-RoPE)}${DECOUPLE_FLAG:+decoupled (3D + new XYZ RoPE)}${VANILLA_FLAG:-${DECOUPLE_FLAG:-4D M-RoPE (image_xyz)}}"
echo "[INFO] Polar coords (M-RoPE)= ${POLAR_FLAG:-disabled (Cartesian)}"
echo "[INFO] Decouple position    = ${DECOUPLE_FLAG:-disabled}"
if [ -n "$DECOUPLE_FLAG" ] || [ -n "$POLAR_FLAG" ]; then
    echo "[INFO] xyz_rope_dim         = ${XYZ_ROPE_DIM:-66 (default)}"
else
    echo "[INFO] xyz_rope_dim         = N/A (no --decouple / --polar)"
fi
echo "[INFO] Datasets              = ${DATASETS[*]}"
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
    --master_port    29502 \
    train_correspondence.py \
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
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    --datasets               "${DATASETS[@]}"          \
    $POLAR_FLAG                                        \
    $DECOUPLE_FLAG                                     \
    $VANILLA_FLAG                                      \
    $XYZ_ROPE_DIM_FLAG                                 \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
