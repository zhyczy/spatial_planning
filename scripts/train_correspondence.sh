#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration (LM answer loss only)
# on MindCube data.  Multi-GPU via torchrun (DDP).
#
# Uses AnswerOnlyModel with 4D M-RoPE (use_xyz=True) by default.
# Pass --vanilla to use original Qwen 3D M-RoPE instead.
# Pass --relative to enable per-query-frame coordinate transforms.
#
# Usage:
#   bash scripts/train_correspondence.sh [num_gpus] [--polar] [--vanilla] [--relative] [--decouple] [--interleave_vision] [--max_samples N]
#
#   num_gpus            — first positional arg, number of GPUs (default: all)
#   --polar             — use the decouple architecture (Qwen 3D M-RoPE in
#                         rotary 64 + new XYZ RoPE in pass-through 64..129)
#                         BUT feed log-spherical (log r, θ=atan2(y,x),
#                         α=atan2(√(x²+y²),z)) into the XYZ RoPE. Mutually
#                         exclusive with --vanilla and --decouple.
#   --vanilla           — use original Qwen 3D M-RoPE (no image_xyz);
#                         disables --polar / --relative / --interleave_vision / --decouple
#   --relative          — per-query-frame coord transform: Q from frame f sees all K in frame-f coords
#   --decouple          — Qwen 3D M-RoPE [11,11,10] in rotary 64 (UNCHANGED)
#                         + new XYZ RoPE in pass-through dims 64..129 with
#                         **Cartesian** xyz. For log-spherical input, use
#                         --polar instead. Mutually exclusive with --vanilla
#                         / --relative / --polar.
#   --interleave_vision — interleaved visual M-RoPE: t at high-freq end, x/y/z round-robin
#                         (only meaningful for the 4D M-RoPE path; no effect
#                         with --vanilla / --decouple / --polar)
#   --max_samples N     — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_correspondence.sh                                  # all GPUs, 4D M-RoPE (Cartesian)
#   bash scripts/train_correspondence.sh 2                                # 2 GPUs, 4D M-RoPE
#   bash scripts/train_correspondence.sh 2 --polar                        # decouple + log-spherical XYZ RoPE
#   bash scripts/train_correspondence.sh 2 --decouple                     # decouple + Cartesian XYZ RoPE
#   bash scripts/train_correspondence.sh 2 --vanilla                      # vanilla 3D M-RoPE
#   bash scripts/train_correspondence.sh 2 --relative                     # relative per-frame coords
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
RELATIVE_FLAG=""
POLAR_FLAG=""
INTERLEAVE_FLAG=""
DECOUPLE_FLAG=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --vanilla)
            VANILLA_FLAG="--vanilla"; shift ;;
        --relative)
            RELATIVE_FLAG="--relative"; shift ;;
        --polar)
            POLAR_FLAG="--polar"; shift ;;
        --interleave_vision)
            INTERLEAVE_FLAG="--interleave_vision"; shift ;;
        --decouple)
            DECOUPLE_FLAG="--decouple"; shift ;;
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

SAVE_STEPS=50
EVAL_STEPS=50

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Mode-specific settings
# =============================================================================

_vanilla_suffix="${VANILLA_FLAG:+_vanilla}"
_relative_suffix="${RELATIVE_FLAG:+_relative}"
_polar_suffix="${POLAR_FLAG:+_polar}"
_interleave_suffix="${INTERLEAVE_FLAG:+_interleave}"
_decouple_suffix="${DECOUPLE_FLAG:+_decouple}"

RUN_NAME="correspondence_mindcube${_relative_suffix}${_polar_suffix}${_interleave_suffix}${_decouple_suffix}${_vanilla_suffix}"
WANDB_RUN_NAME="corr_mindcube_r${LORA_RANK}_ep${EPOCHS}${_relative_suffix}${_polar_suffix}${_interleave_suffix}${_decouple_suffix}${_vanilla_suffix}"

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
echo "[INFO] Relative coords      = ${RELATIVE_FLAG:-disabled}"
echo "[INFO] Polar coords (M-RoPE)= ${POLAR_FLAG:-disabled (Cartesian)}"
echo "[INFO] Interleave vision    = ${INTERLEAVE_FLAG:-disabled (sequential)}"
echo "[INFO] Decouple position    = ${DECOUPLE_FLAG:-disabled}"
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
    --master_port    29500 \
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
    $RELATIVE_FLAG                                     \
    $POLAR_FLAG                                        \
    $INTERLEAVE_FLAG                                   \
    $DECOUPLE_FLAG                                     \
    $VANILLA_FLAG                                      \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
