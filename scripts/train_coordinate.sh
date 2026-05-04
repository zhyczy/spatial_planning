#!/usr/bin/env bash
# =============================================================================
# train_coordinate.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration on VST 500K with:
#   - LM answer CE loss
#   - Per-patch 3D coordinate loss (coord head reads vision-token hidden states)
#
# Multi-GPU via torchrun (DDP). Default mode: 4D M-RoPE + Cartesian coord GT.
# One alternative mode:
#   --decouple  : decouple architecture + Cartesian XYZ RoPE.
#
# Compare against train_atten.sh (per-layer attention bias) and
# train_correspondence.sh (LM-only, no coord head).
#
# Usage:
#   bash scripts/train_coordinate.sh [num_gpus] [--decouple] [--mindcube|--vst]
#                                    [--skip_layers LAYER] [--coord_weight W]
#                                    [--lora_rank R] [--xyz_rope_dim N]
#                                    [--max_samples N]
#
#   num_gpus            — first positional arg, number of GPUs (default: all visible)
#   --decouple          — decouple + Cartesian XYZ RoPE
#   --mindcube          — train on MindCube_train.jsonl + MindCube 3d_results
#   --vst               — train on vst_500k.json + VST 3d_results (default)
#   --skip_layers LAYER — coord head reads from this transformer layer
#                         (-1 = last/Layer 32 post-norm, default; -2 = Layer 31, ...)
#                         Stamped into RUN_NAME (_sl<LAYER>) when non-default.
#   --coord_weight W    — coord-loss weight (default: 1.0). Stamped (_cw<W>).
#   --lora_rank R       — LoRA rank (default: 16). Stamped (_r<R>) when non-default.
#   --xyz_rope_dim N    — total head_dim units for XYZ RoPE under --decouple
#                         (each axis gets N/6 freq bands). Multiple of 6 ≤ 192.
#                         Default 66. Stamped (_xrd<N>) when non-default.
#   --max_samples N     — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_coordinate.sh                       # all GPUs, VST
#   bash scripts/train_coordinate.sh 2                     # 2 GPUs, VST
#   bash scripts/train_coordinate.sh 2 --mindcube          # 2 GPUs, MindCube
#   bash scripts/train_coordinate.sh 1 --skip_layers -2    # use Layer 31
#   bash scripts/train_coordinate.sh 1 --max_samples 6     # quick smoke run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SPATIAL_DIR"

# ── argument parsing ────────────────────────────────────────────────────────

NPROC=""
MAX_SAMPLES=""
DECOUPLE_FLAG=""
SKIP_LAYERS_ARG=""
COORD_WEIGHT_ARG=""
LORA_RANK_ARG=""
XYZ_ROPE_DIM=""
DATASET="vst"
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --decouple)     DECOUPLE_FLAG="--decouple"; shift ;;
        --skip_layers)  SKIP_LAYERS_ARG="$2"; shift 2 ;;
        --coord_weight) COORD_WEIGHT_ARG="$2"; shift 2 ;;
        --lora_rank)    LORA_RANK_ARG="$2"; shift 2 ;;
        --xyz_rope_dim) XYZ_ROPE_DIM="$2"; shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --mindcube)     DATASET="mindcube"; shift ;;
        --vst)          DATASET="vst"; shift ;;
        *)
            if [ $_positional -eq 0 ]; then
                NPROC="$1"
            fi
            _positional=$((_positional + 1))
            shift ;;
    esac
done

if [ -n "$NPROC" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NPROC - 1)))
else
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NPROC - 1)))
fi

# ── hyperparameters ─────────────────────────────────────────────────────────

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
if [ "$DATASET" = "mindcube" ]; then
    JSON_PATH="$SPATIAL_DIR/datasets/train/MindCube/MindCube_train.jsonl"
    VST_RESULTS_DIR="$SPATIAL_DIR/datasets/train/MindCube/3d_results"
else
    JSON_PATH="$SPATIAL_DIR/datasets/train/VST_parsed/vst_500k.json"
    VST_RESULTS_DIR="$SPATIAL_DIR/datasets/train/VST/3d_results"
fi

EPOCHS=1
LR=2e-4
LORA_RANK=16
MAX_IMAGES=8
GRAD_ACCUM=8
NUM_WORKERS=4

ANSWER_WEIGHT=1.0
COORD_WEIGHT=1.0
COORD_UPSCALE=4

SAVE_STEPS=1000
EVAL_STEPS=200

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# Apply CLI overrides
[ -n "$COORD_WEIGHT_ARG" ] && COORD_WEIGHT="$COORD_WEIGHT_ARG"
[ -n "$LORA_RANK_ARG"    ] && LORA_RANK="$LORA_RANK_ARG"

# ── coord-head layer ────────────────────────────────────────────────────────

SKIP_LAYERS="${SKIP_LAYERS_ARG:--1}"
SKIP_LAYERS_FLAG="--skip_layers ${SKIP_LAYERS}"

layer_name() {
    local v="$1"
    if   [ "$v" = "-1" ]; then echo "Layer 32 (post-norm, last)"
    elif [ "$v" = "-2" ]; then echo "Layer 31 (penultimate)"
    else                       echo "Layer $(( 32 + v ))"
    fi
}
SKIP_LAYERS_DISPLAY="$(layer_name "$SKIP_LAYERS")"

# ── run name ────────────────────────────────────────────────────────────────

_decouple_suffix="${DECOUPLE_FLAG:+_decouple}"

_r_suffix=""
[ "$LORA_RANK"    != "16"  ] && _r_suffix="_r${LORA_RANK}"
_cw_suffix=""
[ "$COORD_WEIGHT" != "1.0" ] && _cw_suffix="_cw${COORD_WEIGHT}"
_sl_suffix=""
[ "$SKIP_LAYERS"  != "-1"  ] && _sl_suffix="_sl${SKIP_LAYERS}"
_xrd_suffix=""
if [ -n "$XYZ_ROPE_DIM" ] && [ "$XYZ_ROPE_DIM" != "66" ] \
   && [ -n "$DECOUPLE_FLAG" ]; then
    _xrd_suffix="_xrd${XYZ_ROPE_DIM}"
fi

RUN_NAME="coordinate_${DATASET}${_decouple_suffix}${_r_suffix}${_cw_suffix}${_sl_suffix}${_xrd_suffix}"
WANDB_RUN_NAME="coord_${DATASET}_r${LORA_RANK}_ep${EPOCHS}_coord${COORD_WEIGHT}${_decouple_suffix}${_sl_suffix}${_xrd_suffix}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# ── setup ───────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

_mode_label="4D M-RoPE + Cartesian coord GT (default)"
[ -n "$DECOUPLE_FLAG" ] && _mode_label="decouple 3D M-RoPE + Cartesian XYZ RoPE"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EVAL_STEPS           = $EVAL_STEPS"
echo "[INFO] DATASET              = $DATASET"
echo "[INFO] Dataset              : $(basename "$JSON_PATH")"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Mode                 : $_mode_label"
if [ -n "$DECOUPLE_FLAG" ]; then
    echo "[INFO] xyz_rope_dim         = ${XYZ_ROPE_DIM:-66 (default)}"
fi
echo "[INFO] Coord head            : $SKIP_LAYERS_DISPLAY  (weight=$COORD_WEIGHT)"
echo "[INFO] Loss                 : LM answer CE + per-patch coord loss"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

XYZ_ROPE_DIM_FLAG=""
if [ -n "$XYZ_ROPE_DIM" ]; then
    XYZ_ROPE_DIM_FLAG="--xyz_rope_dim $XYZ_ROPE_DIM"
fi

# ── launch via torchrun (DDP) ───────────────────────────────────────────────

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    "${MASTER_PORT:-29503}" \
    train_coordinate.py \
    --model_path             "$MODEL_PATH"             \
    --json_path              "$JSON_PATH"              \
    --vst_results_dir        "$VST_RESULTS_DIR"        \
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
    --dataset                "$DATASET"                \
    $SKIP_LAYERS_FLAG                                  \
    $DECOUPLE_FLAG                                     \
    $XYZ_ROPE_DIM_FLAG                                 \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
