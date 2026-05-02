#!/usr/bin/env bash
# =============================================================================
# train_enhance.sh
#
# Baseline launcher: frozen Qwen3.5-VL + sinusoidal 3D positional encoding on
# each merged-patch vision token + trainable coord_head (DepthPredictionTransformer).
#
# Compared to train_coordinate.sh: NO LoRA, NO M-RoPE changes — Qwen weights
# stay exactly as pretrained. The 3D PE is parameter-free; the only trained
# module is `coord_head`.
#
# Usage:
#   bash scripts/train_enhance.sh [num_gpus]
#                                 [--skip_layers LAYER]
#                                 [--coord_weight W]
#                                 [--max_samples N]
#
# Examples:
#   bash scripts/train_enhance.sh                       # all GPUs
#   bash scripts/train_enhance.sh 2                     # 2 GPUs
#   bash scripts/train_enhance.sh 1 --max_samples 6     # quick smoke run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SPATIAL_DIR"

# ── argument parsing ────────────────────────────────────────────────────────

NPROC=""
MAX_SAMPLES=""
SKIP_LAYERS_ARG=""
COORD_WEIGHT_ARG=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip_layers)  SKIP_LAYERS_ARG="$2"; shift 2 ;;
        --coord_weight) COORD_WEIGHT_ARG="$2"; shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2"; shift 2 ;;
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
JSON_PATH="$SPATIAL_DIR/datasets/train/VST_parsed/vst_500k.json"
VST_RESULTS_DIR="$SPATIAL_DIR/datasets/train/VST/3d_results"

EPOCHS=1
LR=2e-4
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

[ -n "$COORD_WEIGHT_ARG" ] && COORD_WEIGHT="$COORD_WEIGHT_ARG"

SKIP_LAYERS="${SKIP_LAYERS_ARG:--1}"
SKIP_LAYERS_FLAG="--skip_layers ${SKIP_LAYERS}"

_cw_suffix=""
[ "$COORD_WEIGHT" != "1.0" ] && _cw_suffix="_cw${COORD_WEIGHT}"
_sl_suffix=""
[ "$SKIP_LAYERS"  != "-1"  ] && _sl_suffix="_sl${SKIP_LAYERS}"

RUN_NAME="enhance_vst${_cw_suffix}${_sl_suffix}"
WANDB_RUN_NAME="enhance_vst_ep${EPOCHS}_coord${COORD_WEIGHT}${_sl_suffix}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# ── setup ───────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EVAL_STEPS           = $EVAL_STEPS"
echo "[INFO] Dataset              : VST 500K"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Mode                 : Frozen Qwen3.5 + sinusoidal 3D PE on patches"
echo "[INFO] Coord head            : skip_layers=$SKIP_LAYERS  weight=$COORD_WEIGHT"
echo "[INFO] Loss                 : LM CE (logging) + per-patch coord loss (training)"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

# ── launch via torchrun (DDP) ───────────────────────────────────────────────

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    "${MASTER_PORT:-29504}" \
    train_enhance.py \
    --model_path             "$MODEL_PATH"             \
    --json_path              "$JSON_PATH"              \
    --vst_results_dir        "$VST_RESULTS_DIR"        \
    --output_dir             "$OUTPUT_DIR"             \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
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
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
