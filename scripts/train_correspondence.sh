#!/usr/bin/env bash
# =============================================================================
# train_correspondence.sh
#
# LoRA fine-tuning of SpaForConditionalGeneration (LM answer loss only)
# on VST MCQ subset (vst_mcq.json). Multi-GPU via torchrun (DDP).
#
# Default mode: 4D M-RoPE (use_xyz=True, Cartesian xyz fed into vision-token
# position). Two alternative modes via flags (mutually exclusive):
#   --decouple  : decouple architecture + Cartesian XYZ RoPE
#                 (Qwen 3D M-RoPE [11,11,10] UNCHANGED in rotary 64 dims +
#                 new XYZ RoPE in pass-through 64..129)
#   --vanilla   : original Qwen 3D M-RoPE only (no image_xyz at all)
#
# Compare against train_atten.sh (per-layer attention bias) and
# train_coordinate.sh (per-patch coord head loss).
#
# Usage:
#   bash scripts/train_correspondence.sh [num_gpus] [--decouple|--vanilla]
#                                        [--mindcube|--vst]
#                                        [--xyz_rope_dim N] [--max_samples N]
#
#   num_gpus         — first positional arg, number of GPUs (default: all visible)
#   --decouple       — decouple + Cartesian XYZ RoPE
#   --vanilla        — original Qwen 3D M-RoPE (no image_xyz)
#   --mindcube       — train on MindCube_train.jsonl + MindCube 3d_results (default)
#   --vst            — train on vst_mcq.json + VST 3d_results
#   --xyz_rope_dim N — total head_dim units for XYZ RoPE under --decouple
#                      (each axis gets N/6 freq bands). Multiple of 6 ≤ 192.
#                      Default 66 (= 11 bands per axis). Stamped into RUN_NAME
#                      when non-default and applicable.
#   --max_samples N  — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_correspondence.sh                       # all GPUs, VST, 4D M-RoPE
#   bash scripts/train_correspondence.sh 2                     # 2 GPUs, VST, 4D M-RoPE
#   bash scripts/train_correspondence.sh 2 --mindcube          # 2 GPUs, MindCube, 4D M-RoPE
#   bash scripts/train_correspondence.sh 2 --decouple          # decouple + Cartesian
#   bash scripts/train_correspondence.sh 2 --vanilla           # original 3D M-RoPE
#   bash scripts/train_correspondence.sh 1 --max_samples 6     # quick smoke run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SPATIAL_DIR"

# ── argument parsing ────────────────────────────────────────────────────────

NPROC=""
MAX_SAMPLES=""
VANILLA_FLAG=""
DECOUPLE_FLAG=""
XYZ_ROPE_DIM=""
FREEZE_FLAG=""
DATASET="mindcube"
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --vanilla)      VANILLA_FLAG="--vanilla"; shift ;;
        --decouple)     DECOUPLE_FLAG="--decouple"; shift ;;
        --xyz_rope_dim) XYZ_ROPE_DIM="$2"; shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --freeze)       FREEZE_FLAG="--freeze"; shift ;;
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
    JSON_PATH="$SPATIAL_DIR/datasets/train/VST_parsed/vst_mcq.json"
    VST_RESULTS_DIR="$SPATIAL_DIR/datasets/train/VST/3d_results"
fi

EPOCHS=9
LR=2e-4
WARMUP_STEPS=100
LORA_RANK=16
GRAD_ACCUM=16
NUM_WORKERS=4

SAVE_STEPS=200
EVAL_STEPS=50

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# ── run name ────────────────────────────────────────────────────────────────

_vanilla_suffix="${VANILLA_FLAG:+_vanilla}"
_decouple_suffix="${DECOUPLE_FLAG:+_decouple}"
_freeze_suffix="${FREEZE_FLAG:+_freeze}"

# Stamp xyz_rope_dim only when overridden AND applicable (--decouple).
_xrd_suffix=""
if [ -n "$XYZ_ROPE_DIM" ] && [ "$XYZ_ROPE_DIM" != "66" ] \
   && [ -n "$DECOUPLE_FLAG" ]; then
    _xrd_suffix="_xrd${XYZ_ROPE_DIM}"
fi

RUN_NAME="correspondence_${DATASET}${_decouple_suffix}${_vanilla_suffix}${_xrd_suffix}${_freeze_suffix}"
WANDB_RUN_NAME="corr_${DATASET}_r${LORA_RANK}_ep${EPOCHS}${_decouple_suffix}${_vanilla_suffix}${_xrd_suffix}${_freeze_suffix}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# ── setup ───────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

_mode_label="4D M-RoPE (image_xyz, Cartesian)"
[ -n "$VANILLA_FLAG"  ] && _mode_label="vanilla 3D M-RoPE (no image_xyz)"
[ -n "$DECOUPLE_FLAG" ] && _mode_label="decouple 3D M-RoPE + new XYZ RoPE (Cartesian)"

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
if [ -n "$FREEZE_FLAG" ]; then
    echo "[INFO] FREEZE (linear-attn) = on"
else
    echo "[INFO] FREEZE (linear-attn) = off"
fi
echo "[INFO] Loss                 : LM answer CE only"
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
    --master_port    29502 \
    train_correspondence.py \
    --model_path             "$MODEL_PATH"             \
    --json_path              "$JSON_PATH"              \
    --vst_results_dir        "$VST_RESULTS_DIR"        \
    --output_dir             "$OUTPUT_DIR"             \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
    --warmup_steps           "$WARMUP_STEPS"           \
    --lora_rank              "$LORA_RANK"              \
    --grad_accum             "$GRAD_ACCUM"             \
    --num_workers            "$NUM_WORKERS"            \
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    --dataset                "$DATASET"                \
    $DECOUPLE_FLAG                                     \
    $VANILLA_FLAG                                      \
    $XYZ_ROPE_DIM_FLAG                                 \
    $MAX_SAMPLES_FLAG                                  \
    $FREEZE_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
