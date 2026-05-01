#!/usr/bin/env bash
# =============================================================================
# train_atten.sh
#
# LoRA + per-layer SpatialAttentionBias fine-tuning of stock
# Qwen3_5ForConditionalGeneration (with .model swapped to SpatialAttnVanillaModel)
# on VST 500K (vst_500k.json, 563,190 entries → 551,013 unique ids in
# 3d_results). LM answer loss only — no contrast / match heads.
#
# Mode is fixed: Qwen3.5 ORIGINAL 3D M-RoPE [11,11,10] (UNCHANGED) +
# per-layer 2-layer geometric MLP that produces a per-head additive bias on
# vision-vision attention pairs. Three parameter-free mask layers enforce
# safety / make the bias usable:
#   • vision_mask (in bias module)         — bias zero outside V×V cells
#   • V↔V prefix-mask hole (in TextModel)  — V↔V attention is bidirectional
#   • torch.where defense (in wrapper)     — preserve causal/padding barriers
# The 3-D scene info enters the model only through the bias module —
# never through position_ids. Full design notes:
#   md/model_design/spatial_attention.md
#
# To compare against 4D M-RoPE / decouple / polar variants use train_correspondence.sh.
#
# Usage:
#   bash scripts/train_atten.sh [num_gpus] [--max_samples N]
#
#   num_gpus         — first positional arg, number of GPUs (default: all visible)
#   --max_samples N  — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_atten.sh                       # all GPUs
#   bash scripts/train_atten.sh 2                     # 2 GPUs
#   bash scripts/train_atten.sh 1 --max_samples 6     # quick smoke run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SPATIAL_DIR"

# ── argument parsing ────────────────────────────────────────────────────────

NPROC=""
MAX_SAMPLES=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
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
LORA_RANK=16
MAX_IMAGES=8
GRAD_ACCUM=8
NUM_WORKERS=4

SAVE_STEPS=1000
EVAL_STEPS=1000

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# ── run name ────────────────────────────────────────────────────────────────

RUN_NAME="atten_vst"
WANDB_RUN_NAME="atten_vst_r${LORA_RANK}_ep${EPOCHS}"
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
echo "[INFO] Mode                 : SpatialAttn (3D M-RoPE UNCHANGED + per-layer geometric MLP bias)"
echo "[INFO] Loss                 : LM answer CE only"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

# ── launch via torchrun (DDP) ───────────────────────────────────────────────

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    29501 \
    train_atten.py \
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
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
