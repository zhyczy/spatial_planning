#!/usr/bin/env bash
# =============================================================================
# train_atten.sh
#
# LoRA + per-layer SpatialAttentionBias fine-tuning of stock
# Qwen3_5ForConditionalGeneration (with .model swapped to SpatialAttnVanillaModel)
# on VST MCQ subset (vst_mcq.json). LM answer loss only — no contrast / match
# heads.
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
# To compare against 4D M-RoPE / decouple variants use train_correspondence.sh.
#
# Usage:
#   bash scripts/train_atten.sh [num_gpus] [--mindcube|--vst] [--couple]
#                               [--max_samples N]
#
#   num_gpus         — first positional arg, number of GPUs (default: all visible)
#   --mindcube       — train on MindCube_train.jsonl + MindCube 3d_results (default)
#   --vst            — train on vst_mcq.json + VST 3d_results
#   --couple         — joint-train LoRA + SpatialAttentionBias (historical
#                      behavior). DEFAULT (no flag) = bias-only: freeze LoRA,
#                      route 100% of gradient to the geometric MLP so W₁/W₂
#                      can specialize. Stamped into RUN_NAME (_solo / _couple).
#   --max_samples N  — truncate dataset to N entries (default: all)
#
# Examples:
#   bash scripts/train_atten.sh                       # all GPUs, MindCube, BIAS-ONLY
#   bash scripts/train_atten.sh 4 --couple            # 4 GPUs, MindCube, joint train
#   bash scripts/train_atten.sh 1 --max_samples 6     # quick smoke run, bias-only
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SPATIAL_DIR"

# ── argument parsing ────────────────────────────────────────────────────────

NPROC=""
MAX_SAMPLES=""
FREEZE=0
DATASET="mindcube"
COUPLE=0
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --freeze)      FREEZE=1; shift ;;
        --mindcube)    DATASET="mindcube"; shift ;;
        --vst)         DATASET="vst"; shift ;;
        --couple)      COUPLE=1; shift ;;
        *)
            if [ $_positional -eq 0 ]; then
                NPROC="$1"
            fi
            _positional=$((_positional + 1))
            shift ;;
    esac
done

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Respect a pre-set CUDA_VISIBLE_DEVICES (e.g. for co-locating with another
    # job on a specific GPU). NPROC must be consistent with it.
    _n_visible=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    if [ -z "$NPROC" ]; then NPROC="$_n_visible"; fi
elif [ -n "$NPROC" ]; then
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
BIAS_LR_SCALE=10.0
BIAS_W2_INIT_SCALE=0.01

SAVE_STEPS=200
EVAL_STEPS=50

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# ── run name ────────────────────────────────────────────────────────────────

_freeze_suffix=""
[ "$FREEZE" -eq 1 ] && _freeze_suffix="_freeze"

if [ "$COUPLE" -eq 1 ]; then
    RUN_NAME="atten_couple_${DATASET}${_freeze_suffix}"
    WANDB_RUN_NAME="atten_couple_${DATASET}${_freeze_suffix}_r${LORA_RANK}_ep${EPOCHS}"
else
    RUN_NAME="atten_${DATASET}_solo${_freeze_suffix}"
    WANDB_RUN_NAME="atten_${DATASET}_solo${_freeze_suffix}_r${LORA_RANK}_ep${EPOCHS}"
fi
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# ── setup ───────────────────────────────────────────────────────────────────

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EVAL_STEPS           = $EVAL_STEPS"
echo "[INFO] BIAS_LR_SCALE        = $BIAS_LR_SCALE"
echo "[INFO] BIAS_W2_INIT_SCALE   = $BIAS_W2_INIT_SCALE"
echo "[INFO] FREEZE (linear-attn) = $FREEZE"
echo "[INFO] COUPLE (joint train) = $COUPLE  (0 = bias-only, 1 = LoRA + bias jointly)"
echo "[INFO] DATASET              = $DATASET"
echo "[INFO] Dataset              : $(basename "$JSON_PATH")"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Mode                 : SpatialAttn (3D M-RoPE UNCHANGED + per-layer geometric MLP bias)"
echo "[INFO] Loss                 : LM answer CE only"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

FREEZE_FLAG=""
if [ "$FREEZE" -eq 1 ]; then
    FREEZE_FLAG="--freeze"
fi

COUPLE_FLAG=""
if [ "$COUPLE" -eq 1 ]; then
    COUPLE_FLAG="--couple"
fi

# ── launch via torchrun (DDP) ───────────────────────────────────────────────

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    29502 \
    train_atten.py \
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
    --bias_lr_scale          "$BIAS_LR_SCALE"          \
    --bias_w2_init_scale     "$BIAS_W2_INIT_SCALE"     \
    --dataset                "$DATASET"                \
    $MAX_SAMPLES_FLAG \
    $FREEZE_FLAG \
    $COUPLE_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
