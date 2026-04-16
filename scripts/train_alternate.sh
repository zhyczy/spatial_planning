#!/usr/bin/env bash
# =============================================================================
# train_alternate.sh
#
# Alternating two-phase single-pass rotation-aware coordinate prediction
# training with differentiable M-RoPE.  LoRA fine-tuning of
# SpaForConditionalGeneration (Qwen3.5-VL) with:
#   - CameraTokenRotationEncoder  → predicts canonical rotation R
#     (trained end-to-end via lm_loss; no GT rotation)
#   - DepthPredictionTransformer  → coordinate head in rotated frame
#
# Schedule (per epoch — two full passes over the dataset)
#   Each epoch iterates the dataloader TWICE; pass 0 is Phase A, pass 1
#   is Phase B.  With EPOCHS=N both branches see N full passes total.
#   Phase A (first full pass):
#     * rotation_enc frozen; train LoRA + coord_head
#     * epoch 0   → R = I  (rotation_enc untrained)
#     * epoch ≥ 1 → R from current (frozen) rotation_enc
#   Phase B (second full pass):
#     * LoRA frozen; train rotation_enc via lm_loss only
#     * coord_head still trained by coord_loss (on detached hidden so
#       rotation_enc is NOT updated by the coord_loss path)
#
# Learning rate
#   Phase A and Phase B each run an independent cosine decay over their own
#   optimizer steps.  Set --lr_phase_a / --lr_phase_b to control per-phase
#   base LRs (fall back to --lr / --rotation_enc_lr respectively).
#
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_alternate.sh [num_gpus] [--train_data NAME] \
#       [--max_samples N] [--coord_weight W] [--coord_scale S] \
#       [--no_coord] [--relative] \
#       [--lr_phase_a LR] [--lr_phase_b LR]
#
#   num_gpus         — first positional arg, number of GPUs (default: all)
#   --train_data NAME— dataset name: mindcube | sat (default: mindcube)
#   --max_samples N  — truncate dataset to N entries (default: all)
#   --coord_weight W — weight for coordinate L1 loss (default: 1.0)
#   --coord_scale S  — XYZ discretization multiplier, must match MLLM (default: 100.0)
#   --no_coord       — disable coord loss / coord_head forward entirely
#   --relative       — coord_head predicts original (un-rotated) xyz with
#                      detached cam_feat conditioning from rotation_enc
#   --lr_phase_a LR  — base LR for Phase A (LoRA + coord_head); independent
#                      cosine decay over Phase A optim steps. Defaults to $LR.
#   --lr_phase_b LR  — base LR for Phase B (rotation_enc + coord_head);
#                      independent cosine decay over Phase B optim steps.
#                      Defaults to $ROTATION_ENC_LR.
#
# Examples:
#   bash scripts/train_alternate.sh                   # all GPUs
#   bash scripts/train_alternate.sh 4                 # 4 GPUs
#   bash scripts/train_alternate.sh 4 --train_data sat
#   bash scripts/train_alternate.sh 4 --lr_phase_a 2e-4 --lr_phase_b 5e-5
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

NPROC=""
TRAIN_DATA_CLI=""
MAX_SAMPLES=""
COORD_WEIGHT=""
COORD_SCALE=""
NO_COORD_CLI=false
RELATIVE_CLI=false
LR_PHASE_A_CLI=""
LR_PHASE_B_CLI=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --train_data)
            TRAIN_DATA_CLI="$2"; shift 2 ;;
        --max_samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --coord_weight)
            COORD_WEIGHT="$2"; shift 2 ;;
        --coord_scale)
            COORD_SCALE="$2"; shift 2 ;;
        --no_coord)
            NO_COORD_CLI=true; shift ;;
        --relative)
            RELATIVE_CLI=true; shift ;;
        --lr_phase_a)
            LR_PHASE_A_CLI="$2"; shift 2 ;;
        --lr_phase_b)
            LR_PHASE_B_CLI="$2"; shift 2 ;;
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
TRAINING_DATASET="${TRAIN_DATA_CLI:-mindcube}"   # {mindcube, sat} — via --train_data flag

case "$TRAINING_DATASET" in
    mindcube)
        JSON_PATH="$SPATIAL_DIR/datasets/train/MindCube/MindCube_train.jsonl"
        RESULTS_DIR="$SPATIAL_DIR/datasets/train/MindCube/3d_results"
        ;;
    sat)
        JSON_PATH="$SPATIAL_DIR/datasets/train/SAT/train_36k.json"
        RESULTS_DIR="$SPATIAL_DIR/datasets/train/SAT/3d_results"
        ;;
    *)
        echo "[ERR] Unknown TRAINING_DATASET: $TRAINING_DATASET" >&2
        exit 1
        ;;
esac

EPOCHS=6
NO_COORD=$NO_COORD_CLI  # true → disable coord loss / coord_head entirely
                        # (LM loss only; coord_head frozen by no-grad)
                        # toggle via `--no_coord` CLI flag
RELATIVE=$RELATIVE_CLI  # true → coord_head predicts original (un-rotated) xyz
                        # with detached cam_feat conditioning; toggle via `--relative`
LR=2e-4                 # Phase A base LR fallback (LoRA + coord_head)
                        # used when --lr_phase_a is not given.
ROTATION_ENC_LR=2e-4    # Phase B base LR fallback (rotation_enc + coord_head)
                        # used when --lr_phase_b is not given.
# Phase-specific base LRs. Each phase runs its own cosine decay that counts
# only that phase's optimizer steps. Empty = let train_alternate.py
# fall back to --lr / --rotation_enc_lr respectively.
LR_PHASE_A="$LR_PHASE_A_CLI"
LR_PHASE_B="$LR_PHASE_B_CLI"
LORA_CLIP=1.0           # grad-norm clip for LoRA + coord_head group
ROTATION_ENC_CLIP=0.3   # strict clip for rotation_enc (RoPE high-freq amplification)
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4

SAVE_STEPS=50
EVAL_STEPS=50

# Rotation encoder architecture
# d_model = ROT_NHEAD × mllm_head_dim (e.g. 4 × 256 = 1024 for Qwen3.5-4B)
ROT_NHEAD=4
ROT_DIM_FEEDFORWARD=2048
ROT_NUM_LAYERS=2

# Default loss weights (can be overridden via CLI)
_COORD_WEIGHT="${COORD_WEIGHT:-1.0}"
_COORD_SCALE="${COORD_SCALE:-100.0}"

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Run name / output dir
# =============================================================================

_METHOD="rotation_alternate"
if [ "$NO_COORD" = "true" ]; then
    _METHOD="${_METHOD}_no_coord"
fi
if [ "$RELATIVE" = "true" ]; then
    _METHOD="${_METHOD}_relative"
fi
RUN_NAME="${_METHOD}_${TRAINING_DATASET}"
if [ "$NO_COORD" = "true" ]; then
    WANDB_RUN_NAME="${_METHOD}_${TRAINING_DATASET}_r${LORA_RANK}_ep${EPOCHS}"
else
    WANDB_RUN_NAME="${_METHOD}_${TRAINING_DATASET}_r${LORA_RANK}_ep${EPOCHS}_cw${_COORD_WEIGHT}"
fi
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# =============================================================================
# Setup
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] TRAINING_DATASET     = $TRAINING_DATASET"
echo "[INFO] JSON_PATH            = $JSON_PATH"
echo "[INFO] RESULTS_DIR          = $RESULTS_DIR"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EPOCHS               = $EPOCHS"
echo "[INFO] NO_COORD             = $NO_COORD"
echo "[INFO] RELATIVE             = $RELATIVE"
echo "[INFO] LR_PHASE_A           = ${LR_PHASE_A:-<fallback to \$LR=$LR>}"
echo "[INFO] LR_PHASE_B           = ${LR_PHASE_B:-<fallback to \$ROTATION_ENC_LR=$ROTATION_ENC_LR>}"
echo "[INFO] coord_weight         = $_COORD_WEIGHT"
echo "[INFO] coord_scale          = $_COORD_SCALE"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Optional flags
# =============================================================================

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

NO_COORD_FLAG=""
if [ "$NO_COORD" = "true" ]; then
    NO_COORD_FLAG="--no_coord"
fi

RELATIVE_FLAG=""
if [ "$RELATIVE" = "true" ]; then
    RELATIVE_FLAG="--relative"
fi

LR_PHASE_A_FLAG=""
if [ -n "$LR_PHASE_A" ]; then
    LR_PHASE_A_FLAG="--lr_phase_a $LR_PHASE_A"
fi

LR_PHASE_B_FLAG=""
if [ -n "$LR_PHASE_B" ]; then
    LR_PHASE_B_FLAG="--lr_phase_b $LR_PHASE_B"
fi

# =============================================================================
# Run via torchrun (DDP)
# =============================================================================

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    29501 \
    train_alternate.py \
    --model_path             "$MODEL_PATH"             \
    --training_dataset       "$TRAINING_DATASET"       \
    --json_path              "$JSON_PATH"              \
    --results_dir            "$RESULTS_DIR"            \
    --output_dir             "$OUTPUT_DIR"             \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
    --rotation_enc_lr        "$ROTATION_ENC_LR"        \
    --lora_clip              "$LORA_CLIP"              \
    --rotation_enc_clip      "$ROTATION_ENC_CLIP"      \
    --lora_rank              "$LORA_RANK"              \
    --max_images             "$MAX_IMAGES"             \
    --grad_accum             "$GRAD_ACCUM"             \
    --num_workers            "$NUM_WORKERS"            \
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --coord_weight           "$_COORD_WEIGHT"          \
    --coord_scale            "$_COORD_SCALE"           \
    --rot_nhead              "$ROT_NHEAD"              \
    --rot_dim_feedforward    "$ROT_DIM_FEEDFORWARD"    \
    --rot_num_layers         "$ROT_NUM_LAYERS"         \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG \
    $NO_COORD_FLAG \
    $RELATIVE_FLAG \
    $LR_PHASE_A_FLAG \
    $LR_PHASE_B_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
