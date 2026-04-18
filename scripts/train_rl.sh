#!/usr/bin/env bash
# =============================================================================
# train_rl.sh
#
# Alternating two-phase training with GRPO-based RL in Phase B.
#   Phase A (SFT)  : LoRA + coord_head trainable; encoder frozen at argmax anchor.
#   Phase B (GRPO) : LoRA frozen; encoder heads + coord_head trainable.
#     - head_cls  ← policy gradient   (REINFORCE, reward = 0/1 MCQ accuracy)
#     - head_res  ← RWR weighted SFT  (softmax(-lm_loss/τ) over correct anchors)
#     - coord_head ← standard SFT     (R detached, hidden detached)
#
# Action space
#   --action_space discrete → head_cls only (24 chiral cube anchors, no residual)
#   --action_space hybrid   → head_cls + head_res (24 anchors + ±30° refinement)
#
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_rl.sh [num_gpus] [options]
#
#   --train_data NAME       mindcube | sat    (default: mindcube)
#   --max_samples N         truncate dataset to N
#   --coord_weight W        weight for coordinate L1 loss (default: 1.0)
#   --coord_scale S         XYZ discretization multiplier (default: 100.0)
#   --no_coord              disable coord loss entirely
#   --relative              coord_head predicts original xyz
#   --action_space {discrete, hybrid}    default: hybrid
#   --lr_phase_a LR         Phase A base LR
#   --lr_phase_b LR         Phase B base LR (head_res + rot_bb; head_cls = this × scale)
#   --k_max_grad K          top-K correct anchors for RWR grad forward (default: 4)
#   --rwr_tau T             RWR softmax temperature (default: 1.0)
#   --entropy_beta B        entropy bonus coeff (default: 0.01)
#   --kl_lambda L           KL-to-uniform coeff (default: 0.001)
#   --rl_weight W           scale on (L_rl + L_ent + L_kl) (default: 1.0)
#   --sft_weight W          scale on differentiable SFT (default: 1.0)
#
# Examples:
#   bash scripts/train_rl.sh                   # all GPUs
#   bash scripts/train_rl.sh 4                 # 4 GPUs
#   bash scripts/train_rl.sh 4 --action_space discrete
#   bash scripts/train_rl.sh 4 --lr_phase_a 2e-4 --lr_phase_b 5e-5
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
ACTION_SPACE_CLI=""
LR_PHASE_A_CLI=""
LR_PHASE_B_CLI=""
K_MAX_GRAD_CLI=""
RWR_TAU_CLI=""
ENTROPY_BETA_CLI=""
KL_LAMBDA_CLI=""
RL_WEIGHT_CLI=""
SFT_WEIGHT_CLI=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --train_data)    TRAIN_DATA_CLI="$2";    shift 2 ;;
        --max_samples)   MAX_SAMPLES="$2";       shift 2 ;;
        --coord_weight)  COORD_WEIGHT="$2";      shift 2 ;;
        --coord_scale)   COORD_SCALE="$2";       shift 2 ;;
        --no_coord)      NO_COORD_CLI=true;      shift   ;;
        --relative)      RELATIVE_CLI=true;      shift   ;;
        --action_space)  ACTION_SPACE_CLI="$2";  shift 2 ;;
        --lr_phase_a)    LR_PHASE_A_CLI="$2";    shift 2 ;;
        --lr_phase_b)    LR_PHASE_B_CLI="$2";    shift 2 ;;
        --k_max_grad)    K_MAX_GRAD_CLI="$2";    shift 2 ;;
        --rwr_tau)       RWR_TAU_CLI="$2";       shift 2 ;;
        --entropy_beta)  ENTROPY_BETA_CLI="$2";  shift 2 ;;
        --kl_lambda)     KL_LAMBDA_CLI="$2";     shift 2 ;;
        --rl_weight)     RL_WEIGHT_CLI="$2";     shift 2 ;;
        --sft_weight)    SFT_WEIGHT_CLI="$2";    shift 2 ;;
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
TRAINING_DATASET="${TRAIN_DATA_CLI:-mindcube}"

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
NO_COORD=$NO_COORD_CLI
RELATIVE=$RELATIVE_CLI
ACTION_SPACE="${ACTION_SPACE_CLI:-hybrid}"     # hybrid | discrete

LR=2e-4                   # Phase A fallback
ROTATION_ENC_LR=5e-5      # Phase B fallback (head_res + rot_bb)
LR_PHASE_A="$LR_PHASE_A_CLI"
LR_PHASE_B="$LR_PHASE_B_CLI"

# Per-group grad clipping (RoPE-aware strict on policy/backbone heads)
LORA_CLIP=1.0
ROT_BB_CLIP=0.3
HEAD_CLS_CLIP=0.3
HEAD_RES_CLIP=1.0
HEAD_CLS_LR_SCALE=0.5     # head_cls LR = lr_phase_b × 0.5

LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4
SAVE_STEPS=50
EVAL_STEPS=50
EVAL_TOPK=5                # Pass@K for policy top-K acc at eval

# Rotation encoder architecture
ROT_NHEAD=4
ROT_DIM_FEEDFORWARD=2048
ROT_NUM_LAYERS=2

# RL hyperparameters
K_MAX_GRAD="${K_MAX_GRAD_CLI:-4}"
RWR_TAU="${RWR_TAU_CLI:-1.0}"
ENTROPY_BETA="${ENTROPY_BETA_CLI:-0.01}"
KL_LAMBDA="${KL_LAMBDA_CLI:-0.001}"
RL_WEIGHT="${RL_WEIGHT_CLI:-1.0}"
SFT_WEIGHT="${SFT_WEIGHT_CLI:-1.0}"
RESIDUAL_CLAMP=0.5235987755982988   # π/6 rad ≈ 30°

# Loss weights
_COORD_WEIGHT="${COORD_WEIGHT:-1.0}"
_COORD_SCALE="${COORD_SCALE:-100.0}"

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Run name / output dir
# =============================================================================

_METHOD="rotation_rl_${ACTION_SPACE}"
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
echo "[INFO] ACTION_SPACE         = $ACTION_SPACE"
echo "[INFO] NO_COORD             = $NO_COORD"
echo "[INFO] RELATIVE             = $RELATIVE"
echo "[INFO] LR_PHASE_A           = ${LR_PHASE_A:-<fallback to \$LR=$LR>}"
echo "[INFO] LR_PHASE_B           = ${LR_PHASE_B:-<fallback to \$ROTATION_ENC_LR=$ROTATION_ENC_LR>}"
echo "[INFO] HEAD_CLS_LR_SCALE    = $HEAD_CLS_LR_SCALE"
echo "[INFO] K_MAX_GRAD           = $K_MAX_GRAD"
echo "[INFO] RWR_TAU              = $RWR_TAU"
echo "[INFO] ENTROPY_BETA         = $ENTROPY_BETA"
echo "[INFO] KL_LAMBDA            = $KL_LAMBDA"
echo "[INFO] RL_WEIGHT            = $RL_WEIGHT"
echo "[INFO] SFT_WEIGHT           = $SFT_WEIGHT"
echo "[INFO] EVAL_TOPK            = $EVAL_TOPK"
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
    --master_port    29502 \
    train_rl.py \
    --model_path             "$MODEL_PATH"             \
    --training_dataset       "$TRAINING_DATASET"       \
    --json_path              "$JSON_PATH"              \
    --results_dir            "$RESULTS_DIR"            \
    --output_dir             "$OUTPUT_DIR"             \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
    --rotation_enc_lr        "$ROTATION_ENC_LR"        \
    --lora_clip              "$LORA_CLIP"              \
    --rot_bb_clip            "$ROT_BB_CLIP"            \
    --head_cls_clip          "$HEAD_CLS_CLIP"          \
    --head_res_clip          "$HEAD_RES_CLIP"          \
    --head_cls_lr_scale      "$HEAD_CLS_LR_SCALE"      \
    --lora_rank              "$LORA_RANK"              \
    --max_images             "$MAX_IMAGES"             \
    --grad_accum             "$GRAD_ACCUM"             \
    --num_workers            "$NUM_WORKERS"            \
    --save_steps             "$SAVE_STEPS"             \
    --eval_steps             "$EVAL_STEPS"             \
    --eval_topk              "$EVAL_TOPK"              \
    --coord_weight           "$_COORD_WEIGHT"          \
    --coord_scale            "$_COORD_SCALE"           \
    --rot_nhead              "$ROT_NHEAD"              \
    --rot_dim_feedforward    "$ROT_DIM_FEEDFORWARD"    \
    --rot_num_layers         "$ROT_NUM_LAYERS"         \
    --action_space           "$ACTION_SPACE"           \
    --residual_clamp         "$RESIDUAL_CLAMP"         \
    --k_max_grad             "$K_MAX_GRAD"             \
    --rwr_tau                "$RWR_TAU"                \
    --entropy_beta           "$ENTROPY_BETA"           \
    --kl_lambda              "$KL_LAMBDA"              \
    --rl_weight              "$RL_WEIGHT"              \
    --sft_weight             "$SFT_WEIGHT"             \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG \
    $NO_COORD_FLAG \
    $RELATIVE_FLAG \
    $LR_PHASE_A_FLAG \
    $LR_PHASE_B_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
