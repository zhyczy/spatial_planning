#!/usr/bin/env bash
# =============================================================================
# train_rl.sh
#
# Alternating two-phase training, discrete-action PPO/GRPO in Phase B.
#   Phase A (SFT)  : LoRA + coord_head trainable; encoder frozen at argmax anchor.
#   Phase B (GRPO) : LoRA frozen; encoder heads + coord_head trainable.
#     - head_cls / rot_bb ← PPO (K inner epochs, clipped surrogate + entropy + KL)
#                           over 24 yaw anchors (every 15°); dense reward
#                           R(k) = -w_lm·lm_loss(k) − α·dist(R_k, R_gt) where
#                           dist is a w_yaw/w_pitch/w_roll-weighted cos-distance
#                           over decoupled ZYX Tait-Bryan Euler errors.
#     - coord_head        ← standard SFT at argmax anchor (hidden detached).
#
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_rl.sh [num_gpus] [options]
#
#   --train_data NAME        mindcube | sat    (default: mindcube)
#   --max_samples N          truncate dataset to N
#   --coord_weight W         weight for coordinate L1 loss (default: 1.0)
#   --coord_scale S          XYZ discretization multiplier (default: 100.0)
#   --no_coord               disable coord loss entirely
#   --relative               coord_head predicts original xyz
#   --decouple               Keep Qwen original 3D M-RoPE in rotary 64 dims +
#                            add new XYZ RoPE on pass-through (mirrors
#                            train_alternate.sh --decouple). Mutually
#                            exclusive with --relative.
#   --xyz_rope_dim D         XYZ RoPE dim under --decouple (positive multiple
#                            of 6 ≤ 192; default 66).
#   --lr_phase_a LR          Phase A base LR
#   --lr_phase_b LR          Phase B base LR (rot_bb; head_cls = this × scale / K)
#   --w_lm W                 scale on -lm_loss dense reward (default: 1.0)
#   --alpha_dist A           Axis-decoupled cos-distance penalty coeff
#                            (default: 0.0 — disabled for lm_loss rotation-sensitivity
#                             probe. Set 0.8 to re-enable rotation discriminability.)
#   --w_yaw W                yaw weight in axis-decoupled distance (default: 1.0)
#   --w_pitch W              pitch weight (default: 0.0 — yaw-only policy)
#   --w_roll W               roll weight (default: 0.0 — yaw-only policy)
#   --entropy_beta B         entropy bonus coeff (default: 0.01)
#   --kl_lambda L            KL-to-uniform coeff (default: 0.001)
#   --rl_weight W            scale on (L_rl + L_ent + L_kl) (default: 1.0)
#   --sft_weight W           scale on differentiable SFT (default: 1.0)
#   --ppo_inner_epochs K     PPO inner-loop steps per rollout (default: 3)
#   --ppo_clip_eps E         PPO clipping epsilon (default: 0.2)
#
#   Bucket-aware anchor-prior shaping (MindCube 5-bucket taxonomy A/B/D/E/H;
#   C residue = 571 E-pos-obj samples with no pose-derivable frame):
#   --w_anchor_prior W       master switch (default: 0.0 — disabled; set 1.0 to re-enable)
#   --w_rot W                Buckets B/H/D/E (default: 0.8; pose-derived R_gt,
#                            geodesic θ handles anchor-reliability implicitly)
#   --w_trans W              Bucket A weight  (default: 0.8; identity prior)
#
# Examples:
#   bash scripts/train_rl.sh                                # all GPUs
#   bash scripts/train_rl.sh 4                              # 4 GPUs
#   bash scripts/train_rl.sh 4 --lr_phase_a 2e-4 --lr_phase_b 5e-5
#   bash scripts/train_rl.sh 4 --ppo_inner_epochs 5 --ppo_clip_eps 0.1
#   bash scripts/train_rl.sh 4 --decouple                   # decouple XYZ RoPE
#   bash scripts/train_rl.sh 4 --decouple --xyz_rope_dim 132
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
DECOUPLE_CLI=false
XYZ_ROPE_DIM_CLI=""
LR_PHASE_A_CLI=""
LR_PHASE_B_CLI=""
W_LM_CLI=""
ALPHA_DIST_CLI=""
W_YAW_CLI=""
W_PITCH_CLI=""
W_ROLL_CLI=""
ENTROPY_BETA_CLI=""
KL_LAMBDA_CLI=""
RL_WEIGHT_CLI=""
SFT_WEIGHT_CLI=""
PPO_INNER_EPOCHS_CLI=""
PPO_CLIP_EPS_CLI=""
W_ANCHOR_PRIOR_CLI=""
W_ROT_CLI=""
W_TRANS_CLI=""
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --train_data)        TRAIN_DATA_CLI="$2";        shift 2 ;;
        --max_samples)       MAX_SAMPLES="$2";           shift 2 ;;
        --coord_weight)      COORD_WEIGHT="$2";          shift 2 ;;
        --coord_scale)       COORD_SCALE="$2";           shift 2 ;;
        --no_coord)          NO_COORD_CLI=true;          shift   ;;
        --relative)          RELATIVE_CLI=true;          shift   ;;
        --decouple)          DECOUPLE_CLI=true;          shift   ;;
        --xyz_rope_dim)      XYZ_ROPE_DIM_CLI="$2";      shift 2 ;;
        --lr_phase_a)        LR_PHASE_A_CLI="$2";        shift 2 ;;
        --lr_phase_b)        LR_PHASE_B_CLI="$2";        shift 2 ;;
        --w_lm)              W_LM_CLI="$2";              shift 2 ;;
        --alpha_dist)        ALPHA_DIST_CLI="$2";        shift 2 ;;
        --w_yaw)             W_YAW_CLI="$2";             shift 2 ;;
        --w_pitch)           W_PITCH_CLI="$2";           shift 2 ;;
        --w_roll)            W_ROLL_CLI="$2";            shift 2 ;;
        --entropy_beta)      ENTROPY_BETA_CLI="$2";      shift 2 ;;
        --kl_lambda)         KL_LAMBDA_CLI="$2";         shift 2 ;;
        --rl_weight)         RL_WEIGHT_CLI="$2";         shift 2 ;;
        --sft_weight)        SFT_WEIGHT_CLI="$2";        shift 2 ;;
        --ppo_inner_epochs)  PPO_INNER_EPOCHS_CLI="$2";  shift 2 ;;
        --ppo_clip_eps)      PPO_CLIP_EPS_CLI="$2";      shift 2 ;;
        --w_anchor_prior)    W_ANCHOR_PRIOR_CLI="$2";    shift 2 ;;
        --w_rot)             W_ROT_CLI="$2";             shift 2 ;;
        --w_trans)           W_TRANS_CLI="$2";           shift 2 ;;
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
DECOUPLE=$DECOUPLE_CLI                              # mirrors train_alternate.sh
XYZ_ROPE_DIM="${XYZ_ROPE_DIM_CLI:-66}"              # only used under --decouple

LR=2e-4                   # Phase A fallback
ROTATION_ENC_LR=2e-4      # Phase B fallback (rot_bb)
LR_PHASE_A="$LR_PHASE_A_CLI"
LR_PHASE_B="$LR_PHASE_B_CLI"

# Per-group grad clipping (RoPE-aware strict on policy/backbone heads)
LORA_CLIP=1.0
ROT_BB_CLIP=0.3
HEAD_CLS_CLIP=0.3
HEAD_CLS_LR_SCALE=1.0     # head_cls LR = lr_phase_b × 1.0 (no discount)

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

# RL / PPO hyperparameters
W_LM="${W_LM_CLI:-1.0}"
# Geometric constraints disabled by default for lm_loss rotation-sensitivity probe.
# Re-enable with --alpha_dist 0.8 --w_anchor_prior 1.0 once diagnosis is done.
ALPHA_DIST="${ALPHA_DIST_CLI:-0.0}"
# Axis-decoupled rotation-distance weights (defaults match train_rl.py argparse).
# Policy is yaw-only (R_bins = 24 yaw bins every 15°); dpitch/droll do not
# depend on the anchor k, so keep w_pitch = w_roll = 0.
W_YAW="${W_YAW_CLI:-1.0}"
W_PITCH="${W_PITCH_CLI:-0.0}"
W_ROLL="${W_ROLL_CLI:-0.0}"
ENTROPY_BETA="${ENTROPY_BETA_CLI:-0.01}"
KL_LAMBDA="${KL_LAMBDA_CLI:-0.001}"
RL_WEIGHT="${RL_WEIGHT_CLI:-1.0}"
SFT_WEIGHT="${SFT_WEIGHT_CLI:-1.0}"
PPO_INNER_EPOCHS="${PPO_INNER_EPOCHS_CLI:-3}"
PPO_CLIP_EPS="${PPO_CLIP_EPS_CLI:-0.2}"

# Bucket-aware anchor-prior shaping (defaults match train_rl.py argparse).
# Single w_rot weight for all pose-derived buckets (B/H/D/E); the continuous
# geodesic θ already differentiates anchor reliability implicitly. w_trans
# stays separate because Bucket A uses an identity R_gt (semantically distinct
# from pose-derived rotations).
W_ANCHOR_PRIOR="${W_ANCHOR_PRIOR_CLI:-0.0}"
W_ROT="${W_ROT_CLI:-0.8}"
W_TRANS="${W_TRANS_CLI:-0.8}"

# Loss weights
_COORD_WEIGHT="${COORD_WEIGHT:-1.0}"
_COORD_SCALE="${COORD_SCALE:-100.0}"

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Run name / output dir
# =============================================================================

_METHOD="rotation_rl_discrete"
if [ "$NO_COORD" = "true" ]; then
    _METHOD="${_METHOD}_no_coord"
fi
if [ "$RELATIVE" = "true" ]; then
    _METHOD="${_METHOD}_relative"
fi
if [ "$DECOUPLE" = "true" ]; then
    _METHOD="${_METHOD}_decouple"
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
echo "[INFO] DECOUPLE             = $DECOUPLE"
if [ "$DECOUPLE" = "true" ]; then
    echo "[INFO]   xyz_rope_dim       = $XYZ_ROPE_DIM"
fi
echo "[INFO] LR_PHASE_A           = ${LR_PHASE_A:-<fallback to \$LR=$LR>}"
echo "[INFO] LR_PHASE_B           = ${LR_PHASE_B:-<fallback to \$ROTATION_ENC_LR=$ROTATION_ENC_LR>}"
echo "[INFO] HEAD_CLS_LR_SCALE    = $HEAD_CLS_LR_SCALE"
echo "[INFO] W_LM                 = $W_LM"
echo "[INFO] ALPHA_DIST           = $ALPHA_DIST"
echo "[INFO] W_YAW / W_PITCH / W_ROLL = $W_YAW / $W_PITCH / $W_ROLL"
echo "[INFO] ENTROPY_BETA         = $ENTROPY_BETA"
echo "[INFO] KL_LAMBDA            = $KL_LAMBDA"
echo "[INFO] RL_WEIGHT            = $RL_WEIGHT"
echo "[INFO] SFT_WEIGHT           = $SFT_WEIGHT"
echo "[INFO] PPO_INNER_EPOCHS     = $PPO_INNER_EPOCHS"
echo "[INFO] PPO_CLIP_EPS         = $PPO_CLIP_EPS"
echo "[INFO] W_ANCHOR_PRIOR       = $W_ANCHOR_PRIOR"
echo "[INFO] W_ROT    (B/H/D/E)   = $W_ROT"
echo "[INFO] W_TRANS  (A)         = $W_TRANS"
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

DECOUPLE_FLAG=""
if [ "$DECOUPLE" = "true" ]; then
    DECOUPLE_FLAG="--decouple --xyz_rope_dim $XYZ_ROPE_DIM"
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
    --w_lm                   "$W_LM"                   \
    --alpha_dist             "$ALPHA_DIST"             \
    --w_yaw                  "$W_YAW"                  \
    --w_pitch                "$W_PITCH"                \
    --w_roll                 "$W_ROLL"                 \
    --entropy_beta           "$ENTROPY_BETA"           \
    --kl_lambda              "$KL_LAMBDA"              \
    --rl_weight              "$RL_WEIGHT"              \
    --sft_weight             "$SFT_WEIGHT"             \
    --ppo_inner_epochs       "$PPO_INNER_EPOCHS"       \
    --ppo_clip_eps           "$PPO_CLIP_EPS"           \
    --w_anchor_prior         "$W_ANCHOR_PRIOR"         \
    --w_rot                  "$W_ROT"                  \
    --w_trans                "$W_TRANS"                \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG \
    $NO_COORD_FLAG \
    $RELATIVE_FLAG \
    $DECOUPLE_FLAG \
    $LR_PHASE_A_FLAG \
    $LR_PHASE_B_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
