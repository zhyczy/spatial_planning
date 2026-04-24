#!/usr/bin/env bash
# =============================================================================
# train_geo.sh
#
# Geometry-focused fine-tuning (MVP from md/discussion/learning_real_3d_geometry.md):
#   lm_loss (answer CE)
#     + contrast_weight * contrast_loss   ← §1.A multi-view triplet correspondence
#     + match_weight    * match_loss      ← §1.D xyz-image match classifier
#
# Position embedding: EXACTLY train_correspondence.py --decouple.
#   SpaDecForConditionalGeneration + original mrope_section [11,11,10] UNCHANGED
#   + XYZ RoPE (66 dims, Cartesian, rope_theta=10000) in pass-through dims
#   64..129 + patch_attention_layers_dec + coord_scale=100.0. train_geo.py does
#   not touch the position-embedding path; it only adds contrast/match heads.
#
# Contrast loss is SKIPPED on permuted steps (positive-pair finding needs
# correct xyz); on average contrast fires on ~(1 - match_prob) of steps.
#
# Multi-GPU via torchrun (DDP).
#
# Usage:
#   bash scripts/train_geo.sh [num_gpus] [options]
#
#   --max_samples N          truncate dataset to N entries (default: all)
#   --lora_rank R            LoRA rank (default: 16)
#   --coord_scale S          XYZ RoPE input scale (default: 100.0)
#
#   --lm_weight W            weight on LM answer loss (default: 1.0)
#   --contrast_weight W      weight on §1.A triplet correspondence (default: 1.0;
#                            set 0 to disable → pure lm + §1.D ablation)
#   --match_weight W         weight on §1.D match head CE (default: 0.3;
#                            set 0 to disable → pure lm + §1.A ablation)
#
#   --contrast_margin M      triplet margin (default: 0.5)
#   --contrast_eps E         xyz L2 threshold for positive-pair selection,
#                            scene units — same as coord_scale=100 convention
#                            (default: 0.05, i.e. ~5cm when xyz is in meters)
#   --n_contrast_anchors N   max anchor patches per (i, j) image pair
#                            (default: 64 — N·(N−1)·64 ≤ 768 triplets/step)
#   --match_prob P           prob of permuting xyz for §1.D (default: 0.5).
#                            Contrast is only computed on non-permuted steps,
#                            so setting this higher trades contrast for match.
#
#   --no_contrast            disable §1.A contrast loss entirely (skip
#                            computation, not just weight=0). Stamped as
#                            `_nocontrast` in RUN_NAME.
#   --no_match               disable §1.D match loss entirely (skip permutation
#                            and match head forward). Stamped as `_nomatch`
#                            in RUN_NAME.
#
#   --train_vision           unfreeze ViT (default: frozen)
#
# Examples:
#   bash scripts/train_geo.sh                      # all GPUs, defaults (both on)
#   bash scripts/train_geo.sh 6                    # 6 GPUs, both losses
#   bash scripts/train_geo.sh 6 --no_match         # §1.A contrast only → measures its gain
#   bash scripts/train_geo.sh 6 --no_contrast      # §1.D match only    → measures its gain
#   bash scripts/train_geo.sh 6 --no_contrast --no_match   # lm-only baseline
#   bash scripts/train_geo.sh 6 --lora_rank 32
#   bash scripts/train_geo.sh 1 --max_samples 8    # smoke test
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
LORA_RANK_CLI=""
COORD_SCALE_CLI=""
LM_WEIGHT_CLI=""
CONTRAST_WEIGHT_CLI=""
MATCH_WEIGHT_CLI=""
CONTRAST_MARGIN_CLI=""
CONTRAST_EPS_CLI=""
N_CONTRAST_ANCHORS_CLI=""
MATCH_PROB_CLI=""
TRAIN_VISION_CLI=false
NO_CONTRAST_CLI=false
NO_MATCH_CLI=false
_positional=0

while [ $# -gt 0 ]; do
    case "$1" in
        --max_samples)          MAX_SAMPLES="$2";            shift 2 ;;
        --lora_rank)            LORA_RANK_CLI="$2";          shift 2 ;;
        --coord_scale)          COORD_SCALE_CLI="$2";        shift 2 ;;
        --lm_weight)            LM_WEIGHT_CLI="$2";          shift 2 ;;
        --contrast_weight)      CONTRAST_WEIGHT_CLI="$2";    shift 2 ;;
        --match_weight)         MATCH_WEIGHT_CLI="$2";       shift 2 ;;
        --contrast_margin)      CONTRAST_MARGIN_CLI="$2";    shift 2 ;;
        --contrast_eps)         CONTRAST_EPS_CLI="$2";       shift 2 ;;
        --n_contrast_anchors)   N_CONTRAST_ANCHORS_CLI="$2"; shift 2 ;;
        --match_prob)           MATCH_PROB_CLI="$2";         shift 2 ;;
        --train_vision)         TRAIN_VISION_CLI=true;       shift   ;;
        --no_contrast)          NO_CONTRAST_CLI=true;        shift   ;;
        --no_match)             NO_MATCH_CLI=true;           shift   ;;
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
MAX_IMAGES=10
GRAD_ACCUM=8
NUM_WORKERS=4
SAVE_STEPS=50
EVAL_STEPS=50

LORA_RANK="${LORA_RANK_CLI:-16}"
COORD_SCALE="${COORD_SCALE_CLI:-100.0}"
TRAIN_VISION="$TRAIN_VISION_CLI"
NO_CONTRAST="$NO_CONTRAST_CLI"
NO_MATCH="$NO_MATCH_CLI"

# Geo loss weights (see md/discussion/learning_real_3d_geometry.md §1.A / §1.D)
LM_WEIGHT="${LM_WEIGHT_CLI:-1.0}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT_CLI:-1.0}"
MATCH_WEIGHT="${MATCH_WEIGHT_CLI:-0.3}"
CONTRAST_MARGIN="${CONTRAST_MARGIN_CLI:-0.5}"
CONTRAST_EPS="${CONTRAST_EPS_CLI:-0.05}"
N_CONTRAST_ANCHORS="${N_CONTRAST_ANCHORS_CLI:-64}"
MATCH_PROB="${MATCH_PROB_CLI:-0.5}"

WANDB_PROJECT="spc"
WANDB_ENTITY="actmrv"

# =============================================================================
# Run name / output dir
# =============================================================================

_METHOD="geo_mindcube"
# Stamp ablation switches + non-default weights into the name so A/B runs
# don't clobber each other's output dirs.
_nc_suffix="";  [ "$NO_CONTRAST"    = "true" ] && _nc_suffix="_nocontrast"
_nm_suffix="";  [ "$NO_MATCH"       = "true" ] && _nm_suffix="_nomatch"
_cw_suffix="";  [ "$CONTRAST_WEIGHT" != "1.0" ] && [ "$NO_CONTRAST" != "true" ] \
    && _cw_suffix="_cw${CONTRAST_WEIGHT}"
_mw_suffix="";  [ "$MATCH_WEIGHT"    != "0.3" ] && [ "$NO_MATCH"    != "true" ] \
    && _mw_suffix="_mw${MATCH_WEIGHT}"
_r_suffix="";   [ "$LORA_RANK"       != "16"  ] && _r_suffix="_r${LORA_RANK}"

RUN_NAME="${_METHOD}${_r_suffix}${_nc_suffix}${_nm_suffix}${_cw_suffix}${_mw_suffix}"
WANDB_RUN_NAME="geo_mindcube_r${LORA_RANK}_ep${EPOCHS}${_nc_suffix}${_nm_suffix}_cw${CONTRAST_WEIGHT}_mw${MATCH_WEIGHT}_mp${MATCH_PROB}"
OUTPUT_DIR="$SPATIAL_DIR/train_records/$RUN_NAME"

# =============================================================================
# Setup
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE       = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] JSON_PATH            = $JSON_PATH"
echo "[INFO] RESULTS_DIR          = $MINDCUBE_RESULTS_DIR"
echo "[INFO] MAX_SAMPLES          = ${MAX_SAMPLES:-all}"
echo "[INFO] EPOCHS               = $EPOCHS"
echo "[INFO] LORA_RANK            = $LORA_RANK"
echo "[INFO] TRAIN_VISION         = $TRAIN_VISION"
echo "[INFO] COORD_SCALE          = $COORD_SCALE"
echo "[INFO] LM_WEIGHT            = $LM_WEIGHT"
echo "[INFO] NO_CONTRAST  (§1.A)    = $NO_CONTRAST"
echo "[INFO] NO_MATCH     (§1.D)    = $NO_MATCH"
echo "[INFO] CONTRAST_WEIGHT (§1.A) = $CONTRAST_WEIGHT"
echo "[INFO] MATCH_WEIGHT    (§1.D) = $MATCH_WEIGHT"
echo "[INFO] CONTRAST_MARGIN      = $CONTRAST_MARGIN"
echo "[INFO] CONTRAST_EPS         = $CONTRAST_EPS"
echo "[INFO] N_CONTRAST_ANCHORS   = $N_CONTRAST_ANCHORS"
echo "[INFO] MATCH_PROB           = $MATCH_PROB"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Optional flags
# =============================================================================

MAX_SAMPLES_FLAG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_FLAG="--max_samples $MAX_SAMPLES"
fi

TRAIN_VISION_FLAG=""
if [ "$TRAIN_VISION" = "true" ]; then
    TRAIN_VISION_FLAG="--train_vision"
fi

NO_CONTRAST_FLAG=""
if [ "$NO_CONTRAST" = "true" ]; then
    NO_CONTRAST_FLAG="--no_contrast"
fi

NO_MATCH_FLAG=""
if [ "$NO_MATCH" = "true" ]; then
    NO_MATCH_FLAG="--no_match"
fi

# =============================================================================
# Run via torchrun (DDP)
# =============================================================================

TORCHRUN=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun

$TORCHRUN \
    --nproc_per_node "$NPROC" \
    --master_port    "${MASTER_PORT:-29507}" \
    train_geo.py \
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
    --coord_scale            "$COORD_SCALE"            \
    --lm_weight              "$LM_WEIGHT"              \
    --contrast_weight        "$CONTRAST_WEIGHT"        \
    --match_weight           "$MATCH_WEIGHT"           \
    --contrast_margin        "$CONTRAST_MARGIN"        \
    --contrast_eps           "$CONTRAST_EPS"           \
    --n_contrast_anchors     "$N_CONTRAST_ANCHORS"     \
    --match_prob             "$MATCH_PROB"             \
    --wandb_project          "$WANDB_PROJECT"          \
    --wandb_entity           "$WANDB_ENTITY"           \
    --wandb_run_name         "$WANDB_RUN_NAME"         \
    $MAX_SAMPLES_FLAG \
    $TRAIN_VISION_FLAG \
    $NO_CONTRAST_FLAG \
    $NO_MATCH_FLAG

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
