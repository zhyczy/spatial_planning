#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial Planning — GRPO Pipeline (Steps 1–4)
#
#  Step 1: Iterative GRPO training   (grpo.py)
#  Step 2: Generate instructions     (generate_image_instructions.py)
#  Step 3: Generate images           (image_generation.py)
#  Step 4: Evaluate                  (evaluation.py)
#
#  Quick start:
#    bash scripts/run_grpo_pipeline.sh                          # smoke test, LoRA, all steps
#    bash scripts/run_grpo_pipeline.sh smoke  lora all          # smoke: 50 samples, 1 iter
#    bash scripts/run_grpo_pipeline.sh pilot  lora all          # pilot: 2000 samples, 5 iters
#    bash scripts/run_grpo_pipeline.sh full   lora all          # full:  all ~172K, 5 iters
#    bash scripts/run_grpo_pipeline.sh smoke  lora train        # Step 1 only
#    bash scripts/run_grpo_pipeline.sh smoke  lora eval         # Steps 2-4 only
#
#  Arguments:
#    $1  SCALE      : smoke | pilot | full     (default: smoke)
#    $2  TRAIN_MODE : lora  | full_model       (default: lora)
#    $3  RUN_STEPS  : all   | train | eval     (default: all)
#
#  For eval-only (skip Step 1), set TRAINED_MODEL below to your model path.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
export PYTHONUNBUFFERED=1   # prevent Python stdout block-buffering when redirected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════
SCALE="${1:-smoke}"         # smoke | pilot | full
TRAIN_MODE="${2:-lora}"     # lora  | full_model
RUN_STEPS="${3:-all}"       # all   | train | eval

if [[ "$SCALE" != "smoke" && "$SCALE" != "pilot" && "$SCALE" != "full" ]]; then
    echo "ERROR: SCALE must be 'smoke', 'pilot', or 'full', got '$SCALE'"
    exit 1
fi
if [[ "$TRAIN_MODE" != "lora" && "$TRAIN_MODE" != "full_model" ]]; then
    echo "ERROR: TRAIN_MODE must be 'lora' or 'full_model', got '$TRAIN_MODE'"
    exit 1
fi
if [[ "$RUN_STEPS" != "all" && "$RUN_STEPS" != "train" && "$RUN_STEPS" != "eval" ]]; then
    echo "ERROR: RUN_STEPS must be 'all', 'train', or 'eval', got '$RUN_STEPS'"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Shared model checkpoints
# ══════════════════════════════════════════════════════════════════════════════
PLANNER_MODEL="checkpoints/Qwen3-VL-4B-Instruct"
CRITIC_MODEL="checkpoints/Qwen3-VL-8B-Instruct"
FLUX_CKPT="checkpoints/flux2-klein-4B"
EVAL_MODEL="checkpoints/Qwen3-VL-4B-Instruct"

# ══════════════════════════════════════════════════════════════════════════════
#  Scale-specific config
# ══════════════════════════════════════════════════════════════════════════════
DATASET="sat"
DATA_PATH="datasets/evaluation/SAT/train_action_consequence.json"
IMAGE_ROOT="datasets/evaluation/SAT"

if [[ "$SCALE" == "smoke" ]]; then
    # ── Smoke: 50 samples, 1 iteration — fast multi-GPU sanity check ─────
    MAX_SAMPLES=50
    NUM_ITERATIONS=1
    NUM_ROLLOUTS=4
    EPOCHS_PER_ITER=1
    LEARNING_RATE=1e-4
    GRAD_ACCUM=2
    BATCH_PER_DEVICE=1
    EVAL_LIMIT=10
    OUTPUT_DIR="train_records/grpo_sat_smoke"

elif [[ "$SCALE" == "pilot" ]]; then
    # ── Pilot: 2000 samples, 5 iterations ────────────────────────────────
    MAX_SAMPLES=2000
    NUM_ITERATIONS=5
    NUM_ROLLOUTS=8
    EPOCHS_PER_ITER=1
    LEARNING_RATE=1e-4
    GRAD_ACCUM=4
    BATCH_PER_DEVICE=1
    EVAL_LIMIT=30
    OUTPUT_DIR="train_records/grpo_sat_pilot"

else  # full
    # ── Full scale: all ~172K samples, 5 iterations ───────────────────────
    MAX_SAMPLES=-1
    NUM_ITERATIONS=5
    NUM_ROLLOUTS=8
    EPOCHS_PER_ITER=1
    LEARNING_RATE=5e-5
    GRAD_ACCUM=8
    BATCH_PER_DEVICE=1
    EVAL_LIMIT=-1
    OUTPUT_DIR="train_records/grpo_sat_full"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Train mode config
# ══════════════════════════════════════════════════════════════════════════════
LORA_RANK=64
LORA_ALPHA=16
BETA=0.04
MIN_REWARD_STD=0.05
NUM_GPUS=-1    # -1 = use all available GPUs (auto-detected by grpo.py)

if [[ "$TRAIN_MODE" == "lora" ]]; then
    LORA_ENABLE="True"
else  # full_model
    LORA_ENABLE="False"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Detect available GPUs
# ══════════════════════════════════════════════════════════════════════════════
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)

# ══════════════════════════════════════════════════════════════════════════════
#  Eval-only: set TRAINED_MODEL manually if skipping Step 1
# ══════════════════════════════════════════════════════════════════════════════
TRAINED_MODEL=""

# ══════════════════════════════════════════════════════════════════════════════
#  Print summary
# ══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo " Spatial Planning — GRPO Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo " Scale       : ${SCALE^^}"
echo " Train mode  : ${TRAIN_MODE^^}  (lora_enable=${LORA_ENABLE})"
echo " Steps       : ${RUN_STEPS}"
echo " Dataset     : ${DATASET} (max_samples=${MAX_SAMPLES})"
echo " Iterations  : ${NUM_ITERATIONS}  (rollouts/q=${NUM_ROLLOUTS})"
echo " LR          : ${LEARNING_RATE},  beta=${BETA}"
echo " GPUs        : ${N_GPUS} detected  (num_gpus=${NUM_GPUS})"
echo " Output      : ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Iterative GRPO Training
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_STEPS" == "all" || "$RUN_STEPS" == "train" ]]; then
    echo ""
    echo "── STEP 1: Iterative GRPO Training ─────────────────────────────"
    conda run --no-capture-output -n SPR python -u grpo.py \
        --dataset                     "$DATASET" \
        --data_path                   "$DATA_PATH" \
        --image_root                  "$IMAGE_ROOT" \
        --planner_model_path          "$PLANNER_MODEL" \
        --critic_model_path           "$CRITIC_MODEL" \
        --flux_ckpt                   "$FLUX_CKPT" \
        --max_samples                 "$MAX_SAMPLES" \
        --num_iterations              "$NUM_ITERATIONS" \
        --num_rollouts                "$NUM_ROLLOUTS" \
        --scoring_method              gt_similarity \
        --min_reward_std              "$MIN_REWARD_STD" \
        --beta                        "$BETA" \
        --lora_enable                 "$LORA_ENABLE" \
        --lora_rank                   "$LORA_RANK" \
        --lora_alpha                  "$LORA_ALPHA" \
        --learning_rate               "$LEARNING_RATE" \
        --num_train_epochs            "$EPOCHS_PER_ITER" \
        --per_device_batch_size       "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --num_gpus                    "$NUM_GPUS" \
        --output_dir                  "$OUTPUT_DIR" \
        2>&1 | tee /tmp/grpo_step1_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 1 done ─────────────────────────────────────────────────"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Resolve trained model path for Steps 2–4
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_STEPS" == "all" || "$RUN_STEPS" == "eval" ]]; then

    if [[ -z "$TRAINED_MODEL" ]]; then
        LAST_RUN=$(ls -dt "${OUTPUT_DIR}/${DATASET}_"* 2>/dev/null | head -1)
        if [[ -z "$LAST_RUN" ]]; then
            echo "ERROR: No training run found under ${OUTPUT_DIR}."
            echo "       Set TRAINED_MODEL manually at the top of this script, or run Step 1 first."
            exit 1
        fi
        LAST_ITER=$(( NUM_ITERATIONS - 1 ))
        TRAINED_MODEL="${LAST_RUN}/iter_${LAST_ITER}/model_merged"
        if [[ ! -d "$TRAINED_MODEL" ]]; then
            echo "ERROR: Trained model not found at ${TRAINED_MODEL}"
            echo "       Check that Step 1 completed successfully."
            exit 1
        fi
        echo "Auto-detected trained model: ${TRAINED_MODEL}"
    fi

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — Generate Planning Instructions on Test Set
    # ══════════════════════════════════════════════════════════════════════
    echo ""
    echo "── STEP 2: Generate Instructions (test set) ─────────────────────"
    EVAL_LIMIT_ARG=""
    [[ "$EVAL_LIMIT" != "-1" ]] && EVAL_LIMIT_ARG="--max_samples ${EVAL_LIMIT}"

    conda run --no-capture-output -n SPR python -u generate_image_instructions.py \
        --dataset    mmsibench \
        --data_path  datasets/evaluation/MMSIBench/data/test_data_final.json \
        --image_root datasets/evaluation/MMSIBench \
        --model_path "$TRAINED_MODEL" \
        --model_type qwen-vl \
        ${EVAL_LIMIT_ARG} \
        2>&1 | tee /tmp/grpo_step2_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 2 done ─────────────────────────────────────────────────"

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3 — Generate Images (Flux2Klein)
    # ══════════════════════════════════════════════════════════════════════
    PLANNING_MODEL_NAME="$(basename "$TRAINED_MODEL")"
    echo ""
    echo "── STEP 3: Generate Images (Flux2Klein) ─────────────────────────"
    conda run --no-capture-output -n SPR python -u image_generation.py \
        --planning_model               "$PLANNING_MODEL_NAME" \
        --predicted_instructions_root  results/mmsibench \
        --flux_ckpt                    "$FLUX_CKPT" \
        ${EVAL_LIMIT_ARG} \
        2>&1 | tee /tmp/grpo_step3_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 3 done ─────────────────────────────────────────────────"

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4 — Evaluate (Baseline vs Augmented)
    # ══════════════════════════════════════════════════════════════════════
    GEN_DIR="generated_images/mmsibench/${PLANNING_MODEL_NAME}/$(basename "$FLUX_CKPT")"
    EVAL_OUTPUT="results/eval_grpo_${SCALE}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "── STEP 4: Evaluate ─────────────────────────────────────────────"
    LIMIT_ARG=""
    [[ "$EVAL_LIMIT" != "-1" ]] && LIMIT_ARG="--limit ${EVAL_LIMIT}"

    conda run --no-capture-output -n SPR python -u evaluation.py \
        --model_type qwen3-vl \
        --model_path "$EVAL_MODEL" \
        --data_dir   datasets/evaluation/MMSIBench \
        --gen_dir    "$GEN_DIR" \
        ${LIMIT_ARG} \
        --output_dir "$EVAL_OUTPUT" \
        2>&1 | tee /tmp/grpo_step4_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 4 done ─────────────────────────────────────────────────"
    echo ""
    echo " Results saved to: ${EVAL_OUTPUT}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Pipeline complete."
echo "═══════════════════════════════════════════════════════════════"
