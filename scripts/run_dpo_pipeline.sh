#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial Planning — Full DPO Pipeline (Steps 1–4)
#
#  Step 1: Iterative DPO training   (iterative_dpo.py)
#  Step 2: Generate instructions    (generate_image_instructions.py)
#  Step 3: Generate images          (image_generation.py)
#  Step 4: Evaluate                 (evaluation.py)
#
#  Quick start:
#    bash scripts/run_dpo_pipeline.sh                        # pilot, LoRA, all steps
#    bash scripts/run_dpo_pipeline.sh pilot lora all
#    bash scripts/run_dpo_pipeline.sh pilot full all         # full-model DPO (needs more VRAM)
#    bash scripts/run_dpo_pipeline.sh full  lora all         # full dataset, LoRA
#    bash scripts/run_dpo_pipeline.sh pilot lora train       # Step 1 only
#    bash scripts/run_dpo_pipeline.sh pilot lora eval        # Steps 2-4 only (needs TRAINED_MODEL set)
#
#  Arguments:
#    $1  SCALE      : pilot | full            (default: pilot)
#    $2  TRAIN_MODE : lora  | full_model      (default: lora)
#    $3  RUN_STEPS  : all | train | eval      (default: all)
#
#  For eval-only (skip Step 1), set TRAINED_MODEL below to your model path.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
export PYTHONUNBUFFERED=1   # prevent Python stdout block-buffering when output is redirected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════
SCALE="${1:-pilot}"        # pilot | full
TRAIN_MODE="${2:-lora}"    # lora  | full_model
RUN_STEPS="${3:-all}"      # all   | train | eval

if [[ "$SCALE" != "pilot" && "$SCALE" != "full" ]]; then
    echo "ERROR: SCALE must be 'pilot' or 'full', got '$SCALE'"
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
EVAL_MODEL="checkpoints/Qwen3-VL-4B-Instruct"   # VQA judge for Step 4

# ══════════════════════════════════════════════════════════════════════════════
#  Scale-specific config
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$SCALE" == "pilot" ]]; then
    # ── Pilot: SAT 2000 samples, 5 iterations ───────────────────────────────
    DATASET="sat"
    DATA_PATH="datasets/evaluation/SAT/train_action_consequence.json"
    IMAGE_ROOT="datasets/evaluation/SAT"
    MAX_SAMPLES=2000
    NUM_ITERATIONS=5
    NUM_ROLLOUTS=8
    EPOCHS_PER_ITER=2
    LEARNING_RATE=1e-4
    GRAD_ACCUM=4
    BATCH_PER_DEVICE=2
    EVAL_LIMIT=30
    OUTPUT_DIR="train_records/sat_pilot"   # iterative_dpo.py appends sat_<timestamp> inside

else  # full
    # ── Full scale: SAT all ~172K samples, 5 iterations ─────────────────────
    DATASET="sat"
    DATA_PATH="datasets/evaluation/SAT/train_action_consequence.json"
    IMAGE_ROOT="datasets/evaluation/SAT"
    MAX_SAMPLES=-1   # -1 = all (~172K)
    NUM_ITERATIONS=5
    NUM_ROLLOUTS=8
    EPOCHS_PER_ITER=2
    LEARNING_RATE=5e-5
    GRAD_ACCUM=8
    BATCH_PER_DEVICE=2
    EVAL_LIMIT=-1   # -1 = all
    OUTPUT_DIR="train_records/sat_full"    # iterative_dpo.py appends sat_<timestamp> inside
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Train mode config
# ══════════════════════════════════════════════════════════════════════════════
LORA_RANK=64
LORA_ALPHA=16
BETA=0.1
MIN_SCORE_GAP=0.3
NUM_GPUS=-1    # -1 = use all available GPUs

if [[ "$TRAIN_MODE" == "lora" ]]; then
    LORA_ENABLE="True"
    # LoRA uses torchrun DDP across all NUM_GPUS — multi-GPU supported
else  # full_model
    LORA_ENABLE="False"
    # Full-model DPO uses all CUDA_VISIBLE_DEVICES — make sure you have enough VRAM
    # Recommendation: 4×A100 80GB minimum; use zero3_offload DeepSpeed config
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Eval-only: set TRAINED_MODEL manually if skipping Step 1
# ══════════════════════════════════════════════════════════════════════════════
#  Leave empty ("") to auto-detect from the training output.
TRAINED_MODEL=""

# ══════════════════════════════════════════════════════════════════════════════
#  Print summary
# ══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo " Spatial Planning — DPO Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo " Scale      : ${SCALE^^}"
echo " Train mode : ${TRAIN_MODE^^}  (lora_enable=${LORA_ENABLE})"
echo " Steps      : ${RUN_STEPS}"
echo " Dataset    : ${DATASET} (max_samples=${MAX_SAMPLES})"
echo " Iterations : ${NUM_ITERATIONS}  (rollouts/q=${NUM_ROLLOUTS})"
echo " LR         : ${LEARNING_RATE},  beta=${BETA}"
echo " Output     : ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Iterative DPO Training
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_STEPS" == "all" || "$RUN_STEPS" == "train" ]]; then
    echo ""
    echo "── STEP 1: Iterative DPO Training ──────────────────────────────"
    conda run --no-capture-output -n SPR python -u iterative_dpo.py \
        --dataset           "$DATASET" \
        --data_path         "$DATA_PATH" \
        --image_root        "$IMAGE_ROOT" \
        --planner_model_path "$PLANNER_MODEL" \
        --critic_model_path  "$CRITIC_MODEL" \
        --flux_ckpt          "$FLUX_CKPT" \
        --max_samples        "$MAX_SAMPLES" \
        --num_iterations     "$NUM_ITERATIONS" \
        --num_rollouts       "$NUM_ROLLOUTS" \
        --scoring_method     gt_similarity \
        --min_score_gap      "$MIN_SCORE_GAP" \
        --lora_enable        "$LORA_ENABLE" \
        --lora_rank          "$LORA_RANK" \
        --lora_alpha         "$LORA_ALPHA" \
        --learning_rate      "$LEARNING_RATE" \
        --beta               "$BETA" \
        --num_train_epochs   "$EPOCHS_PER_ITER" \
        --per_device_batch_size      "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --num_gpus           "$NUM_GPUS" \
        --output_dir         "$OUTPUT_DIR" \
        2>&1 | tee /tmp/dpo_step1_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 1 done ─────────────────────────────────────────────────"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Resolve trained model path for Steps 2–4
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$RUN_STEPS" == "all" || "$RUN_STEPS" == "eval" ]]; then

    if [[ -z "$TRAINED_MODEL" ]]; then
        # Auto-detect: find the most recent run dir and pick the last iter's model_merged
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
        2>&1 | tee /tmp/dpo_step2_${SCALE}_${TRAIN_MODE}.log

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
        2>&1 | tee /tmp/dpo_step3_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 3 done ─────────────────────────────────────────────────"

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4 — Evaluate (Baseline vs Augmented)
    # ══════════════════════════════════════════════════════════════════════
    GEN_DIR="generated_images/mmsibench/${PLANNING_MODEL_NAME}/$(basename "$FLUX_CKPT")"
    EVAL_OUTPUT="results/eval_${SCALE}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"

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
        2>&1 | tee /tmp/dpo_step4_${SCALE}_${TRAIN_MODE}.log

    echo "── Step 4 done ─────────────────────────────────────────────────"
    echo ""
    echo " Results saved to: ${EVAL_OUTPUT}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Pipeline complete."
echo "═══════════════════════════════════════════════════════════════"
