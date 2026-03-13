#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial Planning — Eval-Only Pipeline (Steps 2–4, no training)
#
#  Step 2: Generate instructions     (instruction_generation.py)
#  Step 3: Generate videos           (video_generation.py)
#  Step 4: Evaluate                  (eval.py)
#
#  Quick start:
#    bash scripts/eval.sh                               # smoke, qwen3.5 pretrained
#    bash scripts/eval.sh smoke                         # 10 samples, fast sanity check
#    bash scripts/eval.sh full                          # all samples
#    bash scripts/eval.sh smoke qwen3.5                 # pretrained Qwen3.5
#    bash scripts/eval.sh smoke qwen3-vl                # pretrained Qwen3-VL
#
#  Arguments:
#    $1  SCALE              : smoke | full            (default: smoke)
#    $2  MODEL_TYPE         : qwen3.5 | qwen3-vl     (default: qwen3.5)
#    $3  PLANNER_THINKING   : true | false            (default: false)
#    $4  BASELINE_THINKING  : true | false            (default: false)
#
#  Examples:
#    bash scripts/eval.sh smoke qwen3.5 false false   # our method + baseline, no thinking
#    bash scripts/eval.sh smoke qwen3.5 false true    # baseline WITH thinking
#    bash scripts/eval.sh smoke qwen3.5 false false   # baseline WITHOUT thinking
#
#  Set TRAINED_MODEL below to evaluate a fine-tuned checkpoint.
#  If TRAINED_MODEL is empty, falls back to the pretrained MLLM selected by MODEL_TYPE.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
#  *** CONFIGURE: set the model directory to evaluate ***
# ══════════════════════════════════════════════════════════════════════════════
# TRAINED_MODEL="/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/train_records/grpo_sat_pilot/sat_20260309_000605/iter_0/model_merged"
TRAINED_MODEL=""

# ══════════════════════════════════════════════════════════════════════════════
#  Shared model checkpoints
# ══════════════════════════════════════════════════════════════════════════════
VACE_CKPT="checkpoints/Wan2.1-VACE-14B"
EVAL_MODEL="checkpoints/Qwen3.5-4B"

# ══════════════════════════════════════════════════════════════════════════════
#  Detect available GPUs
# ══════════════════════════════════════════════════════════════════════════════
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)

# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════
SCALE="${1:-smoke}"               # smoke | full
MODEL_TYPE="${2:-qwen3.5}"        # qwen3.5 | qwen3-vl
PLANNER_THINKING="${3:-false}"    # true | false — planner (our method) thinking mode
BASELINE_THINKING="${4:-false}"   # true | false — baseline reasoning thinking mode

if [[ "$SCALE" == "smoke" ]]; then
    EVAL_LIMIT=16
elif [[ "$SCALE" == "full" ]]; then
    EVAL_LIMIT=-1
else
    echo "ERROR: SCALE must be 'smoke' or 'full', got '$SCALE'"
    exit 1
fi

if [[ "$MODEL_TYPE" != "qwen3.5" && "$MODEL_TYPE" != "qwen3-vl" ]]; then
    echo "ERROR: MODEL_TYPE must be 'qwen3.5' or 'qwen3-vl', got '$MODEL_TYPE'"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Resolve planning model
#  If TRAINED_MODEL is empty, fall back to the pretrained MLLM.
# ══════════════════════════════════════════════════════════════════════════════
if [[ -z "$TRAINED_MODEL" ]]; then
    if [[ "$MODEL_TYPE" == "qwen3.5" ]]; then
        TRAINED_MODEL="checkpoints/Qwen3.5-4B"
    else
        TRAINED_MODEL="checkpoints/Qwen3-VL-4B"
    fi
    echo "[INFO] TRAINED_MODEL not set — using pretrained ${MODEL_TYPE}: ${TRAINED_MODEL}"
fi

if [[ ! -d "$TRAINED_MODEL" ]]; then
    echo "ERROR: TRAINED_MODEL directory not found: ${TRAINED_MODEL}"
    exit 1
fi

PLANNING_MODEL_NAME="$(basename "$TRAINED_MODEL")"

echo "═══════════════════════════════════════════════════════════════"
echo " Spatial Planning — Eval-Only Pipeline (Steps 2–4)"
echo "═══════════════════════════════════════════════════════════════"
echo " Scale          : ${SCALE^^}"
echo " Model type     : ${MODEL_TYPE}"
echo " Planning model : ${TRAINED_MODEL}"
echo " VACE ckpt      : ${VACE_CKPT}"
echo " Eval model     : ${EVAL_MODEL}"
echo " Eval limit     : ${EVAL_LIMIT}"
echo " GPUs           : ${N_GPUS} detected"
echo " Planner think  : ${PLANNER_THINKING}"
echo " Baseline think : ${BASELINE_THINKING}"
echo "═══════════════════════════════════════════════════════════════"

EVAL_LIMIT_ARG=""
[[ "$EVAL_LIMIT" != "-1" ]] && EVAL_LIMIT_ARG="--max_samples ${EVAL_LIMIT}"

# Build --enable_thinking flag for planner (step 2)
PLANNER_THINKING_ARG=""
[[ "$PLANNER_THINKING" == "true" ]] && PLANNER_THINKING_ARG="--enable_thinking"

# Build --baseline_thinking flag for evaluation (step 4)
BASELINE_THINKING_ARG=""
[[ "$BASELINE_THINKING" == "true" ]] && BASELINE_THINKING_ARG="--baseline_thinking"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Generate Planning Instructions on Test Set
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── STEP 2: Generate Instructions (test set) ─────────────────────"
conda run --no-capture-output -n spi python -u instruction_generation.py \
    --dataset    mmsibench \
    --data_path  datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --model_path "$TRAINED_MODEL" \
    --model_type "$MODEL_TYPE" \
    ${EVAL_LIMIT_ARG} \
    ${PLANNER_THINKING_ARG} \
    2>&1 | tee /tmp/eval_step2_${SCALE}.log

echo "── Step 2 done ─────────────────────────────────────────────────"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Generate Videos (Wan2.1-VACE-14B)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── STEP 3: Generate Videos (Wan2.1-VACE-14B) ───────────────────"
conda run --no-capture-output -n spi python -u video_generation.py \
    --planning_model "$PLANNING_MODEL_NAME" \
    --vace_ckpt      "$VACE_CKPT" \
    ${EVAL_LIMIT_ARG} \
    2>&1 | tee /tmp/eval_step3_${SCALE}.log

echo "── Step 3 done ─────────────────────────────────────────────────"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Evaluate (Baseline vs Augmented)
# ══════════════════════════════════════════════════════════════════════════════
GEN_DIR="generated_videos/mmsibench/${PLANNING_MODEL_NAME}/$(basename "$VACE_CKPT")"
EVAL_OUTPUT="results/eval_${SCALE}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "── STEP 4: Evaluate ─────────────────────────────────────────────"
LIMIT_ARG=""
[[ "$EVAL_LIMIT" != "-1" ]] && LIMIT_ARG="--limit ${EVAL_LIMIT}"

conda run --no-capture-output -n spi python -u eval.py \
    --model_type qwen3.5 \
    --model_path "$EVAL_MODEL" \
    --data_dir   datasets/evaluation/MMSIBench \
    --gen_dir    "$GEN_DIR" \
    ${LIMIT_ARG} \
    ${BASELINE_THINKING_ARG} \
    --output_dir "$EVAL_OUTPUT" \
    2>&1 | tee /tmp/eval_step4_${SCALE}.log

echo "── Step 4 done ─────────────────────────────────────────────────"
echo ""
echo " Results saved to: ${EVAL_OUTPUT}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Pipeline complete."
echo "═══════════════════════════════════════════════════════════════"
