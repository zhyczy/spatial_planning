#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Step 0: Generate DPO preference data for the Planning Model.
#
#  This runs the full pipeline:
#    1. Rollout  – sample K instruction sets per question
#    2. Execute  – generate images via Flux2Klein
#    3. Label    – score rollouts with MLLM judge
#    4. Build    – construct (chosen, rejected) pairs → dpo_train.json
#
#  Usage:
#    bash scripts/run_generate_dpo_data.sh
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_generate_dpo_data.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Configuration (edit these) ─────────────────────────────────────────────
DATASET="mmsibench"
DATA_PATH="datasets/evaluation/MMSIBench/data/test_data_final.json"
IMAGE_ROOT="datasets/evaluation/MMSIBench"

PLANNER_MODEL="checkpoints/Qwen3-VL-4B-Instruct"
JUDGE_MODEL="checkpoints/Qwen3-VL-4B-Instruct"   # can be larger model
FLUX_CKPT="checkpoints/flux2-klein-4B"

NUM_ROLLOUTS=6           # instructions per question
SCORING_METHOD="confidence"   # confidence | explicit | both
MIN_SCORE_GAP=0.5
MAX_SAMPLES=-1           # -1 = all, set small for testing
NUM_INFERENCE_STEPS=28

OUTPUT_DIR="results/dpo_data"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo " DPO Data Generation Pipeline"
echo "========================================"
echo " Dataset         : $DATASET"
echo " Planner model   : $PLANNER_MODEL"
echo " Judge model     : $JUDGE_MODEL"
echo " Flux checkpoint : $FLUX_CKPT"
echo " Rollouts/sample : $NUM_ROLLOUTS"
echo " Scoring method  : $SCORING_METHOD"
echo " Output          : $OUTPUT_DIR"
echo "========================================"

python generate_dpo_data.py \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --image_root "$IMAGE_ROOT" \
    --planner_model_path "$PLANNER_MODEL" \
    --judge_model_path "$JUDGE_MODEL" \
    --flux_ckpt "$FLUX_CKPT" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --scoring_method "$SCORING_METHOD" \
    --min_score_gap "$MIN_SCORE_GAP" \
    --max_samples "$MAX_SAMPLES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --output_dir "$OUTPUT_DIR"
