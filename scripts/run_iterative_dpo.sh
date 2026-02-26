#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  Iterative DPO — Snowball Training for the Spatial Planning Model
#
#  Architecture:
#    • Planning Model (trainable): Qwen3-VL-4B-Instruct — LoRA
#    • Executor (frozen)         : Flux2Klein-4B
#    • Critic  (frozen)          : Qwen3-VL-8B-Instruct
#
#  Usage:
#    # ── Pilot experiment (1700-sample subset, 5 iterations) ──
#    bash scripts/run_iterative_dpo.sh pilot
#
#    # ── Full scale (175K samples, 5 iterations) ──
#    bash scripts/run_iterative_dpo.sh full
#
#    # ── Resume from iteration 3 ──
#    bash scripts/run_iterative_dpo.sh resume results/iterative_dpo/mmsibench_xxx 3
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Mode ──────────────────────────────────────────────────────────────────
MODE="${1:-pilot}"

# ── Shared configuration ─────────────────────────────────────────────────
PLANNER_MODEL="checkpoints/Qwen3-VL-4B-Instruct"
CRITIC_MODEL="checkpoints/Qwen3-VL-8B-Instruct"
FLUX_CKPT="checkpoints/flux2-klein-4B"

NUM_ITERATIONS=5
NUM_ROLLOUTS=8

# LoRA hyperparams
LORA_RANK=64
LORA_ALPHA=16
LEARNING_RATE=1e-5
BETA=0.1          # DPO KL penalty
EPOCHS_PER_ITER=1
MIN_SCORE_GAP=0.3

echo "═══════════════════════════════════════════════════════════"
echo " Iterative DPO — Planning Model (Snowball Training)"
echo "═══════════════════════════════════════════════════════════"

if [[ "$MODE" == "pilot" ]]; then
    # ── Pilot: 1700 samples, 5 iterations × ~340 samples ────────────────
    DATA_PATH="datasets/evaluation/MMSIBench/data/test_data_final.json"
    IMAGE_ROOT="datasets/evaluation/MMSIBench"
    MAX_SAMPLES=1700
    BATCH_PER_DEVICE=2
    GRAD_ACCUM=4

    echo " Mode           : PILOT (${MAX_SAMPLES} samples)"
    echo " Data           : ${DATA_PATH}"
    echo " Shard size     : ~$(( MAX_SAMPLES / NUM_ITERATIONS )) per iteration"
    echo " Rollouts/q     : ${NUM_ROLLOUTS}"
    echo " Iterations     : ${NUM_ITERATIONS}"
    echo "═══════════════════════════════════════════════════════════"

    python iterative_dpo.py \
        --dataset mmsibench \
        --data_path "$DATA_PATH" \
        --image_root "$IMAGE_ROOT" \
        --planner_model_path "$PLANNER_MODEL" \
        --critic_model_path "$CRITIC_MODEL" \
        --flux_ckpt "$FLUX_CKPT" \
        --max_samples "$MAX_SAMPLES" \
        --num_iterations "$NUM_ITERATIONS" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --scoring_method confidence \
        --min_score_gap "$MIN_SCORE_GAP" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --learning_rate "$LEARNING_RATE" \
        --beta "$BETA" \
        --num_train_epochs "$EPOCHS_PER_ITER" \
        --per_device_batch_size "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --output_dir "results/iterative_dpo"

elif [[ "$MODE" == "full" ]]; then
    # ── Full scale: 175K samples, 5 iterations × ~35K samples ───────────
    DATA_PATH="datasets/training/spatial_planning_175k.json"
    IMAGE_ROOT="datasets/training"
    MAX_SAMPLES=-1
    BATCH_PER_DEVICE=2
    GRAD_ACCUM=8

    echo " Mode           : FULL SCALE (175K samples)"
    echo " Data           : ${DATA_PATH}"
    echo " Shard size     : ~35000 per iteration"
    echo " Rollouts/q     : ${NUM_ROLLOUTS}"
    echo " Iterations     : ${NUM_ITERATIONS}"
    echo "═══════════════════════════════════════════════════════════"

    python iterative_dpo.py \
        --dataset mmsibench \
        --data_path "$DATA_PATH" \
        --image_root "$IMAGE_ROOT" \
        --planner_model_path "$PLANNER_MODEL" \
        --critic_model_path "$CRITIC_MODEL" \
        --flux_ckpt "$FLUX_CKPT" \
        --max_samples "$MAX_SAMPLES" \
        --num_iterations "$NUM_ITERATIONS" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --scoring_method both \
        --min_score_gap "$MIN_SCORE_GAP" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --learning_rate "$LEARNING_RATE" \
        --beta "$BETA" \
        --num_train_epochs "$EPOCHS_PER_ITER" \
        --per_device_batch_size "$BATCH_PER_DEVICE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --output_dir "results/iterative_dpo"

elif [[ "$MODE" == "resume" ]]; then
    # ── Resume from a previous run ───────────────────────────────────────
    RESUME_DIR="${2:?Usage: $0 resume <run_dir> <iter>}"
    RESUME_ITER="${3:?Usage: $0 resume <run_dir> <iter>}"

    echo " Mode           : RESUME"
    echo " Run dir        : ${RESUME_DIR}"
    echo " Resume iter    : ${RESUME_ITER}"
    echo "═══════════════════════════════════════════════════════════"

    # Read config from the existing run
    if [[ -f "${RESUME_DIR}/config.json" ]]; then
        echo " Loading config from ${RESUME_DIR}/config.json"
    fi

    python iterative_dpo.py \
        --dataset mmsibench \
        --data_path "$(python -c "import json; print(json.load(open('${RESUME_DIR}/config.json'))['data_path'])")" \
        --image_root "$(python -c "import json; print(json.load(open('${RESUME_DIR}/config.json'))['image_root'])")" \
        --planner_model_path "$PLANNER_MODEL" \
        --critic_model_path "$CRITIC_MODEL" \
        --flux_ckpt "$FLUX_CKPT" \
        --max_samples "$(python -c "import json; print(json.load(open('${RESUME_DIR}/config.json'))['max_samples'])")" \
        --num_iterations "$NUM_ITERATIONS" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --scoring_method confidence \
        --min_score_gap "$MIN_SCORE_GAP" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --learning_rate "$LEARNING_RATE" \
        --beta "$BETA" \
        --num_train_epochs "$EPOCHS_PER_ITER" \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --resume_from "$RESUME_DIR" \
        --resume_iter "$RESUME_ITER"

else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {pilot|full|resume}"
    exit 1
fi
