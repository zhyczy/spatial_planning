#!/usr/bin/env bash
# Baseline QwenVL reasoning evaluation on MMSIBench.
# Automatically shards the dataset across all visible GPUs.
#
# Usage:
#   bash scripts/run_evaluation.sh [MODEL_TYPE] [MODEL_PATH] [OPTIONS...]
#
# Examples:
#   bash scripts/run_evaluation.sh qwen2.5-vl checkpoints/Qwen2.5-VL-3B-Instruct
#   bash scripts/run_evaluation.sh qwen3-vl   checkpoints/Qwen3-VL-4B-Instruct
#   bash scripts/run_evaluation.sh qwen3-vl   checkpoints/Qwen3-VL-8B-Instruct --limit 50
#
#   # Restrict to specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_TYPE="${1:-qwen2.5-vl}"
MODEL_PATH="${2:-checkpoints/Qwen2.5-VL-3B-Instruct}"
shift 2 2>/dev/null || true   # pass remaining args to python

# Resolve relative model path against workspace root
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
fi

DATA_DIR="$ROOT_DIR/datasets/evaluation/MMSIBench"
OUTPUT_DIR="$ROOT_DIR/results/mmsibench"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "?")
CUDA_INFO="${CUDA_VISIBLE_DEVICES:-all $NUM_GPUS GPUs}"

echo "========================================"
echo " Model type  : $MODEL_TYPE"
echo " Model path  : $MODEL_PATH"
echo " Data dir    : $DATA_DIR"
echo " Output dir  : $OUTPUT_DIR"
echo " GPUs        : $CUDA_INFO"
echo "========================================"

cd "$ROOT_DIR"
python evaluation.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --data_dir   "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
