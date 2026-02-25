#!/usr/bin/env bash
# Dual-method QwenVL reasoning evaluation on MMSIBench.
# Runs both baseline (original images) and augmented (+ generated images) inference
# for every sample. Automatically shards the dataset across all visible GPUs.
#
# Usage:
#   bash scripts/run_evaluation.sh [MODEL_TYPE] [MODEL_PATH] [GEN_DIR] [OPTIONS...]
#
# Positional args (all optional, in order):
#   MODEL_TYPE   qwen2.5-vl | qwen3-vl          (default: qwen3-vl)
#   MODEL_PATH   path to checkpoint directory    (default: checkpoints/Qwen3-VL-4B-Instruct)
#   GEN_DIR      path to generated images root   (default: generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B)
#
# Examples:
#   bash scripts/run_evaluation.sh
#   bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct
#   bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
#   bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B --limit 12
#
#   # Restrict to specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_TYPE="${1:-qwen3-vl}"
MODEL_PATH="${2:-checkpoints/Qwen3-VL-4B-Instruct}"
GEN_DIR="${3:-generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B}"
shift 3 2>/dev/null || true   # pass remaining args to python

# Resolve relative paths against workspace root
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
fi
if [[ "$GEN_DIR" != /* ]]; then
    GEN_DIR="$ROOT_DIR/$GEN_DIR"
fi

DATA_DIR="$ROOT_DIR/datasets/evaluation/MMSIBench"
OUTPUT_DIR="$ROOT_DIR/results/mmsibench"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "?")
CUDA_INFO="${CUDA_VISIBLE_DEVICES:-all $NUM_GPUS GPUs}"

echo "========================================"
echo " Model type  : $MODEL_TYPE"
echo " Model path  : $MODEL_PATH"
echo " Data dir    : $DATA_DIR"
echo " Gen dir     : $GEN_DIR"
echo " Output dir  : $OUTPUT_DIR"
echo " GPUs        : $CUDA_INFO"
echo "========================================"

cd "$ROOT_DIR"
python evaluation.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --data_dir   "$DATA_DIR" \
    --gen_dir    "$GEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
