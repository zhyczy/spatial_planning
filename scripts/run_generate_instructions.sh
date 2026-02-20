#!/usr/bin/env bash
# Run image-generation instruction pipeline using Qwen3-VL (or Qwen2.5-VL).
# Runs on all available GPUs (or those in CUDA_VISIBLE_DEVICES).
# Each GPU gets an equal shard of the data; results are merged into results.jsonl.
#
# Usage:
#   bash scripts/run_generate_instructions.sh [DATASET] [MODEL_PATH] [OPTIONS...]
#
# Examples:
#   bash scripts/run_generate_instructions.sh mindcube /path/to/Qwen3-VL-7B-Instruct
#   bash scripts/run_generate_instructions.sh sat      /path/to/Qwen3-VL-7B-Instruct --max_samples 50
#   bash scripts/run_generate_instructions.sh vsibench /path/to/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_generate_instructions.sh mindcube /path/to/Qwen3-VL-7B-Instruct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET="${1:-mindcube}"
MODEL_PATH="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
shift 2 2>/dev/null || true   # remaining args forwarded to python

OUTPUT_DIR="$ROOT_DIR/results"
mkdir -p "$OUTPUT_DIR"

case "$DATASET" in
  mindcube)
    DATA_PATH="datasets/evaluation/MindCube/MindCube_tinybench.jsonl"
    IMAGE_ROOT="datasets/evaluation/MindCube"
    ;;
  sat)
    DATA_PATH="datasets/evaluation/SAT/test.json"
    IMAGE_ROOT="datasets/evaluation/SAT"
    ;;
  vsibench)
    DATA_PATH="datasets/evaluation/vsibench/test.jsonl"
    IMAGE_ROOT="datasets/evaluation/vsibench"
    ;;
  mmsibench)
    DATA_PATH="datasets/evaluation/MMSIBench/data/test_data_final.json"
    IMAGE_ROOT="datasets/evaluation/MMSIBench"   # images are resolved relative to this root
    ;;
  *)
    echo "Unknown dataset: $DATASET. Choose: mindcube | sat | vsibench | mmsibench"
    exit 1
    ;;
esac

# Determine GPU count for display
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "?")
CUDA_INFO="${CUDA_VISIBLE_DEVICES:-all $NUM_GPUS GPUs}"

echo "========================================"
echo " Dataset  : $DATASET"
echo " Model    : $MODEL_PATH"
echo " Data     : $DATA_PATH"
echo " GPUs     : $CUDA_INFO"
echo " Output   : results/$DATASET/<model_name>/<timestamp>/"
echo "========================================"

cd "$ROOT_DIR"
python generate_image_instructions.py \
  --dataset      "$DATASET" \
  --data_path    "$DATA_PATH" \
  --image_root   "$IMAGE_ROOT" \
  --model_path   "$MODEL_PATH" \
  "$@"
