#!/usr/bin/env bash
set -euo pipefail

# Dedicated launcher for Qwen2.5-VL baseline training.
# Usage:
#   bash scripts/train_baseline_qwen25.sh [num_gpus] [optional_model_path] [extra_3drs_args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

NPROC="${1:-}"
if [ -z "$NPROC" ]; then
    NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"

MODEL_OVERRIDE="${2:-}"
EXTRA_ARGS=("${@:3}")

if [ -z "$MODEL_OVERRIDE" ] && [ -d "$SPATIAL_DIR/checkpoints/Qwen2.5-VL-3B-Instruct" ]; then
    MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen2.5-VL-3B-Instruct"
fi

PYTHON=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
RUN_NAME="3drs_qwen2.5vl-3b"
OUTPUT_DIR="$SPATIAL_DIR/train_records_baseline/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

CMD=(
    "$PYTHON" baseline/3drs.py
    --model qwen2.5vl-3b
    --num-gpus "$NPROC"
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    --run-name "$RUN_NAME"
    --output-dir "$OUTPUT_DIR"
    --num-train-epochs 3
    --learning-rate 2e-4
    --frames-upbound 32
    --frame-sampling-strategy uniform
)

if [ -n "$MODEL_OVERRIDE" ]; then
    CMD+=(--model-name-or-path "$MODEL_OVERRIDE")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
