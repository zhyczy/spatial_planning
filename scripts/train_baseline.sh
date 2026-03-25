#!/usr/bin/env bash
# =============================================================================
# train_baseline.sh
#
# 3DRS training launcher.
# Multi-GPU is managed by baseline/3drs.py (internally uses torchrun).
#
# Usage:
#   bash scripts/train_baseline.sh [num_gpus] [model_preset] [model_name_or_path]
#
#   num_gpus           — number of GPUs to use (default: all available)
#   model_preset       — qwen2.5vl-3b | qwen3.5-4b (default: qwen3.5-4b)
#   model_name_or_path — optional override model path/name
#
# Examples:
#   bash scripts/train_baseline.sh
#   bash scripts/train_baseline.sh 8
#   bash scripts/train_baseline.sh 8 qwen2.5vl-3b
#   bash scripts/train_baseline.sh 8 qwen3.5-4b /path/to/local/model
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Arguments
# =============================================================================

# num_gpus: positional arg 1 (default: auto-detect)
if [ -n "${1:-}" ]; then
    NPROC="$1"
    CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))
    export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
else
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NPROC="$N_AVAIL"
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NPROC - 1)))
fi

# model preset: positional arg 2
MODEL_PRESET="${2:-qwen3.5-4b}"

if [[ "$MODEL_PRESET" != "qwen2.5vl-3b" && "$MODEL_PRESET" != "qwen3.5-4b" ]]; then
    echo "[ERROR] model_preset must be one of: qwen2.5vl-3b, qwen3.5-4b"
    exit 1
fi

# optional model path/name override: positional arg 3
MODEL_OVERRIDE="${3:-}"

# =============================================================================
# Paths
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
if [ -z "$MODEL_OVERRIDE" ] && [ -d "$MODEL_PATH" ]; then
    MODEL_OVERRIDE="$MODEL_PATH"
fi

RUN_NAME="3drs_${MODEL_PRESET}"
OUTPUT_DIR="$SPATIAL_DIR/train_records_baseline/$RUN_NAME"

# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS=3
LR=2e-4

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo "[INFO] NPROC_PER_NODE      = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PRESET         = $MODEL_PRESET"
echo "[INFO] MODEL_OVERRIDE       = ${MODEL_OVERRIDE:-<preset default>}"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Build command
# =============================================================================

PYTHON=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python

CMD=(
    "$PYTHON" baseline/3drs.py
    --model "$MODEL_PRESET"
    --num-gpus "$NPROC"
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    --run-name "$RUN_NAME"
    --output-dir "$OUTPUT_DIR"
    --num-train-epochs "$EPOCHS"
    --learning-rate "$LR"
    --frames-upbound 32
    --frame-sampling-strategy uniform
)

if [ -n "$MODEL_OVERRIDE" ]; then
    CMD+=(--model-name-or-path "$MODEL_OVERRIDE")
fi

# =============================================================================
# Run via baseline/3drs.py
# =============================================================================

"${CMD[@]}"

echo "[INFO] Done — $(date '+%Y-%m-%d %H:%M:%S')"
