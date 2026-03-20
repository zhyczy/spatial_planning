#!/usr/bin/env bash
# Run FLUX2-klein-4B image generation conditioned on predicted instructions (multi-GPU).
# Mirrors run_generate_instructions.sh: one subprocess per GPU, samples sharded evenly.
#
# Usage:
#   bash scripts/run_image_generation.sh [PLANNING_MODEL] [OPTIONS...]
#
# PLANNING_MODEL: name of the folder under predicted_instructions/ (or "all")
#
# Examples:
#   # Single planning model (all available GPUs)
#   bash scripts/run_image_generation.sh Qwen3-VL-4B-Instruct
#
#   # All planning models
#   bash scripts/run_image_generation.sh all
#
#   # Specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_image_generation.sh all
#
#   # Limit to 2 GPUs
#   bash scripts/run_image_generation.sh all --num_gpus 2
#
#   # Re-generate everything (ignore existing outputs)
#   bash scripts/run_image_generation.sh all --no_skip_existing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

PLANNING_MODEL="${1:-all}"
shift 1 2>/dev/null || true   # remaining args forwarded to python

FLUX_CKPT="$ROOT_DIR/checkpoints/flux2-klein-4B"
PREDICTED_INSTRUCTIONS_ROOT="$ROOT_DIR/predicted_instructions"
OUTPUT_ROOT="$ROOT_DIR/generated_images"

PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python)"
fi

# Count GPUs for display
if command -v nvidia-smi &>/dev/null; then
    N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "?")
else
    N_GPUS="?"
fi

echo "============================================================"
echo " FLUX2-klein-4B Image Generation  (multi-GPU)"
echo "  Planning model : $PLANNING_MODEL"
echo "  Checkpoint     : $FLUX_CKPT"
echo "  Instructions   : $PREDICTED_INSTRUCTIONS_ROOT"
echo "  Output         : $OUTPUT_ROOT"
echo "  GPUs available : ${CUDA_VISIBLE_DEVICES:-all ($N_GPUS)}"
echo "  Python         : $PYTHON"
echo "============================================================"

if [[ "$PLANNING_MODEL" == "all" ]]; then
    "$PYTHON" "$ROOT_DIR/image_generation.py" \
        --all_planning_models \
        --flux_ckpt "$FLUX_CKPT" \
        --predicted_instructions_root "$PREDICTED_INSTRUCTIONS_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        "$@"
else
    "$PYTHON" "$ROOT_DIR/image_generation.py" \
        --planning_model "$PLANNING_MODEL" \
        --flux_ckpt "$FLUX_CKPT" \
        --predicted_instructions_root "$PREDICTED_INSTRUCTIONS_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        "$@"
fi

echo "Done. Results in: $OUTPUT_ROOT"
