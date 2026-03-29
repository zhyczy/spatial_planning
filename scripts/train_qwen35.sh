#!/usr/bin/env bash
set -euo pipefail

# Dedicated 3DRS launcher for qwen3.5 workflow.
#
# Usage:
#   bash scripts/train_qwen35.sh [num_gpus] [model_name_or_path] [allow_incompatible_qwen35] [extra_3drs_args...]
#
# Positional args:
#   1) num_gpus                  default: auto-detect
#   2) model_name_or_path        default: checkpoints/Qwen3.5-4B if exists
#   3) allow_incompatible_qwen35 default: true
#   4+) extra_3drs_args          optional passthrough args for baseline/3drs.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

usage() {
    sed -n '1,35p' "$0"
}

if [ -n "${1:-}" ]; then
    NPROC="$1"
else
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)
    NPROC="$N_AVAIL"
fi

if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [ "$NPROC" -lt 1 ]; then
    echo "[ERROR] num_gpus must be a positive integer, got: ${NPROC}"
    usage
    exit 1
fi

CUDA_IDS=$(seq -s ',' 0 $((NPROC - 1)))
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"

MODEL_OVERRIDE="${2:-}"

ALLOW_INCOMPATIBLE_RAW="${3:-true}"
ALLOW_INCOMPATIBLE="true"
case "${ALLOW_INCOMPATIBLE_RAW,,}" in
    true|1|yes|y) ALLOW_INCOMPATIBLE="true" ;;
    false|0|no|n|"") ALLOW_INCOMPATIBLE="false" ;;
    *)
        echo "[ERROR] allow_incompatible_qwen35 must be true/false, got: $ALLOW_INCOMPATIBLE_RAW"
        exit 1
        ;;
esac

EXTRA_ARGS=()
if [ "$#" -ge 4 ]; then
    EXTRA_ARGS=("${@:4}")
fi

if [ -z "$MODEL_OVERRIDE" ]; then
    if [ -d "$SPATIAL_DIR/checkpoints/Qwen3.5-4B" ]; then
        MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
    elif [ -d "$SPATIAL_DIR/checkpoints/Qwen3.5-9B" ]; then
        MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen3.5-9B"
    fi
fi

RUN_NAME="3drs_qwen3.5-4b"
OUTPUT_DIR="$SPATIAL_DIR/train_records_baseline/$RUN_NAME"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"

DEFAULT_USE_TEACHER_FEATURE="true"
DEFAULT_REQUIRE_TEACHER_FEATURE="true"
DEFAULT_TEACHER_FEATURE_DIR="/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/train/SPAR_7M/spar/vggt_teacher_features"

USE_TEACHER_FEATURE_RAW="${USE_TEACHER_FEATURE:-$DEFAULT_USE_TEACHER_FEATURE}"
case "${USE_TEACHER_FEATURE_RAW,,}" in
    true|1|yes|y) USE_TEACHER_FEATURE="true" ;;
    false|0|no|n) USE_TEACHER_FEATURE="false" ;;
    *)
        echo "[ERROR] USE_TEACHER_FEATURE must be true/false, got: $USE_TEACHER_FEATURE_RAW"
        exit 1
        ;;
esac

REQUIRE_TEACHER_FEATURE_RAW="${REQUIRE_TEACHER_FEATURE:-$DEFAULT_REQUIRE_TEACHER_FEATURE}"
case "${REQUIRE_TEACHER_FEATURE_RAW,,}" in
    true|1|yes|y) REQUIRE_TEACHER_FEATURE="true" ;;
    false|0|no|n) REQUIRE_TEACHER_FEATURE="false" ;;
    *)
        echo "[ERROR] REQUIRE_TEACHER_FEATURE must be true/false, got: $REQUIRE_TEACHER_FEATURE_RAW"
        exit 1
        ;;
esac

TEACHER_FEATURE_DIR="${TEACHER_FEATURE_DIR:-$DEFAULT_TEACHER_FEATURE_DIR}"

if [ "$USE_TEACHER_FEATURE" = "true" ] && [ "$REQUIRE_TEACHER_FEATURE" = "true" ] && [ ! -d "$TEACHER_FEATURE_DIR" ]; then
    echo "[ERROR] Teacher feature dir not found: $TEACHER_FEATURE_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] python not found in current PATH. Please activate your environment first."
    exit 1
fi

echo "[INFO] NPROC_PER_NODE      = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PRESET         = qwen3.5-4b"
echo "[INFO] MODEL_OVERRIDE       = ${MODEL_OVERRIDE:-<preset default>}"
echo "[INFO] ALLOW_INCOMPATIBLE   = $ALLOW_INCOMPATIBLE"
echo "[INFO] USE_TEACHER_FEATURE  = $USE_TEACHER_FEATURE"
echo "[INFO] REQUIRE_TEACHER_FEAT = $REQUIRE_TEACHER_FEATURE"
echo "[INFO] TEACHER_FEATURE_DIR  = $TEACHER_FEATURE_DIR"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Python               : $PYTHON_BIN"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

CMD=(
    "$PYTHON_BIN" baseline/3drs.py
    --model qwen3.5-4b
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

if [ "$ALLOW_INCOMPATIBLE" = "true" ]; then
    CMD+=(--allow-incompatible-qwen35)
fi

if [ "$USE_TEACHER_FEATURE" = "true" ]; then
    CMD+=(--teacher-feature-dir "$TEACHER_FEATURE_DIR")
    if [ "$REQUIRE_TEACHER_FEATURE" = "true" ]; then
        CMD+=(--require-teacher-feature)
    else
        CMD+=(--no-require-teacher-feature)
    fi
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "[INFO] Done - $(date '+%Y-%m-%d %H:%M:%S')"
