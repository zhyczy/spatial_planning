#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# train_baseline.sh
#
# Unified 3DRS baseline training launcher for Qwen2.5 and Qwen3.5 presets.
# Multi-GPU launch is delegated to baseline/3drs.py.
#
# Usage:
#   bash scripts/train_baseline.sh [num_gpus] [model_preset] [model_name_or_path] [allow_incompatible_qwen35] [extra_3drs_args...]
#
# Positional args:
#   1) num_gpus                  default: auto-detect
#   2) model_preset              default: qwen2.5vl-3b
#                                supported: qwen2.5vl-3b | qwen2.5 | qwen25 |
#                                           qwen3.5-4b | qwen3.5 | qwen35
#   3) model_name_or_path        optional, overrides preset path/id
#   4) allow_incompatible_qwen35 optional, true/false (default: false)
#                                only applies to qwen3.5 preset
#   5+) extra_3drs_args          optional passthrough args for baseline/3drs.py
#
# Examples:
#   bash scripts/train_baseline.sh
#   bash scripts/train_baseline.sh 8 qwen2.5
#   bash scripts/train_baseline.sh 8 qwen3.5 /path/to/ckpt true
#   bash scripts/train_baseline.sh 8 qwen3.5 "" false --global-batch-size 32
#
# Teacher feature env vars (optional):
#   USE_TEACHER_FEATURE=true|false       default: true
#   REQUIRE_TEACHER_FEATURE=true|false   default: true
#   TEACHER_FEATURE_DIR=/abs/path        default: datasets/train/SPAR_7M/spar/vggt_teacher_features
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

usage() {
    sed -n '1,50p' "$0"
}

# =============================================================================
# Arguments
# =============================================================================

# num_gpus: positional arg 1 (default: auto-detect)
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

# model preset: positional arg 2
MODEL_PRESET_RAW="${2:-qwen2.5vl-3b}"

case "$MODEL_PRESET_RAW" in
    qwen2.5vl-3b|qwen2.5|qwen25)
        MODEL_PRESET="qwen2.5vl-3b"
        ;;
    qwen3.5-4b|qwen3.5|qwen35)
        MODEL_PRESET="qwen3.5-4b"
        ;;
    *)
        echo "[ERROR] Unsupported model_preset: $MODEL_PRESET_RAW"
        echo "[ERROR] Choose one of: qwen2.5vl-3b, qwen2.5, qwen25, qwen3.5-4b, qwen3.5, qwen35"
        usage
        exit 1
        ;;
esac

# optional model path/name override: positional arg 3
MODEL_OVERRIDE="${3:-}"

# optional compatibility flag for qwen3.5: positional arg 4
ALLOW_INCOMPATIBLE_RAW="${4:-false}"
ALLOW_INCOMPATIBLE="false"
case "${ALLOW_INCOMPATIBLE_RAW,,}" in
    true|1|yes|y) ALLOW_INCOMPATIBLE="true" ;;
    false|0|no|n|"") ALLOW_INCOMPATIBLE="false" ;;
    *)
        echo "[ERROR] allow_incompatible_qwen35 must be true/false, got: $ALLOW_INCOMPATIBLE_RAW"
        exit 1
        ;;
esac

EXTRA_ARGS=()
if [ "$#" -ge 5 ]; then
    EXTRA_ARGS=("${@:5}")
fi

# =============================================================================
# Paths
# =============================================================================

if [ -z "$MODEL_OVERRIDE" ]; then
    if [ "$MODEL_PRESET" = "qwen2.5vl-3b" ]; then
        if [ -d "$SPATIAL_DIR/checkpoints/Qwen2.5-VL-3B-Instruct" ]; then
            MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen2.5-VL-3B-Instruct"
        fi
    else
        if [ -d "$SPATIAL_DIR/checkpoints/Qwen3.5-4B-Instruct" ]; then
            MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen3.5-4B-Instruct"
        elif [ -d "$SPATIAL_DIR/checkpoints/Qwen3.5-4B" ]; then
            MODEL_OVERRIDE="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"
        fi
    fi
fi

RUN_NAME="3drs_${MODEL_PRESET}"
OUTPUT_DIR="$SPATIAL_DIR/train_records_baseline/$RUN_NAME"

# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS=3
LR=2e-4

# =============================================================================
# Teacher Feature Options
# =============================================================================

USE_TEACHER_FEATURE_RAW="${USE_TEACHER_FEATURE:-true}"
case "${USE_TEACHER_FEATURE_RAW,,}" in
    true|1|yes|y) USE_TEACHER_FEATURE="true" ;;
    false|0|no|n) USE_TEACHER_FEATURE="false" ;;
    *)
        echo "[ERROR] USE_TEACHER_FEATURE must be true/false, got: $USE_TEACHER_FEATURE_RAW"
        exit 1
        ;;
esac

REQUIRE_TEACHER_FEATURE_RAW="${REQUIRE_TEACHER_FEATURE:-true}"
case "${REQUIRE_TEACHER_FEATURE_RAW,,}" in
    true|1|yes|y) REQUIRE_TEACHER_FEATURE="true" ;;
    false|0|no|n) REQUIRE_TEACHER_FEATURE="false" ;;
    *)
        echo "[ERROR] REQUIRE_TEACHER_FEATURE must be true/false, got: $REQUIRE_TEACHER_FEATURE_RAW"
        exit 1
        ;;
esac

TEACHER_FEATURE_DIR="${TEACHER_FEATURE_DIR:-$SPATIAL_DIR/datasets/train/SPAR_7M/spar/vggt_teacher_features}"

if [ "$USE_TEACHER_FEATURE" = "true" ] && [ "$REQUIRE_TEACHER_FEATURE" = "true" ] && [ ! -d "$TEACHER_FEATURE_DIR" ]; then
    echo "[ERROR] Teacher feature dir not found: $TEACHER_FEATURE_DIR"
    echo "[ERROR] Generate features first with:"
    echo "        python extract_vggt_teacher_features.py --data-json datasets/train/SPAR_7M/spar/train_10k.json --output-dir $TEACHER_FEATURE_DIR --checkpoint checkpoints/VGGT-1B/model.pt --device cuda:0"
    exit 1
fi

# =============================================================================
# Setup
# =============================================================================

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] python not found in current PATH. Please activate your environment first (e.g., conda activate spc)."
    exit 1
fi

echo "[INFO] NPROC_PER_NODE      = $NPROC"
echo "[INFO] CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PRESET         = $MODEL_PRESET"
echo "[INFO] MODEL_OVERRIDE       = ${MODEL_OVERRIDE:-<preset default>}"
echo "[INFO] ALLOW_INCOMPATIBLE   = $ALLOW_INCOMPATIBLE"
echo "[INFO] USE_TEACHER_FEATURE  = $USE_TEACHER_FEATURE"
echo "[INFO] REQUIRE_TEACHER_FEAT = $REQUIRE_TEACHER_FEATURE"
echo "[INFO] TEACHER_FEATURE_DIR  = $TEACHER_FEATURE_DIR"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Python               : $PYTHON_BIN"
echo "[INFO] Starting             : $(date '+%Y-%m-%d %H:%M:%S')"

# =============================================================================
# Build command
# =============================================================================

CMD=(
    "$PYTHON_BIN" baseline/3drs.py
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

if [ "$MODEL_PRESET" = "qwen3.5-4b" ] && [ "$ALLOW_INCOMPATIBLE" = "true" ]; then
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

# =============================================================================
# Run via baseline/3drs.py
# =============================================================================

"${CMD[@]}"

echo "[INFO] Done - $(date '+%Y-%m-%d %H:%M:%S')"
