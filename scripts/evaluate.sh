#!/usr/bin/env bash
# =============================================================================
# evaluate.sh
#
# Run evaluation.py over one or more benchmark datasets.
# Datasets can be run sequentially; inference is parallelised across all
# visible GPUs automatically by evaluation.py.
#
# Usage:
#   bash scripts/evaluate.sh [options]
#
# Options:
#   --method    baseline | correspondence | both   (default: both)
#   --ckpt      path to correspondence LoRA checkpoint dir
#               (required when method = correspondence | both)
#   --datasets  comma-separated list of dataset names to run, e.g.
#                 "mindcube,mmsibench"
#               Available: mindcube  mmsibench  sparbench_multi_view
#                          sparbench_single_view  sat_realia你
#               Default: all five datasets
#   --gpus      comma-separated GPU IDs, e.g. "0,1,2,3"
#               Default: auto-detect all available GPUs
#   --limit     truncate each dataset to N samples (debug / smoke test)
#   --output    base output directory (default: train_records/eval_results)
#   --run_name  optional sub-folder name (default: auto timestamp per dataset)
#   --no_coord  pass --no_coord to evaluation.py (zero XYZ, ablation mode)
#   --max_new_tokens  generation budget (default: 512)
#
# Examples:
#   # Run all datasets, both methods, 4 GPUs:
#   bash scripts/evaluate.sh \
#       --ckpt train_records/correspondence/final \
#       --gpus 0,1,2,3
#
#   # Baseline only on two datasets:
#   bash scripts/evaluate.sh \
#       --method baseline \
#       --datasets "mindcube,sat_real"
#
#   # Smoke test — 6 samples per dataset, correspondence only:
#   bash scripts/evaluate.sh \
#       --method correspondence \
#       --ckpt train_records/correspondence/final \
#       --datasets "mmsibench" \
#       --limit 6 \
#       --gpus 0
#
#   # With thinking mode (recommend --max_new_tokens 8192):
#   bash scripts/evaluate.sh \
#       --thinking \
#       --max_new_tokens 8192 \
#       --method baseline \
#       --datasets "mindcube,mmsibench"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Defaults
# =============================================================================

METHOD="both"
CKPT=""
GPUS=""
LIMIT=""
OUTPUT_BASE="$SPATIAL_DIR/eval_results"
RUN_NAME=""
NO_COORD=""
THINKING=""
MAX_NEW_TOKENS=512

# All supported datasets (in evaluation order)
ALL_DATASETS="mindcube mmsibench sparbench_multi_view sparbench_single_view sat_real"

# Dataset → data_dir mapping (relative to SPATIAL_DIR)
declare -A DATASET_DIR
DATASET_DIR["mindcube"]="datasets/evaluation/MindCube"
DATASET_DIR["mmsibench"]="datasets/evaluation/MMSIBench"
DATASET_DIR["sparbench_multi_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sparbench_single_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sat_real"]="datasets/evaluation/SAT"

# =============================================================================
# Parse arguments
# =============================================================================

DATASETS_ARG=""   # user-supplied comma-separated list; empty → all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)        METHOD="$2";          shift 2 ;;
        --ckpt)          CKPT="$2";            shift 2 ;;
        --datasets)      DATASETS_ARG="$2";    shift 2 ;;
        --gpus)          GPUS="$2";            shift 2 ;;
        --limit)         LIMIT="$2";           shift 2 ;;
        --output)        OUTPUT_BASE="$2";     shift 2 ;;
        --run_name)      RUN_NAME="$2";        shift 2 ;;
        --no_coord)      NO_COORD="--no_coord";   shift  ;;
        --thinking)      THINKING="--thinking";   shift  ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2";    shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate arguments
# =============================================================================

if [[ "$METHOD" != "baseline" && "$METHOD" != "correspondence" && "$METHOD" != "both" ]]; then
    echo "[ERROR] --method must be one of: baseline  correspondence  both" >&2
    exit 1
fi

if [[ "$METHOD" == "correspondence" || "$METHOD" == "both" ]]; then
    if [[ -z "$CKPT" ]]; then
        echo "[ERROR] --ckpt is required when --method is '$METHOD'" >&2
        exit 1
    fi
    if [[ ! -d "$CKPT" ]]; then
        echo "[ERROR] Checkpoint directory not found: $CKPT" >&2
        exit 1
    fi
fi

# =============================================================================
# Resolve GPU list
# =============================================================================

if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [[ "$N_GPU" -eq 0 ]]; then
    echo "[ERROR] No CUDA GPUs available." >&2
    exit 1
fi

# =============================================================================
# Resolve dataset list
# =============================================================================

if [[ -n "$DATASETS_ARG" ]]; then
    # Replace commas with spaces
    DATASETS="${DATASETS_ARG//,/ }"
else
    DATASETS="$ALL_DATASETS"
fi

# =============================================================================
# Python path
# =============================================================================

PYTHON=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python

# =============================================================================
# Build common flags
# =============================================================================

MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"

COMMON_FLAGS=(
    --model_path      "$MODEL_PATH"
    --method          "$METHOD"
    --max_new_tokens  "$MAX_NEW_TOKENS"
    --output_dir      "$OUTPUT_BASE"
)

if [[ -n "$CKPT" ]]; then
    COMMON_FLAGS+=(--correspondence_ckpt "$CKPT")
fi

if [[ -n "$NO_COORD" ]]; then
    COMMON_FLAGS+=($NO_COORD)
fi

if [[ -n "$THINKING" ]]; then
    COMMON_FLAGS+=($THINKING)
fi

if [[ -n "$LIMIT" ]]; then
    COMMON_FLAGS+=(--limit "$LIMIT")
fi

# =============================================================================
# Run
# =============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo "=========================================================="
echo "[INFO] evaluate.sh"
echo "[INFO]   Method              : $METHOD"
echo "[INFO]   Datasets            : $DATASETS"
echo "[INFO]   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[INFO]   Num GPUs            : $N_GPU"
if [[ -n "$CKPT" ]]; then
    echo "[INFO]   Correspondence ckpt : $CKPT"
fi
if [[ -n "$LIMIT" ]]; then
    echo "[INFO]   Limit per dataset  : $LIMIT"
fi
echo "[INFO]   Output base         : $OUTPUT_BASE"
echo "[INFO]   Started             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

FAILED_DATASETS=()

for DS in $DATASETS; do
    if [[ -z "${DATASET_DIR[$DS]+x}" ]]; then
        echo "[WARN] Unknown dataset '$DS', skipping." >&2
        continue
    fi

    DATA_DIR="$SPATIAL_DIR/${DATASET_DIR[$DS]}"

    if [[ ! -d "$DATA_DIR" ]]; then
        echo "[WARN] Data directory not found for '$DS': $DATA_DIR — skipping." >&2
        FAILED_DATASETS+=("$DS(missing)")
        continue
    fi

    # Per-dataset run name
    if [[ -n "$RUN_NAME" ]]; then
        DS_RUN_NAME="${RUN_NAME}_${DS}"
    else
        DS_RUN_NAME="${METHOD}_${DS}_${TIMESTAMP}"
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "[INFO] Dataset : $DS"
    echo "[INFO] Data dir: $DATA_DIR"
    echo "[INFO] Run name: $DS_RUN_NAME"
    echo "[INFO] Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------------------------"

    set +e
    $PYTHON evaluation.py \
        "${COMMON_FLAGS[@]}"  \
        --dataset   "$DS"     \
        --data_dir  "$DATA_DIR" \
        --run_name  "$DS_RUN_NAME"

    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] evaluation.py exited with code $EXIT_CODE for dataset '$DS'" >&2
        FAILED_DATASETS+=("$DS(exit=$EXIT_CODE)")
    else
        echo "[INFO] Finished '$DS' — $(date '+%Y-%m-%d %H:%M:%S')"
    fi
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================================="
echo "[INFO] All datasets processed — $(date '+%Y-%m-%d %H:%M:%S')"
if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "[WARN] Failed datasets: ${FAILED_DATASETS[*]}"
    exit 1
else
    echo "[INFO] All datasets completed successfully."
fi
echo "=========================================================="
