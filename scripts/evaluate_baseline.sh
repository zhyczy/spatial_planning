#!/usr/bin/env bash
# =============================================================================
# evaluate_baseline.sh
#
# Run spatial_mllm_eval.py (Spatial-MLLM 135K / 820K) over benchmark datasets.
# Mirrors evaluate.sh structure; inference is parallelised across all visible
# GPUs automatically by spatial_mllm_eval.py.
#
# Usage:
#   bash scripts/evaluate_baseline.sh [options]
#
# Options:
#   --model     135K | 820K | /absolute/path/to/checkpoint
#               (default: 135K)
#   --datasets  comma-separated list of dataset names, e.g. "mindcube,mmsibench"
#               Available: mindcube  mmsibench  sparbench_multi_view
#                          sparbench_single_view  sparbench_mv  sat_real
#               Default: all five datasets
#   --gpus      comma-separated GPU IDs, e.g. "0,1,2,3"
#               Default: auto-detect all available GPUs
#   --limit     truncate each dataset to N samples (debug / smoke test)
#   --output    base output directory (default: eval_results_baseline)
#   --run_name  optional sub-folder name (default: auto timestamp per dataset)
#
# Examples:
#   # Run all datasets with 135K model:
#   bash scripts/evaluate_baseline.sh --model 135K --gpus 0,1,2,3
#
#   # Run 820K on two datasets:
#   bash scripts/evaluate_baseline.sh \
#       --model 820K \
#       --datasets "mindcube,sat_real"
#
#   # Smoke test — 6 samples per dataset, single GPU:
#   bash scripts/evaluate_baseline.sh \
#       --model 135K \
#       --datasets "mindcube" \
#       --limit 6 \
#       --gpus 0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Defaults
# =============================================================================

MODEL="135K"
GPUS=""
LIMIT=""
OUTPUT_BASE="$SPATIAL_DIR/eval_results_baseline"
RUN_NAME=""

# All supported datasets (in evaluation order)
ALL_DATASETS="mindcube mmsibench sparbench_multi_view sparbench_single_view sat_real"

# Dataset → data_dir mapping (relative to SPATIAL_DIR)
declare -A DATASET_DIR
DATASET_DIR["mindcube"]="datasets/evaluation/MindCube"
DATASET_DIR["mmsibench"]="datasets/evaluation/MMSIBench"
DATASET_DIR["sparbench_multi_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sparbench_single_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sparbench_mv"]="datasets/evaluation/SPARBench"
DATASET_DIR["sat_real"]="datasets/evaluation/SAT"

# Dataset → JSONL/JSON file mapping (relative to SPATIAL_DIR)
declare -A DATASET_JSONL
DATASET_JSONL["mindcube"]="datasets/evaluation/MindCube/MindCube_tinybench.jsonl"
DATASET_JSONL["mmsibench"]="datasets/evaluation/MMSIBench/data/test_data_final.json"
DATASET_JSONL["sparbench_multi_view"]="datasets/evaluation/SPARBench/sparbench_multi_view.json"
DATASET_JSONL["sparbench_single_view"]="datasets/evaluation/SPARBench/sparbench_single_view.json"
DATASET_JSONL["sat_real"]="datasets/evaluation/SAT/test.json"

# Dataset → subcommand mapping for spatial_mllm_eval.py
declare -A DATASET_SUBCMD
DATASET_SUBCMD["mindcube"]="mindcube"
DATASET_SUBCMD["mmsibench"]="mmsibench"
DATASET_SUBCMD["sparbench_multi_view"]="sparbench"
DATASET_SUBCMD["sparbench_single_view"]="sparbench"
DATASET_SUBCMD["sat_real"]="sat"

# =============================================================================
# Parse arguments
# =============================================================================

DATASETS_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";        shift 2 ;;
        --datasets) DATASETS_ARG="$2"; shift 2 ;;
        --gpus)     GPUS="$2";         shift 2 ;;
        --limit)    LIMIT="$2";        shift 2 ;;
        --output)   OUTPUT_BASE="$2";  shift 2 ;;
        --run_name) RUN_NAME="$2";     shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Resolve model path
# =============================================================================

case "$MODEL" in
    135K)
        MODEL_PATH="$SPATIAL_DIR/checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
        ;;
    820K)
        MODEL_PATH="$SPATIAL_DIR/checkpoints/Spatial-MLLM-v1.1-Instruct-820K"
        ;;
    *)
        # Treat as a direct path
        MODEL_PATH="$MODEL"
        ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Model directory not found: $MODEL_PATH" >&2
    exit 1
fi

# =============================================================================
# Resolve GPU list
# =============================================================================

if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

N_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$N_GPU" -eq 0 ]]; then
    echo "[ERROR] No CUDA GPUs available." >&2
    exit 1
fi

# =============================================================================
# Resolve dataset list
# =============================================================================

if [[ -n "$DATASETS_ARG" ]]; then
    DATASETS="${DATASETS_ARG//,/ }"
else
    DATASETS="$ALL_DATASETS"
fi

# =============================================================================
# Build common flags
# =============================================================================

COMMON_FLAGS=(
    --model_path  "$MODEL_PATH"
    --output_dir  "$OUTPUT_BASE"
)

if [[ -n "$LIMIT" ]]; then
    COMMON_FLAGS+=(--limit "$LIMIT")
fi

# =============================================================================
# Run
# =============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME=$(basename "$MODEL_PATH")

echo "=========================================================="
echo "[INFO] evaluate_baseline.sh"
echo "[INFO]   Model               : $MODEL_PATH"
echo "[INFO]   Datasets            : $DATASETS"
echo "[INFO]   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[INFO]   Num GPUs            : $N_GPU"
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
        DS_RUN_NAME="${MODEL_NAME}_${DS}_${TIMESTAMP}"
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "[INFO] Dataset : $DS"
    echo "[INFO] Data dir: $DATA_DIR"
    echo "[INFO] Run name: $DS_RUN_NAME"
    echo "[INFO] Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------------------------"

    DATA_JSON="$SPATIAL_DIR/${DATASET_JSONL[$DS]}"
    if [[ ! -f "$DATA_JSON" ]]; then
        echo "[WARN] Data file not found for '$DS': $DATA_JSON — skipping." >&2
        FAILED_DATASETS+=("$DS(missing_data)")
        continue
    fi

    SUBCMD="${DATASET_SUBCMD[$DS]}"

    # Build dataset-specific flags
    DS_FLAGS=()
    case "$SUBCMD" in
        mindcube)
            DS_FLAGS+=(--data_jsonl "$DATA_JSON" --data_dir "$DATA_DIR") ;;
        mmsibench)
            DS_FLAGS+=(--data_json "$DATA_JSON") ;;
        sparbench)
            DS_FLAGS+=(--data_json "$DATA_JSON") ;;
        sat)
            DS_FLAGS+=(--data_json "$DATA_JSON" --data_dir "$DATA_DIR") ;;
    esac

    set +e
    python baseline/spatial_mllm_eval.py "$SUBCMD" \
        "${COMMON_FLAGS[@]}"  \
        "${DS_FLAGS[@]}"      \
        --model_name "$DS_RUN_NAME"

    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] spatial_mllm_eval.py exited with code $EXIT_CODE for dataset '$DS'" >&2
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
