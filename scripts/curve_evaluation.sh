#!/usr/bin/env bash
# =============================================================================
# curve_evaluation.sh
#
# Evaluate checkpoints at multiple training steps and plot accuracy-step curves.
# Calls curve_evaluation.py which internally invokes evaluation.py for each
# (step, dataset) pair and saves plots to --output.
#
# Usage:
#   bash scripts/curve_evaluation.sh [options]
#
# Options:
#   --ckpt_dir  path to train_records directory containing step_N sub-dirs
#               (required)
#   --start     first step to evaluate      (default: 200)
#   --end       last step to evaluate       (default: 1000)
#   --step      step stride                 (default: 50)
#   --method    evaluation method           (default: coordinate)
#               choices: baseline vanilla position_embedding coordinate decouple atten
#   --datasets  comma-separated dataset list (single name OK, e.g. "mindcube")
#               (default: mindcube,sat_real,spinbench,robospatial,
#                         viewspatial,omnispatial_pt,embspatial)
#   --gpus      comma-separated GPU IDs     (default: all visible)
#   --limit     truncate each dataset to N samples (debug / smoke test)
#   --output    output directory for results and plots
#               (default: eval_results/curves/<ckpt_dir_basename>)
#   --max_new_tokens  generation budget     (default: 4096)
#
# Examples:
#   # Full sweep, 500→1000 step 50, all default datasets:
#   bash scripts/curve_evaluation.sh \
#       --ckpt_dir train_records/coordinate_no_cam_mindcube \
#       --gpus 0,1,2,3
#
#   # Custom range and step size:
#   bash scripts/curve_evaluation.sh \
#       --ckpt_dir train_records/coordinate_no_cam_mindcube \
#       --start 200 --end 1200 --step 100 \
#       --datasets "mindcube,sat_real" \
#       --gpus 0,1
#
#   # Smoke test (6 samples per dataset):
#   bash scripts/curve_evaluation.sh \
#       --ckpt_dir train_records/coordinate_no_cam_mindcube \
#       --start 500 --end 600 --step 50 \
#       --limit 6 --gpus 0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Defaults
# =============================================================================

CKPT_DIR=""
START=200
END=1000
STEP_SIZE=50
METHOD="coordinate"
DATASETS="mindcube,sat_real,spinbench,robospatial,viewspatial,omnispatial_pt,embspatial"
GPUS=""
LIMIT=""
OUTPUT=""
MAX_NEW_TOKENS=4096

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt_dir)      CKPT_DIR="$2";       shift 2 ;;
        --start)         START="$2";           shift 2 ;;
        --end)           END="$2";             shift 2 ;;
        --step)          STEP_SIZE="$2";       shift 2 ;;
        --method)        METHOD="$2";          shift 2 ;;
        --datasets)      DATASETS="$2";        shift 2 ;;
        --gpus)          GPUS="$2";            shift 2 ;;
        --limit)         LIMIT="$2";           shift 2 ;;
        --output)        OUTPUT="$2";          shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate
# =============================================================================

if [[ -z "$CKPT_DIR" ]]; then
    echo "[ERROR] --ckpt_dir is required." >&2
    exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[ERROR] Checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
fi

VALID_METHODS="baseline vanilla position_embedding coordinate decouple atten"
if ! echo "$VALID_METHODS" | grep -qw "$METHOD"; then
    echo "[ERROR] --method must be one of: $VALID_METHODS" >&2
    exit 1
fi

# =============================================================================
# GPU setup
# =============================================================================

if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

PYTHON=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python

N_GPU=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())")
if [[ "$N_GPU" -eq 0 ]]; then
    echo "[ERROR] No CUDA GPUs available." >&2
    exit 1
fi

# =============================================================================
# Build command
# =============================================================================

CMD=(
    "$PYTHON" curve_evaluation.py
    --ckpt_dir        "$CKPT_DIR"
    --start           "$START"
    --end             "$END"
    --step_size       "$STEP_SIZE"
    --method          "$METHOD"
    --datasets        "$DATASETS"
    --max_new_tokens  "$MAX_NEW_TOKENS"
)

if [[ -n "$GPUS" ]]; then
    CMD+=(--gpus "$GPUS")
fi

if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
fi

if [[ -n "$OUTPUT" ]]; then
    CMD+=(--output_dir "$OUTPUT")
fi

# =============================================================================
# Info banner
# =============================================================================

echo "=========================================================="
echo "[INFO] curve_evaluation.sh"
echo "[INFO]   ckpt_dir            : $CKPT_DIR"
echo "[INFO]   steps               : $START → $END (stride $STEP_SIZE)"
echo "[INFO]   method              : $METHOD"
echo "[INFO]   datasets            : $DATASETS"
echo "[INFO]   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[INFO]   Num GPUs            : $N_GPU"
if [[ -n "$LIMIT" ]]; then
    echo "[INFO]   Limit per dataset  : $LIMIT"
fi
if [[ -n "$OUTPUT" ]]; then
    echo "[INFO]   Output dir         : $OUTPUT"
else
    echo "[INFO]   Output dir         : curve_evaluation_results/$(basename "$CKPT_DIR")"
fi
echo "[INFO]   Started             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

# =============================================================================
# Run
# =============================================================================

"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[INFO] curve_evaluation completed successfully — $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "[ERROR] curve_evaluation exited with code $EXIT_CODE — $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "=========================================================="

exit $EXIT_CODE
