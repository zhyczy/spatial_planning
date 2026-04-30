#!/usr/bin/env bash
# =============================================================================
# xyz_validation.sh
#
# Run xyz_validation.py to ablate whether a trained SPA checkpoint actually
# consumes the per-patch 3D coordinates fed into its xyz pathway. For each
# dataset, runs two passes on the same checkpoint:
#   • PASS 1 — normal xyz   : image_xyz from <data_dir>/3d_results/<id>/
#   • PASS 2 — image_xyz=0  : torch.zeros_like(...) before any consumer
# Per-dataset Δaccuracy (and Δcoord_mae for coordinate method) is the
# evidence: a near-zero delta means the xyz pathway is dead at inference.
#
# Usage:
#   bash scripts/xyz_validation.sh \
#       --method   {position_embedding|coordinate|polar|decouple|atten} \
#       --ckpt     <ckpt dir> \
#       [--datasets "mindcube,sat_real,spinbench"]   # default: mindcube
#       [--gpus 0,1,2,3]                              # default: all visible
#       [--limit N]                                   # debug truncation
#       [--output <base dir>]                         # default: vis_results/
#       [--run_name <name>]                           # sub-folder per dataset
#       [--xyz_rope_dim N]                            # polar/decouple only, default 66
#       [--max_new_tokens N]                          # default 512
#       [--only normal|zero|both]                     # default both
#
# Methods that ingest xyz at inference (others have no point ablating):
#   position_embedding   — SPA LoRA + 4D M-RoPE (xyz fed through RoPE; no coord head)
#   coordinate           — SPA LoRA + 4D M-RoPE + coord head (Cartesian readout)
#   polar                — SpaDec + LoRA: log-spherical XYZ RoPE in pass-through, θ=1000
#   decouple             — SpaDec + LoRA: Cartesian XYZ RoPE in pass-through, θ=10000
#   atten                — Qwen3.5 + LoRA + per-layer SpatialAttentionBias on V↔V
#                          (3D M-RoPE unchanged; matches train_atten.py)
#
# Examples:
#   # MindCube, coord ckpt, all GPUs, both passes:
#   bash scripts/xyz_validation.sh \
#       --method coordinate \
#       --ckpt   train_records/coordinate_no_cam_mindcube/step_1000
#
#   # atten ckpt, MindCube + SAT_real, 4 GPUs:
#   bash scripts/xyz_validation.sh \
#       --method   atten \
#       --ckpt     train_records/app/atten_mindcube/step_1000 \
#       --datasets "mindcube,sat_real" \
#       --gpus     0,1,2,3
#
#   # decouple ckpt, only the zero pass (assume normal already done):
#   bash scripts/xyz_validation.sh \
#       --method   decouple \
#       --ckpt     train_records/correspondence_mindcube_decouple/step_1000 \
#       --only     zero
#
#   # smoke test — 6 samples, 1 GPU:
#   bash scripts/xyz_validation.sh \
#       --method   coordinate \
#       --ckpt     train_records/coordinate_no_cam_mindcube/step_1000 \
#       --limit    6 \
#       --gpus     0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIAL_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SPATIAL_DIR"

# =============================================================================
# Defaults
# =============================================================================

METHOD=""
CKPT=""
GPUS=""
LIMIT=""
OUTPUT_BASE="$SPATIAL_DIR/vis_results"
RUN_NAME=""
ONLY="both"
XYZ_ROPE_DIM=""
MAX_NEW_TOKENS=512
DATASETS_ARG=""

# Default dataset for this ablation: just MindCube (smallest meaningful set).
DEFAULT_DATASETS="mindcube"

# Dataset → data_dir mapping (matches scripts/evaluate.sh).
declare -A DATASET_DIR
DATASET_DIR["mindcube"]="datasets/evaluation/MindCube"
DATASET_DIR["mmsibench"]="datasets/evaluation/MMSIBench"
DATASET_DIR["sparbench_multi_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sparbench_single_view"]="datasets/evaluation/SPARBench"
DATASET_DIR["sparbench_mv"]="datasets/evaluation/SPARBench"
DATASET_DIR["sat"]="datasets/evaluation/SAT"
DATASET_DIR["sat_real"]="datasets/evaluation/SAT"
DATASET_DIR["spinbench"]="datasets/evaluation/spinbench_data"
DATASET_DIR["robospatial"]="datasets/evaluation/RoboSpatial"
DATASET_DIR["viewspatial"]="datasets/evaluation/ViewSpatial-Bench"
DATASET_DIR["omnispatial_pt"]="datasets/evaluation/OmniSpatial"
DATASET_DIR["embspatial"]="datasets/evaluation/EmbSpatial-Bench"

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)         METHOD="$2";         shift 2 ;;
        --ckpt)           CKPT="$2";           shift 2 ;;
        --datasets)       DATASETS_ARG="$2";   shift 2 ;;
        --gpus)           GPUS="$2";           shift 2 ;;
        --limit)          LIMIT="$2";          shift 2 ;;
        --output)         OUTPUT_BASE="$2";    shift 2 ;;
        --run_name)       RUN_NAME="$2";       shift 2 ;;
        --only)           ONLY="$2";           shift 2 ;;
        --xyz_rope_dim)   XYZ_ROPE_DIM="$2";   shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate arguments
# =============================================================================

VALID_METHODS="position_embedding coordinate polar decouple atten"
if [[ -z "$METHOD" ]]; then
    echo "[ERROR] --method is required (one of: $VALID_METHODS)" >&2
    exit 1
fi
if ! echo "$VALID_METHODS" | grep -qw "$METHOD"; then
    echo "[ERROR] --method must be one of: $VALID_METHODS" >&2
    echo "        baseline / vanilla don't ingest xyz; nothing to ablate." >&2
    exit 1
fi

if [[ -z "$CKPT" ]]; then
    echo "[ERROR] --ckpt is required" >&2
    exit 1
fi
if [[ ! -d "$CKPT" ]]; then
    echo "[ERROR] Checkpoint directory not found: $CKPT" >&2
    exit 1
fi

case "$ONLY" in
    normal|zero|both) ;;
    *) echo "[ERROR] --only must be normal|zero|both (got '$ONLY')" >&2; exit 1 ;;
esac

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
    DATASETS="${DATASETS_ARG//,/ }"
else
    DATASETS="$DEFAULT_DATASETS"
fi

for ds in $DATASETS; do
    if [[ -z "${DATASET_DIR[$ds]+x}" ]]; then
        echo "[ERROR] Unknown dataset '$ds'." >&2
        echo "        Known: ${!DATASET_DIR[*]}" >&2
        exit 1
    fi
done

# =============================================================================
# Resolve sub-folder name (per-method timestamp by default)
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$RUN_NAME" ]]; then
    CKPT_TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
    RUN_NAME="xyz_val_${METHOD}_${CKPT_TAG}_${TIMESTAMP}"
fi

# =============================================================================
# Optional flags
# =============================================================================

LIMIT_FLAG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

XYZ_ROPE_DIM_FLAG=""
if [[ -n "$XYZ_ROPE_DIM" ]]; then
    XYZ_ROPE_DIM_FLAG="--xyz_rope_dim $XYZ_ROPE_DIM"
fi

# =============================================================================
# Banner
# =============================================================================

echo "[INFO] =============================================================="
echo "[INFO] xyz_validation"
echo "[INFO] =============================================================="
echo "[INFO] method               : $METHOD"
echo "[INFO] ckpt                 : $CKPT"
echo "[INFO] datasets             : $DATASETS"
echo "[INFO] CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[INFO] N_GPU                : $N_GPU"
echo "[INFO] only                 : $ONLY"
echo "[INFO] limit                : ${LIMIT:-all}"
echo "[INFO] xyz_rope_dim         : ${XYZ_ROPE_DIM:-66 (default; only used by polar/decouple)}"
echo "[INFO] max_new_tokens       : $MAX_NEW_TOKENS"
echo "[INFO] output base          : $OUTPUT_BASE"
echo "[INFO] run name             : $RUN_NAME"
echo "[INFO] starting             : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] =============================================================="

# =============================================================================
# Run per-dataset
# =============================================================================

PYTHON=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
SCRIPT="$SPATIAL_DIR/xyz_validation.py"
MODEL_PATH="$SPATIAL_DIR/checkpoints/Qwen3.5-4B"

for ds in $DATASETS; do
    DATA_DIR="$SPATIAL_DIR/${DATASET_DIR[$ds]}"
    OUT_DIR="$OUTPUT_BASE/$RUN_NAME/$ds"

    echo ""
    echo "[INFO] === [$(date '+%H:%M:%S')] $METHOD  $ds  start ==="
    echo "[INFO]     data_dir   : $DATA_DIR"
    echo "[INFO]     output_dir : $OUT_DIR"

    mkdir -p "$OUT_DIR"

    "$PYTHON" "$SCRIPT" \
        --method         "$METHOD" \
        --ckpt           "$CKPT" \
        --model_path     "$MODEL_PATH" \
        --dataset        "$ds" \
        --data_dir       "$DATA_DIR" \
        --output_dir     "$OUT_DIR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --only           "$ONLY" \
        $XYZ_ROPE_DIM_FLAG \
        $LIMIT_FLAG \
        2>&1 | tee "$OUT_DIR/run.console.log"

    echo "[INFO] === [$(date '+%H:%M:%S')] $METHOD  $ds  done ==="
done

echo ""
echo "[INFO] all done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] results under: $OUTPUT_BASE/$RUN_NAME/"
