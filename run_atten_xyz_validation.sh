#!/bin/bash
# atten ckpt — xyz=0 vs normal xyz ablation on MindCube test set.
# Tests whether train_records/app/atten_mindcube/step_1000 actually uses
# the per-patch 3D coordinates that SpatialAttentionBias receives.
set -u

PY=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
ROOT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning
SCRIPT=$ROOT/validation.py

CKPT=$ROOT/train_records/app/atten_mindcube/step_1000
MODEL=$ROOT/checkpoints/Qwen3.5-4B
DATA_DIR=$ROOT/datasets/evaluation/MindCube
OUT=$ROOT/vis_results/atten_xyz0_step_1000

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

mkdir -p "$OUT"

echo "=== [$(date)] atten xyz=0 ablation start ==="
echo "    ckpt    : $CKPT"
echo "    dataset : mindcube  ($DATA_DIR)"
echo "    output  : $OUT"
echo "    GPUs    : $CUDA_VISIBLE_DEVICES"

$PY "$SCRIPT" \
    --ckpt        "$CKPT" \
    --model_path  "$MODEL" \
    --dataset     mindcube \
    --data_dir    "$DATA_DIR" \
    --output_dir  "$OUT" \
    --only        both \
    2>&1 | tee "$OUT/run.console.log"

echo "=== [$(date)] atten xyz=0 ablation done ==="
