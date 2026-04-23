#!/bin/bash
# Run coord rotation-robustness sweep on MindCube, SAT, SpinBench
# in sequence on 8 GPUs. Outputs under vis_results/.
set -u

CKPT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/train_records/coordinate_no_cam_mindcube/step_1000
MODEL=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/checkpoints/Qwen3.5-4B
PY=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
SCRIPT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/validate_coord_rotation_robustness.py
OUT_ROOT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/vis_results

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "$OUT_ROOT"

echo "=== [$(date)] MindCube start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MindCube \
  --dataset mindcube \
  --output_dir "$OUT_ROOT/R_sweep_mindcube" \
  2>&1 | tee "$OUT_ROOT/mindcube_run.log"
echo "=== [$(date)] MindCube done ==="

echo "=== [$(date)] SAT start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SAT \
  --dataset sat \
  --output_dir "$OUT_ROOT/R_sweep_sat" \
  2>&1 | tee "$OUT_ROOT/sat_run.log"
echo "=== [$(date)] SAT done ==="

echo "=== [$(date)] SpinBench start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/spinbench_data \
  --dataset spinbench \
  --output_dir "$OUT_ROOT/R_sweep_spinbench" \
  2>&1 | tee "$OUT_ROOT/spinbench_run.log"
echo "=== [$(date)] SpinBench done ==="
