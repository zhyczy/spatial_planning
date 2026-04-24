#!/bin/bash
# Rotation-robustness sweep on correspondence_mindcube_decouple/step_1000.
# This ckpt is LoRA-only (no coord_head.pt), so we pass --no_coord_head
# and only measure QA accuracy under R.
# Runs on 6 GPUs sequentially across MindCube / SAT / SpinBench.
set -u

CKPT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/train_records/correspondence_mindcube_decouple/step_1000
MODEL=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/checkpoints/Qwen3.5-4B
PY=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
SCRIPT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/validate_coord_rotation_robustness.py
OUT_ROOT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/vis_results

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

mkdir -p "$OUT_ROOT"

echo "=== [$(date)] corresp_decouple MindCube start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MindCube \
  --dataset mindcube \
  --output_dir "$OUT_ROOT/R_sweep_corresp_decouple_mindcube" \
  --no_coord_head \
  2>&1 | tee "$OUT_ROOT/corresp_decouple_mindcube_run.log"
echo "=== [$(date)] corresp_decouple MindCube done ==="

echo "=== [$(date)] corresp_decouple SAT start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SAT \
  --dataset sat \
  --output_dir "$OUT_ROOT/R_sweep_corresp_decouple_sat" \
  --no_coord_head \
  2>&1 | tee "$OUT_ROOT/corresp_decouple_sat_run.log"
echo "=== [$(date)] corresp_decouple SAT done ==="

echo "=== [$(date)] corresp_decouple SpinBench start ==="
$PY $SCRIPT \
  --ckpt "$CKPT" --model_path "$MODEL" \
  --data_dir /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/spinbench_data \
  --dataset spinbench \
  --output_dir "$OUT_ROOT/R_sweep_corresp_decouple_spinbench" \
  --no_coord_head \
  2>&1 | tee "$OUT_ROOT/corresp_decouple_spinbench_run.log"
echo "=== [$(date)] corresp_decouple SpinBench done ==="
