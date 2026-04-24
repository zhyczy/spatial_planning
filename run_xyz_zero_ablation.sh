#!/bin/bash
# xyz=0 ablation: measure QA acc (and coord_mae for coord model) when the
# per-patch xyz is replaced with zeros at inference time.
# Under R·0 = 0, xyz=0 is rotation-invariant by construction → only test R=I.
#
# Compares against baseline R=I from previous R-sweep runs.
#
# Two models:
#   (A) coord_no_cam        : 4D M-RoPE ckpt with coord_head (measure both QA & coord_mae)
#   (B) corresp_decouple    : decouple arch, no coord_head (QA only)
set -u

MODEL=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/checkpoints/Qwen3.5-4B
PY=/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python
SCRIPT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/validate_coord_rotation_robustness.py
OUT_ROOT=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/vis_results

CKPT_COORD=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/train_records/coordinate_no_cam_mindcube/step_1000
CKPT_DECO=/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/train_records/correspondence_mindcube_decouple/step_1000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
mkdir -p "$OUT_ROOT"

DATASETS_DIRS=(
  "mindcube|/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MindCube"
  "sat|/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SAT"
  "spinbench|/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/spinbench_data"
)

# ── (A) coordinate_no_cam: xyz=0 ablation WITH coord_head ─────────────────
for entry in "${DATASETS_DIRS[@]}"; do
  DS="${entry%%|*}"; DIR="${entry##*|}"
  echo "=== [$(date)] COORD  $DS  xyz=0  start ==="
  $PY $SCRIPT \
    --ckpt "$CKPT_COORD" --model_path "$MODEL" \
    --data_dir "$DIR" --dataset "$DS" \
    --output_dir "$OUT_ROOT/xyz0_coord_${DS}" \
    --zero_xyz --identity_only \
    2>&1 | tee "$OUT_ROOT/xyz0_coord_${DS}_run.log"
  echo "=== [$(date)] COORD  $DS  xyz=0  done ==="
done

# ── (B) correspondence_decouple: xyz=0 ablation, QA-only ──────────────────
for entry in "${DATASETS_DIRS[@]}"; do
  DS="${entry%%|*}"; DIR="${entry##*|}"
  echo "=== [$(date)] DECO   $DS  xyz=0  start ==="
  $PY $SCRIPT \
    --ckpt "$CKPT_DECO" --model_path "$MODEL" \
    --data_dir "$DIR" --dataset "$DS" \
    --output_dir "$OUT_ROOT/xyz0_deco_${DS}" \
    --decouple --no_coord_head --zero_xyz --identity_only \
    2>&1 | tee "$OUT_ROOT/xyz0_deco_${DS}_run.log"
  echo "=== [$(date)] DECO   $DS  xyz=0  done ==="
done

echo "=== [$(date)] xyz=0 ablation all done ==="
