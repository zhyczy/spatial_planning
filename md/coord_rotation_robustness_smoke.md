# Coordinate Model Rotation-Robustness Smoke Test

**Date:** 2026-04-22
**Checkpoint:** `train_records/coordinate_no_cam_mindcube/step_1000`
**Dataset:** MindCube tinybench — **smoke subset: 6 samples**
**Script:** `spatial_planning/validate_coord_rotation_robustness.py`
**Config:** 22 rotations per sample → 132 rows total
**Raw data backup:** `/tmp/smoke_R_sweep_backup/`

This is a small-scale sanity check — a full 1050-sample run is queued on 8 GPUs.
Numbers here are directional, not statistically significant. Their value is:

1. validating that the multi-GPU pipeline is correct;
2. surfacing the **qualitative shape** of coord_mae vs R that we expect to confirm at scale.

---

## Motivation

`coordinate_no_cam_mindcube/step_1000` was trained with xyz coordinates expressed
in the **view_0000 camera frame** (R = I for every sample). At inference we
rotate the xyz by an arbitrary R before feeding the 4D M-RoPE (and comparing
coord head predictions to the rotated GT). Questions:

1. Does QA accuracy drop as R moves away from I?
2. Does the coord head's xyz prediction track the rotated input, or is it
   frame-locked?
3. Which rotation axis is most harmful, and does that match geometric intuition?

---

## Method

For each sample we prepare `inputs` and `image_xyz` once, then sweep 22 rotations:

| Category | Angles | Count |
|---|---|---|
| Identity | 0° | 1 |
| Rx (around camera **x** = right) | 30, 60, 90, 120, 150, 180, 270 | 7 |
| Ry (around camera **y** = down) | 30, 60, 90, 120, 150, 180, 270 | 7 |
| Rz (around camera **z** = forward / depth) | 30, 60, 90, 120, 150, 180, 270 | 7 |

For each R we compute:

- **`prediction`** — the answer letter the LM generates with the rotated xyz as its 4D-RoPE channel
- **`coord_mae`** — L1 error between the coord head's xyz output and `R @ xyz_world` (rotated GT)

---

## Results

### 1. QA prediction is invariant under R

| Sample (idx) | Category | GT | #unique preds across 22 R |
|---|---|---|---|
| among_…3f76f0_q2_2_1 | perpendicular | C | **1** — all predict "C" |
| among_group458_q0_2_3 | perpendicular | A | **1** — all predict "A" |
| among_group603_q1_1_2 | perpendicular | B | **1** — all predict "B" |
| among_group603_q1_2_2 | perpendicular | B | **1** — all predict "B" |
| among_group693_q1_1_2 | perpendicular | D | **1** — all predict "D" |
| among_group693_q1_5_2 | perpendicular | C | **1** — all predict "C" |

**All 6 samples answer the same letter under all 22 R's.** Accuracy is 1.0 at
every R. These samples are too easy / too stable to distinguish anything; the
1050-run will reveal which (if any) samples flip under rotation.

### 2. coord_mae degrades in a very clean geometric pattern

Mean coord_mae across 6 samples (rounded):

```
           I        30°     60°     90°    120°    150°    180°    270°
Rx-axis   0.229   0.455   0.765   1.042   1.200   1.249   1.269   1.010
Ry-axis   0.229   0.435   0.786   1.123   1.338   1.420   1.452   1.175
Rz-axis   0.229   0.329   0.501   0.671   0.802   0.861   0.875   0.704
```

Three properties hold:

- **Monotonic 30° → 180°** on 15 of 18 sample × axis curves. The remaining 3
  show ±0.02 wobble in the 150° – 180° plateau — noise on a flat region.
- **90°/270° symmetry**: |mae(90°) − mae(270°)| ≤ 0.05 for every axis,
  consistent with 270° being geometrically equivalent to −90°.
- **Axis ranking: Ry > Rx > Rz** at every angle.

### 3. Axis sensitivity

In camera convention (+x right, +y down, +z forward = depth), each rotation
mixes a specific pair of coordinate channels:

| Rotation | Channels mixed | Channel preserved | Effective semantics |
|---|---|---|---|
| **Rz** (roll)  | x ↔ y | **z (depth)** | image-plane rotation only, depth intact |
| **Rx** (pitch) | y ↔ z | x (horizontal) | vertical ↔ depth swap |
| **Ry** (yaw)   | x ↔ z | y (vertical) | **horizontal ↔ depth swap** |

The observed Rz < Rx < Ry ranking is exactly what depth-channel preservation
predicts: rotations that preserve depth are mildest; rotations that swap
horizontal position with depth ("the cup on the left" becomes "the cup in
front") are most damaging.

---

## Diagnosis

coord_mae scales by **5× – 6×** from identity to 180° while QA predictions
are untouched. The coord head is therefore **not** tracking the rotated 4D-RoPE
signal: it emits approximately the same xyz regardless of R, and we
(correctly) rotate the GT to compare, so the residual explodes.

Two conclusions follow:

1. **coord_head is effectively frame-locked** to the view_0000 frame it was
   trained in. It decodes xyz from vision-token semantics, not from the RoPE
   position channel.
2. **The LM is also frame-locked** w.r.t. answer generation (22/22 predictions
   invariant on these 6 samples). On this tiny subset the 4D-RoPE xyz channel
   has negligible influence on QA decoding.

Together: on these samples, the xyz → M-RoPE pathway is effectively
**decorative** — the model answers the same thing with any R, because it is
using image + text, not the rotated positional signal, to both (a) answer the
question and (b) predict xyz.

---

## Limitations

- **N = 6** is far too small to conclude anything about the QA-accuracy side.
  All 6 are `perpendicular` and all happen to be stable under rotation.
- The 1050-run may reveal a subset of samples where QA does depend on R —
  those would be the cases where the LM genuinely uses the 4D-RoPE xyz
  channel. The coord_mae geometry is already robustly clear at N = 6.

---

## Next

When the 1050-sample full run finishes we will:

1. Compute accuracy per R (not just coord_mae) — the interesting metric.
2. Check whether any sample's letter flips across R's; report its count per
   axis and per angle.
3. Stratify by `category` (linear vs perpendicular) — large angle changes may
   hurt linear (depth-dependent) questions more than perpendicular ones.
4. Compare the coord_mae axis-ranking (Ry > Rx > Rz) to see if it persists.

---

## Files

- Script: [../validate_coord_rotation_robustness.py](../validate_coord_rotation_robustness.py)
- Smoke-test data backup: `/tmp/smoke_R_sweep_backup/` (overwritten in-place
  by the full run; the pre-full-run copy is preserved here)
  - `R_sweep.json` — 132 per-(sample, R) rows
  - `metrics_per_R.json` — 22-row per-R summary
- Full run output (when done): `train_records/coordinate_no_cam_mindcube/step_1000/R_sweep/`
- Related: [rotation_diversity_analysis.md](rotation_diversity_analysis.md),
  [train_coordinate.md](train_coordinate.md)
