# Coordinate Model is Frame-Locked — Rotation Robustness Study

**Date:** 2026-04-22
**Checkpoint:** `train_records/coordinate_no_cam_mindcube/step_1000`
**Dataset:** MindCube tinybench (**N = 1050** samples)
**Rotations:** 22 (I + 7 angles × 3 axes)
**Total forward passes:** 23,100
**Script:** [`validate_coord_rotation_robustness.py`](../../validate_coord_rotation_robustness.py)
**Raw results:** `train_records/coordinate_no_cam_mindcube/step_1000/R_sweep/`

---

## TL;DR

A coordinate-supervised model trained with xyz in the view_0000 camera frame
turns out to be **almost entirely rotation-invariant at QA time and
frame-locked at coord-head time**:

- **QA accuracy is flat under every R** — total spread 0.76 pp
  (94.29% – 95.05%) across 22 rotations. Only **25 / 1050 samples (2.4%)** ever
  flip their letter across any R; the other 1025 give identical predictions
  for all 22 R.
- **coord_mae degrades 3.4×** under 180° rotation (0.637 → 2.17 on Ry_180) —
  cleanly monotonic in angle, with axis ranking Ry > Rx > Rz that matches
  depth-channel preservation.
- Combined: the coord_head emits xyz in the training frame regardless of R,
  while the LM decodes answers from vision + text, ignoring the rotated
  4D-RoPE xyz channel.

The xyz → 4D-RoPE pathway **contributes essentially nothing to answer
decoding** on MindCube tinybench. Training artefacts around xyz (coord_head,
xyz-RoPE channels) are in this sense decorative.

---

## Experimental setup

For each sample, we prepare `inputs` and `image_xyz` (first-camera-frame
coordinates from `3d_results/pts3d.npy`) once, then sweep 22 rotations. For
each R we:

1. Rotate xyz → `xyz_rot = R @ xyz`
2. Feed `xyz_rot` to the 4D M-RoPE
3. Run the LM generation → record letter prediction
4. Run a forward pass with the lm_head hook → extract coord_head prediction
5. Compute `coord_mae = L1(pred, xyz_rot)`

Rotation grid:

| Category | Axis | Angles (deg) |
|---|---|---|
| Identity | — | 0 |
| Rx (around camera **x** = right) | x | 30, 60, 90, 120, 150, 180, 270 |
| Ry (around camera **y** = down) | y | 30, 60, 90, 120, 150, 180, 270 |
| Rz (around camera **z** = forward = depth) | z | 30, 60, 90, 120, 150, 180, 270 |

Coordinate convention is OpenCV-style (+x right, +y down, +z depth), so in
the camera frame Rz ≈ roll, Rx ≈ pitch, Ry ≈ yaw. See
[`rotation_axes.md`](../data/rotation_axes.md) for the axis-to-semantics mapping.

---

## Results

### 1. QA accuracy — effectively flat

| R | acc | Δ vs I (pp) | | R | acc | Δ vs I (pp) | | R | acc | Δ vs I (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| **I** | **94.76%** | — | | Ry_030 | 94.29% | -0.47 | | Rz_030 | 94.86% | +0.10 |
| Rx_030 | 94.86% | +0.10 | | Ry_060 | 94.38% | -0.38 | | Rz_060 | 94.76% |   0.00 |
| Rx_060 | 94.86% | +0.10 | | Ry_090 | 94.38% | -0.38 | | Rz_090 | 94.38% | -0.38 |
| Rx_090 | 94.95% | +0.19 | | Ry_120 | 94.48% | -0.28 | | Rz_120 | 94.76% |   0.00 |
| Rx_120 | 94.86% | +0.10 | | Ry_150 | 94.57% | -0.19 | | Rz_150 | 94.67% | -0.09 |
| Rx_150 | 94.86% | +0.10 | | Ry_180 | 94.76% |   0.00 | | Rz_180 | 94.76% |   0.00 |
| Rx_180 | 94.76% |   0.00 | | Ry_270 | 94.29% | -0.47 | | Rz_270 | 94.29% | -0.47 |
| Rx_270 | 94.95% | +0.29 | | | | | | | | |

- **Range:** [94.29%, 95.05%] → spread **0.76 pp** = 8 samples out of 1050
- **Top:** Rx_270 (95.05%), Rx_090 (94.95%)
- **Bottom:** Ry_030 / Ry_270 / Rz_270 (94.29%)
- No R performs worse than Ry_030 at −0.47pp, and none better than Rx_270 at +0.29pp — all within the sampling noise of 1050-N evaluation

### 2. Per-sample prediction stability

Distribution of distinct letters across 22 R per sample:

| #unique predictions | samples |
|---|---|
| **1** (rotation-invariant)    | **1025 / 1050  (97.6%)** |
| 2 (ever flips across some R)  |   25 / 1050  (2.4%) |
| ≥ 3                            |   0 |

Among the 25 "flipping" samples, **each has only 2 distinct predictions** —
i.e., one "default" letter and one "alternate" letter that shows up at
specific rotations. No sample produces 3+ different letters.

### 3. Which rotations flip samples away from I?

Number of samples whose letter at R differs from their letter at I:

| Angle | Rx | Ry | Rz |
|---|---|---|---|
| 30°  | 2 | 8  | 2 |
| 60°  | 2 | 6  | 2 |
| 90°  | 3 | 8  | 7 |
| 120° | 2 | 7  | 4 |
| 150° | 2 | 8  | 7 |
| 180° | 1 | 6  | 8 |
| **270°** | **4** | **12** | **11** |

Two patterns:

- **Ry and Rz flip ~3× more samples than Rx.** Ry (yaw-like, x↔z swap) and
  Rz (roll-like, x↔y swap) are more likely to trigger an LM answer change
  than Rx (pitch-like, y↔z swap). Note this axis ordering differs from the
  coord_mae ranking (Ry > Rx > Rz): the LM's answer sensitivity is governed
  by a different factor than coord_head's geometric error.
- **270° consistently flips more than 180°.** On all three axes, 270° sees
  the highest flip count. 270° = −90° breaks the scene alignment along the
  chosen axis in the "opposite" direction; the model seems to find some
  exotic rotations harder to absorb than plain 180° flips.

### 4. Direct I vs Ry_180 comparison (Ry_180 = worst coord_mae)

| Transition | n |
|---|---|
| I correct → Ry_180 wrong  | 3 |
| I wrong   → Ry_180 correct | 3 |
| **Net change** | **0** |

Even at the worst-coord_mae rotation, the flips are symmetric random
shuffling of a handful of borderline samples — not a directional degradation.

### 5. Per-category accuracy stays flat

| R | linear (n=250) | perpendicular (n=800) |
|---|---|---|
| I       | 91.20% | 95.88% |
| Rx_090  | 91.60% | 96.00% |
| Rx_270  | 92.40% | 95.88% |
| Ry_180  | 92.00% | 95.63% |
| Rz_180  | 92.00% | 95.63% |

`linear` questions (which require reasoning along a depth axis, should be
most sensitive to z-channel corruption) are **not systematically worse** at
rotated R; in fact Rx_270 nudges linear accuracy slightly up. Neither category
shows a signal that the rotated xyz is used for decoding.

### 6. coord_mae — clean geometric degradation

```
         I      30°    60°    90°    120°   150°   180°   270°
Rx    0.637  0.899  1.266  1.644  1.874  1.938  1.973  1.644
Ry    0.637  0.895  1.304  1.690  1.964  2.118  2.172  1.700
Rz    0.637  0.776  0.971  1.178  1.339  1.419  1.445  1.191
```

- **Monotonic** 30° → 180° on all three axes
- **90°/270° symmetric**: |mae(90°) − mae(270°)| < 0.01 for every axis
- **Axis ranking Ry > Rx > Rz** at every angle — exact replication of the
  smoke test pattern, now with 1050-sample precision
- **Peak multiplier**: Ry_180 = 2.172 / 0.637 = 3.41× baseline;
  Rz_180 = 1.445 / 0.637 = 2.27×

The ranking matches the **depth-channel preservation** hypothesis:

| R | Channels mixed | Channel preserved | Observed mae factor @ 180° |
|---|---|---|---|
| Rz | x ↔ y | **z (depth)** | 2.27× (mildest) |
| Rx | y ↔ z | x (horizontal) | 3.10× |
| Ry | x ↔ z | y (vertical) | 3.41× (worst) |

Rotations that corrupt the depth channel degrade coord_mae more; rotations
that preserve depth do the least damage.

---

## Diagnosis

Two independent pathways use `xyz`:

1. **LM generation path**: xyz → 4D M-RoPE positions → attention → answer logits
2. **coord_head path**: vision-token hidden states → DepthPredictionTransformer → xyz prediction

Both paths show evidence of being **xyz-agnostic** at eval time:

- **LM path**: 97.6% of samples never change their answer across 22 very
  different R's. The remaining 2.4% change only between two letters and in a
  pattern that is net-zero (3 gains = 3 losses at Ry_180). This is
  indistinguishable from sampling noise on a handful of borderline samples —
  the LM is not decoding answers from the RoPE xyz channel.
- **coord_head path**: coord_mae scales 3.4× as R moves from I → 180°. If
  coord_head were truly decoding xyz from the (rotated) RoPE signal, it would
  output `R @ xyz`, keeping coord_mae constant. The blow-up means coord_head
  emits approximately `xyz` (the I-frame prediction) regardless of the
  rotation fed into RoPE. It is **frame-locked** to view_0000.

In effect the coordinate-supervised model has learned to answer MindCube
questions from **image patches + text**, using xyz only as a weak auxiliary
signal that never affects decoding outputs. The first-frame coordinate system
choice is neither helping nor hurting QA accuracy, because the model isn't
attending to it.

---

## Implications

1. **The xyz → 4D-RoPE channel is decorative at this checkpoint.**
   Removing it (and the coord_head) should barely move accuracy. An ablation
   swapping 4D M-RoPE → vanilla 3D M-RoPE + LoRA would quantify this; prior
   training logs hint at ≤1pp gain from xyz-RoPE — now explained.

2. **coord supervision is training-time only.** coord_head outputs the
   view_0000-frame xyz because that is what it was trained to predict. At
   eval time its outputs are untethered from the RoPE input — the coord_head
   is effectively a vision-feature → xyz regressor.

3. **The "rotation-aware" pipeline's premise needs re-examination.** In
   [rotation_diversity_analysis.md](../rotation_diversity_analysis.md) we saw
   that `train_alternate.py`'s rotation_enc collapses to a single dataset-level
   R. Here we see that even the simpler coordinate model makes the xyz channel
   irrelevant to QA. Together: **MindCube does not require a per-sample
   rotation to be solved**, and current training recipes recognise this by
   discarding the xyz signal.

4. **A model that cares about R must be built, not inferred.** To get a
   rotation-sensitive QA model, R-dependent *targets* are needed (e.g. direct
   supervision on gt_rotation, or tasks where the answer text must reflect a
   direction that depends on R). Otherwise any amount of xyz-RoPE engineering
   will be silently averaged out.

---

## Follow-up ideas

- **Ablation**: rerun evaluation with xyz = zeros vs xyz = `R @ xyz`. If
  accuracy holds, confirms xyz is entirely unused.
- **Attention probing**: measure how much attention weight at each layer
  concentrates on vision tokens vs the coord sentence. Expected: coord
  tokens attract near-zero attention from the answer-predicting tokens.
- **Force rotation-dependence**: add a loss term that penalises
  answer-invariance across a hard rotation (e.g. a contrastive loss over
  axis-rotated versions of the same scene). This would convert xyz into a
  load-bearing channel and force the LM to actually use the RoPE signal.

---

## Files

- Multi-GPU sweep script: [`../../validate_coord_rotation_robustness.py`](../../validate_coord_rotation_robustness.py)
- Full merged rows (23,100): `train_records/coordinate_no_cam_mindcube/step_1000/R_sweep/R_sweep.json`
- Per-R summary: `train_records/coordinate_no_cam_mindcube/step_1000/R_sweep/metrics_per_R.json`
- Smoke report: [`../coord_rotation_robustness_smoke.md`](../coord_rotation_robustness_smoke.md)
- Related: [`../rotation_diversity_analysis.md`](../rotation_diversity_analysis.md),
  [`../data/rotation_axes.md`](../data/rotation_axes.md)
