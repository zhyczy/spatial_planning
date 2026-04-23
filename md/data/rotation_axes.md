# Rotation Axes — Anchor Group, Label Coverage, Camera Geometry

**Date:** 2026-04-22
**Scope:** `CameraTokenRotationEncoderRL` anchor space vs. dataset supervision.

Cross-references:
- Training data: [train_datasets.md](train_datasets.md)
- Eval data: [eval_datasets.md](eval_datasets.md)
- Downstream constant-R collapse: [../rotation_diversity_analysis.md](../rotation_diversity_analysis.md)

---

## 1. Summary (one screen)

| Layer | Yaw | Pitch | Roll |
|---|---|---|---|
| 24-anchor head (chiral cube group O) + 3-DOF residual | full | full | full (SO(3) complete) |
| Training labels (pose-derived, MindCube B/D/E) | ✓ | ✓ | ✓ |
| Training labels (text, SAT + SPAR view_change_infer) | 4,937 | 71 | 0 |
| Evaluation labels (across all 9 benchmarks) | present | **absent** | **absent** |
| Input geometry (SPARBench camera pairs) | 59% | 15% | 26% |

Gap: anchor *space* is 3D; training *labels* are ~99% yaw; eval *labels* are 100% yaw → the non-yaw capacity of the head has no gradient and no test.

---

## 2. The 24-anchor group

Built at [src/models/rotation_rope_llm.py:1039-1065](../../src/models/rotation_rope_llm.py#L1039-L1065) via `_build_chiral_cube_group()`. All signed permutation matrices with `det = +1` — the orientation-preserving symmetries of the cube (isomorphic to S₄).

Decomposition:

| Category | N | Description |
|---|---:|---|
| identity | 1 | R = I |
| face-90° about ±X / ±Y / ±Z | 6 | 2 per principal axis |
| face-180° about X / Y / Z | 3 | 1 per principal axis |
| body-diagonal-120° | 8 | around the 4 body diagonals, ±120° each |
| edge-180° | 6 | around the 6 face-diagonals |
| **total** | **24** | |

Only 9 of 24 are pure single-axis rotations (identity + face rotations). The other 15 mix all three axes — rotation along body/face diagonals has non-trivial components on every axis.

Combined with `head_res: Linear(d_model, 72)` ([src/models/rotation_rope_llm.py:1182](../../src/models/rotation_rope_llm.py#L1182)), which gives a per-anchor 3-DOF axis-angle residual, the predictor spans **full SO(3)**.

**Common misconception:** the 24 anchors are *not* yaw-only. The confusion arose because MindCube training GT is yaw-only, not because of the head structure.

---

## 3. Training supervision per axis

### 3.1 Pose-derived labels (MindCube B/H/D/E, 8,127 samples)

`R_gt` is a real 3×3 rotation from `camera_pose.npy` files. All three axes are structurally present — the B-bucket composes `C2W[0]ᵀ · C2W[N]` (continuous, any-axis) with a discrete yaw turn. See [train_datasets.md §1.1](train_datasets.md#11-bucket-taxonomy-text-derived-full-10k-distribution).

### 3.2 Text labels (SAT + SPAR_view_change_infer)

Strict regex hits:

| Source | Yaw | Pitch | Roll |
|---|---:|---:|---:|
| SAT train_36k | 4,821 | 0 | 0 |
| SPAR train_10k view_change_infer | 116 | 71 | 0 |
| **Total text-level** | **4,937** | **71** | **0** |

**Pitch-labeled training fraction ≈ 0.13%** (71 / 56,810).
**Roll-labeled training fraction = 0%.**

### 3.3 Aggregate gradient landscape

- Anchor 0 (I): hit by MindCube A (1,302 samples) as explicit prior, plus near-identity pose cases.
- Anchors 5/6/16 (face 90°/180° yaw): hit by MindCube B/H via `R_turn`.
- Anchors 1-4, 7-15, 17-23 (off-yaw + mixed): gradient only through `C2W[0]ᵀ · C2W[N]` in B/D/E, which for MindCube scenes is a near-horizontal rotation — effectively still yaw-dominated.

---

## 4. Evaluation supervision per axis

All 9 benchmarks: **yaw-only**. See [eval_datasets.md §4](eval_datasets.md#4-rotation-axis-keyword-hits-per-eval-strict-regex-noun-exclusion) for strict-regex counts.

No eval benchmark has:
- "tilt up / down by N°" prediction labels
- "roll left / right" prediction labels
- continuous 3-DOF rotation labels

---

## 5. SPARBench camera-pair geometry (input-side, non-label)

Although SPARBench has no rotation-axis labels, its camera pose pairs do span all three axes.

**Method:** For every `*_mv` record (3 views), compute `R_rel = R0ᵀ · Ri` for i ∈ {1, 2}. Decompose axis-angle `(axis, θ)`; per-axis contribution = `|axis_k| · θ` in degrees. Camera frame = OpenCV (X right, Y down, Z forward); verified that `R0 ≈ I` (‖R0−I‖ ≈ 0.0006) and camera-Y aligns with world vertical.

### 5.1 Per-axis magnitudes (degrees)

| Task | pairs | mean \|Δθ\| | ⟨pitch⟩ | ⟨yaw⟩ | ⟨roll⟩ | **pitch : yaw : roll (%)** |
|---|---:|---:|---:|---:|---:|---|
| obj_spatial_relation_oc_mv | 800 | 53.1° | 11.71° | 44.03° | 20.32° | **15.4 / 57.9 / 26.7** |
| obj_spatial_relation_oo_mv | 722 | 58.1° | 10.55° | 49.67° | 21.47° | **12.9 / 60.8 / 26.3** |
| spatial_imagination_oc_mv | 688 | 53.3° | 10.95° | 44.78° | 19.87° | **14.5 / 59.2 / 26.3** |
| spatial_imagination_oo_mv | 714 | 52.2° | 12.09° | 43.64° | 18.35° | **16.3 / 58.9 / 24.8** |

Pooled:

| Pool | pairs | mean \|Δθ\| | pitch : yaw : roll (%) |
|---|---:|---:|---|
| spatial_relation (oc + oo) mv | 1,522 | 55.4° | 14.2 / 59.3 / 26.5 |
| spatial_imagination (oc + oo) mv | 1,402 | 52.7° | 15.4 / 59.1 / 25.5 |

### 5.2 Dominant-axis distribution (which axis carries the largest contribution)

| Task | yaw-dominant | pitch-dominant | roll-dominant |
|---|---:|---:|---:|
| obj_spatial_relation_oc_mv | 79.4% | 15.4% | 5.2% |
| obj_spatial_relation_oo_mv | 83.4% | 13.2% | 3.5% |
| spatial_imagination_oc_mv | 81.5% | 15.0% | 3.5% |
| spatial_imagination_oo_mv | 79.7% | 16.7% | 3.6% |

### 5.3 \|Δθ\| distribution

```
                                 0–10°  10–30°  30–60°  60–90°  90–120° 120–150° 150–180°
obj_spatial_relation_oc_mv        4.9%   26.1%   35.5%   17.8%    8.8%    4.5%    2.5%
obj_spatial_relation_oo_mv        4.8%   26.2%   30.6%   17.7%   10.2%    5.5%    4.8%
spatial_imagination_oc_mv         5.8%   28.2%   32.4%   17.6%    8.7%    3.9%    3.3%
spatial_imagination_oo_mv         6.0%   26.3%   35.4%   17.8%    7.7%    4.5%    2.2%
```

Most pairs land in 30°–90°; ~15% exceed 90°.

---

## 6. Implications

1. **Head capacity is not the bottleneck.** 24-anchor + residual spans full SO(3); pitch/roll are representable.
2. **Supervision is the bottleneck.** 0.13% pitch-labeled training, 0% roll-labeled training, 0% non-yaw eval.
3. **The constant-R collapse in [rotation_diversity_analysis.md](../rotation_diversity_analysis.md)** (angle to R̄ = 0.07° across 1050 MindCube tinybench samples) is consistent with this: absent per-axis supervision, the head has no reason to diversify.
4. **To activate non-yaw capacity** any of the following would help:
   - Train on the full SPAR_7M `view_change_infer` split (not just the 508 in train_10k); build held-out from it as pitch eval.
   - Add a MindCube-style bucket that uses D/E pose-derived `R_gt` and weights pitch/roll components of the geodesic loss (currently [train_rl.py:490-495](../../train_rl.py#L490-L495) comments note `w_pitch = w_roll = 0` by default, making shaping yaw-only).
   - Introduce a new relative-pose task directly from scene data (ScanNet / ScanNet++) that grounds a 3-DOF rotation to multi-view evidence.
5. **Evaluating** non-yaw learning requires new held-out benchmarks; none of the 9 current evals measure it.
