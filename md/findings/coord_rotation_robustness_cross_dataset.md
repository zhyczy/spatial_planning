# Coordinate Model Rotation Robustness — Cross-Dataset Study

**Date:** 2026-04-23
**Checkpoint:** `train_records/coordinate_no_cam_mindcube/step_1000` (retrained 2026-04-22 22:02)
**Datasets:** MindCube tinybench, SAT, SpinBench
**Rotations:** 22 per sample (I + 7 angles × {Rx, Ry, Rz})
**Script:** [`validate_coord_rotation_robustness.py`](../../validate_coord_rotation_robustness.py)
**Raw results:** `vis_results/R_sweep_mindcube/`, `vis_results/R_sweep_sat/`, `vis_results/R_sweep_spinbench/`

---

## TL;DR

Running the 22-rotation sweep on three benchmarks of increasing difficulty
confirms and extends the MindCube-only finding: the coordinate-supervised
model is **essentially rotation-invariant at QA time on all three datasets**,
with a characteristic axis-dependent geometric degradation in coord_mae that
is clean in-distribution (MindCube, SpinBench) and washed out under severe
distribution shift (SAT).

Key headline numbers:

| Dataset | N (questions) | I acc | acc spread | flip rate | coord_mae I→180° (worst axis) |
|---|---|---|---|---|---|
| **MindCube**  | 1050 | 95.33% | **0.76 pp** | **2.9%**  | 3.50× (Ry_180: 0.631 → 2.207) |
| **SAT**       | 150  | 67.33% | **2.00 pp** | **8.7%**  | 1.21× (Rx_120: 107.1 → 129.5) |
| **SpinBench** | 2661 | 60.72% | **1.20 pp** | **13.4%** | 2.14× (Ry_150: 1.175 → 2.509) |

- **Accuracy spread ≤ 2pp** on every dataset, independent of R — the LM does
  not use the rotated 4D-RoPE xyz channel for answer decoding.
- **Flip rate (share of questions whose letter ever changes across 22 R)**
  grows with task difficulty: 2.9% → 8.7% → 13.4%. But flips cancel pairwise,
  so accuracy stays flat.
- **Axis ranking Rx (safest) < Ry/Rz (most disruptive)** is consistent across
  the two in-distribution datasets.

---

## 1. Accuracy invariance — the LM does not see R

Full per-(axis, angle) accuracy table (Δ from I in percentage points):

| axis | deg | MindCube Δacc | SAT Δacc | SpinBench Δacc |
|---|---:|---:|---:|---:|
| Rx |  30 | +0.10 |  0.00 |  0.00 |
| Rx |  60 |  0.00 |  0.00 |  0.00 |
| Rx |  90 |  0.00 |  0.00 | −0.15 |
| Rx | 120 |  0.00 |  0.00 | −0.18 |
| Rx | 150 | −0.19 | +0.67 | +0.15 |
| Rx | 180 | −0.10 | **+1.33** | +0.26 |
| Rx | 270 | −0.19 | **+1.33** | +0.22 |
| Ry |  30 |  0.00 | +0.67 | +0.04 |
| Ry |  60 | −0.10 | **+1.33** | +0.55 |
| Ry |  90 | −0.67 | +0.67 | +0.47 |
| Ry | 120 | −0.67 | +0.67 | +0.55 |
| Ry | 150 | −0.57 | +0.67 | +0.26 |
| Ry | 180 | −0.48 | −0.67 | −0.26 |
| Ry | 270 | −0.48 | +0.67 | −0.37 |
| Rz |  30 |  0.00 | −0.67 | −0.15 |
| Rz |  60 | −0.57 |  0.00 | +0.07 |
| Rz |  90 | −0.19 | −0.67 | +0.07 |
| Rz | 120 | −0.67 |  0.00 | +0.22 |
| Rz | 150 | −0.48 |  0.00 | +0.22 |
| Rz | 180 | −0.38 |  0.00 | −0.07 |
| Rz | 270 | −0.57 | −0.67 | **−0.66** |

Every cell is within ±2 pp of I (and ±1 pp on the two larger datasets).
This is within the sampling noise of the evaluation set — there is no R
that systematically wins or loses on any dataset.

### Why SAT's small absolute numbers hide biggest relative noise
SAT's N = 150 means a **single flipped sample shifts accuracy by 0.67 pp**.
So its +1.33 pp peaks are 2 flips going the "good" way and its −0.67 pp dips
are 1 flip going the "bad" way — none of this is a real trend. The 8.7% flip
rate at N = 150 → only 13 questions ever flip across all 22 R; the visible
Δacc values are just which direction those flips happen to point at each R.

---

## 2. Question-level flip rate grows with difficulty — but cancels

How many **unique (image_set, question)** pairs ever produce a different
letter across the 22 rotations?

| Dataset | Questions | Flipped | % flipped | I acc |
|---|---|---|---|---|
| MindCube  | 1050 |  30 |  2.9% | 95.33% |
| SAT       |  150 |  13 |  8.7% | 67.33% |
| SpinBench | 2661 | 357 | 13.4% | 60.72% |

Flip rate scales with **1 − accuracy** — harder questions are more
borderline, more borderline questions are more sensitive to perturbations
like a rotated RoPE signal.

But the flips are **pairwise symmetric**: for every "I correct → R wrong"
there is roughly a matching "I wrong → R correct". On MindCube this was
explicitly 3 gains / 3 losses at Ry_180 (see
[`coord_rotation_robustness.md`](coord_rotation_robustness.md)); the same
pattern holds across all three datasets, explaining why the accuracy
spread is bounded at ≤ 2 pp even with 13% flip rate on SpinBench.

**Interpretation:** rotation does not *degrade* reasoning — it *randomises*
answers on the subset of questions the model was already uncertain about.
Whatever weak signal the LM was using for those borderline cases is
evidently unrelated to the 4D-RoPE xyz channel.

---

## 3. Which rotations trigger flips — Ry / Rz dominate, Rx barely matters

Top 5 R's that disagree with I on each dataset (count = # of questions
flipping from I's answer at that R):

| Rank | MindCube | SAT | SpinBench |
|---|---|---|---|
| 1 | **Rz_150** (10) | **Ry_090** (7) | **Rz_120** (163) |
| 2 | **Ry_150** (9)  | **Ry_120** (5) | **Ry_090** (160) |
| 3 | **Ry_090** (9)  | **Ry_150** (5) | **Rz_180** (157) |
| 4 | Rz_270 (9)      | Rz_180 (4)     | Ry_180 (156) |
| 5 | Ry_120 (8)      | Ry_060 (4)     | Ry_120 (155) |

**Rx appears zero times in the top-5 of any dataset.** Ry and Rz dominate.
This aligns with the coordinate-channel analysis (camera frame +x right,
+y down, +z depth):

| Rotation | Channels mixed | Channel preserved | Intuition |
|---|---|---|---|
| **Rx** (pitch) | y ↔ z | x (horizontal) | vertical ↔ depth swap |
| **Ry** (yaw)   | x ↔ z | y (vertical)   | **horizontal ↔ depth swap** |
| **Rz** (roll)  | x ↔ y | z (depth)      | image-plane rotation |

Ry directly attacks the horizontal-depth geometry that underlies most
spatial QA ("left of the chair", "in front of the table"), so it flips the
most borderline questions. Rz rotates the visual frame (affecting
"left/right"/"up/down" lexical cues in the image content encoded by vision
tokens), so it also flips many. Rx mostly corrupts the vertical-depth
relationship, which is less often queried.

---

## 4. coord_mae geometry — clean in-distribution, collapsed on SAT

Peak coord_mae multiplier relative to I (and the axis that produces it):

| Dataset | I mae | Peak | Multiplier | Peak axis | Clean monotonic? |
|---|---|---|---|---|---|
| MindCube  | 0.631 | 2.207 | **3.50×** | Ry_180 | **Yes** (all 3 axes) |
| SAT       | 107.1 | 129.5 | **1.21×** | Rx_120 | No (scattered) |
| SpinBench | 1.175 | 2.509 | **2.14×** | Ry_150 | **Yes** (all 3 axes) |

Axis ranking at 180°:

| Dataset | Rz_180 mae | Rx_180 mae | Ry_180 mae | Ranking |
|---|---|---|---|---|
| MindCube  | 1.475  | 1.995  | **2.207**  | **Rz < Rx < Ry** ✅ |
| SAT       | 108.18 | 109.86 | 109.93     | All within 2% — no geometric signal |
| SpinBench | 1.451  | 2.415  | **2.360**  | **Rz < Ry ≈ Rx** |

On MindCube the ranking is textbook: Rz preserves depth (mildest),
Ry destroys horizontal↔depth correlation (worst), Rx in between.

On SpinBench, Ry and Rx are tied (both ~2.4) — in multi-object "is X left of
Y?" scenes the model's coord_head predictions seem equally perturbed by
horizontal↔depth swaps and vertical↔depth swaps. Rz still safest.

On SAT the **coord_head is out-of-distribution even at I** (baseline mae 107
vs MindCube's 0.63 — a 170× scale difference from different reconstruction
geometry and scene scale). The R perturbation (~20% relative change) is
dwarfed by the out-of-domain base noise; no geometric pattern survives.

---

## 5. Combined diagnosis — confirmed across three datasets

Two independent pieces of evidence **agree on every dataset**:

**(a) QA-path evidence**: Accuracy varies ≤ 2 pp across 22 rotations on every
dataset, with no systematic direction. The LM is not decoding answers from
the rotated 4D-RoPE xyz channel.

**(b) coord_head evidence**: In-distribution mae rises 2–3.5× under 180°
rotation with clean monotonic angle dependence and the expected
depth-channel-preservation ranking (Rz cheapest, Ry most damaging). If
coord_head were really decoding the rotated xyz, rotating both input and GT
should *preserve* mae. The blow-up proves coord_head is emitting the
training-frame xyz regardless of R — **frame-locked to view_0000**.

The invariance is not an artifact of MindCube being too easy; it holds at
60% accuracy on SpinBench and under severe distribution shift on SAT. The
flip rate does rise with difficulty (2.9 → 13.4%), but those flips are
zero-sum — the model treats R as noise on borderline samples, not as a
usable feature.

**Operational takeaway:** the xyz → 4D-RoPE pathway is decorative in this
training recipe, on all three benchmarks. Any rotation-aware reasoning would
require (a) a loss function that forces R into the decoding path (e.g.
direct supervision on gt_rotation or a contrastive loss across rotated
scenes), or (b) an LM architecture where attention is forced to attend to
the RoPE xyz channels (rather than optimising around them).

---

## 6. Practical implications for the rotation / alternate pipeline

The earlier studies (see [`rotation_diversity_analysis.md`](../rotation_diversity_analysis.md)
and the original MindCube-only robustness write-up) had established:

- **rotation_alternate** (xyz-RoPE + rotation_enc) collapses to a single
  canonical R ≈ Rot_ZYX(106.5°, 25.1°, 54.0°) — no per-sample rotation
  signal in the gradient.
- **coordinate_no_cam** (xyz-RoPE + coord_head, no rotation_enc) is
  frame-locked on MindCube.

This cross-dataset run adds:

- Frame-locking is **not MindCube-specific**. Across three benchmarks with
  different scene types (MindCube: synthetic cube; SAT: single-frame iPhone
  with depth sensor; SpinBench: multi-view objects), the xyz channel is
  ignored by the LM.
- Difficulty changes the **symptoms** (more borderline flips) but not the
  **disease** (LM doesn't use xyz for answers).

This narrows the remaining plausible claim for xyz supervision to *"it
helps training stability / convergence"*, not *"it is used at inference"*.
That should be tested with an ablation: retrain with xyz → all-zeros at
both training and eval, and compare final accuracy curves. If it matches
the current model, xyz is entirely vestigial and can be removed.

---

## Files

- Cross-dataset sweep results: `vis_results/R_sweep_mindcube/`,
  `vis_results/R_sweep_sat/`, `vis_results/R_sweep_spinbench/`
  - Each dir: `R_sweep.json` (all rows), `metrics_per_R.json` (per-R summary),
    `results_cuda{0-7}.json` (per-worker), `configuration.json`
- Sweep runner: [`../../validate_coord_rotation_robustness.py`](../../validate_coord_rotation_robustness.py)
- MindCube-only report: [`coord_rotation_robustness.md`](coord_rotation_robustness.md)
- Smoke report: [`../coord_rotation_robustness_smoke.md`](../coord_rotation_robustness_smoke.md)
- Related: [`../rotation_diversity_analysis.md`](../rotation_diversity_analysis.md),
  [`../data/rotation_axes.md`](../data/rotation_axes.md)
