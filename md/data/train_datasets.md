# Training Datasets — Inventory & Rotation Supervision

**Date:** 2026-04-22
**Scope:** `spatial_planning/datasets/train/{MindCube, SAT, SPAR_7M}/`

Cross-references:
- Rotation anchor structure: [rotation_axes.md](rotation_axes.md)
- Evaluation side: [eval_datasets.md](eval_datasets.md)

---

## 1. MindCube — `MindCube/MindCube_train.jsonl`

- **N = 10,000** samples, keyed by `id`, 3D per-sample data under `3d_results/<id>/view_XXXX/camera_pose.npy`.
- World frame = `view_0000` camera frame (i.e. `camera_pose[0] ≈ I`). See memory `project_mindcube_world_frame`.

### 1.1 Bucket taxonomy (text-derived, full 10k distribution)

Priority chain **B > H > D > E > A > C** (see [src/dataset/train_dataset.py:1000-1011](../../src/dataset/train_dataset.py#L1000-L1011)).

| Bucket | Tag | N | Trigger | R_gt derivation |
|---|---|---:|---|---|
| B | rotation_described | 864 | `type ∈ {four_view, three_view, two_view_{cw,ccw,opp}}` **and** `"same direction as shown in image N"` | `R_gt = (C2W[0]ᵀ · C2W[N]) · R_turn`, `R_turn ∈ {I, R_bins[5/6/16]}` selected from tail `turn 90/180 deg {left/right/around}` |
| H | hypothetical_action | 3,004 | non-B-type + `"image N"` + `"(then I\|, I) turn {left\|right}"` | `R_gt = (C2W[0]ᵀ · C2W[N]) · R_bins[{5,6}]` (±90° yaw) |
| D | viewpoint_anchor | 2,943 | `"from the viewpoint (presented) in image N"` | `R_gt = C2W[0]ᵀ · C2W[N]` (continuous, no quantization) |
| E | multi_view_scene | 1,316 | leftover `"image N"` anchor (no turn verb) | same as D |
| A | motion_query | 1,302 | `"in which direction did I move"` | **`R_gt = I` hard-coded** (no pose read) |
| C | residue | 571 | nothing matches | `R_gt = None`, `has_gt = False` |

Counts verified by running the extractors over the full 10k JSONL; match [src/dataset/train_dataset.py:899](../../src/dataset/train_dataset.py#L899) comment ("571 in training set" for E-pos-obj).

### 1.2 What each bucket actually learns

- **B/H**: true 3D relative pose (anchor half) composed with the precisely-defined 90°/180° yaw. Anchor is **not** snapped to `R_bins`; only the action verb is discrete.
- **D/E**: true 3D relative pose `C2W[0]ᵀ · C2W[N]`, completely continuous.
- **A**: no rotation label. `R_gt = I` is a "stay in view-0 frame" prior. The translation-direction MCQ is supervised via ordinary `lm_loss` per anchor.
- **C**: no rotation supervision at all. 571 "positioned where X" (E-pos-obj) samples — pose-independent, anchored to an object rather than a viewpoint.

### 1.3 E-pos-obj subset

Extracted to `MindCube_train.E_pos_obj.jsonl` (571 lines, original schema preserved) using regex `"positioned\s+where"` — matches the C-bucket residue exactly (zero cross-contamination with A/B/D/E/H under text classification).

### 1.4 Rotation axes in labels (strict regex)

All 10k entries: **yaw only**. Zero pitch/roll keyword hits in question or answer. Matches SAT; contrasts with SPAR.

---

## 2. SAT — `SAT/train_36k.json`

- **N = 36,810** samples, list-of-dicts schema.
- Question types: `other` 24,540, plus 5 spatial types × 2,454 each (`obj_movement`, `action_sequence`, `action_consequence`, `goal_aim`, `perspective`).

### 2.1 Rotation axes in labels

| Axis | Count | Notes |
|---|---:|---|
| yaw | 4,821 (13.1%) | Concentrated in `action_consequence / action_sequence / perspective` — phrases like "turn 40°", "rotated left", "turned right by 90°" |
| pitch | 0 | No tilt/look-up/down labels |
| roll | 0 | No roll/bank labels |

`action_sequence` GT vocabulary: only `rotated left / rotated right / moved forward / moved left / moved right / did not move` (pure 2D SE(2)).

### 2.2 Dataset-side rotation GT

`SAT_Train_Dataset_Rotation` at [src/dataset/train_dataset.py:1024+](../../src/dataset/train_dataset.py#L1024) reads `pts3d.npy` per view but does **not** derive an R_gt like MindCube does — SAT's rotation supervision in `train_rl.py` currently relies on the bucket machinery being a no-op for SAT (bucket = "C"). See [train_rl.py:515-522](../../train_rl.py#L515-L522) for the per-bucket weight lookup.

---

## 3. SPAR — `SPAR_7M/spar/train_10k.json`

- **N = 10,000** samples, 14 sub-tasks.
- **Only training set with pitch supervision.**

### 3.1 Task distribution

| Task | N | Has explicit rotation axis label? |
|---|---:|---|
| depth_prediction_oc_mv | 998 | — |
| distance_infer_center_oo_mv | 986 | — |
| depth_prediction_oo_mv | 959 | — |
| distance_prediction_oo_mv | 929 | — |
| distance_prediction_oc_mv | 905 | — |
| distance_infer_center_oc_mv | 880 | — |
| position_matching | 851 | — |
| obj_spatial_relation_oo_mv | 750 | — |
| camera_motion_infer | 649 | no axis — 2D image-plane observer projection |
| obj_spatial_relation_oc_mv | 552 | — |
| spatial_imagination_oc_mv | 512 | — |
| **view_change_infer** | **508** | **yes: yaw + pitch** |
| spatial_imagination_map_mv | 264 | — |
| spatial_imagination_oo_mv | 257 | — |

### 3.2 `view_change_infer` answer template

Instruction verbatim: `rotate_<down_or_up>:<degrees>, rotate_<right_or_left>:<degrees>`.

Strict regex hits across the full 10k:
| Axis | Count | Source |
|---|---:|---|
| yaw | 116 | `Rotate left/right`, `Turn left N degrees` |
| pitch | 71 | `Tilt up/down by N degrees`, `rotate_up/down` |
| roll | 0 | absent |

All hits lie inside `view_change_infer`. The other 9,492 samples have no explicit axis labels in their questions / answers.

---

## 4. Cross-dataset summary — rotation label coverage

| Dataset | File | N | Yaw labels | Pitch labels | Roll labels |
|---|---|---:|---:|---:|---:|
| MindCube | `MindCube_train.jsonl` | 10,000 | via B/H/D/E (≈ 8,127 pose-derivable) | 0 | 0 |
| SAT | `train_36k.json` | 36,810 | 4,821 text-level | 0 | 0 |
| SPAR | `spar/train_10k.json` | 10,000 | 116 (view_change_infer only) | 71 (view_change_infer only) | 0 |

**Training total with pitch supervision: 71 / 56,810 ≈ 0.13%.**
**Training total with roll supervision: 0.**

---

## 5. Practical implications for `train_rl.py`

1. `w_rot` shaping in [train_rl.py:514-522](../../train_rl.py#L514-L522) only fires for MindCube B/H/D/E (≈ 8,127 / 56,810 samples). SAT + SPAR default to bucket "C" (no R_gt) — rotation reward degenerates to `-w_lm · lm_loss(k)` only.
2. `w_trans` shaping fires for MindCube A (1,302 samples) with `R_gt = I` — pushes policy toward `R_bins[0]`.
3. Even if `view_change_infer` pitch labels are added (via a new bucket), no current evaluation exercises pitch — held-out eval must be synthesized. See [eval_datasets.md §5](eval_datasets.md#5-rotation-axis-coverage-in-eval).
4. 24-anchor head's pitch/roll bins (R_bins[1-2,5-14,17-23]) receive no gradient from MindCube/SAT and near-zero from SPAR. See [rotation_axes.md §2](rotation_axes.md#2-the-24-anchor-group).
