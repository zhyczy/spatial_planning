# Evaluation Datasets — Task Taxonomy & Rotation Coverage

**Date:** 2026-04-22 (last updated 2026-04-27)
**Scope:** `spatial_planning/datasets/evaluation/`

Cross-references:
- Training side: [train_datasets.md](train_datasets.md)
- Rotation anchor details: [rotation_axes.md](rotation_axes.md)
- Spatial-attention model design: [../model_design/spatial_attention.md](../model_design/spatial_attention.md)

---

## 0. Evaluation pipeline composition

`evaluation.py` supports **11 datasets** via `--dataset`. Driver scripts
(`scripts/evaluate.sh`, `curve_evaluation.{py,sh}`) default to a 7-dataset
sweep:
`mindcube · sat_real · spinbench · robospatial · viewspatial · omnispatial_pt · embspatial`.
The other 4 (`mmsibench · sat · sparbench_{multi_view,single_view,mv}`) are
opt-in via `--datasets`.

`All_Angles_Bench` is present on disk and covered in §2.1 for taxonomy /
rotation-coverage analysis but is **not** wired into `evaluation.py`.
Three datasets that are wired into `evaluation.py` — `viewspatial`,
`omnispatial_pt`, `embspatial`, plus the combined `sparbench_mv` —
do not yet have entries in §2 (only their answer-format properties are
covered in §7).

---

## 1. Inventory

| Benchmark | Main file | N |
|---|---|---:|
| All_Angles_Bench | `test_processed.jsonl` | 2,132 |
| MindCube | `MindCube.jsonl` | 21,154 |
| MMSIBench | `data/test_data_final.json` | 1,000 |
| RoboSpatial | `test_processed.jsonl` | 350 |
| SAT (val) | `val.json` | 4,001 |
| SPARBench (mv) | `sparbench_multi_view.json` | 1,462 |
| SPARBench (sv) | `sparbench_single_view.json` | 1,038 |
| spinbench | `test.jsonl` | 2,739 |

---

## 2. Per-benchmark task taxonomy

### 2.1 All_Angles_Bench (N = 2,132) — multi-view scene QA

| N | category | Core capability |
|---:|---|---|
| 494 | relative_distance | Which view has closer/farther object |
| 476 | manipulation | "If object rotated 90° clockwise in View 1, what happens in View 2?" |
| 383 | attribute_identification | Cross-view object identity |
| 352 | relative_direction | Object facing direction in one view given another |
| 251 | counting | Multi-view counting |
| 176 | camera_pose_estimation | Reconstruct top-down camera layout from views |

**Rotation-axis labels:** 8 yaw (all `manipulation`). `relative_direction` and `camera_pose_estimation` are implicitly 3D-rotation tasks but without axis keywords.

### 2.2 MindCube (N = 21,154) — same-scene multi-view QA

By `type`:

| N | type | Task |
|---:|---|---|
| 4,284 × 4 | `0/1/2/3_frame` | **Translation direction 4-choice** (Directly left / Directly right / Diag-fwd-left / Diag-fwd-right). Answer is the camera's own move direction between two views. |
| 1,068 | general | Scene/attribute QA |
| 960 + 625 + 284 | `2 / 3 / 1` (frame count) | Same translation-direction MCQ with different image counts |
| 360 + 345 | `four_view / three_view` | Spatial relations under known 90° yaw views |
| 220 + 120 + 36 | `two_view_{cw,ccw,opposite}` | Spatial relations under one specified 90° / 180° yaw turn |

**The dominant MindCube task (~17,000 samples) is translation-direction MCQ, not yaw prediction.** Yaw appears only as scene-setup context in the few hundred `two_view_*` / `four_view` / `three_view` variants.

### 2.3 MMSIBench (N = 1,000) — multi-image spatial reasoning

| N | type | Task |
|---:|---|---|
| 198 | MSR | Multi-step compound spatial reasoning |
| 94 | Positional Relationship (Obj.–Obj.) | Cardinal-direction inference (N/S/E/W) |
| 93 | Positional Relationship (Cam.–Cam.) | Second-camera position relative to first |
| 86 | Positional Relationship (Cam.–Obj.) | Object direction relative to camera |
| 85 | Positional Relationship (Obj.–Reg.) | Object-in-region direction |
| 83 | Positional Relationship (Cam.–Reg.) | Camera-relative region direction |
| 81 | Positional Relationship (Reg.–Reg.) | Region-to-region direction |
| 76 | Motion (Obj.) | Object displacement direction across frames |
| 74 | Motion (Cam.) | Camera displacement / rotation direction |
| 66 | Attribute (Appr.) | Object attributes |
| 64 | Attribute (Meas.) | Size comparison |

**Rotation-axis labels:** 64 yaw (mostly `Motion (Cam.)` / `MSR`), 3 "looking down" (pitch scene-context, not prediction labels), 0 roll.

### 2.4 RoboSpatial (N = 350) — single-image robot spatial grounding

| N | category | Task |
|---:|---|---|
| 123 | configuration | Yes/No spatial-relation |
| 122 | context | Pointing: output (x, y) points in empty-space regions |
| 105 | compatibility | Yes/No fit-checking |

No rotation labels.

### 2.5 SAT val (N = 4,001)

| N | question_type | Task |
|---:|---|---|
| 1,336 | action_consequence | "If I turn left by N°, will I face X?" |
| 779 | perspective | "If I move to X and turn N°, does Y get closer?" |
| 647 | obj_movement | Object-moved detection |
| 647 | action_sequence | Infer camera action from two frames (rotated/moved) |
| 592 | goal_aim | Which way to turn to face target |

4/5 categories probe yaw (often with explicit N° values: 40°, 90°, 180°). No pitch/roll.

### 2.6 SPARBench (mv = 1,462, sv = 1,038)

mv:

| N | task | Task |
|---:|---|---|
| 400 | obj_spatial_relation_oc_mv | Object position relative to observer's main view |
| 361 | obj_spatial_relation_oo_mv | Object-to-object relation in observer frame |
| 357 | spatial_imagination_oo_mv | How A-B relation changes after observer moves + reorients |
| 344 | spatial_imagination_oc_mv | How observer-object relation changes |

sv mirrors with single images.

**Zero rotation-axis keywords in labels.** Answer vocabulary is discrete spatial-relation words (front/back/left/right/above/below/closer/further). Note: SPAR-train's `view_change_infer` (which has `rotate_up/down` pitch labels) is **deliberately excluded** from SPARBench — bench only includes spatial-relation / imagination tasks.

### 2.7 spinbench (N = 2,739) — fine-grained rotation / viewpoint tests

By `metadata.task_type` (top families):

| Family | Task | Rotation-related? |
|---|---|---|
| `infinigen_spatial_relation_grounding_*` (~590) | left/right/far/near in scene | No — static placement |
| `infinigen_spatial_relationship_front_behind` (~276) | front/behind static | No |
| `infinigen_spatial_relationship_dynamic_{front_back,left_right}` (~156) | object displacement direction | Translation, not rotation |
| `infinigen_mental_rotation` + `object_mental_rotation` + `car_mental_rotation` (218) | "Object rotates N°, pick correct image" | **Yes, yaw** |
| `{object,car}_rotation_classification_*` (189) | "CW or CCW between two views" | **Yes, yaw** |
| `{object,car,face}_canonical_view_selection_*` (~295) | Select front/back/left/right view | **Yes, yaw** |
| `infinigen_rotation_selection_*` (~300) | Select rotated-view match, with/without occlusion | **Yes, yaw** |
| `face_rotation_classification_*` (164) | Head-turn direction | **Yes, yaw** |
| `infinigen_spatial_relation_transformation_*` (290) | Scene relation under viewpoint change | **Yes, yaw** |
| `{object,car,face}_identity_*` (~300) | Which image shows different object | Implicit viewpoint |

**Largest rotation-focused eval** (~1,100 yaw-related samples) but strictly yaw / view-azimuth. No pitch/roll.

---

## 3. Capability dimensions (pooled across the 8 evals in §1)

| Dimension | Approx N | Typical datasets |
|---|---:|---|
| Continuous-angle yaw reasoning (N°) | ~3,500 | SAT val 2,136; spinbench ≈1,100; MMSIBench ≈150 |
| Discrete direction words (front/back/left/right/up/down) | ~6,000 | MMSIBench 527; SPARBench 2,500 |
| Cross-view identity / canonical-view selection | ~1,500 | All_Angles_Bench 735; spinbench ≈400; SPARBench imagination 701 |
| Translation / motion direction | ~17,500 | MindCube 17,000; MMSIBench 150; SAT obj_movement 647 |
| Metric (size / distance / count) | ~1,650 | All_Angles_Bench 745; RoboSpatial 105 |
| Pointing / yes-no / attributes | ~700 | RoboSpatial 350; MMSIBench 130; SAT subsets |

---

## 4. Rotation-axis keyword hits per eval (strict regex, noun-exclusion)

Regex conventions: yaw = {turn/rotate left|right, clockwise/counterclockwise, heading, azimuth}; pitch = {tilt/look up|down, pitch, rotate_up/down, elevation}; roll = {roll left|right|sideways, bank, lean}.

| Dataset | N | yaw | pitch | roll |
|---|---:|---:|---:|---:|
| All_Angles_Bench | 2,132 | 8 | 0 | 0 |
| MindCube | 21,154 | 6,327 | 0 | 0 |
| MMSIBench | 1,000 | 64 | 3† | 0 |
| RoboSpatial | 350 | 0 | 0 | 0 |
| SAT val | 4,001 | 2,136 | 0 | 0 |
| SPARBench mv | 1,462 | 0 | 0 | 0 |
| SPARBench sv | 1,038 | 0 | 0 | 0 |
| spinbench | 2,739 | 377 | 0 | 0 |

† 3 hits are scene-context descriptions ("standing on stairs, looking down"), not prediction labels.

---

## 5. Rotation-axis coverage in eval

Across **all 8 evaluation benchmarks, the only rotation axis probed is yaw**. Zero pitch / roll prediction tasks exist. Consequences:

1. Even if training data grows pitch/roll labels (e.g. full SPAR_7M `view_change_infer`), there is no held-out eval to measure generalization.
2. `CameraTokenRotationEncoderRL`'s 24-anchor head can collapse its non-yaw bins (15 / 24 mixed-axis anchors) to a constant without any eval penalty — consistent with the dataset-constant-R finding in [../rotation_diversity_analysis.md](../rotation_diversity_analysis.md).
3. Adding a pitch/roll eval requires synthesizing held-out from SPAR-train `view_change_infer` or creating a relative-pose task from ScanNet/ScanNet++.

---

## 6. MindCube — clarification on "yaw" count

The 6,327 yaw-keyword hits in MindCube are misleading. Breakdown:

| Source of yaw keyword | Approx N | Is yaw being *predicted*? |
|---|---:|---|
| Intro text "after turning 90 degrees" (scene setup) | ~5,700 | No — just describes how the views were captured |
| `two_view_{cw,ccw,opposite}` / `four_view` / `three_view` questions | ~700 | Scene relation under fixed yaw — yaw is input, answer is spatial relation |
| Actual "which way did I rotate" style | 0 | Not present in MindCube eval |

The dominant eval task in MindCube (~17,000 samples under `0/1/2/3_frame` etc.) asks for **translation direction**, not rotation. See [train_datasets.md §1.2](train_datasets.md#12-what-each-bucket-actually-learns) for the training-side A-bucket counterpart.

---

## 7. Answer formats (verified empirically against on-disk files)

Every dataset wired into `evaluation.py` emits the loader's `answer` field in
one of two shapes — single ASCII letter, or numeric string. `evaluation.py`
dispatches scoring via the per-sample `format_type` field
(`select` → `extract_answer_letter` + exact match;
 `fill` → `extract_answer_number` + `_mra_score`).

| Dataset | n | format | Verification |
|---|---:|---|---|
| mmsibench             | 1,000 | 100% single A/B/C/D | 265 A · 255 C · 250 B · 230 D |
| mindcube (tinybench)  | 1,050 | 100% single A/B/C/D | 361 B · 342 A · 210 C · 137 D |
| spinbench             | 2,739 | 100% single A/B/C/D | 1131 B · 1130 A · 391 C · 87 D |
| sat                   |   150 | 100% single letter (all 'A' — likely a pruned subset) | suspicious distribution |
| sparbench_multi_view  | 1,462 | 100% single A/B/C/D | 382 A · 361 B · 361 C · 358 D |
| sparbench_single_view | 1,038 | 100% single A/B/C/D | 273 D · 257 B · 254 A · 254 C |
| sparbench_mv          | 3,152 | **mixed** — 1,798 `select` + 1,354 `fill` | letters A–D for select; numerics like `1.2`, `2.6` for fill |
| viewspatial           | 5,712 | 100% single A/B/C/D | letter parsed from `"X. text"` answer string |
| omnispatial_pt        |   561 | 100% single A/B/C/D | derived from int answer-index |
| embspatial            | 3,640 | 100% single A/B/C/D | derived from int answer-index over 4 options |
| robospatial           |   350 | yes/no + (x,y) pointing | dispatched via `format_type="robospatial"` |

**MindCube + SpinBench** are also the two datasets the inline eval inside
[../../train_atten.py](../../train_atten.py) runs after each `eval_steps`. The
inline eval reports `{ds}_acc` as **first-token argmax accuracy** —
implicitly assumes the answer is exactly one token. Since every
MindCube/SpinBench answer is a single capital letter and Qwen tokenizes
A/B/C/D as single tokens, **first-token accuracy ≡ answer accuracy**
on these two datasets. This equivalence breaks if a future eval is wired in
where answers are multi-token (e.g. parenthesized `(A)`, free-form text,
multi-digit numerics, or any tokenizer that introduces a leading-space
variant) — at that point switch to all-position match or `.generate()` +
exact match. See [../model_design/spatial_attention.md §10](../model_design/spatial_attention.md#10-open-questions--future-work).

`evaluation.py` itself does NOT have this issue — it always uses
`.generate()` followed by regex extraction, so multi-token answers are
handled correctly when present.
