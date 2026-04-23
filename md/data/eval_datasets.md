# Evaluation Datasets — Task Taxonomy & Rotation Coverage

**Date:** 2026-04-22
**Scope:** `spatial_planning/datasets/evaluation/`

Cross-references:
- Training side: [train_datasets.md](train_datasets.md)
- Rotation anchor details: [rotation_axes.md](rotation_axes.md)

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
| vsibench | `test.jsonl` | 5,130 |

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

### 2.8 vsibench (N = 5,130) — video spatial understanding

| N | question_type | Task |
|---:|---|---|
| 953 | object_size_estimation | Object dimension (cm) |
| 834 | object_abs_distance | Object-object distance (m) |
| 710 | object_rel_distance | Closest object to X |
| 618 | obj_appearance_order | First-appearance order in video |
| 565 | object_counting | Count objects |
| 378 | object_rel_direction_medium | left/right/back (≥135° threshold) |
| 373 | object_rel_direction_hard | front-left / front-right / back-left / back-right |
| 288 | room_size_estimation | Room area |
| 217 | object_rel_direction_easy | Binary left/right |
| 194 | route_planning | Fill-in turn back / left / right |

96% is metric/counting/direction-word; 194 `route_planning` is the only yaw-like block.

---

## 3. Capability dimensions (pooled across all 9 evals)

| Dimension | Approx N | Typical datasets |
|---|---:|---|
| Continuous-angle yaw reasoning (N°) | ~3,500 | SAT val 2,136; spinbench ≈1,100; MMSIBench ≈150 |
| Discrete direction words (front/back/left/right/up/down) | ~7,000 | vsibench 968; MMSIBench 527; SPARBench 2,500 |
| Cross-view identity / canonical-view selection | ~1,500 | All_Angles_Bench 735; spinbench ≈400; SPARBench imagination 701 |
| Translation / motion direction | ~17,500 | MindCube 17,000; MMSIBench 150; SAT obj_movement 647; vsibench 194 |
| Metric (size / distance / count) | ~5,000 | vsibench 3,368; All_Angles_Bench 745; RoboSpatial 105 |
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
| vsibench | 5,130 | 194 | 0 | 0 |

† 3 hits are scene-context descriptions ("standing on stairs, looking down"), not prediction labels.

---

## 5. Rotation-axis coverage in eval

Across **all 9 evaluation benchmarks, the only rotation axis probed is yaw**. Zero pitch / roll prediction tasks exist. Consequences:

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
