# 3D-Reconstruction Data Preprocessing Pipeline

**Date:** 2026-04-29
**Scope:** Image preprocessing chain from parquet/PNG input to per-pixel pts3d on disk, applied uniformly across single- and multi-image entries (VST training data + eval benchmarks).

Cross-references:
- MapAnything reshape internals: [../base/mapanything.md](../base/mapanything.md)
- Qwen3.5-VL image processing: [../base/qwen3.5.md §2](../base/qwen3.5.md#2-vision-side--qwen3_5visionmodel-and-image-input-pipeline)
- Source: [`coord_esti.py`](../../coord_esti.py), [`src/data_process/estimate_3D.py`](../../src/data_process/estimate_3D.py)

---

## 0. TL;DR

Every reconstruction sample — single-view (1 image) or multi-view (≥2 images), training (VST) or eval (MindCube / SAT / SPARBench / SpinBench / …) — flows through the **same Step-A normalization** before reconstruction. Pts3d lands on disk at the reconstruction model's native output shape, which by Step A is always one of 10 entries in `RESOLUTION_MAPPINGS[518]`.

| Step | Operation | Where |
|---|---|---|
| **A. MapAnything reshape** | `find_closest_aspect_ratio(W/H, 518)` + `crop_resize_if_necessary` (Lanczos rescale + center crop) | Lands input on one of 10 shapes from `RESOLUTION_MAPPINGS[518]` (max-side ≤ 518) |
| **B. Reconstruction model** | Depth Pro (single-view) or MapAnything (multi-view) | pts3d emerges at the Step-A shape (numpy H × W × 3) |
| **C. Qwen-aligned resample (deferred)** | `_resample_to_qwen_aligned`: bilinear pts3d/depth, nearest mask, LANCZOS image, intrinsics × scale | **Not applied at save time** (default `qwen_align=False`). The dataloader applies it at load time, so the same `3d_results/` can feed downstream models with different patch grids (Qwen3.5-VL / SigLIP / DINOv2 / VGGT …). Pass `--qwen_align` to the CLI to bake /28 alignment into saved files. |

The pipeline guarantees every `image.png` and `pts3d.npy` on disk are at **the exact same (H, W)** — pixel-aligned. Step A is applied inside `CoordEstimator`, **not** in iterators, so adding new datasets means only writing an iterator that yields raw decoded RGB.

The **only difference between si_*** and **mi_*** is which model runs in Step B. Step A is identical and produces the same shape distribution.

---

## 1. Why this design

### 1.1 Reconstruction model contract — Step A

MapAnything (multi-view) is a Vision Transformer with `dinov2_vitg14` backbone. Its position embeddings were trained on a fixed 10-shape table (`RESOLUTION_MAPPINGS[518]`), all with `max(W, H) = 518`. Inputs outside this table are out-of-distribution; `crop_resize_if_necessary` is the contract enforcer.

Depth Pro (single-view) is more flexible — it has its own internal preprocessing to 1536² and resamples back to input size — but we apply Step A to single-view inputs as well so that:

1. **Cross-pipeline shape uniformity**: si_*** and mi_*** produce pts3d in the same 10-shape pool; the LLM sees a single, narrow shape distribution at train and eval time.
2. **Bounded disk + compute**: high-res eval images (1296×968 for SPARBench / ViewSpatial) get capped at max-side 518 before reconstruction, which is ~6× less area than parquet-native.
3. **Training/eval consistency**: shapes the LLM saw in training (always /14, max 518) match shapes it sees at eval.

Step A lives in **two places inside CoordEstimator** (and only there):
- `_views_from_arrays` ([coord_esti.py:507-548](../../coord_esti.py#L507-L548)) — for multi-view, called from `_estimate_multi_view`
- `_estimate_single_view` ([coord_esti.py:374-388](../../coord_esti.py#L374-L388)) — for single-view, applied at function entry before Depth Pro

Iterators in `estimate_3D.py` only decode raw RGB and yield numpy arrays; they no longer know about MapAnything's contract.

See [../base/mapanything.md §8](../base/mapanything.md#8-unified-vst-pipeline--single--and-multi-image-entries-at-the-same-resolution-distribution).

### 1.2 Why Qwen-align is deferred to dataloader (default `qwen_align=False`)

`resize_xyz` in [`src/dataset/train_dataset.py:43`](../../src/dataset/train_dataset.py#L43) block-averages pts3d to the LLM patch grid using **integer stride**. If pts3d's H or W isn't a multiple of 28 (= patch_size 14 × merge_size 2), the integer-stride truncation drops edge pixels and accumulates a per-patch offset that misaligns bottom/right patches with their Qwen vision tokens.

The earlier pipeline solved this by applying `_resample_to_qwen_aligned` at save time — pts3d on disk was always /28. We've moved this to the dataloader because:

1. **Qwen smart_resize is model-specific.** Other downstream models (other VLMs, depth visualization, geometry analysis) don't share its (patch=14, merge=2) → /28 grid. Saving /28 locks the data into Qwen's contract.
2. **Saved data is more "raw".** MapAnything outputs (H, W) ∈ {518×392, 392×518, 518×518, …} are the model's actual output. Saving at /28 introduces an extra bilinear pass before disk that smooths out edges.
3. **AR distortion.** smart_resize rounds H and W independently — e.g. 480×640 → 476×644 stretches AR by ~1.5%. Doing this once at load time is no worse than at save time, but makes the saved data faithful to the reconstruction model.
4. **One interpolation instead of two.** With save-time alignment, the dataloader still has to block-average /28 → LLM grid (e.g. 504/28=18). With load-time alignment, the dataloader can resample directly from MapAny shape (e.g. 518) to LLM grid (e.g. 18) in a single bilinear / area pool.

The `qwen_align` flag is preserved (default False) at three plumbing points so save-time alignment can be re-enabled if needed:
- `CoordEstimator.estimate(qwen_align=False)`
- `_estimate_single_view(qwen_align=False)` / `_estimate_multi_view(qwen_align=False)`
- `process_dataset(qwen_align=False)` + CLI `--qwen_align`

See [../base/qwen3.5.md §2.2-2.4](../base/qwen3.5.md#22-the-smart_resize-formula) for `smart_resize` math.

---

## 2. Implementation map

```
parquet bytes / on-disk image
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ Iterator (estimate_3D.py)                                  │
│                                                            │
│  Pure decode + yield. No preprocessing logic.              │
│  Each iterator yields (entry_id, [uint8 RGB arrays]).      │
│                                                            │
│  _iter_vst_train          (VST si_*** + mi_***)            │
│  _iter_mindcube           (MindCube tinybench, multi-view) │
│  _iter_mindcube_train                                      │
│  _iter_mmsibench                                           │
│  _iter_sat                (mixed 1- and ≥2-view)           │
│  _iter_sparbench(*)       (multi/single/mv variants)       │
│  _iter_viewspatial        (single-view)                    │
│  _iter_omnispatial_pt     (single-view)                    │
│  _iter_embspatial         (single-view)                    │
│  _iter_spinbench          (mixed 1-5 unique views)         │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ CoordEstimator.estimate(images, qwen_align=False)          │
│  (coord_esti.py)                                           │
│                                                            │
│  if len(images) == 1:                                      │
│      _estimate_single_view                                 │
│        ├─ Step A: find_closest_aspect_ratio                │
│        │  + crop_resize_if_necessary on input image         │
│        ├─ Depth Pro inference + interp back to (H, W)      │
│        └─ result kept at MapAny shape (no Step C)          │
│                                                            │
│  else:                                                     │
│      _estimate_multi_view                                  │
│        ├─ _views_from_arrays: Step A for each view         │
│        │  (find_closest_aspect_ratio + crop_resize)         │
│        ├─ MapAnything inference                            │
│        └─ each result kept at MapAny shape (no Step C)     │
│                                                            │
│  qwen_align=True path: each result is run through          │
│  _resample_to_qwen_aligned() before return.                │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ save_results (coord_esti.py:save_results)                  │
│                                                            │
│  Per view (at MapAny shape, one of 10 RESOLUTION_MAPPINGS):│
│     pts3d.npy        (H, W, 3) float32                     │
│     depth.npy        (H, W)    float32                     │
│     mask.npy         (H, W)    bool                        │
│     intrinsics.npy   (3, 3)    float32                     │
│     camera_pose.npy  (4, 4)    float32                     │
│     image.png        (H, W, 3) uint8   PNG (lossless)      │
│  Per entry:                                                │
│     cameras.json     (extrinsics + intrinsics manifest)    │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Per-dataset reshape outcomes

`find_closest_aspect_ratio` lookup table from [../base/mapanything.md §2](../base/mapanything.md#2-the-10-allowed-output-shapes-resolution_set518). Saved pts3d shape is the (H, W) entry from this table — **no /28 rounding**.

### 3.1 Unified compliance table (training + eval)

All entries land on `RESOLUTION_MAPPINGS[518]` because Step A is now applied uniformly for all paths.

| Source | views | Typical input (W×H) | AR | Saved (W×H) | In 10-shape table? |
|---|---|---|---|---|---|
| **VST mi_*** | ≥2 | 640×480 / 644×476 / 1920×1440 | 1.33-1.35 | (518, 392) | ✓ |
| VST si_distance | 1 | 1024×768 | 1.333 | (518, 392) | ✓ |
| VST si_measurement | 1 | 1024×768 | 1.333 | (518, 392) | ✓ |
| VST si_depth_comparison | 1 | 1024×768 / 644×448 | 1.33 / 1.44 | (518, 392) / (518, 336) | ✓ |
| VST si_scene_caption | 1 | 1024×768 / 730×530 | ~1.33 | (518, 392) | ✓ |
| MindCube tinybench (portrait) | 4 | 480×640 (74%) | 0.75 | (392, 518) | ✓ |
| MindCube tinybench (16:9) | 2 | 480×270 (26%) | 1.778 | (518, 294) | ✓ |
| MindCube_train | 2-4 | 480×640 | 0.75 | (392, 518) | ✓ |
| MMSIBench | 2-10 | 1280×1024 / 1296×968 | 1.25-1.34 | (518, 392) | ✓ |
| sparbench_multi_view | 2-3 | 1296×968 | 1.339 | (518, 392) | ✓ |
| sparbench_mv | 2-3 | 1296×968 | 1.339 | (518, 392) | ✓ |
| SAT (≥2) | 2 | 512×512 | 1.0 | (518, 518) | ✓ |
| **SAT (1 img)** | 1 | 512×512 | 1.0 | **(518, 518)** ← was (512, 512) | ✓ |
| **sparbench_single_view** | 1 | 1296×968 | 1.339 | **(518, 392)** ← was (1296, 968) | ✓ |
| **ViewSpatial** | 1 | 1296×968 | 1.339 | **(518, 392)** ← was (1296, 968) | ✓ |
| **OmniSpatial PT** | 1 | 165 sizes | varies | per-entry → table row | ✓ |
| **EmbSpatial** | 1 | 640×480 | 1.333 | **(518, 392)** ← was (640, 480) | ✓ |
| **spinbench (infinigen)** | 1-4 | 200×200 (1405 entries) | 1.0 | **(518, 518)** ← was (200, 200) | ✓ |
| **spinbench (abo)** | 2-5 | max-side 224, AR varies (617 entries) | varies | per-entry → table row | ✓ |
| **spinbench (faces)** | 2-4 | 640×480 (339 entries) | 1.333 | **(518, 392)** | ✓ |
| **spinbench (cars)** | 2-5 | 376×250 (258 entries) | 1.504 | **(518, 336)** | ✓ |
| **spinbench (?)** | 5 | unknown (120 entries) | varies | per-entry → table row | ✓ |

Boldface marks the entries whose **saved shape changed** vs. the previous pipeline (which kept single-view eval at parquet-native and applied save-time Qwen-align). All these need re-running to match the new convention.

### 3.2 SpinBench specifics

- **Entry IDs are 12-char hex hashes** mapped from `test_idx_to_3d_id.json` (not sequential indices), so 3d_results/ subdir names match what downstream training/eval scripts already index.
- **View count = number of unique images per entry**: SpinBench's `test.jsonl` has 4 image paths per entry but they often duplicate (different multiple-choice options reusing the same view). `_iter_spinbench` deduplicates with `dict.fromkeys` (preserving order).
- View-count distribution on disk: 4-view (39%), 3-view (19.5%), 2-view (17.5%), 5-view (13.5%), 1-view (10.5%).
- 1-view entries are mostly `infinigen` (200×200 input → (518, 518) after Step A); these were the previously non-conforming `(200, 200)` entries on disk.

---

## 4. Disk and runtime profile

### 4.1 Per-view sizes (4:3 entry, pts3d shape (392, 518, 3))

| File | Shape | Bytes |
|---|---|---|
| pts3d.npy | (392, 518, 3) float32 | 2.43 MB |
| depth.npy | (392, 518) float32 | 0.81 MB |
| mask.npy | (392, 518) bool | 0.20 MB |
| image.png | (392, 518, 3) uint8 PNG | ≈ 295 KB |
| intrinsics.npy | (3, 3) float32 | 164 B |
| camera_pose.npy | (4, 4) float32 | 192 B |
| **Per-view total** | | **≈ 3.7 MB** |

~3% larger than the previous Qwen-aligned (392, 504) layout. For other 10-shape entries (392×518 transposed, 518×518, etc.) the ratio is similar.

### 4.2 VST total disk projection

| Group | Views | Per-view (avg) | Subtotal |
|---|---|---|---|
| mi_*** (4 subsets) | 526K | 2.8 MB pts3d/depth/mask + 0.20 MB PNG / view | ≈ 1.55 TB |
| si_*** (4 subsets) | 300K | same | ≈ 0.88 TB |
| **Total VST 3d_results** | 826K | | **≈ 2.43 TB** |

### 4.3 Runtime estimate (6 GPU shards)

Empirical from earlier mi_* run: ~67 entries/min/shard (= ~10K entries/hour total) when NFS isn't contended.

- mi_***: 526K views / 6 GPUs ≈ 88K views/shard ≈ 9 hours/shard
- si_***: 300K views / 6 GPUs ≈ 50K views/shard, Depth Pro is faster (~1s/view) ≈ 5 hours/shard
- **Total wallclock: ~14 hours** for full VST run on 6× RTX PRO 6000 (best case, NFS uncontended)

Eval datasets are tiny in comparison — full eval re-run typically <1 hour on a single GPU.

---

## 5. Why image.png is saved at MapAny shape (not parquet original)

The dataloader needs the image and pts3d at the same shape so the patch-level correspondence holds (downstream vision token (i, j) ↔ pts3d patch (i, j) covering the same scene region). Three options:

| Option | Disk | Pros | Cons |
|---|---|---|---|
| **Save MapAny image.png (current)** | +320 GB | Dataloader reads image.png directly; patch-aligned with pts3d; model-agnostic | image.png is post-Step-A, so re-decoding at parquet resolution is impossible |
| Save Qwen-aligned image.png (`--qwen_align`) | +316 GB | Same as current but pre-rounded to /28 | Locks data to Qwen's grid; AR distortion baked in |
| Skip image.png, dataloader reapplies Step A on parquet | 0 GB | Always have parquet original | Dataloader must import `mapanything` and reapply preprocessing per sample |

We chose option 1 — the saved image is post-Step-A but **not** Qwen-aligned. Step A is the same for all downstream consumers (it's a property of the reconstruction model), so it's safe to bake in. Qwen-align is model-specific, so it's deferred.

PNG (lossless) is preferred over JPG q=95 because the post-Step-A image is the **canonical input** the dataloader will feed downstream — JPG quantization would compound with downstream resizes and add a small but unnecessary loss layer.

---

## 6. Verification snippets

### 6.1 Per-entry shape sanity check

```python
import numpy as np
from PIL import Image
import os

entry_dir = ".../3d_results/si_distance/154751"
for v in sorted(os.listdir(entry_dir)):
    if not v.startswith("view_"): continue
    pts = np.load(os.path.join(entry_dir, v, "pts3d.npy"), mmap_mode="r")
    img = np.asarray(Image.open(os.path.join(entry_dir, v, "image.png")))
    assert pts.shape[:2] == img.shape[:2], f"shape mismatch: pts={pts.shape}, img={img.shape}"
    print(f"{v}: pts3d {pts.shape}, image.png {img.shape}, MATCH")
```

### 6.2 RESOLUTION_MAPPINGS[518] compliance check

```python
import numpy as np, os
# 10-shape table (H, W) — note (W=518, H=168) and (W=168, H=518) both exist
VALID_HW = {
    (252, 518), (294, 518), (336, 518), (392, 518),
    (518, 518),
    (518, 392), (518, 336), (518, 294), (518, 252), (518, 168), (168, 518),
}
root = "datasets/evaluation/spinbench_data/3d_results"  # adjust as needed
bad = []
for sub in os.listdir(root):
    sd = os.path.join(root, sub)
    if not os.path.isdir(sd): continue
    for v in os.listdir(sd):
        if v.startswith("view_"):
            p = os.path.join(sd, v, "pts3d.npy")
            if os.path.exists(p):
                hw = tuple(np.load(p, mmap_mode="r").shape[:2])
                if hw not in VALID_HW:
                    bad.append((sub, v, hw))
                    break
print(f"non-conforming entries: {len(bad)}")
for b in bad[:5]: print(" ", b)
```

### 6.3 Dataloader-side Qwen alignment (one bilinear, no two-step)

When the dataloader feeds Qwen3.5-VL, it should resample MapAny pts3d (e.g. 392×518) directly to the Qwen LLM grid (e.g. 14×18 or 14×19 depending on smart_resize) in **one** interpolation. Don't first round to /28 (504) and then block-average — that's two bilinears stacked.

---

## 7. Routing summary

| Path | Decision | Implementation |
|---|---|---|
| Reconstruction model | `len(images) == 1` → Depth Pro; else MapAnything | [`coord_esti.py:325-337`](../../coord_esti.py#L325-L337) |
| Step A (MapAny reshape) | Always, both single- and multi-view, inside CoordEstimator | [`coord_esti.py:_estimate_single_view`](../../coord_esti.py#L374-L388); [`_views_from_arrays`](../../coord_esti.py#L507-L548) |
| Step C (Qwen-aligned resample) | **Deferred to dataloader** by default. CLI `--qwen_align` re-enables save-time. | [`coord_esti.py:_resample_to_qwen_aligned`](../../coord_esti.py#L142-L216) |
| Image format on disk | PNG (lossless), at MapAny shape | [`coord_esti.py:save_results`](../../coord_esti.py#L546-L605) (default `save_image=True`) |
| Skip-existing | If `cameras.json` present | [`estimate_3D.py:process_dataset`](../../src/data_process/estimate_3D.py) |

---

## 8. Key invariants

1. **`pts3d.npy` and `image.png` are pixel-aligned on disk** — same `(H, W)` for every view.
2. **`(H, W) ∈ RESOLUTION_MAPPINGS[518]`** — one of 10 entries, max-side 518, all /14 multiples.
3. **`max(H, W) ≤ 518`** — bounded by the MapAnything reshape contract (Step A).
4. **Single shape distribution across training and eval** — si_*, mi_*, multi-view eval, and single-view eval (including the previously parquet-native SAT-1, SPARBench-1, ViewSpatial, OmniSpatial, EmbSpatial, SpinBench) all sample from the same 10-shape pool.
5. **The only per-pipeline difference is the per-pixel xyz signal source** (monocular Depth Pro vs multi-view MapAnything). Image shape, pts3d shape, and downstream patch grid are identical for entries with matching parquet AR.

---

## 9. Migration notes (from the previous /28 + parquet-native pipeline)

What changed at the code level:

- **`coord_esti.py`**:
  - `_estimate_single_view` now applies Step A at function entry (lines 383-388). Before, single-view had no Step A — Depth Pro saw parquet-native images.
  - `estimate()` / `_estimate_single_view()` / `_estimate_multi_view()` gained `qwen_align: bool = False` parameter; `_resample_to_qwen_aligned` is called only if True.

- **`src/data_process/estimate_3D.py`**:
  - `_iter_vst_train` no longer applies Step A — it now just decodes RGB. The function used to apply `find_closest_aspect_ratio` + `crop_resize_if_necessary` for `si_*` only; that's now redundant because `_estimate_single_view` handles all single-view inputs.
  - `_iter_spinbench` added; `DATASETS["spinbench"]` registered.
  - `process_dataset` gained `qwen_align: bool = False` parameter; CLI gained `--qwen_align` flag.

What this means for existing data:

- **VST si_*** previously preprocessed: shape unchanged (Step A was already applied in iterator before, now applied in estimator — math is the same).
- **VST mi_*** previously preprocessed with the old pipeline: shape was (392, 504, 3) Qwen-aligned. New default produces (392, 518, 3) MapAny-native. **Re-run required** to mix old and new entries.
- **Single-view eval (SAT-1 / SPARBench-1 / ViewSpatial / OmniSpatial PT / EmbSpatial)** previously at parquet-native: now at MapAny shape. **Re-run required.**
- **SpinBench**: the ~926 1-unique-image (`infinigen`) entries previously had non-conforming (200, 200) shape. **Re-run required**; new shape will be (518, 518).
