# MapAnything Image Preprocessing

**Source:** [`mapanything/utils/cropping.py`](../../external/map-anything/mapanything/utils/cropping.py), [`mapanything/utils/image.py`](../../external/map-anything/mapanything/utils/image.py)
**Invoked from:** [`coord_esti.py:_build_views`](../../coord_esti.py#L380-L399)

---

## TL;DR

Before MapAnything inference, every input image is reshaped to a **fixed grid from `RESOLUTION_MAPPINGS[518]`** via `crop_resize_if_necessary`. The reshape is **Lanczos rescale (preserving aspect ratio) then center-crop**. Output H, W are always multiples of 14 (dinov2 patch size). Aspect ratio is preserved exactly; a small (~0.4%) edge-pixel strip is cropped.

| Step | Operation | Effect |
|---|---|---|
| 1 | `find_closest_aspect_ratio(W/H, 518)` | Pick `(target_w, target_h)` from a 10-entry table |
| 2 | `rescale_image_and_other_optional_info(...)` | Lanczos resize so the image *just contains* the target box |
| 3 | Center-crop to `(target_w, target_h)` | Trim symmetric edge strip (≤ a few px) |

---

## 1. Why preprocessing is mandatory

MapAnything's vision encoder is **dinov2_vitg14** (patch_size = 14). The encoder requires `H % 14 == 0` and `W % 14 == 0`. Additionally, the model was trained on a fixed table of (W, H) shapes derived from `resolution_set=518` — feeding shapes outside this table is undefined behavior.

The function `crop_resize_if_necessary` is what enforces this. It is **not invoked automatically by the model**; the caller must invoke it. In our code, that caller is [`CoordEstimator._build_views`](../../coord_esti.py#L380-L399).

---

## 2. The 10 allowed output shapes (`resolution_set=518`)

Built from `RESOLUTION_MAPPINGS[518]` in [`mapanything/utils/image.py`](../../external/map-anything/mapanything/utils/image.py#L40):

| AR (W/H) | (W × H) | Pixels | Patch grid (W/14 × H/14) |
|---|---|---|---|
| 0.486 | 252 × 518 | 130,536 | 18 × 37 |
| 0.567 | 294 × 518 | 152,292 | 21 × 37 |
| 0.649 | 336 × 518 | 174,048 | 24 × 37 |
| 0.757 | 392 × 518 | 203,056 | 28 × 37 |
| 1.000 | 518 × 518 | 268,324 | 37 × 37 |
| **1.321** | **518 × 392** | **203,056** | **37 × 28** |
| 1.542 | 518 × 336 | 174,048 | 37 × 24 |
| 1.762 | 518 × 294 | 152,292 | 37 × 21 |
| 2.056 | 518 × 252 | 130,536 | 37 × 18 |
| 3.083 | 518 × 168 | 87,024  | 37 × 12 |

`find_closest_aspect_ratio(input_AR)` picks the entry minimizing `|key - input_AR|`. There is **no exact 4:3 (1.333) entry** — the closest is 1.321 (518×392).

---

## 3. Concrete walkthrough — 480×640 input

For a typical MindCube image (H=480, W=640, AR = 640/480 = **1.333**):

### Step 1 — Pick target shape

```python
from mapanything.utils.image import find_closest_aspect_ratio
target_w, target_h = find_closest_aspect_ratio(640/480, resolution_set=518)
# → (518, 392), AR = 1.321
```

### Step 2 — Lanczos rescale to bounding box

```python
scale_final = max(target_w / W, target_h / H)
            = max(518/640, 392/480)
            = max(0.8094, 0.8167) = 0.8167
new_size = floor((640, 480) * 0.8167) = (522, 392)
image = image.resize((522, 392), resample=Lanczos)
```

The rescale picks the **larger** of the two scale factors, ensuring the rescaled image *contains* the target box. Aspect ratio is preserved exactly (uniform scale).

### Step 3 — Center-crop to target

```python
# image is now 522 × 392, target is 518 × 392
# Crop horizontally: drop (522 - 518)/2 = 2 px from each side
crop_bbox = (2, 0, 520, 392)
image = image.crop(crop_bbox)   # → 518 × 392
```

### End state

| Attribute | Value |
|---|---|
| Output (W × H) | **518 × 392** |
| Patch grid (W/14 × H/14) | **37 × 28** = 1,036 patches |
| Aspect ratio | 518/392 = 1.321 (vs input 1.333; exact-preserved by step 2, then **unchanged by symmetric crop**) |
| Cropped pixels | 4 px width on the 522×392 rescaled image (≈ 2.45 px on each side at the original 640 scale) |
| Cropped fraction | ~0.4% of image area |
| Information loss | (a) Lanczos high-frequency smoothing from ×0.8167 downsample — minimal; (b) symmetric edge strip ~0.4% — geometric loss |

---

## 4. Where this matters in our pipeline

[`coord_esti.py:_build_views`](../../coord_esti.py#L380-L399) handles preprocessing for the multi-view path:

```python
aspect_ratios = [arr.shape[1] / arr.shape[0] for arr in arrays]
avg_ar = sum(aspect_ratios) / len(aspect_ratios)   # use mean AR across views
target_w, target_h = find_closest_aspect_ratio(avg_ar, resolution_set=518)
for arr in arrays:
    pil_resized = crop_resize_if_necessary(pil, resolution=(target_w, target_h))[0]
```

→ `pred["img_no_norm"]` returned by MapAnything is at `(target_h, target_w)`. `pts3d`, `depth`, `mask` are also at this shape, guaranteed consistent on disk.

---

## 5. Resolution-set alternatives

`RESOLUTION_MAPPINGS` also exposes `512` and `504` keys with similar structure. We use `518` because it's the dinov2_vitg14 default training resolution. Switching to `504` would shrink output patches (36×27 vs 37×28) and reduce per-view disk by ~5%.

---

## 6. Can MapAnything output a larger pts3d? Theory vs trained distribution

A natural question: do we have to use the 10 shapes in `RESOLUTION_MAPPINGS[518]`? For 1920×1440 inputs the model throws away ~80% of the input pixels (Lanczos rescale ×0.27 → 392×518). It would be ideal to feed MapAnything a larger /14 shape (e.g. 1148×840) and get pts3d at higher native resolution.

### Theory: dinov2_vitg14 supports any /14 shape

dinov2_vitg14 uses **interpolatable 2D positional embeddings** (`interpolate_pos_encoding=True`); at forward time the trained pos embedding is bilinearly interpolated to whatever input grid the current image produces. So architecturally:

```python
# dinov2 backbone accepts any (H, W) where H % 14 == 0 and W % 14 == 0
# e.g. (840, 1148): 840=14*60, 1148=14*82 → grid 60×82 ✓ forwards fine
```

As long as the input is a /14 multiple, `model.forward()` does not raise a shape error. MapAnything's downstream attention / decoder are also shape-agnostic (attention does not depend on a fixed token count).

### Practice: trained distribution caps `max(H, W) ≤ 518`

The MapAnything checkpoint we use was trained on the fixed 10 shapes in `RESOLUTION_MAPPINGS[518]` — **all of them satisfy `max(W, H) = 518`**. A target like 1148×840 is **2.2× extrapolation** beyond the largest training shape.

DINOv2's positional-encoding interpolation typically holds up well within ~1.5× extrapolation, then degrades:

| Extrapolation factor | Expected behavior |
|---|---|
| ≤ 1.5× | essentially loss-free |
| ~ 2.0× | edge-token geometric drift ~ 5–10% |
| ≥ 2.5× | visible degradation (global shift, gradient collapse, smeared depth at edges) |

We have **not validated** MapAnything at scales like 1148×840. Whether it produces metric-quality pts3d at that scale is unverified, so the 10 default shapes remain the safe default.

### Implications

- **Bilinear-upsampling MapAnything's pts3d (392×518 → 840×1148)**: cheap, safe, but adds no real geometric detail (interpolation cannot invent information).
- **Feeding MapAnything (840×1148) directly**: would produce native high-resolution pts3d, but lives in untested territory. Empirical ablation needed before relying on it.

Until we run that ablation, treat MapAnything as a **fixed-resolution** geometry source (≤ 518 max-side) and reconcile any downstream resolution differences via post-resampling.

---

## 7. The 518 cap is a class-wide DINOv2 problem, not a MapAnything quirk

If we ever swap MapAnything for another modern reconstruction transformer (VGGT, π³, DUSt3R, MASt3R, …), the same 518-side ceiling will follow us — they all share the same DINOv2 backbone family. The only escape is to switch to a non-DINOv2 model (e.g. Depth Pro).

### VGGT — confirmed in our repo at [`vggt/utils/load_fn.py`](../../external/vggt/vggt/utils/load_fn.py#L97-L165)

```python
def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    mode="crop" (default): Sets width to 518px and center crops height if needed.
    mode="pad":            Preserves all pixels by making the largest dimension 518px.
    """
    target_size = 518
```

Same 518 ceiling as MapAnything; the difference is the framing strategy:

| Mode | Behavior | Field-of-view loss |
|---|---|---|
| `crop` | Uniform rescale to W=518, then center-crop H to 518 if oversize | Vertical strip cropped on landscape/tall images |
| `pad` | Uniform rescale so `max(W, H) = 518`, then zero-pad short side to 518×518 | None, but padded pixels become wasted vision tokens |

VGGT's `pad` mode is **more aspect-ratio-flexible** than MapAnything's 10-entry table: it accepts any input AR by padding instead of cropping. But the padding bytes still occupy tokens that carry no geometry — analogous to MapAnything's edge crop, just shifted from "missing pixels" to "useless pixels".

### π³ (not integrated, inferred from the paper)

π³ ("Permutation-Equivariant Visual Geometry Grounded Transformer", VGGT successor, 2025) is a DINOv2_vitl14-backed model in the same family. Architecturally it inherits the same 518 max-side training distribution; we have not validated it independently.

### Cross-model summary

| Model | Backbone | Patch | Max side | Aspect-ratio handling |
|---|---|---|---|---|
| **MapAnything** | dinov2_vitg14 | 14 | **518** | 10 fixed shapes from `RESOLUTION_MAPPINGS[518]` (Lanczos rescale + center-crop) |
| **VGGT** | dinov2_vitl14 | 14 | **518** | `crop`: rescale W to 518 + crop H; `pad`: rescale max-side to 518 + pad short side |
| **π³** (inferred) | dinov2_vitl14 | 14 | **518** | likely same family of DINOv2-driven constraints |
| **DUSt3R** | croco_vit_b16 | 16 | ~512 | scale + crop; multi-scale variants |
| **MASt3R** | croco_vit_b16 | 16 | ~512 | inherits DUSt3R |
| **Depth Pro** (the exception) | Apple custom ViT | 16 | **1536** | scale + crop/pad to 1536² internally |

### Why Depth Pro escapes the 518 ceiling

Depth Pro is **not** built on DINOv2. It is Apple's monocular-depth model trained with a custom ViT backbone at 1536², so its preprocessing routine (`_depth_pro_transform`) targets that much higher resolution and resamples the prediction back to the input's native size.

For our pipeline this is why the **si_*** (Depth Pro) path can keep pts3d at parquet-native resolution (e.g. 1920×1440 in si_depth_comparison) while the **mi_*** (MapAnything) path is locked at 392×518 even when the parquet is 1920×1440 — Depth Pro absorbs the high-res input losslessly, MapAnything throws ~80% of pixels away in `crop_resize_if_necessary`.

### Practical takeaways

- **Don't expect a different DINOv2-backed model to lift the resolution ceiling.** VGGT / π³ would inherit the same constraint. Pre-resize / post-resample alignment work has to happen regardless of which one we use.
- **High-resolution inputs (≥ 1920×1440)** belong on Depth Pro (or a future non-DINOv2 reconstruction model) if we want the geometry to carry the full input detail. Routing them through MapAnything is information-wasteful.
- **For multi-view tasks** where MapAnything is mandatory (no monocular alternative gives multi-view-consistent pts3d), the 80%-information-loss on high-res inputs is currently unavoidable; this is why we keep mi_*** at the default 392×518 and reconcile downstream.
- **Tile-based workarounds** (split a 2048×1536 image into 4 overlapping 518×518 tiles, run MapAnything on each, stitch the outputs) exist but require careful pose alignment and are not in our pipeline today.

---

## 8. Unified VST pipeline — single- and multi-image entries at the same resolution distribution

To keep training stable, our VST pipeline forces single-image (Depth Pro) and multi-image (MapAnything) entries into the **same effective resolution distribution**, instead of letting si_*** run at parquet-native (768×1024 / 1440×1920) while mi_*** sits at MapAnything's 518 cap. The mismatch otherwise propagates downstream and forces the model to learn a conditional scale per entry.

### Apply MapAnything preprocessing identically to both paths

```
parquet original
    ↓ MapAnything preprocessing — find_closest_aspect_ratio + crop_resize_if_necessary
       (Lanczos rescale + center-crop to one of the 10 RESOLUTION_MAPPINGS[518] shapes)
    ↓
       — applied automatically inside CoordEstimator._build_views for mi_*
       — applied explicitly inside estimate_3D._iter_vst_train for si_*,
         so Depth Pro sees the same shape pool as MapAnything
    ↓
pts3d at one of MapAnything's 10 fixed shapes
```

Without this normalization, si_*** outputs span the parquet's native resolutions (e.g. 768×1024, 1440×1920, 530×730 …) while mi_*** outputs are confined to the 10 RESOLUTION_MAPPINGS[518] shapes (max-side 518). Downstream code has to reason about wildly different geometry resolutions per entry. Forcing both paths through the same preprocessor collapses everything to that 10-shape distribution.

### Distribution across VST and eval benchmarks

| benchmark / subset | parquet AR | MapAny target (W × H) | pts3d numpy (H, W) |
|---|---|---|---|
| VST mi_camera_motion / mi_correspondence / mi_object_object_relation | 1.33 | 518 × 392 | (392, 518) |
| VST mi_scene_caption | 1.33 (mostly) | 518 × 392 | (392, 518) |
| VST si_distance / si_measurement | 1.33 | 518 × 392 | (392, 518) |
| VST si_depth_comparison (1024×768 / 1920×1440) | 1.33 | 518 × 392 | (392, 518) |
| VST si_depth_comparison (644×448) | 1.438 | 518 × 336 | (336, 518) |
| VST si_scene_caption | 1.33 (mostly) | 518 × 392 | (392, 518) |
| MindCube tinybench (480×640 portrait) | 0.75 | 392 × 518 | (518, 392) |
| MindCube tinybench (480×270, 16:9) | 1.78 | 518 × 294 | (294, 518) |
| MMSIBench / ViewSpatial / SPARBench | ~1.34 | 518 × 392 | (392, 518) |
| **SAT** (512×512 square) | 1.000 | 518 × 518 | (518, 518) |
| EmbSpatial-Bench | 1.33 | 518 × 392 | (392, 518) |
| OmniSpatial PT | 165 ARs | varies (10 possible) | mixed across all 10 |

→ Every entry lands on exactly **one of the 10 RESOLUTION_MAPPINGS[518] shapes**. VST training plus most eval benchmarks concentrate on `(392, 518)` (landscape 4:3). MindCube tinybench is portrait-dominated, SAT is square. OmniSpatial spans the whole table.

### Implementation references

- [`estimate_3D.py:_iter_vst_train`](../../src/data_process/estimate_3D.py) — applies MapAnything preprocessing to si_*** inputs before yielding (so Depth Pro sees the same shape pool as MapAnything).
- [`coord_esti.py:_build_views`](../../coord_esti.py#L380-L399) — applies MapAnything preprocessing to mi_*** inputs internally (the model's own contract).
