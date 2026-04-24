# Toward learning actual 3D geometry: four training-recipe ingredients

Follow-up to [findings/xyz_zero_ablation.md §10](../findings/xyz_zero_ablation.md). That analysis showed the current coord and decouple checkpoints learn "MindCube-3D" (a distribution-bound statistical pattern of xyz values) rather than geometric reasoning. This document works through the four training recipes §10.3 named, with concrete implementation sketches, failure modes, and a dependency order.

The ingredients:

1. **Explicit geometric supervision that punishes shortcut use**
2. **Rotation-equivariance augmentation**
3. **Cross-dataset training**
4. **Hold-out geometric transfer test**

## 1. Explicit geometric supervision

### Why the current training is insufficient

The two loss signals in use:

- **LM loss (decouple's only loss)**: penalizes wrong QA letter. The model can answer correctly without consulting xyz at all, so this loss provides no pressure to use the xyz channel. Confirmed by the xyz = 0 ablation (decouple MindCube: Δacc = +0.10 pp).
- **coord_head regression loss (coord model)**: trains `coord_head(hidden_state) → xyz_GT`. Two separate leaks let the model bypass xyz:
  - `coord_head` reads from the `<|image_pad|>` **vision token hidden state**, not from xyz directly. The vision tokens already encode a rough spatial layout (patch index, etc.), so the head learns to regress xyz from vision features.
  - Confirmed: zeroing xyz at inference only degrades coord_mae by 1.85× on MindCube and actually reduces it 50× on SAT — the head is vision-driven.

"Punishing shortcut use" means making the shortcut cost more than the geometric path. Right now the shortcut is free; gradient descent takes it.

### Four concrete approaches

#### 1.A Multi-view contrastive correspondence (highest priority)

MindCube already provides 4 views per scene. Use corresponding 3D points across views as positive pairs:

```python
for (image_i, xyz_i), (image_j, xyz_j) in same_scene_different_view_pairs:
    # p_i, p_j point to the SAME physical 3D point
    p_i, p_j = find_correspondence(xyz_i, xyz_j, eps=0.05)  # via xyz L2 match
    # and p_i, p_k is a negative pair in the same image
    p_k = random_other_patch(image_i)

    loss_contrast = triplet_loss(
        anchor = hidden[image_i][p_i],
        positive = hidden[image_j][p_j],
        negative = hidden[image_i][p_k],
        margin = 0.5,
    )
```

- **Supervision pathway**: hidden state → LM → direct gradient on "must encode 3D identity"
- **Data cost**: ~0 (MindCube already has 4-view alignment; correspondences auto-generated from xyz distance)
- **Tolerates xyz noise**: only needs `dist(correspondent) < dist(non-correspondent)` to hold — ordinal, not metric
- **Risk**: negative-sample design matters; too easy → trivially solved, too hard → no learning

#### 1.B xyz masking + reconstruction

Training-time: randomly mask k % of image patches' xyz (set to zero or a `<mask>` token). Require `coord_head` to reconstruct the masked xyz from (visible xyz + vision features):

```python
mask = torch.rand(n_patches) < 0.25
xyz_input = image_xyz.clone()
xyz_input[mask] = 0.0
# forward as usual
# reconstruction loss:
loss_recon = MSE(coord_head(hidden[mask]), image_xyz_gt[mask])
```

- Analog of Masked Autoencoders for 3D position
- Forces the model to build **cross-patch geometric relationships** (otherwise reconstruction from partial xyz is impossible)
- Hyperparameter: mask rate (tunable; 15–30 %)

#### 1.C Pose prediction from xyz + images

Predict the 6-DoF camera pose for each image, given image + xyz:

```python
pose_pred = pose_head(hidden_states_per_image)
loss_pose = SO3_loss(pose_pred[:, :3], pose_gt[:, :3]) + MSE(pose_pred[:, 3:], pose_gt[:, 3:])
```

- **Pose is structurally geometric** — cannot be inferred from vision alone (up to scale/translation ambiguity) or from xyz alone (without images) in general. Forces joint xyz-vision reasoning.
- **Data cost**: MindCube has `camera_pose.npy`; SAT/SpinBench may not.
- **Risk**: pose regression has known difficulties (rotation representation choice — 6D rep or quaternion + hemisphere ambiguity).

#### 1.D xyz-image consistency classification (anti-shortcut)

With 50 % probability, permute the batch's xyz assignment: image A gets image B's xyz. Add a binary label `xyz_matches_image ∈ {0, 1}` and a classification head:

```python
if random.random() < 0.5:
    perm = torch.randperm(batch_size)
    image_xyz_batch = image_xyz_batch[perm]
    xyz_matches = (perm == torch.arange(batch_size)).long()
else:
    xyz_matches = torch.ones(batch_size).long()

pred = match_head(pooled_hidden)
loss_match = CE(pred, xyz_matches)
```

- The **only way** to classify match/mismatch is to check whether xyz is geometrically consistent with the image. Shortcut-proof.
- Cheapest of the four; 1 extra forward per sample not needed.
- Weak signal alone, but excellent as an auxiliary anti-shortcut term.

### Failure modes and tradeoffs

| Method | Training time cost | Data cost | Forces xyz usage? | Implementation complexity |
|---|---|---|---|---|
| 1.A contrastive | ~1 extra forward per view-pair | 0 (auto from MindCube) | **yes, strongly** | medium |
| 1.B mask/reconstruct | 1 extra forward | 0 | medium — head can still cheat via vision | low |
| 1.C pose prediction | negligible | need pose GT | medium | medium (rotation repr) |
| 1.D match/mismatch | negligible | 0 | weak alone | very low |

**Recommendation**: start with 1.A + 1.D. Both are cheap; both are hard to cheat; both can be added to `train_correspondence.py` with < 100 LOC.

## 2. Rotation-equivariance augmentation

### Why the current training is insufficient

Every training sample's xyz is in the fixed **first-camera frame**. The model never sees the same images paired with a different xyz coordinate system. Consequences:

- Model has no reason to learn that xyz is a **frame-relative** quantity — it just learns the specific distribution of values
- R-invariance at test time is therefore **not** evidence of rotational reasoning; it is evidence that the model doesn't use xyz's frame-sensitive meaning at all
- Without R-variation in training, even a contrastive loss (§1.A) can be solved via frame-specific features (i.e., remembering absolute xyz values rather than relative 3D positions)

### Two approaches

#### 2.A Input R-augmentation (simple form)

At each batch step, sample `R ∈ SO(3)` and apply it to `image_xyz`:

```python
R = random_rotation_matrix()
image_xyz_aug = torch.einsum("ij,bnj->bni", R, image_xyz)
# Labels that are geometric (coord_head GT, pose GT) also rotate:
xyz_gt_aug = torch.einsum("ij,bnj->bni", R, xyz_gt)
pose_gt_aug = R @ pose_gt  # rotate the extrinsics similarly
# QA labels are unchanged (answers are about image content)
```

- **Must be paired with §1**. In isolation, R-augmentation on QA-only training will just reinforce "ignore xyz" (that's the easiest way to stay R-invariant). Contrastive loss (1.A) is what forces the model to actually *use* the rotated xyz.
- Sampling distribution choice:
  - Full SO(3): requires more capacity but is the right test bed
  - SO(2) about the up-axis (yaw only): matches natural variation in indoor scenes; cheaper
  - Small perturbations around I: weak signal, not recommended

#### 2.B Covariance loss (stronger form)

Run the forward pass twice, once with xyz and once with `R · xyz`, and constrain the outputs to be R-related rather than R-invariant:

```python
out_1 = model(image, xyz)
out_2 = model(image, R @ xyz)

# QA logits should match (answers are R-invariant)
loss_qa_inv = KL(logits_1, logits_2)

# coord_head output should rotate with R
loss_coord_cov = MSE(coord_1, torch.einsum("ji,bnj->bni", R, coord_2))
#                                              ^^ R transpose → R⁻¹

# hidden states can be required to be R-equivariant via a learned transform
# (optional, expensive)
```

- **Covariance > invariance**: forcing "xyz rotates → output rotates" is much stronger than "output unchanged" (invariance is a special case achievable by ignoring the input).
- 2× training cost (two forward passes).

### Data-label conflict caveat

MindCube QA has view-dependent phrasing: "from image 1's perspective, the cup is on the **left** of the box." If we rotate xyz but keep the QA label, we're lying to the model: the new "left" should be different under the rotated frame.

- For QA labels that reference **relative positions within an image** (most of MindCube): actually unchanged, because rotating xyz doesn't rotate the images themselves. The question "what's in image 1?" is image-centric, not xyz-centric. Safe.
- For QA labels that reference the **world frame** explicitly (e.g., "which object is most north"): not safe; these have to be re-labeled or filtered.

### Priority

**Do not deploy §2 without §1 being in place.** Otherwise the R-augmentation training bias strengthens the "ignore xyz" solution.

## 3. Cross-dataset training

### Why the current training is insufficient

Training only on MindCube means the model's entire notion of "xyz" is MindCube's coordinate convention. Evidence:

- SAT baseline coord_mae = 107 (scene scale off by orders of magnitude)
- xyz = 0 on coord MindCube is catastrophic (−21 pp), but on coord SAT actually helps (+10.67 pp) — because SAT's own xyz is worse than nothing from the model's perspective
- Vanilla (no xyz) outperforms decouple and coord on SAT (78 % vs 67 %)

Without cross-dataset exposure, "generalization to SAT" is functionally impossible: the model has no pressure to learn anything beyond MindCube-3D.

### Three approaches

#### 3.A Naive dataset mixing

```python
train_loader = ConcatDataset([MindCubeTrain, SATTrain, SpinBenchTrain, RoboSpatialTrain])
```

- Simplest; lowest implementation cost
- **Problem**: xyz distributions differ by orders of magnitude across datasets (SAT mae=107 is evidence). Mixed training will learn "dataset ID → xyz-interpretation" as a shortcut, not universal geometry.
- **Mitigation**: dataset-agnostic loss only; no dataset-ID signal reaches the model
- Still suboptimal without 3.B

#### 3.B Canonical xyz normalization (recommended)

Preprocess each dataset once to a shared canonical frame:

```python
def canonicalize(xyz_raw, scene_bbox=None):
    # 1. translate: center at scene centroid
    center = xyz_raw.mean(dim=[-2, -1], keepdim=False)  # per scene
    xyz = xyz_raw - center
    # 2. scale: normalize to unit bounding box
    scale = xyz.abs().max()
    xyz = xyz / scale
    # 3. rotate: align up-axis convention
    # (use first image's camera down vector as +y; dataset-specific)
    R_canonical = compute_canonical_rotation(dataset, scene)
    xyz = torch.einsum("ij,bnj->bni", R_canonical, xyz)
    return xyz, {"center": center, "scale": scale, "R": R_canonical}
```

- Stores the un-canonicalization transform so test-time predictions can be mapped back if needed
- One-time preprocessing pass over all 4 datasets
- Per-scene normalization (not per-dataset) so within-scene relative geometry is preserved

**Engineering cost**: medium (need to write `canonicalize()` for each dataset's specific convention). Benefits every subsequent experiment.

#### 3.C Scale/translation augmentation

At train time, sample random `s ~ U[0.5, 2]`, `t ~ N(0, σ²)` and apply `xyz_aug = s · (xyz + t)`. Simulates cross-dataset variation without requiring multiple datasets.

- Cheap; no new data
- **Cannot simulate coordinate-system convention differences** (e.g., +y up vs +z up, metric vs unitless)
- Useful complement to 3.B but not a replacement

### Priority

**3.B is the right path** for long-term work. 3.C is cheap and can ship fast while 3.B is being implemented.

## 4. Hold-out geometric transfer test

### Why the current metrics can't answer "did it learn geometry?"

Current evaluation surfaces:

- **QA accuracy (multiple choice)**: solvable by content features alone; 4-way accuracy = 67 % on SAT for vanilla-no-xyz suggests the task has substantial shortcut-solvability
- **coord_mae**: can be water-passed by a vision-driven coord_head (the actual observed behavior — 1.85× degradation with xyz zeroed instead of catastrophic)
- **R-sweep spread**: can be zero both because (a) model doesn't use xyz at all, and (b) model uses xyz in a frame-invariant way. These two are not distinguishable from R-sweep alone.

We need tasks where **"if xyz is wrong, the answer is wrong, period."**

### Five candidate tasks

#### 4.A Relative pose between two images (highest priority — lowest cost)

```
Input: image_1, image_2, xyz_1, xyz_2  (from same scene, different views)
Output: R, t such that camera_2 = R · camera_1 + t
Metric: angular error on R, L2 error on t/||t||
```

- **Why geometric**: R and t cannot be inferred without matching 3D points across views; the matching requires xyz to be correct.
- **Data cost**: MindCube has `camera_pose.npy` per view → GT is free.
- **Prediction head**: rotation via 6D representation (Zhou et al., 2019) or unit quaternion, translation as 3-vector.

#### 4.B Novel-view consistency

Train on 3 of 4 MindCube views per scene; test on the held-out view's QA and coord prediction.

- **Why geometric**: model must infer the unseen viewpoint from only the seen 3 views + xyz. Content shortcuts fail (never saw this viewpoint's content).
- **Data cost**: zero (split existing data).
- **Metric**: QA acc on held-out view; coord_mae on held-out view's patches.

#### 4.B* Out-of-distribution viewpoint

Harder version of 4.B: test views come from **a viewpoint distribution not present in training** (e.g., train uses eye-level; test uses top-down).

- **Why**: the true test of "3D understanding" is whether the model can handle unseen camera poses at all, not just unseen content from seen pose distributions.
- **Data cost**: requires generating held-out views → need a 3D renderer pipeline for MindCube assets, or use a separate dataset with wide viewpoint variation.

#### 4.C Occlusion reasoning

"In image 1, is object A occluded by object B?" / "What fraction of object A is visible?"

- **Why geometric**: occlusion is a 3D relationship (requires depth ordering).
- **Data cost**: needs per-object segmentation + depth, which can be auto-generated from the existing xyz if objects are delineated.

#### 4.D Cross-image 3D correspondence

"Given patch P in image 1, which patch in image 2 corresponds to the same 3D point?"

- **Why geometric**: shortcut-proof — same scene often has visually similar patches (e.g., multiple instances of the same object class). Only the correct 3D correspondence resolves the ambiguity.
- **Data cost**: auto-labeled from xyz proximity.
- **Metric**: top-1 / top-5 correspondence accuracy at pixel tolerance.

#### 4.E Metric distance

"What is the distance (in meters) between cup and book in image 1?"

- **Why geometric**: requires absolute 3D reasoning with correct scale.
- **Data cost**: needs object box / instance annotation — not auto-labeled from just xyz.
- **Metric**: mean absolute error on predicted distance.

### Comparison

| Task | Geometric necessity | Auto-labeled? | Implementation complexity |
|---|---|---|---|
| 4.A pose | high | yes (camera_pose.npy) | low (add a head + loss) |
| 4.B novel view | medium | yes (split) | low (re-split data) |
| 4.B* OOD view | **very high** | no (need renderer) | high |
| 4.C occlusion | high | needs seg+depth | medium |
| 4.D correspondence | high | yes (xyz match) | low (add metric) |
| 4.E distance | high | needs object boxes | high |

**Recommendation**: implement 4.A and 4.D first (both auto-labeled, both strong geometric necessity, both low complexity). If those reveal "model can't do geometry," the other three become irrelevant; if they reveal "model can do some geometry," invest in 4.B*.

## Dependency order

```
           ┌──────────────────────────────┐
           │ 4. Hold-out geometric test   │ ← evaluator
           │   (4.A pose + 4.D correspon) │   (without this, we cannot tell
           │                              │    whether any §1/§2/§3 change
           └──────────────┬───────────────┘    actually helps)
                          │ requires trustworthy evaluator
                          ▼
           ┌──────────────────────────────┐
           │ 3. Cross-dataset (3.B)       │ ← data ground
           │   canonical xyz norm         │   (without this, §1/§2 can only
           │                              │    learn "MindCube-3D")
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │ 1. Explicit geom supervision │ ← core inductive bias
           │   (1.A contrastive + 1.D     │
           │    xyz-match classifier)     │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │ 2. R-equivariance aug        │ ← final touch
           │   (2.A input R + rotate      │   (strengthens §1 — do not run
           │    labels; paired with 1.A)  │    standalone)
           └──────────────────────────────┘
```

### Minimum viable path

Two items must come first, in this order:

1. **§4 (hold-out geometric test)** — especially 4.A (pose) and 4.D (correspondence), both auto-labeled. Without a geometry-necessary evaluator we can't know whether any training intervention works.
2. **§1.A (contrastive correspondence)** — cheapest, most direct supervision that punishes shortcut use.

Run these two on the existing decouple / coord checkpoints first. That determines the baseline geometric ability (expected: near-zero for both, from the xyz=0 evidence). Any subsequent training intervention then has a numeric target.

### Full path

Only after §4 shows the baseline, consider the investment order §3 → §1 → §2 as drawn in the dependency diagram. §2 without §1 is actively counterproductive (reinforces shortcut); §3 alone without §1/§2 is just mixing OOD garbage into training.

## Risks and open questions

1. **Is there enough capacity in LoRA r=16 for genuine geometry?** Possibly not. The current LoRA is small (a few M params). Learning a 3D prior from scratch with that budget may be infeasible; full fine-tuning or larger LoRA (r=64+) may be needed for any of §1–§3 to show real gains.

2. **Does pretrained Qwen's vision encoder already have a 3D prior?** If yes, the vision tokens already carry 3D info and xyz is genuinely redundant — the ablation results become much less diagnostic. Quick probe: compare vision-only (frozen Qwen + linear probe) on 4.A pose vs vision+xyz. If vision alone scores well, the "learn geometry from xyz" framing is the wrong framing; we should instead ask "how to make the vision encoder geometric-aware."

3. **What's the right xyz representation?** The whole campaign uses Cartesian (and log-polar for §5 of spa_emb.py). But depth + 2D patch index might be a better parametrization for an image-conditioned model than world-frame xyz. Worth an ablation before committing to 3.B's full canonicalization pipeline.

4. **Does any of this actually matter for the end task?** If the target is "answer QA about rotational cubes," we may not need geometric 3D at all — MindCube's QA is 95 % solvable with content shortcuts. The geometry investment is only justified if the downstream application (robotics, navigation, etc.) requires real 3D reasoning. Worth re-confirming the end-task requirements before investing.
