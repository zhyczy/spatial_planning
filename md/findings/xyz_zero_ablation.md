# xyz=0 ablation: does the model actually use the per-patch xyz channel?

Companion to [coord_rotation_robustness_cross_dataset.md](coord_rotation_robustness_cross_dataset.md). The rotation sweep showed near-zero QA spread under 22 R's on both checkpoints — but that does not by itself prove xyz is unused. R·0 = 0 for the zero-xyz input, so zeroing the xyz collapses the 22-R grid to a single evaluation that directly probes "how load-bearing is the xyz channel?"

## 1. Experimental design

Two checkpoints, three datasets, single condition (R = I):

| model | ckpt | xyz enters model via | has coord_head? |
|---|---|---|---|
| **coord** | `coordinate_no_cam_mindcube/step_1000` | 4D M-RoPE `[2, 10, 10, 10]` rotary region | yes |
| **decouple** | `correspondence_mindcube_decouple/step_1000` | XYZ RoPE in pass-through dims 64..129 (rotary `[11,11,10]` untouched) | no |

Datasets: MindCube (1050), SAT (150), SpinBench (2739).

Change: `--zero_xyz` flag in `validate_coord_rotation_robustness.py` replaces every per-patch xyz tensor with a zero tensor of the same shape before the forward pass. Everything else (images, prompts, position-id computation path, coord_head loading) is unchanged. `--identity_only` skips the 22-R grid.

Baseline for comparison: the R = I row from the earlier rotation sweeps (`R_sweep_{dataset}/` for coord, `R_sweep_corresp_decouple_{dataset}/` for decouple).

## 2. QA accuracy: baseline vs xyz = 0

| model | dataset | N | baseline | xyz = 0 | **Δacc** |
|---|---|---:|---:|---:|---:|
| **coord (4D)** | MindCube | 1050 | 95.33% | 74.29% | **−21.05 pp** |
|  | SAT | 150 | 67.33% | 78.00% | **+10.67 pp** |
|  | SpinBench | 2739 | 60.72% | 60.06% | −0.66 pp |
| **decouple** | MindCube | 1050 | 95.24% | 95.33% | +0.10 pp |
|  | SAT | 150 | 67.33% | 76.67% | +9.33 pp |
|  | SpinBench | 2739 | 65.61% | 66.41% | +0.80 pp |

## 3. Per-sample answer stability

Same samples, compare baseline vs xyz = 0 predictions. "Changed" = the predicted letter flipped.

| model | dataset | N | same | changed | change % |
|---|---|---:|---:|---:|---:|
| **coord** | MindCube | 1050 | 790 | 260 | **24.76 %** |
|  | SAT | 150 | 128 | 22 | 14.67 % |
|  | SpinBench | 1836 | 1013 | 823 | **44.83 %** |
| **decouple** | MindCube | 1050 | 1043 | 7 | **0.67 %** |
|  | SAT | 150 | 124 | 26 | 17.33 % |
|  | SpinBench | 1836 | 1706 | 130 | 7.08 % |

Coord model changes ~25 % of MindCube answers and ~45 % of SpinBench answers when xyz is zeroed. Decouple changes <1 % on MindCube.

## 4. coord_mae (coord model only)

| dataset | baseline | xyz = 0 | ratio |
|---|---:|---:|---:|
| MindCube | 0.631 | 1.171 | **1.85 ×** |
| SAT | 107.08 | 1.913 | 0.02 × |
| SpinBench | 1.175 | 1.002 | 0.85 × |

MindCube's coord prediction degrades predictably (≈2 × worse) when xyz is withheld at inference. SAT's baseline coord_mae of 107 is catastrophic (OOD scene scale / convention mismatch — pre-existing issue, not this ablation's). The SpinBench ratio <1 says xyz was slightly **degrading** coord prediction on SpinBench.

## 5. Three conclusions

### 5.1 Decouple's XYZ RoPE channel is near-zero contribution on in-distribution MindCube

On in-distribution MindCube, zeroing the xyz input changes:
- Accuracy by +0.10 pp (well within sampling noise).
- 0.67 % of per-sample predictions (7 out of 1050).

So on MindCube the XYZ RoPE channel contributes ≈ 0 at inference.

On OOD (SAT +9.33 pp, SpinBench +0.80 pp) the story is different — zeroing xyz changes QA non-trivially on SAT, so the XYZ channel *is* firing on OOD inputs, just in a harmful (training-distribution-specialized) way. See §9 for the fuller picture that brings vanilla into the comparison.

### 5.2 Coord model's 4D M-RoPE xyz is load-bearing, but not for the geometric reason you'd expect

MindCube drops 21 pp when xyz is zeroed. 25 % of answers flip. This is decisive in the other direction: the coord model **does** use xyz heavily.

But the mechanism is not "encoding physical 3D geometry." Evidence:
- `coord_mae` only degrades 1.85 × (0.63 → 1.17) on MindCube — coord head still produces reasonable coords without the xyz input, so it's mostly reading from vision features.
- When xyz = 0, all patches of an image get the *same* (0, 0, 0) position code in the rotary region. Only the t axis (image index) differentiates tokens. **All within-image spatial structure in the position codes is destroyed.**
- The LM loses its learned per-image patch-space attention skeleton → QA collapses.

So xyz in the 4D M-RoPE rotary region functions as a **spatial-structure key** that lets the model resolve "which patch is where within image k." It doesn't need to be geometrically correct — it needs to be **consistent with training distribution**. Which leads to:

### 5.3 On OOD, xyz is actively harmful for the coord model

The smoking gun: coord MindCube → **−21 pp** (xyz is necessary); coord SAT → **+10.67 pp** (xyz is harmful); coord SpinBench → ≈ 0 (xyz is ≈ neutral, but 45 % per-sample churn).

This is textbook distribution shift:
- MindCube xyz comes from the same `estimate_3D.py` pipeline with the same camera-intrinsics conventions as training → xyz is a "familiar key" that gates the learned attention.
- SAT xyz comes from a different 3D pipeline with different scene scale / convention (witness the baseline `coord_mae = 107`: coord head itself is scrambled by SAT xyz) → xyz is a "wrong key" that disrupts the learned attention.

So the coord model has **overfit to the training distribution's xyz statistics**. Replacing bad xyz with zero is a cheap out-of-distribution "safety net."

## 6. Reconciling with the rotation-sweep result

The rotation sweep showed both models are ≈ R-invariant for QA. It was tempting to read this as "neither model uses xyz heavily." The xyz = 0 ablation shows that was wrong for the coord model on MindCube:

- coord MindCube: R-sweep spread ~2 pp, but xyz=0 Δacc **−21 pp** → xyz is load-bearing; R just happens to preserve the relative-position structure the model uses.
- decouple MindCube: R-sweep spread ~0.67 pp, xyz=0 Δacc +0.10 pp → both ≈ 0 here (but see §9 for why "≈ 0" is scope-limited to MindCube).

Rotation preserves pairwise position differences modulo axis mixing and applies the same R to all patches of one image, so intra-image relative geometry is largely invariant. Zeroing destroys that structure. R-invariance therefore signals "the model uses relative position structure, not absolute frame" — not "the model doesn't use xyz."

## 7. Implications

1. **For the coord model on OOD:** the pipeline is already in a regime where xyz is hurting more than helping. Either normalize SAT/SpinBench xyz to training convention, or adopt an architecture where xyz is not load-bearing.
2. **For the rotation-robustness report:** the "≈ R-invariant QA" statement needs the caveat "but coord model collapses to 74 % when xyz is zeroed" — R-invariance is not the same as xyz-independence.

## 8. Artifacts

- Ablation runs: `vis_results/xyz0_{coord,deco}_{mindcube,sat,spinbench}/`
- Baseline runs (R = I row of the R-sweep): `vis_results/R_sweep_{dataset}/` (coord), `vis_results/R_sweep_corresp_decouple_{dataset}/` (decouple)
- Launcher: `run_xyz_zero_ablation.sh`
- Script: `validate_coord_rotation_robustness.py --zero_xyz --identity_only`

## 9. Follow-up: "xyz-insensitive at inference" ≠ "xyz-insensitive during training"

The xyz = 0 ablation answers the inference-time question. Bringing the **vanilla** checkpoint (`correspondence_mindcube_vanilla/step_1000`, no XYZ channel anywhere) into the comparison exposes a subtler training-time effect.

### 9.1 Three-model comparison at step 1000

All three checkpoints share identical LoRA config (r = 16, α = 32, all 32 layers, 7 target modules):

| model | MindCube | SAT | SpinBench |
|---|---:|---:|---:|
| **vanilla** (no XYZ anywhere) | 83.14% | **78.00%** | 63.64% |
| **decouple** (XYZ in pass-through) | **95.24%** | 67.33% | 65.61% |
| decouple xyz = 0 | **95.33%** | 76.67% | 66.41% |
| **coord** (XYZ in rotary) | 95.33% | 67.33% | 60.72% |
| coord xyz = 0 | 74.29% | 78.00% | 60.06% |

### 9.2 OOD: decouple xyz = 0 ≈ vanilla

On SAT and SpinBench, zeroing xyz on decouple recovers vanilla-like numbers (±2 pp). This confirms §5.1: at inference, the decouple XYZ RoPE channel contributes ≈ 0 to OOD behavior.

### 9.3 MindCube: decouple xyz = 0 beats vanilla by 12 pp

Here the earlier story breaks. Zeroing xyz on decouple yields 95.33 % on MindCube, versus vanilla's 83.14 %. **Same LoRA config, neither model is using xyz at inference (xyz was never even wired up for vanilla; decouple has it zeroed), yet decouple is 12 pp better.**

So the decouple architecture's advantage on MindCube is not from "using xyz at inference." Something about the training process itself produced a stronger LoRA, even though that LoRA ultimately routes QA through xyz-free channels.

### 9.4 Candidate explanations for the 12-pp MindCube gap

**A. SpaDecAttentionWrapper isn't numerically identical to vanilla attention even when XYZ RoPE outputs zero.**
The wrapper (`src/models/spa_emb_dec.py:240-305`) reimplements the full attention forward (manual q_proj/k_proj, manual q_norm/k_norm, manual RoPE application) instead of calling the stock `Qwen3_5Attention.forward`. Even with XYZ RoPE producing zero contribution, the sequence of operations and intermediate dtypes differs from vanilla's path. Under bfloat16 this can change training gradients enough to guide LoRA toward different minima.

**B. Training-time XYZ exposure acts as a regularizer / gradient shaper.**
Every image patch carries a non-zero xyz through the pass-through 64..129 dims during training. Gradients flow through the XYZ RoPE → attention → logits path. The model eventually learns to "route around" this channel (pushes attention weights to near zero on it), but the training trajectory is different from vanilla's, leading to LoRA weights that happen to fit MindCube better. The +12 pp is then a form of **training-distribution overfitting**: MindCube's xyz provided a shortcut signal during optimization, even if the final model doesn't need it at test time.

**C. Upstream data-pipeline differences.** The vanilla training path may skip `image_xyz` loading entirely, producing subtle differences in dataloading (batch composition, max_images, augmentation) that change the effective training schedule.

**D. Run-to-run variance.** Different random seeds, different nondeterministic kernel ordering. Unlikely to explain 12 pp alone but could contribute.

### 9.5 Refined interpretation of the whole campaign

| model | MindCube | SAT | SpinBench | what inference xyz does |
|---|---:|---:|---:|---|
| vanilla | 83 | **78** | 64 | n/a |
| decouple baseline | **95** | 67 | 66 | ≈ nothing (ablation verified) |
| decouple xyz = 0 | **95** | 77 | 66 | n/a |
| coord baseline | **95** | 67 | 61 | load-bearing (−21 pp if zeroed) |
| coord xyz = 0 | 74 | **78** | 60 | n/a |

Two striking patterns emerge:

- **Vanilla's SAT at 78 % is the "uncontaminated ceiling" on OOD.** Both decouple and coord drop to 67 % on SAT with their xyz channel active (decouple's supposedly-unused XYZ RoPE included). Zeroing xyz on either recovers to ≈ 77-78 %. **Training-time xyz access degrades OOD generalization, period** — it doesn't matter whether xyz enters through the rotary or pass-through region.
- **The 12-pp MindCube gain for decouple/coord over vanilla is training-distribution overfitting**, not generic "adding geometry helps." Evidence: on SAT the same models lose 11 pp relative to vanilla; on SpinBench the gap is modest; and zeroing xyz at inference does not dissolve the MindCube gain, so the overfitting is baked into the LoRA weights themselves.

The updated bottom line:

> Whether xyz enters through the rotary region (coord) or the pass-through region (decouple), if training exposes the model to xyz, the LoRA **always** specializes toward the training distribution's xyz statistics. Inference-time xyz may or may not matter thereafter (decouple: doesn't; coord: does), but the OOD penalty is visible either way. Vanilla, which never saw xyz, is the only configuration that maintains the base Qwen's OOD behavior on SAT.

### 9.6 Proposed test to disambiguate A vs B

To separate "architectural numerical difference at XYZ=0" (A) from "training-time xyz regularization" (B), train:

**decouple-zero-xyz-at-train**: same SpaDecForConditionalGeneration + SpaDecAttentionWrapper architecture, but feed zero tensors as image_xyz during training as well.

- If MindCube ≈ 95 % → explanation **A** (the architecture alone drives the gain, xyz is incidental).
- If MindCube ≈ 83 % → explanation **B** (the training-time xyz signal is what makes the difference).

Either outcome sharpens the story; B would also suggest the decouple design is just a roundabout overfitting machine on this dataset.

## 10. Did the model learn geometry?

The user's sharper question: once we see that xyz works as a training-distribution shortcut and not a geometric channel, is the honest conclusion that **neither checkpoint has learned 3D geometry**?

Collected against a list of what *real* geometry learning would look like, the evidence is uniformly negative:

| Predicted signature if the model learned 3D geometry | Observed |
|---|---|
| **QA drifts under 22 R's** — rotating the scene changes what a camera would see; a geometry-aware model's answer should move or at least shift confidence. | Both models ≤ 1-4 pp spread across 22 R's → ≈ R-invariant. ✗ |
| **xyz = 0 collapses QA toward task floor** — zeroing the per-token 3D coordinate should destroy any geometric reasoning. | Decouple MindCube Δ = +0.10 pp (predictions 99.3 % identical). ✗ |
| **xyz = 0 collapses coord_mae** — if coord_head is decoding geometry from xyz, zeroing xyz should make it unable to produce sensible coords. | MindCube only 1.85 × worse (0.63 → 1.17); SAT 50 × **better** (107 → 1.9). ✗ |
| **Geometry transfers across datasets** — a learned 3D prior should help OOD scenes with their own valid xyz. | Decouple SAT = 67 % vs vanilla 78 %: xyz exposure *hurts* OOD by 11 pp. ✗ |
| **xyz content matters independent of its distribution** — the exact per-patch (x,y,z) values should guide attention geometrically. | What matters is whether xyz matches training distribution statistics (SAT xyz scrambles the coord head to mae = 107; zeroing cleans it up). The contents are a distribution-match key, not a geometry signal. ✗ |

Five independent lines of evidence all say no.

### 10.1 What xyz actually does in each model

Restating the mechanisms precisely:

- **Decouple (train uses xyz, inference ignores it)**: xyz shaped gradient paths during training → LoRA weights absorbed MindCube-specific patterns that are content-correlated with xyz statistics. At inference the XYZ RoPE output can be zeroed without moving QA because the *LoRA* is already carrying the specialization. This is **overfitting via an auxiliary input**, not a geometry prior.

- **Coord (train and inference both use xyz)**: xyz serves as a *position-indexing lookup key* in the rotary region. Image patches of the same image get different xyz → they become position-distinguishable to the LM, and attention develops a learned per-patch pattern that depends on xyz being "the training-distribution kind." When xyz = 0 all patches in an image collapse to the same rotary position code → intra-image position structure is lost → QA drops 21 pp. But coord_mae barely moves (1.85 ×): coord_head is reading mostly from vision features anyway; xyz-as-key is a cheap extra cue, not the actual predictor.

Neither mechanism requires the model to understand what xyz *means*. They only require xyz to be *consistent* with training.

### 10.2 Why "regularization" is the wrong label

Regularization's fingerprint: widens the model's competence — **improves OOD generalization** at a possible in-distribution cost.

What we observe:

| | MindCube (in-dist) | SAT (OOD) |
|---|---:|---:|
| vanilla (no xyz) | 83 % | **78 %** |
| decouple / coord (xyz at train) | **95 %** | 67 % |
| delta | **+12 pp** | **−11 pp** |

The swap is symmetric around vanilla: +12 pp in-dist, −11 pp OOD. This is the **exact opposite** of regularization. It's the fingerprint of a **training-distribution specialization**, a.k.a. overfitting via shortcut features.

A cleaner vocabulary:

- Not "xyz regularizes the model" → **"xyz provides a shortcut feature that the LoRA latches onto."**
- Not "model learned 3D" → **"model learned 'MindCube-3D': the statistical pattern of xyz values in MindCube scenes."**
- Not "xyz gives the model geometry" → **"xyz gives the model a cheap, distribution-bound index that happens to help on in-distribution and hurts on OOD."**

### 10.3 What a geometry-learning pipeline would need to look like

If we want an actual geometric prior in the model, the ablations above prescribe what evidence would be necessary:

1. **Explicit geometric supervision that punishes shortcut use.** Contrastive correspondence loss, pose-from-xyz prediction with held-out poses, or augmentation that varies xyz independently of content — these pull the model off the "xyz = identity key" shortcut.
2. **Rotation-equivariant training.** Data augmentation that actually rotates xyz (and the inferred-to-be-rotated labels) so the model cannot be frame-locked. Without this, the 22-R invariance will persist because the training distribution is frame-constant.
3. **Cross-dataset training.** Training on MindCube alone means "MindCube-3D" is all the model can learn. Mixing SAT / SpinBench / RoboSpatial with different xyz conventions forces the LoRA to find a representation that isn't distribution-bound.
4. **Hold-out geometric transfer test.** Evaluate on a held-out 3D task (e.g., occluded-object localization, relative-depth ordering) that cannot be solved without physical 3D reasoning. Current R-sweep and xyz = 0 are necessary but not sufficient probes.

None of these were present in the current training recipe, which is consistent with the observation that no geometry was learned.

### 10.4 Bottom line

- **Coord**: xyz used as a distribution-bound spatial-index key, not as geometry. Fails OOD because the key is wrong.
- **Decouple**: xyz not used at inference at all; shaped LoRA during training; specialization benefits vanish OOD.
- **Vanilla**: no xyz anywhere. The OOD ceiling of this recipe is **78 % SAT, 64 % SpinBench**. Any "gains" above that on MindCube purchased by adding xyz are training-distribution shortcuts, not genuine geometric competence.

If the goal is a model that reasons about 3D scenes, this campaign argues the current coord/decouple paths are not producing one — they are producing a MindCube-tuned QA model that happens to ingest xyz as a side channel. Any next step toward real geometry learning needs the four ingredients in §10.3.
