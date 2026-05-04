# `atten` (SpatialAttentionBias) couple ckpt — xyz validation across MindCube + SpinBench

Companion to [xyz_zero_ablation.md](xyz_zero_ablation.md), but for the `atten`
method (per-layer learnable bias on V↔V attention, Qwen 3D M-RoPE unchanged)
trained jointly with LoRA. Goal: quantify how much of the headline 98% on
MindCube tinybench is real geometric reasoning vs. dataset-shape memorization,
and locate where the geometric pathway is alive vs. dead.

## 0. ckpt under test

`train_records/atten_couple_mindcube/step_624_final` (May 4, 2026).

Training config (from `train.log`):
- joint training (`couple=True`): LoRA on q/k/v/o/gate/up/down + per-layer
  `SpatialAttentionBias` MLP both trainable
- 6 epochs over MindCube_train.jsonl (10K samples), 6×NVIDIA RTX PRO 6000
- LR = 2e-4 (LoRA), 2e-3 (bias module via `bias_lr_scale=10`)
- W₂ init = `N(0, 0.01²)` (`bias_w2_init_scale=0.01`)
- step_624 = end of cosine schedule, LR landed at 0 → fully converged

8 of 36 LLM layers carry SpatialAttentionBias (the standard-attn layers
L03/07/11/15/19/23/27/31; the 24 linear-attn layers are skipped because
their kernel ignores the additive attention_mask).

## 1. MindCube tinybench (in-distribution)

200 samples, full ablation = pass-1 (real xyz) + pass-2 (image_xyz=0).

| pass | correct / 200 | accuracy |
|---|---:|---:|
| normal xyz | 196 | **98.0%** |
| xyz = 0    | 171 | 85.5% |
| **Δ accuracy** |  | **−12.5%** |

Per-sample comparison:

| metric | value |
|---|---:|
| predictions differ between passes | 25 / 200 |
| xyz HELPS (normal-correct, zero-wrong) | **25** |
| xyz HURTS (zero-correct, normal-wrong) | **0** |
| both wrong but different | 0 |
| net wins | **+25** |

**Every single xyz-driven prediction flip is a correct one.** This is the
textbook "xyz is real signal, not noise" pattern.

## 2. SpinBench full set (out-of-distribution)

2739 samples (1836 unique image-sets, mostly 4-frame perspective-taking).

### 2.1 Strict vs lenient extraction

| extractor | normal | zero | Δ |
|---|---:|---:|---:|
| strict (`<answer>X</answer>` only — eval default) | 33.88% | 31.25% | −2.63% |
| **lenient (tag OR raw `^[A-F]`)** | **66.5%** | **64.9%** | **−1.64%** |

The strict-extraction 34% number is misleading: SpinBench's single-image
prompts include "Only answer with a single capital letter from (A, B)…" and
the model **correctly obeys** by emitting raw `A` / `B`, but the eval-default
extractor only recognizes `<answer>X</answer>` and scores those 926 raw-letter
answers as empty strings → 0% on single-image. All numbers below use the
lenient extractor unless noted.

### 2.2 single (1 image) vs multi (≥2 images)

| bucket | n | normal | zero | Δ | wins / losses | net |
|---|---:|---:|---:|---:|---|---:|
| **single (n=1)** | 926 | **82.2%** | 82.4% | **+0.22%** | 5 / 7 | **−2** |
| **multi (n≥2)** | 1813 | 58.5% | 55.9% | **−2.59%** | 116 / 69 | **+47** |
| ALL | 2739 | 66.5% | 64.9% | −1.64% | 121 / 76 | +45 |

**The xyz pathway is dead on single-image inputs** (Δ ≈ 0, 5 vs 7 = pure
noise). It only carries signal in multi-image settings, where Δ is a real
−2.6%.

### 2.3 Fine-grained by exact image count

| n_imgs | n | normal | zero | Δ | net wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 926 | 82.2% | 82.4% | +0.22% | −2 |
| 2 | 509 | 52.1% | 49.3% | −2.75% | +14 |
| 3 | 341 | **85.9%** | 85.0% | −0.88% | +3 |
| 4 | 745 | 58.0% | 54.6% | **−3.36%** | **+25** |
| 5 | 218 | 32.6% | 30.3% | −2.29% | +5 |

3-image bucket is highest accuracy but smallest Δ — the model solves these
by image content alone (LLM identifies objects directly), xyz barely
contributes. 4-image bucket has the largest Δ (−3.36%, net +25 wins) and
matches the MindCube training distribution most closely (74% of MindCube
training is 4-frame). 5-image is OOD relative to MindCube training and
accuracy collapses to 32.6%.

## 3. MindCube vs SpinBench side-by-side

| metric | MindCube tinybench | SpinBench (lenient) |
|---|---:|---:|
| accuracy normal xyz | **98.0%** | 66.5% |
| Δ accuracy (zero − normal) | −12.5% | −1.64% |
| net wins (xyz fix − xyz break) | +25 | +45 |
| **net wins / n** | **+12.5%** | **+1.6%** |

xyz-pathway "signal density" on out-of-distribution data is **8× lower**
than in-distribution.

## 4. Mechanistic check — B-probe on the same ckpt

Per-layer SpatialAttentionBias output, evaluated on 8 real MindCube tinybench
samples (real xyz vs `xyz=0`):

| layer | ‖B‖_F | max\|B\| | std(B) | rel_Δ = ‖B−B₀‖/‖B‖ | corr(B, B₀) |
|---:|---:|---:|---:|---:|---:|
| L03 | 3236 | 7.6 | 1.45 | 0.91 | +0.52 |
| L07 | 1547 | 5.0 | 0.71 | 1.02 | +0.05 |
| L11 | 2488 | 6.5 | 1.00 | 1.20 | −0.40 |
| L15 | 3585 | 6.7 | 1.63 | 0.93 | +0.34 |
| L19 | 2893 | **11.1** | 1.29 | 1.05 | −0.34 |
| L23 | 1693 | 4.2 | 0.71 | 0.78 | +0.78 |
| L27 | 1075 | 2.9 | 0.47 | 0.82 | +0.73 |
| L31 | 153 | 0.5 | 0.07 | 0.96 | +0.32 |
| **mean** | **2084** | **5.56** | **0.96** | **+0.25** |

**At the per-tensor level B looks random**: rel_Δ ≈ 0.96 and corr(real, zero)
≈ +0.25 mean B(xyz_real) and B(xyz=0) are roughly orthogonal in Frobenius
sense. Yet system-level Δaccuracy is −12.5% on MindCube and −2.6% on
SpinBench-multi. Resolution: **LoRA learned to read out the useful directions
from the random-looking B**, not the bias MLP itself.

L31 is largely silent (max\|B\| = 0.5, ‖B‖ 10–20× smaller than L03/L19) —
the model is suppressing geometric bias at the deepest layer.

## 5. Weight-matrix training movement

‖·‖_F means across the 8 SpatialAttentionBias layers, compared to init:

| matrix | init | trained | movement |
|---|---:|---:|---|
| W₁ (4 → 128, Kaiming bound 0.5) | 6.53 | 6.62 | **+1.4%** |
| b₁ (U(−0.5, 0.5)) | 3.27 | 3.39 | +3.7% |
| W₂ (128 → 16, N(0, 0.01²) init) | 0.45 | 1.41 | **+213%** |
| b₂ (zeros) | 0.00 | 0.10 | — |

**W₁/b₁ essentially did not train.** Per-layer `max|W₁| = 0.50–0.54` —
still pinned at the Kaiming-uniform init bound. The first layer of the
geometric MLP is a fixed random projection. Only W₂ (and weakly b₂) moved.

This means the MLP is operating as a **random-feature regressor**: a fixed
random basis projects (n_x, n_y, n_z, d) into 128 dims, and W₂ + LoRA learn
to read out useful directions from this fixed basis. Why W₁ doesn't train:

- with W₂ small at init (‖W₂‖_F ≈ 0.45), `∂L/∂W₁ ∝ W₂ᵀ · ∂L/∂(post-act)` is
  small from step 0 — gradient flow is bottlenecked
- LoRA on q/k/v/o (21M trainable params) can absorb the task signal more
  efficiently than a 5K-param geometric MLP, so most of the gradient flows
  there

## 6. What this means

1. **MindCube 98% is partly real, partly memorized.** The xyz pathway is
   genuinely consumed (Δ=−12.5%, all flips are gains). But on out-of-
   distribution SpinBench, accuracy drops 32 pp and xyz contribution shrinks
   8× in density. The model learned "MindCube-flavored multi-frame
   perspective comparison" rather than generic geometric reasoning.

2. **xyz contribution is multi-image only.** Single-image V↔V attention
   bias is structurally weak (all patches share the same camera frame, so
   pairwise geometry is trivial), and LoRA never had to learn how to use it
   during training (MindCube is 100% multi-image). On SpinBench single-image
   depth-comparison questions, the model still gets 82.2% — but purely from
   image features, with xyz contributing 0.

3. **The "geometric MLP" is barely geometric.** W₁ is at random-init; only
   W₂ trained. The system works because LoRA learned to decode a fixed
   random projection of (n_x, n_y, n_z, d). This is a fragile architecture:
   if LoRA capacity were reduced or the LoRA were frozen, the xyz pathway
   would collapse back to noise (the 5e-5 LR ckpt's 0% Δ confirms this).

4. **B amplitude is now uncomfortably large.** max\|B\| = 5.56 mean,
   L19 hits 11.1. Pre-softmax attention logits are O(1); B on V↔V cells is
   ~5–10× larger. If W₂ keeps growing on longer training, B can saturate
   softmax and break pretrained attention. L31's near-silent behavior may
   already be a regularization response.

## 7. Follow-ups

Priority A (cheap diagnostics):

- **Repeat xyz_validation on MMSIBench / SAT** to confirm the OOD pattern
  is general and not SpinBench-specific.
- **Fix the eval extractor** in `evaluation.py`: fall back to "first
  capital letter" when `<answer>X</answer>` is missing. Affects SpinBench /
  SAT / EmbSpatial scoring.

Priority B (potentially better xyz pathway):

- **Bias-only training** (`--couple` OFF, the new default introduced in
  this branch) — freeze LoRA, force all gradient through W₁/W₂. Test
  whether W₁ actually moves and whether B becomes correlated with task
  geometry rather than random projections.
- **Two-stage training**: bias-only for ~1 epoch (lets W₁ specialize),
  then unfreeze LoRA for joint training (let LoRA fine-tune around the
  now-meaningful B). May break the random-feature ceiling.
- **B output normalization** (LayerNorm or `tanh × scale` after MLP) to
  prevent max\|B\| from growing unboundedly with training.

Priority C (architectural):

- **Multi-image bias on linear-attn layers** would 4× the wrapped layer
  count, but requires a custom kernel that respects additive bias.
- **Per-layer learnable scalar gate** initialized to 1 — let the model
  decide which layers actually want B (currently L31 has manually
  suppressed itself; making this explicit could help others do the same
  cleanly).

## 8. Reproduction

```bash
# MindCube tinybench (in-distribution)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/xyz_validation.sh \
    --method atten \
    --ckpt   train_records/atten_couple_mindcube/step_624_final \
    --datasets mindcube --gpus 0,1,2,3 --limit 200 \
    --output vis_results --run_name xyz_val_atten_couple_mc_step624

# SpinBench full set (out-of-distribution)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/xyz_validation.sh \
    --method atten \
    --ckpt   train_records/atten_couple_mindcube/step_624_final \
    --datasets spinbench --gpus 0,1,2,3 \
    --output vis_results --run_name xyz_val_atten_couple_spinbench

# image-count bucketing (after each xyz_validation)
python probe_xyz_by_imgcount.py \
    --val_dir vis_results/xyz_val_atten_couple_spinbench/spinbench \
    --jsonl   datasets/evaluation/spinbench_data/test.jsonl

# B-probe (per-layer SpatialAttentionBias output stats)
python probe_atten_bias.py \
    --ckpt   train_records/atten_couple_mindcube/step_624_final \
    --data_dir datasets/evaluation/MindCube \
    --n_samples 8 --num_heads 16 --hidden_dim 128 \
    --out vis_results/xyz_val_atten_couple_mc_step624/probe_B.json
```

## 9. Output artifacts

- MindCube validation: `vis_results/xyz_val_atten_couple_mc_step624/mindcube/`
- SpinBench validation: `vis_results/xyz_val_atten_couple_spinbench/spinbench/`
- B-probe stats: `vis_results/xyz_val_atten_couple_mc_step624/probe_B.json`
- Helper scripts: `probe_atten_bias.py`, `probe_xyz_by_imgcount.py`


## 10. Cross-dataset task analysis (SpinBench task-types + SAT_real + EmbSpatial)

Question being asked: is the +47-net-wins on SpinBench actually distributed
evenly across tasks, or concentrated on a few task families that happen to
match MindCube? And does the "single-image xyz dead" pattern hold on other
datasets, or is it a SpinBench artifact?

### 10.1 SpinBench by task-type cluster

53 raw `metadata.task_type` values clustered into 8 semantic groups:

| cluster | task | n | normal | zero | Δacc | wins/losses | net | wins/n |
|---|---|---:|---:|---:|---:|---|---:|---:|
| **A** | spatial-relation grounding (1-img, L/R/F/B/Far/Near) | 636 | **94.3%** | 94.0% | −0.31% | 2/0 | +2 | +0.31% |
| **B** | dynamic spatial relations (multi-img) | 156 | 50.0% | 50.0% | 0.00% | 15/15 | 0 | 0% |
| **C** | perspective-taking transformation (1-img, w/wo premise) | 290 | 55.5% | 56.9% | **+1.38%** | 3/7 | **−4** | −1.38% |
| **D** | mental rotation (multi-img, image-only) | 218 | 32.6% | 30.3% | −2.29% | 27/22 | +5 | +2.29% |
| **E** | rotation selection (4-img, occlusion variants) | 323 | 35.6% | 31.0% | **−4.64%** | **28/13** | **+15** | **+4.64%** |
| **F** | canonical view selection (multi-img) | 358 | 60.9% | 57.3% | **−3.63%** | 22/9 | +13 | +3.63% |
| **G** | rotation classification (multi-img) | 353 | 53.0% | 49.0% | **−3.97%** | 21/7 | +14 | +3.97% |
| **H** | identity matching (multi-img) | 405 | **96.8%** | 96.8% | 0.00% | 3/3 | 0 | 0% |
| **TOTAL** | | 2739 | 66.5% | 64.9% | −1.64% | 121/76 | +45 | +1.64% |

Three findings on SpinBench alone:

1. **Three clusters carry 93% of net wins** (E + F + G = +42 / +45). All
   three involve **multi-view rotation / canonical-view geometry** — the
   structural match to MindCube training (74% 4-frame, all
   "perpendicular" category). xyz pathway is not generic; it activates
   only on tasks that look like training.
2. **Cluster C is a regression** (Δ=+1.38%, **net −4 wins**, 7 losses vs
   3 wins). Single-image perspective-taking ("if I were where X is, where
   would Y be") is the *only* task type where xyz **actively hurts** on
   SpinBench. Same shape as the SAT findings below.
3. **Clusters A and H are at task ceiling** (94.3% / 96.8%). xyz cannot
   contribute (no headroom). Δ≈0 here is "no-op", not "broken".

### 10.2 Cross-dataset replication: SAT_real + EmbSpatial

Same atten_couple_mindcube/step_624_final ckpt, same xyz=0 ablation:

| dataset | n | normal | zero | **Δacc** | net wins | net/n | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| MindCube tinybench | 200 | 98.0% | 85.5% | **−12.50%** | +25 | +12.5% | xyz strongly helps |
| SpinBench | 2739 | 66.5% | 64.9% | **−1.64%** | +45 | +1.6% | xyz net helps |
| EmbSpatial-Bench (all 1-img) | 3640 | 76.9% | 76.9% | **+0.05%** | −2 | −0.05% | xyz dead (noise floor) |
| **SAT_real (mixed 1+2 img)** | 150 | 67.3% | **74.0%** | **+6.67%** | **−10** | **−6.7%** | **xyz harmful** |

SAT_real broken down further:

| bucket | n | normal | zero | **Δacc** | wins/losses | net |
|---|---:|---:|---:|---:|---|---:|
| n_imgs=1 (overall) | 104 | 67.3% | 68.3% | +0.96% | 1/2 | −1 |
| **n_imgs=2 (overall)** | 46 | 67.4% | **87.0%** | **+19.57%** | **2/11** | **−9** |
| —— ego_movement (n=2) | 23 | **60.9%** | **95.7%** | **+34.78%** | **0/8** | **−8** |
| —— obj_movement (n=2) | 23 | 73.9% | 78.3% | +4.35% | 2/3 | −1 |
| —— action_conseq (n=1) | 37 | 81.1% | 81.1% | 0.00% | 0/0 | 0 |
| —— goal_aim (n=1) | 34 | 79.4% | 82.4% | +2.94% | 1/2 | −1 |
| —— perspective (n=1) | 33 | 39.4% | 39.4% | 0.00% | 0/0 | 0 |

EmbSpatial broken down by spatial relation (all single-image):

| relation | n | normal | zero | Δacc | net |
|---|---:|---:|---:|---:|---:|
| right | 620 | 79.2% | 79.7% | +0.48% | −3 |
| left | 616 | 89.8% | 89.6% | −0.16% | +1 |
| close | 612 | 66.3% | 65.5% | −0.82% | +5 |
| under | 602 | 88.5% | 89.0% | +0.50% | −3 |
| above | 596 | 80.7% | 80.7% | 0.00% | 0 |
| far | 594 | 56.2% | 56.6% | +0.34% | −2 |

All 6 relations land within ±1% — statistically equivalent to zero. The
xyz pathway on a 3640-sample full single-image dataset is **completely
inert**: any 50/50 of the 50 sample swings are pure noise.

### 10.3 The headline outlier — SAT ego_movement (Δ = +34.78%)

```
Task: 2-frame ego-motion reasoning ("how does the scene change after
      the camera moves?")

Normal xyz: 14 / 23 = 60.9%
xyz = 0   : 22 / 23 = 95.7%
Δ accuracy: +34.78%        ← turning xyz off nearly doubles accuracy
wins/losses: 0 / 8         ← every xyz-driven prediction flip is WRONG
```

Pure destructive signal: 8 prediction changes, 0 of them correct. The
xyz pathway in this task isn't carrying information — it's
systematically pulling the answer toward the wrong choice. Mechanism:
MindCube training is 100% "camera orbits stationary object" 4-frame
geometry; SAT ego_movement is "camera translates, scene fixed" 2-frame.
The V↔V correspondence pattern the LoRA learned to decode for the
former is misapplied to the latter.

### 10.4 Revised hypothesis — it's *training-distribution* dependence, not *image-count* dependence

The original SpinBench-only reading was "xyz dead on single-image, xyz
helps on multi-image". Cross-dataset data refines this to three distinct
regimes:

| regime | xyz behavior | datasets confirming |
|---|---|---|
| **single-image (any dataset)** | dead — Δ ≈ 0, no help and no harm | EmbSpatial all (Δ=+0.05%), SpinBench A (Δ=−0.31%), SAT n=1 (Δ=+0.96%), SpinBench C (Δ=+1.38%) |
| **multi-image, in-distribution** (4-frame rotation/canonical view, matches MindCube) | active — Δ clearly negative | SpinBench E (−4.64%), F (−3.63%), G (−3.97%), MindCube tinybench (−12.5%) |
| **multi-image, out-of-distribution** (2-frame ego/object motion, dynamic spatial relations) | **harmful** — Δ clearly positive | **SAT n=2 (+19.57%), SAT ego_movement (+34.78%)**, SpinBench C 1-img perspective transform (+1.38%) |

→ xyz is **not** a generic geometric prior. It is a learned correlate of
"4-frame perpendicular layout with object correspondences", and the
LoRA decoder is trained to consume it under that specific structure.
When the test sample's geometry differs from this template (whether
because there's only 1 view, or because the views relate via camera
translation rather than rotation), the LoRA decoder produces a
correlated-but-wrong signal that the rest of the model trusts and acts
on.

### 10.5 Per-task pattern stability (SpinBench → SAT)

SpinBench cluster C "perspective-taking transformation" and SAT
"perspective" / "ego_movement" are **the same task family** with
different difficulty:

| dataset / task | n | normal | zero | Δacc | xyz net wins |
|---|---:|---:|---:|---:|---:|
| SpinBench C (1-img perspective transform) | 290 | 55.5% | 56.9% | **+1.38%** | −4 |
| SAT perspective (n=1) | 33 | 39.4% | 39.4% | 0.00% | 0 |
| SAT ego_movement (n=2) | 23 | **60.9%** | **95.7%** | **+34.78%** | **−8** |
| SAT obj_movement (n=2) | 23 | 73.9% | 78.3% | +4.35% | −1 |

The "xyz hurts perspective-style tasks" signal is present on **both**
SpinBench and SAT, with the harm magnitude scaling with how multi-view
the task is. This rules out it being a SpinBench-specific artifact and
makes the regression a robust property of the trained pathway.

### 10.6 Implications for follow-up fixes

The original §7 "follow-up A: bias-only training" is still valid, but
this analysis adds two more concrete actionables:

- **Inference-time gating** (cheap): turn off `image_xyz` when the input
  image-count or task-type signature suggests OOD multi-image (e.g.
  n_imgs=2 with ego-motion-style prompts). Mechanism does not require
  retraining; expected gain is dataset-dependent (SAT total 67.3% →
  ≥75%; SpinBench may dip 1–2 pp).
- **Training-distribution diversification** (expensive): augment
  MindCube with VST `qa`/`depth`/`distance` (single-image), VST
  `correspondence` ego-motion (camera translate), and possibly a small
  fraction of frame-dropped multi-image samples. Expected gain: the
  (single | OOD multi-image) regimes go from "broken" to "neutral or
  helpful", at small cost to the in-distribution +47-wins channel.

### 10.7 Output artifacts (cross-dataset)

- SAT_real validation: `vis_results/xyz_val_atten_couple_taskcheck/sat_real/`
- EmbSpatial validation: `vis_results/xyz_val_atten_couple_taskcheck/embspatial/`
- SpinBench task-cluster bucketing reproducible via the helper script in
  `probe_xyz_by_imgcount.py` (extended in-place to also bucket by
  `category`/task_type — 53 raw types collapse into 8 clusters by name
  prefix).

### 10.8 Reproduction

```bash
# Cross-dataset replication run (4 GPUs, ~25 min):
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/xyz_validation.sh \
    --method atten \
    --ckpt   train_records/atten_couple_mindcube/step_624_final \
    --datasets "sat_real,embspatial" \
    --gpus 0,1,2,3 \
    --output vis_results --run_name xyz_val_atten_couple_taskcheck

# Per-task breakdown (analysis-only, no GPU): use the snippet in
# probe_xyz_by_imgcount.py main(); SAT id resolver and EmbSpatial all-1-img
# resolver are inline in the analysis cell that produced §10.1–10.5.
```
