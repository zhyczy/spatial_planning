# Rotation Encoder Case-by-Case Diversity Analysis

**Date:** 2026-04-22
**Checkpoint:** `train_records/rotation_alternate_mindcube/step_1600`
**Dataset:** MindCube tinybench (1050 samples)
**Script:** `spatial_planning/validate_rotation_diversity.py`

---

## Motivation

Earlier experiments produced an unresolved puzzle:

1. Replacing (h, w) grid-RoPE with world **xyz-RoPE** in [train_alternate.py](../train_alternate.py) improved in-domain eval on MindCube / SpinBench.
2. But Phase B RL (`train_rl.py`, PPO over 24 discrete cube anchors) **hurt** eval performance rather than helping.
3. Hypothesis: SFT's rotation_enc may be learning a dataset-level constant R, not a per-sample rotation. If so, "case-by-case selection via PPO" is structurally impossible because there is no per-sample signal in `lm_loss`.

This validation quantitatively tests whether rotation_enc is case-invariant or case-dependent.

---

## Method

1. Load SFT checkpoint `step_1600` (LoRA + rotation_enc + coord_head).
2. For every sample in MindCube tinybench (1050 items), forward through `CameraTokenRotationEncoder` and record the predicted rotation matrix R (3×3).
3. Compute:
   - **Angle from identity** `θ = acos((tr R − 1) / 2)` — how far from I
   - **Dataset-mean rotation** `R̄` via SVD projection of the element-wise mean onto SO(3)
   - **Angle to R̄** per sample — measures case-specific deviation around canonical
   - **Frobenius deviation** `‖R − R̄‖_F`
   - **ZYX Euler decomposition** (yaw, pitch, roll) + per-axis std across samples
   - **Per-category breakdown** (linear vs perpendicular)

---

## Results

### Summary statistics (N = 1050)

| Metric | Value | Interpretation |
|---|---|---|
| **Angle to identity** | 106.38° ± 0.03° | R is a definite canonical far from I — not a trivial degenerate |
| **Angle to R̄** | **0.07° ± 0.04°** (max 0.22°) | All samples predict essentially the same R |
| **Frobenius ‖R − R̄‖** | 0.0017 ± 0.0011 | Numerical noise, not meaningful variance |
| **Yaw std** | 0.03° | Invariant to input |
| **Pitch std** | 0.04° | Invariant to input |
| **Roll std** | 0.06° | Invariant to input |
| **R̄ Euler (ZYX)** | **(106.53°, 25.08°, 54.04°)** | The dataset canonical the model collapsed to |

### Per-category breakdown

| Category | n | Angle of category-R̄ vs global-R̄ | Within-category std |
|---|---|---|---|
| linear        | 250 | 0.03° | 0.03° |
| perpendicular | 800 | 0.02° | 0.04° |

The category-level R̄ coincides with the global R̄ to within 0.03° — no category-dependent specialization either.

---

## Raw log

```
13:58:40  INFO  ======================================================================
13:58:40  INFO  ROTATION DIVERSITY REPORT  (N=1050 samples, ckpt=.../step_1600)
13:58:40  INFO  ======================================================================
13:58:40  INFO
13:58:40  INFO  [1] Angle to identity  (does R ≈ I?)
13:58:40  INFO      angle_I_deg:  mean=106.38  std=  0.03  min=106.31  max=106.48
13:58:40  INFO
13:58:40  INFO  [2] Angle to dataset-mean R  (per-sample variation around canonical)
13:58:40  INFO      angle_bar_deg: mean=  0.07  std=  0.04  min=  0.01  max=  0.22
13:58:40  INFO      ↑ If ≈ 0, R is essentially constant across the eval set.
13:58:40  INFO
13:58:40  INFO  [3] Frobenius ||R - R_bar||_F  (0 if all equal; max ≈ 2√2 ≈ 2.83)
13:58:40  INFO      fro_norm:      mean=0.0017  std=0.0011  min=0.0001  max=0.0055
13:58:40  INFO
13:58:40  INFO  [4] Per-axis (ZYX Euler) std across samples  (deg)
13:58:40  INFO      yaw   std=  0.03
13:58:40  INFO      pitch std=  0.04
13:58:40  INFO      roll  std=  0.06
13:58:40  INFO
13:58:40  INFO  [5] Euler of R_bar  (the 'dataset canonical' the model collapsed to)
13:58:40  INFO      yaw=106.53  pitch= 25.08  roll= 54.04  deg
13:58:40  INFO
13:58:40  INFO  [6] Per-category R_bar angle-to-global-bar  (does R depend on category?)
13:58:40  INFO      linear                          n= 250  bar-angle-to-global=  0.03°  within-cat std= 0.03°
13:58:40  INFO      perpendicular                   n= 800  bar-angle-to-global=  0.02°  within-cat std= 0.04°
```

Per-sample dump: [../train_records/rotation_alternate_mindcube/step_1600/R_diversity.json](../train_records/rotation_alternate_mindcube/step_1600/R_diversity.json)

---

## Diagnosis

**rotation_enc is a numerically exact constant function on the eval distribution.** The 2.5B-parameter shallow M-RoPE transformer with `CameraTokenRotationEncoder` acts as a 3×3 constant matrix — a 10⁻³-magnitude scatter around the single SO(3) point

```
R̄ ≈ Rot_ZYX(yaw=106.53°, pitch=25.08°, roll=54.04°)
```

This is **not** a training bug — it is the inevitable equilibrium under three facts established in earlier analysis:

1. **`lm_loss` is only weakly rotation-sensitive** (per-sample std across 24 anchors ≈ 0.015, relative ~6%).
2. **Per-sample gradient `∂lm_loss/∂R` is mostly noise.** When backpropagated over the dataset it averages out, leaving only a dataset-consistent direction.
3. **SFT backprop *exploits* this weak consistent signal** to pull rotation_enc toward a single R that makes all scenes slightly better under xyz-RoPE. The model's capacity to do anything else is suppressed by LoRA-only trainable params and the weakness of the per-sample signal.

---

## Implications for the RL pipeline

The RL experiment (`train_rl.py`) is trying to learn **case-by-case** anchor selection from per-sample `-w_lm · lm_loss(k)` reward. This validation shows the premise is structurally broken:

- SFT already discovered that the only useful rotation signal is a **single constant R** (0.07° scatter across 1050 samples).
- Any PPO policy that outputs different R for different samples is, by definition, worse than the SFT consensus.
- PPO's advantage normalization amplifies noise (per-sample reward std ~0.015 → advantage ~O(1)) and **destroys** the SFT consensus without replacing it with anything meaningful.
- Observed mode collapse (`unique_k=4/24`) and eval regression after Phase B are not bugs — they are the expected outcome of optimizing per-sample with no per-sample signal.

**PPO over 24 discrete anchors with lm_loss reward cannot work in this setup.**

---

## Recommended next steps

### A. Cheap ablation (highest information gain per compute)

Replace rotation_enc entirely with the **fixed constant** `R̄ = Rot_ZYX(106.53°, 25.08°, 54.04°)` and re-run eval:

- If accuracy within 1pt of step_1600 → **delete rotation_enc, save 40MB + Phase B compute**
- The rotation path collapses to "xyz-RoPE + a fixed canonical rotation + LoRA", which is a much cleaner architecture

### B. Reframe the RL problem

If case-by-case rotation is truly desired, PPO-from-lm_loss is the wrong tool. Viable alternatives:

1. **Direct supervision from `gt_rotation`** (when dataset bucket provides it) — gives real per-sample gradient
2. **`coord_loss` as signal** (target = R @ xyz_gt is structurally R-dependent, gradient O(dist(R, R_gt)))
3. **Change the LM's input** so rotation becomes answer-relevant:
   - longer responses containing directional words
   - unfreeze Q/K projections so xyz-RoPE channels can actually be attended to
   - explicit direction head trained on gt_rotation

### C. Drop rotation entirely

If A shows the constant R is redundant with what LoRA already learned, the whole "rotation-aware MLLM" framing may be dispensable for this task. The improvement from xyz-RoPE vs (h,w)-grid RoPE is a distance-structure gain, not a rotation-reasoning gain.

---

## Files

- Script: [../validate_rotation_diversity.py](../validate_rotation_diversity.py)
- Per-sample JSON: `train_records/rotation_alternate_mindcube/step_1600/R_diversity.json`
- Related: [train_alternate.md](train_alternate.md), [Train_rotation.md](Train_rotation.md)
