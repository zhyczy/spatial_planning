# `train_geo.py`: Multi-view Contrastive + xyz-Match Fine-Tuning

## 1. Purpose

`train_geo.py` is the MVP implementation of the training recipe in [../discussion/learning_real_3d_geometry.md](../discussion/learning_real_3d_geometry.md). That discussion argues the coord head in `train_coordinate.py` **cheats** — it regresses xyz from vision features directly rather than forcing the LM to use the xyz channel, so it doesn't actually teach the model to reason about 3D. The doc names four anti-shortcut supervision recipes; `train_geo.py` implements two of them:

- **§1.A Multi-view contrastive correspondence** — positive pairs = same 3D point across views
- **§1.D xyz-image match classifier** — 50 % of steps permute the per-image xyz assignment; a head must classify match vs mismatch

Relation to the other training scripts:

| Script | Auxiliary loss | Shortcut-proof? | Extra trainable params |
|---|---|---|---|
| `train_correspondence.py --decouple` | — | — | 0 (LoRA only) |
| `train_coordinate.py --decouple` | Per-patch xyz L1 regression | No (vision-driven) | ~4M (`DepthPredictionTransformer`) |
| **`train_geo.py`** | §1.A triplet + §1.D match | **Yes (both)** | **5,122** (`nn.Linear(2560, 2)` match head) |

## 2. Loss

```
total_loss = lm_weight · lm_loss
           + contrast_weight · contrast_loss    [§1.A]
           + match_weight    · match_loss       [§1.D]
```

Defaults: `lm_weight=1.0`, `contrast_weight=1.0`, `match_weight=0.3`.

Any term can be turned off at the `lm_*` level:
- `--no_contrast` — skip §1.A computation entirely (not just weight=0)
- `--no_match` — skip §1.D and don't allocate the match head

## 3. Position-embedding backbone — `train_correspondence.py --decouple` verbatim

`train_geo.py` does **not** modify the position-embedding path. It uses:

| | Value |
|---|---|
| Backbone | `SpaDecForConditionalGeneration` |
| `mrope_section` | `[11, 11, 10]` **UNCHANGED** (Qwen 3D M-RoPE in rotary 64 dims) |
| `partial_rotary_factor` | 0.25 (unchanged) |
| XYZ RoPE dims | `64..129` in pass-through (66 dims) |
| `rope_theta` (XYZ RoPE) | 10000 (Cartesian) |
| `coord_scale` | 100.0 |
| `polar` | `False` (never passed) |
| Post-LoRA hook | `patch_attention_layers_dec(spa)` |
| Gradient checkpointing | `use_reentrant=False` |
| LoRA config | `r=16, α=2r, {q,k,v,o,gate,up,down}_proj, dropout=0.05, bias="none"` |

See [decouple_xyz_rope.md](decouple_xyz_rope.md) for the XYZ RoPE algebra; it is unchanged here.

## 4. §1.A Multi-view contrastive correspondence

### 4.1 What it does

Within one training sample's N views (MindCube: N=4), build **positive pairs** (patch p in image i, patch q in image j) that point to the **same 3D location**, plus a **negative** (random other patch k ≠ p in image i). Minimize triplet margin loss on the last-layer vision-token hidden states so same-3D-point patches get pulled closer than unrelated patches from the same view.

### 4.2 Data flow

```
image_xyz[i] ∈ ℝ^(H_i·W_i × 3)            # per-patch 3D coords (no-grad)
per_image[i] ∈ ℝ^(H_i·W_i × 2560)         # last-layer hidden state, gradient-carrying

for each ordered (i, j), i ≠ j:
    dmat   = cdist(image_xyz[i], image_xyz[j])              # (n_i, n_j)       no-grad
    dmin[p], qmin[p] = dmat[p].min()                         # nearest neighbor q in j
    valid  = { p | dmin[p] < contrast_eps }                   # only real 3D-point matches
    sample P ≤ n_contrast_anchors anchors from valid
    k      = uniform random(0..n_i-1), k ≠ p                  # same-image negative

    L_ij = mean_p  max(0, margin
                           + ‖per_image[i][p] − per_image[j][q]‖²     # anchor–positive
                           − ‖per_image[i][p] − per_image[i][k]‖²)    # anchor–negative

contrast_loss = mean over (i, j) pairs
```

No `match_head`-style pooling, no projection head — the triplet acts directly on the 2560-dim hidden states. Any linear projection would just let the gradient route around the constraint.

### 4.3 When it fires

- **Train**: only on non-permuted steps (permutation is §1.D's thing; positive-pair finding depends on *correct* xyz). With `match_prob=0.5`, contrast fires on ~50 % of steps.
- **Eval**: always fires (see §7; eval is always un-permuted).
- **`--no_contrast`**: never fires; the triplet loop is skipped.

### 4.4 Hyperparameters

| Flag | Default | Meaning |
|---|---|---|
| `--contrast_margin` | 0.5 | Triplet margin (d_an − d_ap must exceed this to get zero loss) |
| `--contrast_eps` | 0.05 | xyz L2 threshold for positive-pair selection (scene units; ≈ 5 cm when xyz is in meters) |
| `--n_contrast_anchors` | 64 | Max anchors per (i, j) pair (bounds cost: N·(N−1)·64 ≤ 768 triplets/step) |

## 5. §1.D xyz-image match classifier

### 5.1 What it does

Anti-shortcut binary supervision: with probability `match_prob`, permute the per-image xyz assignment within the sample (image k gets image π(k)'s xyz). A match head reads the per-image pooled last-layer hidden state and predicts `{matches (1), mismatches (0)}`. To classify correctly under permutation, the model must **actually check** whether the xyz channel is geometrically consistent with the image content — there's no vision-only shortcut.

### 5.2 Data flow

```
# Decide permutation (training only; see §7 for eval)
if use_match and training and N ≥ 2 and random() < match_prob:
    π ← non-identity random permutation of {0,…,N-1}
    image_xyz_fwd[k] = image_xyz[π(k)]         # swap
    match_label[k]   = 1 if π(k) == k else 0
else:
    image_xyz_fwd = image_xyz
    match_label   = [1, 1, …, 1]

# Backbone forward with image_xyz_fwd through decouple's XYZ RoPE
hidden = spa_model(image_xyz=image_xyz_fwd, …).last_hidden_state   # via lm_head pre-hook

# Per-image pooling + linear classifier
pooled[k] = mean(hidden[vis_pos of image k])       # (N, 2560)
m_logits  = match_head(pooled)                      # (N, 2)
match_loss = CrossEntropy(m_logits, match_label)
```

### 5.3 When it fires

- **Train**: every step when `use_match=True` and N ≥ 2. Permutes on `match_prob` fraction of steps; the other fraction has label=[1,…,1] (trivial but still provides grad).
- **Eval**: fires in the forward, but permutation is **gated by `self.training`** so eval never permutes. The eval `match_loss` is against label=[1,…,1], which is degenerate (near-zero as the head learns). It is **excluded from wandb and INFO logs on eval** to avoid misleading flat curves.
- **`--no_match`**: match head not allocated (`self.match_head = None`); permutation skipped; term omitted from total loss.

### 5.4 Hyperparameters

| Flag | Default | Meaning |
|---|---|---|
| `--match_prob` | 0.5 | Probability of permuting xyz on a training step. Higher → more permuted steps (stronger match signal, but contrast fires less often since contrast is skipped on permuted steps) |
| `--match_weight` | 0.3 | Weight on match CE (binary CE magnitude is small; 0.3 is comparable to `contrast_weight=1.0` after normalization) |

## 6. New trainable parameters

Only one new module on top of the decouple backbone + LoRA:

```python
self.match_head = nn.Linear(hidden_dim=2560, 2).to(bfloat16)
#   weight: (2, 2560) = 5120
#   bias:   (2,)      = 2
#                       -----
#                       5122 params
```

Allocated only when `use_match=True`. Checkpointed separately as `match_head.pt` alongside the LoRA adapter.

Contrast has **zero** new parameters — triplet distances operate directly on hidden states.

## 7. Train / eval parity

Both train and eval call `GeoModel.forward` with the same signature:

```python
_, loss, loss_dict = model(
    input_ids, attention_mask, pixel_values, image_grid_thw,
    image_xyz,         # list of (llm_H_k, llm_W_k, 3), Cartesian, first-cam frame
    labels,
)
```

The only behavioral difference between train and eval is **one gated branch** inside `GeoModel.forward`:

```python
if self.use_match and self.training and N >= 2 and random() < match_prob:
    # permute
```

Because `self.training=False` during eval, `do_permute` is always `False` on eval:

|  | Train | Eval |
|---|---|---|
| `image_xyz` fed to `spa_model` | permuted with prob `match_prob`, else correct | always correct |
| `contrast_loss` | fires on non-permuted steps (~50 %) | fires every step |
| `match_loss` | mixed label (real discriminability) | label always 1 (trivial) |

This design is deliberate:

1. We want the eval `contrast_loss` to be meaningful → eval must use un-permuted xyz.
2. We don't want eval metrics to jitter step-to-step from random permutations.
3. The match head's **training** signal requires both classes (achieved via training-time permutation); its **eval** utility is limited and therefore suppressed from the wandb dashboard.

### Dataset choice

| | Train | Eval (mindcube & spinbench) |
|---|---|---|
| Class | `MindCube_Train_Dataset` | `Eval_Dataset_Coord` (mirrors `train_coordinate.py`) |
| Fields used | `image_xyz`, `labels`, `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` | same |
| Extra fields returned | — | `image_xyz_hires` (unused by GeoModel) |

### wandb metrics per eval

```
eval/{ds}_lm_loss        ← real — answer quality
eval/{ds}_acc            ← real — top-1 on first answer token
eval/{ds}_contrast_loss  ← real — cross-view geometry signal
# match_loss / match_acc are filtered out (see §5.3)
```

## 8. Ablation flags

All combinations share the same decouple backbone and LoRA config; they differ only in which aux losses are active.

| Command | `use_contrast` | `use_match` | Effective objective |
|---|---|---|---|
| `bash scripts/train_geo.sh 6` | ✓ | ✓ | lm + contrast + match (full) |
| `bash scripts/train_geo.sh 6 --no_match` | ✓ | ✗ | lm + contrast only (§1.A gain) |
| `bash scripts/train_geo.sh 6 --no_contrast` | ✗ | ✓ | lm + match only (§1.D gain) |
| `bash scripts/train_geo.sh 6 --no_contrast --no_match` | ✗ | ✗ | lm only (baseline — equivalent to `train_correspondence.py --decouple`) |

Run-name stamping (only on non-default values):
`_r{rank}`, `_nocontrast`, `_nomatch`, `_cw{w}`, `_mw{w}`.

## 9. Hidden-state capture

Same lm_head pre-hook pattern used in `train_coordinate.py` (see [train_coordinate.md §3](train_coordinate.md#3-architecture)):

```python
for name, mod in spa_model.named_modules():
    if name.endswith("lm_head"):
        mod.register_forward_pre_hook(self._capture_last_hidden)
        break
# in forward:
outputs = self.spa_model(..., output_hidden_states=False, ...)
hidden = self._last_hidden        # post-norm last hidden state
```

This avoids `output_hidden_states=True`, which both costs activation memory and triggers a NaN bug on `SpaDecTextModel` + gradient checkpointing (documented in `train_coordinate.md §6.2`).

## 10. DDP notes

```python
DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```

- Contrast has **no learnable params** → no unused-parameter risk
- `match_head` (when enabled) fires on every step with N ≥ 2
- LoRA params in q/k/v/o/gate/up/down_proj always receive `lm_loss` gradient

`find_unused_parameters=True` was originally set out of caution but PyTorch's warning correctly flagged it as a performance hit with no benefit. Now `False`. Caveat: using `max_images=1` **and** `use_match=True` would leave `match_head` without grad on that step → DDP error; avoid that combination (the defaults have `max_images=4`, so this is only a smoke-test edge case).

## 11. Checkpoint layout

```
{output_dir}/
  step_{global_step}/
      adapter_model.safetensors       # LoRA adapter (peft.save_pretrained)
      adapter_config.json
      match_head.pt                   # nn.Linear state_dict (if use_match=True)
      …tokenizer files
  step_{last}_final/
  train.log / train_rank{k}.log
  wandb/
```

## 12. What this recipe does *not* implement

From the discussion doc's four §1 ingredients:
- ~~§1.B xyz masking + reconstruction~~ — needs a coord head; would cannibalize `train_coordinate.py`'s machinery
- ~~§1.C pose prediction~~ — needs per-view pose GT + 6D rotation head
- From §2: no R-equivariance augmentation (doc explicitly says do not run standalone — wait until §1 + §3 are in place)
- From §3: no cross-dataset canonicalization
- From §4: no held-out geometric transfer test harness (pose / correspondence eval lives separately)

The MVP here is the simplest 2-ingredient combination that's (a) cheap to train, (b) provably shortcut-resistant, (c) fits in one script without touching the position-embedding path.

## 13. Related docs

- Position embedding (unchanged): [decouple_xyz_rope.md](decouple_xyz_rope.md)
- Alternative supervision (baseline comparison): [train_coordinate.md](train_coordinate.md)
- Motivation and recipe menu: [../discussion/learning_real_3d_geometry.md](../discussion/learning_real_3d_geometry.md)
