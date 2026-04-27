# `--polar`: Decoupled log-spherical XYZ RoPE

**See also**:
- [`train_correspondence_algorithms.md`](train_correspondence_algorithms.md) — high-level overview of all training modes and compatibility matrix
- [`decouple_xyz_rope.md`](decouple_xyz_rope.md) — base architecture that `--polar` extends (with Cartesian input and θ=10000)

## 1. What `--polar` does now

`--polar` is the log-spherical variant of the decoupled architecture from [`decouple_xyz_rope.md`](decouple_xyz_rope.md). It **reuses the same decoupled architecture** (Qwen 3D M-RoPE in the rotary 64 dims untouched + new XYZ RoPE in pass-through dims 64..129 + 126 content dims untouched) but:

1. **Feeds log-spherical `(log r, θ, α)` into the XYZ RoPE** instead of raw Cartesian `(x, y, z)`.
2. **Tightens the XYZ RoPE's `rope_theta` to 1000** (down from `--decouple`'s 10000), matching polar's narrower dynamic range.

`--polar` and `--decouple` are **mutually exclusive** (see §5).

## 2. Head-dim layout (same as `--decouple`)

```
head_dim = 256:
┌────── rotary 64 dims ──────┐┌──── new XYZ RoPE 66 dims ────┐┌── pass-through 126 ──┐
│ Qwen original 3D M-RoPE    ││ log r: 22d  θ: 22d  α: 22d   ││ untouched            │
│ mrope_section = [11,11,10] ││ symmetric per-axis spectrum  ││ content-only         │
│ rope_theta = 1e7 (Qwen)    ││ rope_theta = 1000 (polar)    ││                      │
│ dims 0..63                 ││ dims 64..129                 ││ dims 130..255        │
└────────────────────────────┘└──────────────────────────────┘└──────────────────────┘
     (UNCHANGED, Qwen-3D)            (log-spherical input)         (UNCHANGED)
```

## 3. The log-spherical transform

Applied on-the-fly inside `SpaXYZRotaryEmbedding.forward(..., polar=True)`:

```python
# input per-token xyz (batch, seq, 3) — Cartesian, meters
r       = sqrt(x² + y² + z²)
log_r   = log(clamp(r, 1e-8))                      # ∈ ℝ    scale-invariant
θ       = atan2(y, x)                              # ∈ [-π, π]   azimuth
α       = atan2(√(x²+y²), z)                       # ∈ [0,  π]   inclination

# Text tokens (xyz = 0) stay at (0, 0, 0) → identity rotation preserved
log_r, θ, α = torch.where(r == 0, 0, ·)

# Scaling and freq application (same as Cartesian decouple)
scaled  = stack([log_r, θ, α]) × coord_scale       # (batch, seq, 3)
```

Convention matches `xyz_to_polar` in `src/dataset/train_dataset.py` — so if you ever need to predict these quantities from the coord head, the channel order and semantics align.

### Why log r (not raw r)

RoPE attention depends on **relative** position difference. `log r_i − log r_j = log(r_i / r_j)` is **invariant** to a global scene scaling `r → k·r`. Raw `r` would produce per-dataset attention patterns (e.g. MindCube r ≈ O(1 m) vs SAT r ≈ O(100 m)). Log r gives one consistent encoding across all scene scales.

### Why raw radians (not normalized to [0, 1])

Earlier draft scaled `θ` by `(θ+π)/(2π)` and `α` by `α/π` into `[0, 1]`. This created an asymmetry: `coord_scale=100` corresponded to 3.6° per unit for `θ` but 1.8° for `α`. Using raw radians: one `coord_scale` value gives the same angular resolution to both.

## 4. Why `rope_theta = 1000` for polar (not 10000)

The polar channels have a **narrower natural dynamic range** than raw Cartesian:

| Channel | Natural Δ (patch-level) | Max Δ | Dynamic span |
|---------|------------------------|-------|--------------|
| log r  | 0.01 – 0.3 | ~7 (for r ∈ [0.1, 100m]) | ~700× |
| θ      | 0.01 – 0.3 rad | ~2π ≈ 6.3 | ~600× |
| α      | 0.01 – 0.3 rad | ~π ≈ 3.1 | ~300× |

Compare with raw Cartesian `(x, y, z)`: `|x|, |y|, |z|` can reach ±10m, so max Δ ≈ 20m. The dynamic range is similar order but polar's log r compresses very far distances. More importantly, polar has a **hard upper bound** on two of three channels (angles) whereas Cartesian is open-ended.

Given this narrower effective range, a smaller `rope_theta` matches better:

### Band-utilization comparison at `scale = 100`

At patch-level Δ_natural = 0.1 (so Δp = 10):

| θ | inv_freq range | bands where 0.01 ≤ Δp × inv_freq ≤ 2π |
|---|---------------|---------------------------------------|
| 10000 | [1.0, 2.31e-4] (4329×) | **8 / 11** |
| **1000** | **[1.0, 1.87e-3] (534×)** | **10 / 11** ✓ |
| 100 | [1.0, 0.0152] (66×) | 7 / 11 |

At max Δ_natural = 6 (so Δp = 600) — checking whether the top bands still monotonic:

| θ | # bands still monotonic (Δp × inv_freq ≤ 2π) |
|---|---------------------------------------------|
| 10000 | 5 / 11 |
| **1000** | **3 / 11** |
| 100 | 1 / 11 |

`θ = 1000` trades a little far-range coverage (5 → 3 monotonic bands) for much better near-range activation (8 → 10 useful bands). Polar's typical use case — distinguishing nearby patches across the scene — weights toward near-range, so `θ = 1000` wins.

### The 11 per-axis frequencies at θ = 1000

```
inv_freq_axis[k] = 1 / 1000^(k/11)

 k | inv_freq  | wavelength at scale=100 → natural Δ it covers
---|-----------|----------------------------------------------
 0 | 1.000     | 0.063            (ultra-local Δ < 0.06)
 1 | 0.533     | 0.118            (0.03 - 0.5)
 2 | 0.285     | 0.220            (0.05 - 0.9)
 3 | 0.152     | 0.414            (0.1 - 1.7)
 4 | 0.0811    | 0.775            (0.2 - 3.1)
 5 | 0.0432    | 1.45             (0.4 - 5.8)
 6 | 0.0230    | 2.73             (0.7 - 10.9)
 7 | 0.0123    | 5.12             (1.3 - 20.5)
 8 | 0.00656   | 9.58             (2.4 - 38.3)
 9 | 0.00350   | 17.93            (4.6 - 71.7)
10 | 0.00187   | 33.55            (8.7 - 134.2)
```

With θ=1000, the 11 bands span roughly natural Δ from 0.03 (band 1 useful floor) to 134 (band 10 aliasing ceiling), covering both fine patch variations and full-scene extents. θ=10000 would stretch this to ~1300, but most of that extra range is wasted on log r values that don't exist in realistic scenes.

## 5. Mutual exclusivity

`--polar` raises `ValueError` when combined with `--vanilla` or `--decouple`. (`--polar` already implies the decouple architecture; `--decouple` specifies the Cartesian variant — so `--polar --decouple` is redundant and disallowed.)

```
[train_correspondence.py] build_model:
    if polar and vanilla:  raise ValueError(...)
    if decouple and vanilla: raise ValueError(...)
    if polar and decouple: raise ValueError("--polar already implies decouple ...")
```

| Flag combination | Architecture | XYZ channel input |
|------------------|--------------|-------------------|
| (none) | `SpaForConditionalGeneration` 4D M-RoPE | Cartesian on rotary 64 |
| `--vanilla` | `Qwen3_5ForConditionalGeneration` | — |
| `--decouple` | `SpaDecForConditionalGeneration` θ=10000 | Cartesian on dims 64..129 |
| **`--polar`** | `SpaDecForConditionalGeneration` θ=1000 | **log-spherical on dims 64..129** |

## 6. Implementation details

### 6.1 The polar flag flow

```
build_model(polar=True)
    use_decouple = True            ← --polar implies decouple
    polar_xyz    = True
        ↓
    SpaDecForConditionalGeneration.from_pretrained(...)
        ↓
    if polar_xyz:                  ← swap in the θ=1000 rotary emb
        spa.model.language_model.xyz_rotary_emb = SpaXYZRotaryEmbedding(
            xyz_dim=66, rope_theta=1000.0, default_coord_scale=100.0,
        )
        ↓
    patch_attention_layers_dec(spa)
        ↓
    AnswerOnlyModel(spa, polar=True, coord_scale=100)
        ↓ forward-time:
    fwd_kwargs["polar"] = True     ← in AnswerOnlyModel.forward
        ↓
    SpaDecForConditionalGeneration.forward(polar=True)
        ↓
    SpaDecModel.forward:
        language_model._polar = True   ← stashed for the text model to read
        ↓
    SpaDecTextModel.forward:
        xyz_rotary_emb(xyz_pos, coord_scale=100, polar=True)
            ↓ inside SpaXYZRotaryEmbedding.forward:
            convert xyz → (log r, θ, α) if polar=True
            multiply by scale, inv_freq → cos / sin
```

### 6.2 Key code locations

- **Conversion math**: [`src/models/spa_emb_dec.py:132-160`](../../src/models/spa_emb_dec.py) (inside `SpaXYZRotaryEmbedding.forward`)
- **θ=1000 swap**: [`train_correspondence.py:170-205`](../../train_correspondence.py) in `build_model`
- **Forward plumbing**: `SpaDecForConditionalGeneration.forward` → `SpaDecModel.forward` → `SpaDecTextModel` both accept `polar` kwarg
- **Mutex check**: [`train_correspondence.py:140-150`](../../train_correspondence.py)

### 6.3 Why swap the rotary emb after `from_pretrained` (not configure at `__init__`)

`SpaDecTextModel.__init__` hardcodes `rope_theta=10000.0` because it's called through `._from_config(config.text_config)` during `from_pretrained` — there's no clean way to pass a custom theta through `transformers`'s model-loading path without subclassing further.

`xyz_rotary_emb` has **no trainable parameters** (only a non-persistent buffer `inv_freq_axis`), so replacing it post-`from_pretrained` is safe and doesn't interact with LoRA wrapping (which happens later).

### 6.4 Text tokens stay at identity rotation

```python
# SpaXYZRotaryEmbedding.forward polar branch
r = sqrt(x² + y² + z²)
zero_mask = (r == 0)                                    # text tokens
log_r = where(zero_mask, 0, log(clamp(r, 1e-8)))
θ     = where(zero_mask, 0, atan2(y, x))
α     = where(zero_mask, 0, atan2(sqrt(x²+y²), z))
# → after scaling by coord_scale and multiplying by inv_freq, all zero
# → cos = 1, sin = 0 for every frequency band
# → identity rotation on dims 64..129 for text tokens
```

Same invariance as `--decouple`: text-text attention is exactly preserved from Qwen pretraining; only image-patch tokens get non-trivial XYZ RoPE rotation.

## 7. Expected behaviour vs `--decouple`

Same architecture, same mutex rules, same KV-cache semantics. Differences only in:

| Aspect | `--decouple` | `--polar` |
|--------|-------------|-----------|
| XYZ RoPE input | `(x, y, z)` raw Cartesian | `(log r, θ, α)` log-spherical |
| `rope_theta` | 10000 | **1000** |
| Scene-scale invariance | Relies on coord_scale tuning per dataset | **Yes — log r is scale-invariant by design** |
| θ/α axis symmetry | x/y/z already symmetric in Cartesian | **Also symmetric (raw rad × same scale)** |
| Band utilization at patch-level Δ | 8/11 | **10/11** |
| Band aliasing at max Δ | 5/11 monotonic | 3/11 monotonic |

In short, `--polar` is the better default for multi-scene training; `--decouple` is retained as a more conservative Cartesian baseline.

## 8. Usage

```bash
# log-spherical variant (default for polar)
bash scripts/train_correspondence.sh 2 --polar

# Cartesian decouple (for comparison / ablation)
bash scripts/train_correspondence.sh 2 --decouple

# --polar --decouple raises ValueError (see §5)
```

Shell script auto-suffixes `_polar` to `RUN_NAME` / `WANDB_RUN_NAME`. The `[INFO]` banner shows:

```
[INFO] Mode                 = polar (decouple + log-spherical XYZ RoPE)
[XYZ RoPE] theta swapped to 1000 for polar mode
```

## 9. Related docs

- [`decouple_xyz_rope.md`](decouple_xyz_rope.md) — the underlying decoupled architecture (sections 4, 4.5, 5 on mutex, KV-cache, text preservation all apply to `--polar` verbatim)
- [`../findings/qwen35_rope_decoupling.md`](../findings/qwen35_rope_decoupling.md) — Qwen3.5's partial-rotary design (why there's a "pass-through" region to carve from)
- [`../findings/rope_4d_xyz_mismatch.md`](../findings/rope_4d_xyz_mismatch.md) — why the original 4D M-RoPE approach (with `[2, 10, 10, 10]` sequential section) fails for continuous xyz; motivates the decoupled XYZ RoPE and the polar variant.

## 10. Open knobs and future tuning

- **coord_scale** (currently 100): the XYZ values before RoPE multiplication. Controls which freq bands land in the monotonic region. Can be switched to per-axis if one channel needs different resolution.
- **rope_theta** (currently 1000 for polar, 10000 for decouple): the freq-band geometric base. Lower θ → tighter band spacing, better near-range coverage, worse far-range.
- **xyz_dim** (currently 66 / 11 bands per axis): could grow to 78 (13 bands) or 60 (10 bands) — needs divisibility by 6 for symmetric split-half pairing.
- **Polar vs Cartesian per-axis**: possible hybrid — log r on one axis, Cartesian on others. Not implemented; unclear benefit.
