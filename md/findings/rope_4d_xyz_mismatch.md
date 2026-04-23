# 4D M-RoPE for continuous xyz: frequency-band mismatch analysis

## 1. RoPE recap

### 1.1 Rotation formula

For each pair of dimensions `(d_{2k}, d_{2k+1})` inside one attention head, RoPE rotates the 2-vector by an angle that scales linearly with the token's position `p`:

```
φ_k(p) = p × inv_freq[k]
```

The inverse-frequency table is geometric in `k`:

```
inv_freq[k] = 1 / θ^(2k / d_rot)          k ∈ {0, 1, ..., d_rot/2 − 1}
```

- `θ = rope_theta` — global base (Qwen3.5: `1e7`)
- `d_rot = rotary_dim = head_dim × partial_rotary_factor` — how many head dims actually get rotated
- `d_rot / 2` — number of frequency bands (each band rotates a 2-dim pair)

For Qwen3.5-4B: `head_dim = 256`, `partial_rotary_factor = 0.25`
→ `d_rot = 64`, **32 frequency bands**.

### 1.2 M-RoPE extension to N dimensions

`mrope_section = [s_0, s_1, ..., s_{N-1}]` with `sum(s_i) = d_rot/2` partitions the 32 bands among N semantic axes. Each axis `i` supplies its own position value to `s_i` bands.

Our 4D M-RoPE (`[t, x, y, z]`):
- Default: `[2, 10, 10, 10]` sequential (t gets bands 0-1, x gets 2-11, y gets 12-21, z gets 22-31)
- With `--interleave_vision`: t gets bands 0-1, then x/y/z round-robin through bands 2-31

### 1.3 Concrete inv_freq table for Qwen3.5 (θ=1e7, d_rot=64)

| k | inv_freq | wavelength = 2π / inv_freq |
|---|---------|---------------------------|
| 0  | 1.0      | 6.28       |
| 1  | 0.606    | 10.4       |
| 2  | 0.367    | 17.1       |
| 5  | 0.082    | 77         |
| 10 | 6.68e-3  | 940        |
| 11 | 4.05e-3  | 1,550      |
| 15 | 5.48e-4  | 11,500     |
| 20 | 4.50e-5  | 139,600    |
| 21 | 2.73e-5  | 230,000    |
| 25 | 3.70e-6  | 1.7 M      |
| 30 | 3.03e-7  | 20.7 M     |
| 31 | 1.84e-7  | 34 M       |

### 1.4 High-vs-low frequency categorization

| Category | k range | inv_freq | wavelength | Good for Δp ∈ |
|----------|---------|----------|------------|---------------|
| **High**      | 0–5   | 1.0 → 0.08     | 6 → 77            | 1–10 |
| **Mid-high**  | 6–11  | 0.05 → 4e-3    | 130 → 1,500       | 10–100 |
| **Mid**       | 12–17 | 2.5e-3 → 2e-4  | 2.5K → 30K        | 100–1,000 |
| **Mid-low**   | 18–23 | 1e-4 → 1e-5    | 50K → 600K        | 1K–10K |
| **Low**       | 24–31 | 6e-6 → 1.4e-7  | 1M → 34M          | 10K–128K |

Rule of thumb: higher `k` → smaller `inv_freq` → longer wavelength → more useful for separating far-apart positions; lower `k` → faster rotation → resolves adjacent positions.

## 2. Useful-band criterion

For two positions `p_1, p_2` and a band `k`, the rotation angle difference is:

```
Δφ_k = (p_1 − p_2) × inv_freq[k] = Δp × inv_freq[k]
```

Band `k` contributes distinguishable signal for the pair iff:

- `Δφ_k ≥ ~0.01 rad` (below this, `cos Δφ ≈ 1`, `sin Δφ ≈ 0` — effectively DC under bf16/fp32)
- `Δφ_k ≤ ~2π` (above this, phase wraps — relative-distance order is destroyed)

Rearranging the lower bound:

```
Δp_min(k)  ≈  0.01 / inv_freq[k]
```

Any Δp below this threshold is indistinguishable at band `k`.

## 3. How Qwen's original 3D M-RoPE avoids the mismatch

Original config: `mrope_section = [11, 11, 10]` for `(t_frame, h, w)`. For text tokens all three dims equal the scalar text position `p`; for image tokens `(t, h, w)` are all integer indices.

### 3.1 Matching the band spectrum to Qwen's intended Δp range

Typical Δp for different comparisons in Qwen:

| Comparison | Δp | Needed band |
|------------|----|-------------|
| Adjacent text tokens                 | 1        | k ≈ 0–5    |
| Text across 1K tokens                | 1,000    | k ≈ 12–15  |
| Text across 10K tokens               | 10,000   | k ≈ 18–22  |
| Text across 128K (max context)       | 128,000  | k ≈ 29–31  |

Δp dynamic range = `1 → 128,000`, i.e. **~5 orders of magnitude** — which matches the inv_freq spectrum's ~7-order span.

### 3.2 The 128K ↔ rope_theta = 1e7 connection

Qwen's choice of `θ = 1e7` makes the lowest band's wavelength just larger than its max context:

```
wavelength[31] = 2π / inv_freq[31] ≈ 34 M       (= 2π × θ^((d_rot−2)/d_rot))
```

`34 M > 128 K`, so position p=128K produces `Δφ ≈ 0.018 rad` at the lowest band — barely above the useful-band threshold. Qwen specifically tuned `rope_theta` so **even the lowest band remains (marginally) useful at 128K context**.

### 3.3 What about spatial h, w? Those bands ARE mostly wasted

`h, w ∈ [0, ~24]` for image patches. The bands assigned to them (11–21 in `[11,11,10]`) have wavelengths 1.5K → 230K — far exceeding any possible Δh. So the low-freq end of h/w's allocation is **unused** in original Qwen too.

But this doesn't hurt: there's no physical need to distinguish "patch at row 0 vs row 10,000" (no such thing), so wasting a few bands on h/w is harmless as long as text position benefits from the full multi-scale decomposition.

## 4. Our 4D xyz: where the mismatch arises

We replaced `(t, h, w)` with `(t, x, y, z)` where x, y, z are **continuous 3D world coordinates in meters**, scaled by `coord_scale`.

### 4.1 Dynamic range of Δp for xyz

Typical xyz in indoor scenes: ±10 m. With `coord_scale = 100`:

| Comparison | Δxyz_natural (m) | Δp (× 100) |
|------------|------------------|------------|
| Adjacent patches within image | 0.1 – 0.3    | 10 – 30 |
| Patches across an image        | 1 – 3        | 100 – 300 |
| Across different images        | 1 – 10       | 100 – 1,000 |
| **Max physical Δ**             | ~20 m        | **~2,000** |

Δp dynamic range: `10 → 2,000` — only **~2 orders of magnitude**, vs Qwen's 5 for text.

### 4.2 The low-freq bands are physically unreachable

Using the criterion `Δp_min(k) ≈ 0.01 / inv_freq[k]`:

| Band k | Δp_min needed | Natural Δxyz at scale=100 (m) |
|--------|---------------|-------------------------------|
| 11 | 4.5      | 0.045 (patch-level ✓) |
| 15 | 18       | 0.18 (within-image ✓) |
| 20 | 220      | 2.2 (across-image, borderline) |
| 21 | 370      | 3.7 (typical across-image) |
| 25 | 2,700    | **27 m — beyond typical scene extent** |
| 31 | **70,000** | **700 m — physically impossible** |

Bands **25–31** require Δxyz beyond any realistic indoor-scene extent — they're **dead under any reasonable scale**.

### 4.3 The default `[2, 10, 10, 10]` sequential assignment

Because sequential assignment gives each axis a contiguous band slice:

| Axis | Bands | Lowest inv_freq | Δp needed (nat. m × scale=100) | Status |
|------|-------|----------------|-------------------------------|--------|
| t (image index) | 0–1   | 0.61   | any Δt ≥ 0.02 | ✓ |
| x / log ρ       | 2–11  | 4.05e-3 | Δx ≥ 0.025 m | ✓ all 10 useful |
| y / θ           | 12–21 | 2.73e-5 | Δy ≥ 3.7 m   | ~5 of 10 useful |
| **z / α**       | 22–31 | 1.84e-7 | Δz ≥ **544 m** | ✗ **10 of 10 dead** |

Under sequential layout, the z axis (or α in polar mode) is **effectively unused**.

### 4.4 The key mismatch summary

| | Qwen original | Our 4D xyz |
|--|--------------|------------|
| Position value type | Integer indices | Continuous physical coords |
| Δp dynamic range | 1 → 128K (5 orders) | 10 → 2K (2 orders) |
| `rope_theta` | 1e7 (tuned for 128K) | 1e7 (inherited, mismatched) |
| Max Δp × min inv_freq | 128K × 1.4e-7 ≈ 0.018 ✓ | 2K × 1.4e-7 ≈ 3e-4 ✗ |

Inheriting Qwen's long-context frequency ladder while using it for short-dynamic-range physical coordinates guarantees that the low-freq end is wasted.

## 5. Polar (log-spherical) mode doesn't fix the root cause

With `--polar`: `(t, log r, θ, α)` where `θ, α` are raw radians ∈ `[-π, π]` and `[0, π]`.

### 5.1 Dynamic ranges in polar

| Channel | Natural Δ range | With scale=100 Δp range |
|---------|-----------------|------------------------|
| log r | 0.01 → 3 | 1 → 300 |
| θ (azimuth) | 0.01 → π | 1 → 314 |
| α (inclination) | 0.01 → π | 1 → 314 |

Still only **~2.5 orders of magnitude** — no better than Cartesian.

### 5.2 Polar maps `z` onto α which inherits the dead low-freq bands

Under sequential `[2, 10, 10, 10]`, α is pinned to bands 22-31, the same dead region. Polar merely changes *what* the dead bands ignore — not the fact that they're dead.

### 5.3 log-r's scale invariance is orthogonal to the band-utilization problem

`log r` solves a *different* problem (different scenes have different radial scales; raw r produces per-scene attention patterns). It does NOT widen the dynamic range enough to reach the lowest freq bands.

## 6. Mitigations, ranked by impact

### 6.1 `--interleave_vision` (biggest win, minimal cost)

Instead of each axis owning a contiguous band slice, interleave round-robins x/y/z through all bands 2-31. Each axis then gets the same mix of high/mid/low freqs. Because of the narrow Δp range, each axis ends up using ~4-6 of its assigned 10 bands productively, but the usable bands are now the same for every axis (no axis is starved). **Required for polar mode** — otherwise α is pinned to dead bands.

### 6.2 `coord_scale` tuning

Scale shifts the useful-band window left/right on the frequency ladder. Larger scale → useful window shifts toward lower freqs (more low-end activation, more high-end aliasing). Cannot widen the dynamic range, only reposition it.

Rough targeting: let `scale × Δp_natural_typical × inv_freq[k_center] ≈ 0.1 rad`, where `k_center` is the middle of the axis's assigned band range. For sequential mode with `Δp_natural ≈ 0.3 m`, typical good scale is 100–1000.

### 6.3 Lower `rope_theta` for vision tokens (biggest fundamental win, most invasive)

Reducing `θ` from `1e7` toward `1e4` shortens all wavelengths, compressing the useful window into a narrower physical Δp range — matching continuous 3D coords.

**Cost**: text tokens share the same rotary_emb. Changing θ breaks long-context text RoPE. Would require either:
- Separate rotary_emb for visual vs text tokens (major refactor)
- Two-pass attention (expensive)
- Accepting degraded text long-range attention

Not pursued in the current codebase.

### 6.4 `--full` (partial_rotary_factor=1.0) does NOT fix this

Increasing rotary_dim from 64 to 256 gives 128 freq bands instead of 32, but `rope_theta` is unchanged → the frequency range is the same, just sampled 4× denser. The lowest band still has inv_freq ≈ 1e-7. The 96 extra bands mostly fall inside the already-dead low-freq region.

See earlier analysis: `--full` gives ~1.7× more usable bands, not 4×, and breaks the pretrained content/position split (the 192 previously-non-rotary content dims now rotate).

## 7. Practical recommendations

1. **Always use `--interleave_vision` with 4D xyz M-RoPE.** Sequential `[2, 10, 10, 10]` wastes the z/α axis entirely.
2. **For `--polar`, use `--interleave_vision`.** Otherwise α channel's coord loss is paired with a dead position encoding, and the model cannot attend over α differences at all.
3. Treat `coord_scale` as a per-axis knob (`--coord_scale_xyz`). Different axes have different natural Δ and benefit from independent scales.
4. Float position_ids (implemented): removes quantization loss at scale=100, especially important for polar where angular 1° resolution was otherwise limited.
5. Accept that a non-trivial fraction of bands (roughly the low-freq half) will remain unused. The problem is fundamental: the frequency ladder was designed for 5-order dynamic ranges that physical 3D coords don't have.

## Appendix: quick numerical checker

Given `rope_theta=1e7`, `d_rot=64`:

```
inv_freq[k]   = 1e7 ** (-k/32)
wavelength[k] = 2 * pi / inv_freq[k]
usable(k, Δp) = 0.01 ≤ Δp * inv_freq[k] ≤ 2 * pi
```

For your `Δxyz_natural`, compute `Δp = coord_scale × Δxyz_natural`, then iterate over bands `k ∈ axis_range` and count how many satisfy the usable criterion. That gives you the "effective band count" for a given axis under a given scale.
