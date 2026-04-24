# Qwen3.5 RoPE: head-dim decoupling and M-RoPE extension

How Qwen3.5 splits each attention head's `head_dim` into a **rotary portion** (position-carrying) and a **pass-through portion** (content-only), then further subdivides the rotary portion into frequency bands assigned to semantic position axes.

## 1. TL;DR — numeric breakdown for Qwen3.5-4B

| Quantity | Value | Formula |
|----------|-------|---------|
| `head_dim` | 256 | per-head total dims |
| `partial_rotary_factor` | 0.25 | config.text_config.rope_parameters |
| `rotary_dim` | **64** | = `head_dim × 0.25` |
| Pass-through dims | **192** | = `head_dim − rotary_dim` (75%) |
| `rotary_dim / 2` | **32** | number of RoPE frequency bands |
| `rope_theta` | 1e7 | geometric base for `inv_freq` |

## 2. Two-layer decoupling

```
┌────────── per-head dim: head_dim = 256 ──────────┐
│                                                   │
│  ┌──────── rotary ────────┐ ┌──── pass-through ──┐│
│  │       64 dims (25%)     │ │   192 dims (75%)   ││
│  │   position-carrying     │ │  content-only       ││
│  │   via RoPE rotation     │ │  (never rotated)    ││
│  └─────────────────────────┘ └─────────────────────┘│
└───────────────────────────────────────────────────┘
```

### 2.1 Where this split happens in code

[modeling_qwen3_5.py:662-668](/egr/research-actionlab/caizhon2/miniconda3/envs/spc/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py#L662):

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    rotary_dim = cos.shape[-1]                              # = 64
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)    # rotate first 64
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)          # concat back with pass-through
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

The first 64 dims of Q and K get RoPE applied; the last 192 dims are returned untouched.

### 2.2 Attention score decomposition

For any two tokens `i, j`, the head's QK inner product splits:

```
⟨Q_i, K_j⟩ = ⟨Q_i^rot, K_j^rot⟩  +  ⟨Q_i^pass, K_j^pass⟩
              └── position-dependent ──┘    └──── content-only ────┘
                depends on (p_i − p_j)        absolute similarity
```

- **Rotary project** (64 dims): produces `f(p_i − p_j)` — decays/structure with relative distance.
- **Pass-through project** (192 dims): pure dot-product of content features — not modulated by position.

This is the "decoupling": position and content contribute additively rather than multiplicatively, and the model can weigh them via learning.

## 3. Why Qwen chose 0.25 (not 1.0)

Early full-rotary designs (GPT-J, early LLaMA) applied RoPE to every head dim. This means content features are **always** modulated by position. A word at position 10 vs position 1000 has totally different representations even when its meaning is identical.

Partial rotary (NeoX style, inherited by Qwen 2/3/3.5) reserves a pure content channel:

| Model family | `partial_rotary_factor` | Rotary share | Pass-through share |
|--------------|------------------------|--------------|-------------------|
| GPT-J, early LLaMA | 1.0 | 100% | 0% |
| LLaMA 2, 3 | 1.0 | 100% | 0% |
| **GPT-NeoX** | **0.25** | 25% | 75% |
| **Qwen 2 / 3 / 3.5** | **0.25** | 25% | 75% |

Empirically the NeoX/Qwen split gives better long-context attention: distant tokens with the same semantics can still attend to each other strongly (via the 192 content dims) while local position still matters (via the 64 rotary dims).

## 4. Inside the rotary portion: 32 bands via split-half pairing

### 4.1 `inv_freq` shape

[modeling_qwen3_5.py:222-224](/egr/research-actionlab/caizhon2/miniconda3/envs/spc/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py#L222):

```python
dim = int(head_dim * partial_rotary_factor)  # = 64
inv_freq = 1.0 / base ** (torch.arange(0, dim, 2) / dim)
# → inv_freq has shape (32,)
```

Only `rotary_dim / 2 = 32` frequencies.

### 4.2 `cos` / `sin` shape

The rotary embedding builds:
```python
freqs = inv_freq × position              # (batch, seq, 32)
emb   = cat((freqs, freqs), dim=-1)     # (batch, seq, 64) — duplicated
cos, sin = emb.cos(), emb.sin()
```

`cos[..., k]` and `cos[..., k + 32]` are the same value — the same frequency appears twice.

### 4.3 `rotate_half` + duplicated cos = split-half pair rotation

[modeling_qwen3_5.py:630-634](/egr/research-actionlab/caizhon2/miniconda3/envs/spc/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py#L630):

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]   # dims 0..31
    x2 = x[..., x.shape[-1] // 2 :]   # dims 32..63
    return torch.cat((-x2, x1), dim=-1)  # returns (-x2, x1)
```

Combined with `q_embed = q_rot * cos + rotate_half(q_rot) * sin`, each 2D rotation pairs dim `k` with dim `k + 32` (within the 64 rotary dims):

```
pair k:  (d_k, d_{k+32})   rotated by angle  p × inv_freq[k]

For k = 0..31 — total 32 independent 2D rotations inside the rotary region.
```

### 4.4 Why split-half (not adjacent) pairing?

Equivalent math, different indexing convention. HuggingFace picked split-half because the implementation (`rotate_half`) is a single slice+negate+concat — no element-wise gather needed. Performance reason, not mathematical.

## 5. M-RoPE: the 32 bands get partitioned across N axes

Standard 1D RoPE: all 32 bands use the same scalar `position`.

M-RoPE (Multi-dim): split `mrope_section = [s_0, s_1, ..., s_{N−1}]` with `sum = 32`, and band `k` uses the position value from whichever axis owns that band.

### 5.1 Original Qwen 3D (text + image)

`mrope_section = [11, 11, 10]`:
- bands 0-10 → `t` (temporal / text position)
- bands 11-21 → `h` (patch row or text position)
- bands 22-31 → `w` (patch col or text position)

For **text tokens**, all three position values equal the token's scalar position → all 32 bands see the same value → multi-scale decomposition over distance.

For **image tokens**, `(t, h, w) = (start_pos, patch_row, patch_col)` → the 11-21 and 22-31 bands pick up spatial structure.

### 5.2 Our 4D extension (t, x, y, z)

`mrope_section = [2, 10, 10, 10]` sequential (default):

| Axis | Bands | Rotary-dim indices (pair form) |
|------|-------|-------------------------------|
| t | 0-1 | (d₀, d₃₂), (d₁, d₃₃) |
| x / log ρ | 2-11 | (d₂, d₃₄) ... (d₁₁, d₄₃) |
| y / θ | 12-21 | (d₁₂, d₄₄) ... (d₂₁, d₅₃) |
| z / α | 22-31 | (d₂₂, d₅₄) ... (d₃₁, d₆₃) |

Implemented in our `SpaTextRotaryEmbedding.apply_interleaved_mrope` (despite its name, the default behaviour is **sequential**; `--interleave_vision` switches to true interleave).

## 6. Full visual summary of head-dim layout

```
head_dim = 256 per-head breakdown:

┌─────────────────────────────── rotary (64 dims, 25%) ───────────────────────────────┐
│                                                                                      │
│ first half (dims 0..31) mirrors second half (dims 32..63) via split-half pairing:    │
│                                                                                      │
│ dim index:  0   1   2   3   ...  11  12  ...  21  22  ...  31                        │
│             ↕   ↕   ↕   ↕        ↕   ↕        ↕   ↕        ↕                         │
│ pair with: 32  33  34  35  ...  43  44   ...  53  54   ...  63                       │
│                                                                                      │
│ band k:     0   1  │ 2 ............... 11 │ 12 ........... 21 │ 22 ........... 31    │
│ uses pos:   t   t  │ x / log ρ            │ y / θ              │ z / α                │
│             └──────┴───────────┴──────────┴───────────┴────────┴───────────┘         │
│                 mrope_section = [2, 10, 10, 10]   — sum = 32                         │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── pass-through (192 dims, 75%) ────────────────────────────┐
│                                                                                      │
│  dims 64..255 — never touched by RoPE.                                               │
│  Encodes: word identity, syntactic role, semantic similarity,                        │
│           cross-image object correspondence — everything position-invariant.         │
│                                                                                      │
│  ⟨Q_pass_i, K_pass_j⟩ does NOT depend on (p_i − p_j).                                │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Band-to-dim mapping summary

| mrope_section entry | # bands | # rotary dims occupied | Indices |
|---------------------|---------|------------------------|---------|
| `[0] = 2` (t) | 2 | 4 | (0,32), (1,33) |
| `[1] = 10` (x) | 10 | 20 | (2,34) through (11,43) |
| `[2] = 10` (y) | 10 | 20 | (12,44) through (21,53) |
| `[3] = 10` (z) | 10 | 20 | (22,54) through (31,63) |
| **Total** | **32** | **64** | = rotary_dim |

(Each band pairs 2 dims for a 2D rotation, so "bands × 2 = rotary dims occupied".)

## 7. What `--full` changes

Setting `partial_rotary_factor = 1.0`:

```
Before (--full off):
┌─── rotary 64 ───┐┌──── pass-through 192 ────┐   ← 32 bands, 75% content channels
│                 ││                            │
└─────────────────┘└────────────────────────────┘

After (--full on):
┌─────────────── rotary 256 ───────────────┐       ← 128 bands, 0% content channels
│                                           │
└───────────────────────────────────────────┘
```

- rotary_dim 64 → 256 (all head dims rotate)
- freq bands 32 → 128 (4× denser sampling of the same frequency range)
- pass-through disappears — content features now modulated by position

Downstream effect: the 192 previously-position-invariant content dims become position-dependent. Pre-trained word-identity signals scramble. LoRA (only adapting Q/K/V/O low-rank deltas) cannot fully compensate.

See [rope_4d_xyz_mismatch.md](rope_4d_xyz_mismatch.md) for why `--full` does not fix the xyz mismatch problem.

## 8. `inv_freq` concrete numbers

With `rope_theta = 1e7`, `rotary_dim = 64`:

`inv_freq[k] = (1e7)^(-k/32)`

| k | inv_freq | wavelength = 2π / inv_freq |
|---|----------|----------------------------|
| 0  | 1.0      | 6.28      |
| 5  | 0.082    | 77        |
| 10 | 6.68e-3  | 940       |
| 15 | 5.48e-4  | 11,500    |
| 20 | 4.50e-5  | 139,600   |
| 25 | 3.70e-6  | 1.7 M     |
| 31 | 1.84e-7  | 34 M      |

Band 31's wavelength (34 M) was chosen via `rope_theta = 1e7` so that Qwen's max context (128K) fits inside a single cycle — see the Qwen context-length tuning discussion.

## 9. Key takeaways

1. **Two-layer decoupling is the core design**:
   - 25% of head_dim → RoPE → position signal
   - 75% of head_dim → pass-through → content signal
   - Attention mixes both additively

2. **Within rotary, split-half pairing**: dim k ↔ dim k+32 rotates together with `inv_freq[k]`.

3. **M-RoPE partitions the 32 bands, not the 64 rotary dims directly** — each "mrope_section entry of s" takes `2s` actual rotary dims.

4. **`partial_rotary_factor = 0.25`** is a deliberate Qwen/NeoX design. Changing it to 1.0 (`--full`) destroys the content pass-through channel.

5. **Pass-through enables long-context content similarity**. Without it, distant same-meaning tokens cannot attend to each other without position modulation distorting their features.

## 10. M-RoPE vs traditional 2D-RoPE: do the axes share frequency bands?

A frequently-conflated question: in multi-axis RoPE schemes, do the position axes **share** the same `inv_freq` ladder, or do they **partition** it? This is the core architectural split between traditional 2D-RoPE (early ViT position embeddings, some spatial transformer works) and Qwen's M-RoPE. Getting this right is load-bearing for understanding why 4D xyz has the "dead z bands" problem that the original 3D (t,h,w) design never had.

### 10.1 Traditional 2D-RoPE: each axis gets its own copy of the full `inv_freq` table

Under the 2D-RoPE convention used in much of the pre-M-RoPE literature:

```
head_dim = d
           └─ split into two equal halves:
              ┌── x-half (dims 0..d/2-1) ──┐ ┌── y-half (dims d/2..d-1) ──┐
              │                            │ │                              │
              │  runs 1D-RoPE driven by x  │ │  runs 1D-RoPE driven by y    │
              │  with inv_freq[0..d/4-1]   │ │  with inv_freq[0..d/4-1]      │
              │                            │ │  ←── SAME inv_freq table ──→ │
              └────────────────────────────┘ └──────────────────────────────┘
```

- The **same** `inv_freq` table (length `d/4`) is used by both axes.
- Each axis gets a full multi-scale frequency decomposition: high-freq bands for near-Δ, low-freq bands for far-Δ.
- Cost: each axis only has `d/2` worth of dims to spend. The head's effective per-axis expressive capacity is halved.

Both axes can resolve the same Δ range with the same granularity. There is **no asymmetry by construction**.

### 10.2 Qwen M-RoPE: the single `inv_freq` table is partitioned across axes

Under Qwen's M-RoPE convention (see section 5 above, restated here for contrast):

```
head_dim = d         rotary_dim = d_rot (= d/4 in Qwen3.5)
                     └── single inv_freq table of length d_rot/2 = 32 ──┐
                                                                         │
mrope_section = [s_0, s_1, s_2]   with s_0 + s_1 + s_2 = 32              │
                                                                         │
┌────────────────────────────── 32 bands ─────────────────────────────┐  │
│                                                                     │  │
│  ┌── s_0 bands ──┐ ┌── s_1 bands ──┐ ┌── s_2 bands ──┐               │  │
│  │  axis t owns  │ │  axis h owns  │ │  axis w owns   │               │  │
│  │  band 0..s_0  │ │  s_0..s_0+s_1 │ │  ... to 31     │               │  │
│  │  (high freq)  │ │  (mid freq)   │ │  (low freq)    │               │  │
│  └───────────────┘ └───────────────┘ └────────────────┘               │  │
└─────────────────────────────────────────────────────────────────────┘
```

- There is **one** `inv_freq` table. It is **partitioned**, not replicated.
- Axis t drives only its allocated band slice; bands outside that slice are **not reachable** by t's position value.
- Different axes therefore **see different frequency subranges** of the ladder.

This is an "asymmetric" design: t's best-resolvable Δ range differs from h's best-resolvable Δ range.

### 10.3 Side-by-side comparison

| | Traditional 2D-RoPE | Qwen M-RoPE |
|---|---|---|
| Number of `inv_freq` tables | 1 (duplicated per axis) | 1 (partitioned across axes) |
| Bands accessible per axis | Full `d/4` bands | `s_i` bands (from `mrope_section`) |
| Wavelength range per axis | Identical across axes | Different per axis |
| Per-axis resolvable Δ | Same for all axes | Different per axis |
| Head dims spent per axis | `d/2` | `2·s_i` rotary dims |
| Frequency symmetry | Symmetric | Asymmetric by design |
| Per-axis bandwidth tuning | Not possible | Via `mrope_section` |

### 10.4 Why Qwen chose the asymmetric design

For Qwen's original video+text use case, the three axes `(t, h, w)` have **fundamentally different Δ dynamic ranges**:

| Axis | Typical Δ | Max Δ |
|------|-----------|-------|
| t (frame / text position) | 1 | 128K |
| h (patch row) | 1 | ~24 |
| w (patch col) | 1 | ~24 |

Under 2D-RoPE-style symmetric allocation, t, h, w would all use the same inv_freq ladder — but h and w never exceed ~24, so the low-freq bands (wavelength → millions) are permanently wasted on them. Qwen's partition instead gives t the full frequency sweep (bands 0-10 → high-freq useful for small Δt; in 3D with t as text position, bands 0-10 at θ=1e7 actually don't resolve the full 128K — they alias — but the outer bands 22-31 would handle 128K if assigned; see below).

Actually the 3D layout `[11, 11, 10]` partitions such that **text** tokens (whose `(t, h, w) = (p, p, p)`) effectively recover all 32 bands since every band sees `p`. It's only for **image** tokens (where the three axis values differ) that the partition becomes operationally meaningful. The design therefore optimizes for:
- Text: full multi-scale decomposition (all 32 bands see `p`).
- Image: patch layout signal (h/w get a slice each), at the cost of not spending bandwidth redundantly on axes with small Δ.

See also [rope_4d_xyz_mismatch.md §3.3](rope_4d_xyz_mismatch.md) for why the wasted low-freq bands on h/w are harmless in the original design.

### 10.5 Why the asymmetric design breaks for continuous xyz

Our 4D extension `(t, x, y, z)` with physical coordinates has a fundamentally different Δ structure:

| Axis | Typical Δ (scale=100) | Max Δ | Symmetry |
|------|----------------------|-------|----------|
| t (image/frame idx) | 1 | ~50 | asymmetric w.r.t. xyz |
| x (meters × 100) | 10 | ~2000 | **isotropic with y, z** |
| y (meters × 100) | 10 | ~2000 | **isotropic with x, z** |
| z (meters × 100) | 10 | ~2000 | **isotropic with x, y** |

The three spatial axes have **identical dynamic ranges**. The physical world is (to a first approximation) isotropic: there is no preferred direction, so no reason for x to deserve high-freq bands while z is starved to low-freq bands.

But sequential `mrope_section = [2, 10, 10, 10]` inherits the Qwen partition pattern anyway:
- x: bands 2-11 (high to mid freq) → 10/10 usable for xyz's Δ range
- y: bands 12-21 (mid freq) → 10/10 usable
- z: bands 22-31 (low freq) → **only 3/10 usable** (see [rope_4d_xyz_mismatch_viz.png](rope_4d_xyz_mismatch_viz.png) Panel C)

The asymmetric allocation applies a non-uniform frequency budget to axes that have uniform Δ requirements. z loses.

### 10.6 Two remedies, from the 2D-RoPE vs M-RoPE lens

**Option A: interleave within M-RoPE (`--interleave_vision`, already implemented).**

Round-robin xyz through bands 2-31 so each axis gets a stratified sample:
- x: bands {2, 5, 8, 11, 14, 17, 20, 23, 26, 29}
- y: bands {3, 6, 9, 12, 15, 18, 21, 24, 27, 30}
- z: bands {4, 7, 10, 13, 16, 19, 22, 25, 28, 31}

Each axis now gets a mix of high/mid/low frequencies — 7-8 of 10 usable under the xyz Δ range. Still each band is owned by exactly one axis (partition, not replication), so this is NOT 2D-RoPE — it is M-RoPE with a symmetrized partition that **mimics** the symmetric coverage of 2D-RoPE.

**Option B: true 2D-RoPE-style replication for xyz (not implemented).**

Split the rotary_dim into 3 equal sub-heads, each running its own 1D-RoPE driven by one axis with **the full `inv_freq` table replicated**:

```
rotary_dim 64:
┌── x sub-head (21 dims, 10 bands) ──┐
│  runs 1D-RoPE with inv_freq[0..9] driven by x position │
├── y sub-head (21 dims, 10 bands) ──┤
│  runs 1D-RoPE with inv_freq[0..9] driven by y position │
├── z sub-head (22 dims, 11 bands) ──┤
│  runs 1D-RoPE with inv_freq[0..10] driven by z position│
└─────────────────────────────────────┘
```

All three axes see the same frequency ladder, removing the asymmetry entirely. This matches the isotropy of physical space but requires a new `apply_rotary_pos_emb` that knows about per-axis inv_freq lookups (rather than selecting from the shared cos/sin via `mrope_section`).

### 10.7 Summary

- **Traditional 2D-RoPE**: each axis gets an independent copy of the **full** inv_freq ladder → symmetric, isotropic, but costs head-dim.
- **Qwen M-RoPE**: the single inv_freq ladder is **partitioned** across axes → asymmetric, per-axis frequency tuning via `mrope_section`, efficient for axes with heterogeneous Δ ranges.
- **Our 4D xyz case**: the three spatial axes have **isotropic** Δ ranges, but we inherited Qwen's **asymmetric** partition → z axis gets starved into a dead frequency slice.
- **Interleave** is a cheap fix that symmetrizes the partition within M-RoPE.
- **Real 2D-RoPE-style replication** would be architecturally cleaner for isotropic spatial axes, at the cost of head-dim fan-out.

See [rope_4d_xyz_mismatch.md](rope_4d_xyz_mismatch.md) for the quantitative band-utilization analysis and [rope_4d_xyz_mismatch_viz.py](rope_4d_xyz_mismatch_viz.py) / [rope_4d_xyz_mismatch_viz.png](rope_4d_xyz_mismatch_viz.png) for the accompanying figures.
