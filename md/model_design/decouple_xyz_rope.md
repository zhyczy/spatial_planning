# `--decouple`: Decoupled XYZ RoPE in the Pass-Through Region

## 1. Motivation

Earlier modes (`default` 4D M-RoPE, `--polar`, `--relative`) all **modify** Qwen3.5's pretrained `mrope_section` (`[11, 11, 10]` → `[2, 10, 10, 10]`) to inject xyz into the existing 64 rotary dims. This:

- Disrupts the pretrained allocation: `t / h / w` bands are repurposed for `t / x / y / z`
- Forces LoRA to relearn what was already in pretraining
- Squeezes new spatial info into a fixed budget already used for sequential / 2D position

`--decouple` takes the **additive** approach: leave Qwen's first 25% (rotary 64 dims) **completely untouched**, and carve out 66 dims of the 192 pass-through region for a brand-new XYZ RoPE.

## 2. Head-dim Layout

For Qwen3.5-4B (`head_dim = 256, partial_rotary_factor = 0.25`):

```
head_dim = 256:
┌────── rotary 64 dims ──────┐┌──── new XYZ RoPE 66 dims ────┐┌── pass-through 126 ──┐
│ Qwen original 3D M-RoPE    ││ x: 22d   y: 22d   z: 22d     ││ untouched            │
│ mrope_section = [11,11,10] ││ symmetric per-axis spectrum  ││ content-only         │
│ rope_theta = 1e7 (Qwen)    ││ rope_theta = 10000 (new)     ││                      │
│ dims 0..63                 ││ dims 64..129                 ││ dims 130..255        │
└────────────────────────────┘└──────────────────────────────┘└──────────────────────┘
       (UNCHANGED)                  (NEW — this design)            (UNCHANGED)
```

Two RoPE rotations now apply to Q/K, in series:

1. Qwen's standard `apply_rotary_pos_emb` rotates dims 0..63 with the original 3D position
2. `_apply_xyz_rotary` rotates dims 64..129 with per-token (x, y, z)
3. Dims 130..255 pass through unrotated (content-only)

Attention score for any pair (i, j):
```
⟨Q_i, K_j⟩ = ⟨Q_i^orig,  K_j^orig⟩      ← depends on (text_pos_i − text_pos_j) etc.
            + ⟨Q_i^xyz,   K_j^xyz⟩      ← depends on (xyz_i − xyz_j)
            + ⟨Q_i^pass,  K_j^pass⟩     ← content similarity, position-independent
```

The model can independently weight three orthogonal channels.

## 3. New XYZ RoPE Design

### 3.1 Symmetric per-axis spectrum (key fix)

x, y, z **share the same** 11-band `inv_freq_axis`:

```
n_per_axis = 11
per_axis_dim = 22  (= 2 × n_per_axis, "head dim equivalent" for one axis)

inv_freq_axis[k] = 1 / rope_theta ** (2k / per_axis_dim)
                 = 1 / 10000 ** (k / 11)            for k = 0..10
```

Output cos/sin (dim = 66) is built by concatenating three identical per-axis blocks:

```
freqs = [x · inv_freq_axis | y · inv_freq_axis | z · inv_freq_axis]    # (B, S, 33)
emb   = cat(freqs, freqs, dim=-1)                                       # (B, S, 66)
cos, sin = emb.cos(), emb.sin()
```

Why symmetric (not 33 sequential bands across axes):
- An earlier draft assigned bands 0..10 → x, 11..21 → y, 22..32 → z. This gave **x high-freq, z low-freq** asymmetrically. The 3D coordinate space has no preferred axis, so this asymmetry is unjustified and hurts learning.
- With symmetric 11 bands per axis, each axis covers the full wavelength spectrum identically.

### 3.2 Numerical defaults

| Param | Value | Why |
|---|---|---|
| `rope_theta` | **10000** | Smaller than Qwen's 1e7; concentrates wavelength range on scene-relevant scales (cm to ~300m), not 128K-token text contexts |
| `coord_scale` | **100** | Same convention as the existing 4D / polar / coordinate pipelines (cm-equivalent xyz units) |
| `xyz_dim` | 66 | 3 axes × (2 × 11 bands) = 66 |
| `xyz_offset` | 64 | Start of pass-through region |

### 3.3 Resulting per-axis spectrum

With `cs=100, θ=10000`:

| band k | inv_freq_axis[k] | wavelength (m) |
|---|---|---|
| 0 | 1.000 | 0.063 |
| 1 | 0.433 | 0.145 |
| 2 | 0.187 | 0.336 |
| 3 | 0.081 | 0.775 |
| 4 | 0.035 | 1.79 |
| 5 | 0.0152 | 4.13 |
| 6 | 0.00658 | 9.55 |
| 7 | 0.00285 | 22.06 |
| 8 | 0.00123 | 50.96 |
| 9 | 0.000534 | 117.7 |
| 10 | 0.000231 | **271.98** |

**Range: 0.063 m → 272 m, ~4330× geometric span.**

This range is designed for cross-dataset generalization across MindCube (±5m indoor), SAT (±10m), VSI-Bench (±5m), SpinBench, RoboSpatial (±0.5m tabletop), etc.

### 3.4 Text-token identity rotation

For text tokens we set `xyz = (0, 0, 0)`:
```
freqs = 0 × inv_freq = 0    →  cos = 1, sin = 0    →  identity rotation
```

So text tokens are unaffected by the XYZ RoPE. Only image patch tokens get a non-trivial spatial-position-dependent rotation.

## 4. Architecture (file: `src/models/spa_emb_dec.py`)

```
SpaXYZRotaryEmbedding(nn.Module)
    Computes (xyz_cos, xyz_sin) of shape (B, S, 66) from per-token
    (B, S, 3) xyz tensor.  Forward accepts a runtime coord_scale override.

_apply_xyz_rotary(q, k, xyz_cos, xyz_sin, offset)
    Helper: split-half rotation applied locally to dims [offset : offset+66]
    of q and k.  Independent of the original rotary at dims [0 : offset].

SpaDecAttentionWrapper(nn.Module)
    Wraps Qwen3_5Attention.  Replicates its forward but inserts an extra
    XYZ rotation step AFTER the standard apply_rotary_pos_emb.

SpaDecTextModel(Qwen3_5TextModel)
    Reads self._xyz_pos and self._coord_scale (set by SpaDecModel.forward),
    computes xyz_position_embeddings, forwards to every decoder layer
    via the extra kwarg `xyz_position_embeddings`.

SpaDecModel(Qwen3_5Model)
    Computes per-token xyz tensor in forward() (text → (0,0,0); image
    patches → image_xyz) and stashes it on self.language_model._xyz_pos.

SpaDecForConditionalGeneration(Qwen3_5ForConditionalGeneration)
    Top-level.  Routes image_xyz and coord_scale to SpaDecModel.

patch_attention_layers_dec(model)
    Replaces every self_attn with SpaDecAttentionWrapper.  Call AFTER LoRA.
```

Side-channel design: rather than threading xyz through every forward signature, `SpaDecModel.forward` stashes `_xyz_pos` and `_coord_scale` on `language_model`, and `SpaDecTextModel.forward` reads them — same pattern as `SpaRelativeModel._pf_cache`.

### 4.1 KV-cache compatibility

During autoregressive decode (single new token, `past_key_values is not None`):
- The new token is text → `xyz = (0, 0, 0)` → identity rotation → cached K's XYZ rotation is preserved
- `SpaDecTextModel.forward` falls back to zeros when `_xyz_pos` shape doesn't match the current `seq_len` (decode case)
- No special wrapper logic needed — the wrapper just sees `xyz_position_embeddings` with all-1 cos and all-0 sin

## 4.5 Per-token Position Embeddings under `--decouple`

Under `--decouple`, every token is rotated through **two parallel RoPE channels**.
The third channel (pass-through) carries no position info.

### Channel 1 — Qwen original 3D M-RoPE (dims 0..63)

Unchanged from pretraining. `Qwen3_5Model.get_rope_index` emits a
`(3, batch, seq_len)` position_ids tensor of `(t, h, w)`, then
`Qwen3_5TextRotaryEmbedding` produces cos/sin applied to Q/K[0..63].
mrope_section = `[11, 11, 10]`: bands 0..10 → t, 11..21 → h, 22..31 → w.

| Token type | (t, h, w) value |
|---|---|
| **Text** at seq position p | `(p, p, p)` — all three dims equal → standard 1D RoPE behaviour |
| **Image patch** at row i, col j of image k | `(start_pos_k, start_pos_k + i, start_pos_k + j)` |
|   `start_pos_k` | running position counter at the time image k is encountered; advances by `max(H, W) // spatial_merge_size` after each image |

For text tokens `t = h = w = p`, so all 32 RoPE bands see the same value
(equivalent to ordinary 1D positional encoding). For image tokens, (h, w)
encode the patch's 2D location inside the image grid.

### Channel 2 — New XYZ RoPE (dims 64..129)

Per-token `(x, y, z)` tensor stashed on `language_model._xyz_pos`,
fed to `SpaXYZRotaryEmbedding` → `_apply_xyz_rotary` rotates Q/K[64..129].

| Token type | (x, y, z) value | Rotation effect |
|---|---|---|
| **Text** | `(0, 0, 0)` | `freqs = 0 × inv_freq = 0` → cos=1, sin=0 → **identity** (dims 64..129 untouched) |
| **Image patch** at row i, col j of image k | `image_xyz[k][i, j, :]` (per-patch 3D scene coord, see below) | Non-trivial RoPE rotation in 11 symmetric bands per axis |

#### How `image_xyz[k][i, j, :]` is computed

Defined in [`resize_xyz`](../../src/dataset/train_dataset.py) — for each LLM
patch (i, j) of image k:

```
image_xyz[k][i, j] = mean over valid pixels in the (stride_h × stride_w)
                     pixel block that this patch covers in the
                     original-resolution xyz map (pts3d.npy)
```

Concretely:
- Source: per-pixel xyz at the model's processed image resolution
  (`pts3d.npy` from MoGe / 3D estimator)
- Block size: `stride_h = H_src // llm_H`, `stride_w = W_src // llm_W`
  (e.g. for Qwen3.5 ViT patch_size=14 and spatial_merge_size=2, one LLM
  patch covers ~28×28 pixels on the source image)
- Mask: validity mask from the 3D source filters out unreliable pixels
  (e.g. background, sky, missing depth)
- Aggregation: `(masked_sum / valid_count)` over the block
- **Edge case**: if a patch has **no valid pixels**, it falls back to
  `(0, 0, 0)` — same as a text token, so the XYZ RoPE acts as identity for
  that patch too (silently drops bad patches from the spatial channel)

### Channel 3 — Pure content (dims 130..255)

No rotation at all. 126 dims of pure content features, untouched by either
position channel. Same as Qwen's original pass-through behaviour, just
shrunk from 192 to 126 dims.

### Combined attention score

For any pair of tokens (i, j):

```
⟨Q_i, K_j⟩  =  ⟨Q_i^[0..63],   K_j^[0..63]⟩      ← Qwen 3D M-RoPE: depends on (t,h,w) deltas
            +  ⟨Q_i^[64..129], K_j^[64..129]⟩    ← new XYZ RoPE:   depends on (x,y,z) deltas
            +  ⟨Q_i^[130..255], K_j^[130..255]⟩  ← pure content:   no position dependence
```

### Invariance: text-text attention exactly preserved

For two text tokens (no image involved):
- Channel 1: phase difference = `(p_i − p_j) × Qwen_inv_freq[k]` — identical to Qwen pretraining
- Channel 2: both have `xyz=(0,0,0)` → cos=1, sin=0 → `Q_i^[64..129] · K_j^[64..129]` is
  the **un-rotated** dot product, equivalent to plain content similarity
- Channel 3: pure content, identical to Qwen pretraining

Net effect: `⟨Q_i, K_j⟩_decouple = ⟨Q_i, K_j⟩_Qwen + ⟨content_i, content_j⟩_extra`.
The added term is **position-independent**, so the model never confuses
text-text relations with spatial ones — it just gains 66 extra content dims
on the text side. Pretrained text capability is untouched.

### Where each cross-attention sees XYZ signal

| Q ↔ K pair | XYZ RoPE active? | What it encodes |
|---|---|---|
| text ↔ text | ✗ (both identity) | nothing — degenerates to extra content channel |
| text → image | one-sided | text Q sees image K's xyz, image Q sees text K's identity |
| image ↔ image (same image) | ✓ | inter-patch relative xyz within the image |
| image ↔ image (cross-image) | ✓ | **inter-image relative xyz — the primary target signal** |

This is the design's payoff: the model learns cross-image 3D spatial
relations via the new channel while keeping all of Qwen's text-handling
capability intact.

## 4.6 Formal RoPE Formulation

### Notation

- $d_h = 256$: per-head dim
- $d_r = 64$: original Qwen rotary dim
- $d_x = 66$: new XYZ RoPE dim
- $d_p = 126$: pure pass-through dim   ($d_h = d_r + d_x + d_p$)

Per-token coordinates:
- $(t_i, h_i, w_i) \in \mathbb{N}^3$: Qwen M-RoPE position
  - Text token at seq position $p$: $(t_i, h_i, w_i) = (p, p, p)$
  - Image patch (row $r$, col $c$ of image $\ell$): $(t_i, h_i, w_i) = (s_\ell,\ s_\ell+r,\ s_\ell+c)$
- $(x_i, y_i, z_i) \in \mathbb{R}^3$: 3D scene coordinates
  - Text: $(0, 0, 0)$
  - Image patch: mean xyz of valid pixels in the LLM patch's source block

### Channel 1 — Qwen 3D M-RoPE (dims 0..63)

Inverse frequencies (Qwen default, $\theta_{\text{Qwen}} = 10^7$):

$$\beta_k = \theta_{\text{Qwen}}^{-2k/d_r},\qquad k = 0, 1, \ldots, 31$$

Band-to-axis assignment (M-RoPE section $[11, 11, 10]$):

$$
p^{(k)}_i = \begin{cases}
t_i, & k \in \{0, \ldots, 10\} \\
h_i, & k \in \{11, \ldots, 21\} \\
w_i, & k \in \{22, \ldots, 31\}
\end{cases}
\qquad
\phi^{\text{orig}}_{i,k} = p^{(k)}_i \cdot \beta_k
$$

2D rotation on pair $(k,\ k + 32)$ of Q (same for K):

$$
\begin{pmatrix} q'_{i,k} \\ q'_{i,k+32} \end{pmatrix}
=
\begin{pmatrix} \cos \phi^{\text{orig}}_{i,k} & -\sin \phi^{\text{orig}}_{i,k} \\
                \sin \phi^{\text{orig}}_{i,k} &  \cos \phi^{\text{orig}}_{i,k} \end{pmatrix}
\begin{pmatrix} q_{i,k} \\ q_{i,k+32} \end{pmatrix}
$$

### Channel 2 — New XYZ RoPE (dims 64..129)

Per-axis inverse frequencies (shared across $x$, $y$, $z$), $\theta_{\text{xyz}} = 10^4$, $D_a = 22$:

$$\beta'_m = \theta_{\text{xyz}}^{-2m/D_a},\qquad m = 0, 1, \ldots, 10$$

Band-to-axis assignment (sequential, but all three axes see the same $\beta'$ spectrum):

$$
\xi^{(k)}_i = \begin{cases}
x_i, & k \in \{0, \ldots, 10\} \\
y_i, & k \in \{11, \ldots, 21\} \\
z_i, & k \in \{22, \ldots, 32\}
\end{cases}
\qquad
m(k) = k \bmod 11
$$

Rotation angle (with $c = \text{coord\_scale} = 100$):

$$\phi^{\text{xyz}}_{i,k} = c \cdot \xi^{(k)}_i \cdot \beta'_{m(k)}$$

2D rotation on the absolute Q dim pair $(64 + k,\ 97 + k)$ (split-half pairing within the 66-dim slice):

$$
\begin{pmatrix} q'_{i,\,64+k} \\ q'_{i,\,97+k} \end{pmatrix}
=
\begin{pmatrix} \cos \phi^{\text{xyz}}_{i,k} & -\sin \phi^{\text{xyz}}_{i,k} \\
                \sin \phi^{\text{xyz}}_{i,k} &  \cos \phi^{\text{xyz}}_{i,k} \end{pmatrix}
\begin{pmatrix} q_{i,\,64+k} \\ q_{i,\,97+k} \end{pmatrix}
$$

For text tokens, $(x_i, y_i, z_i) = (0, 0, 0)$, so $\phi^{\text{xyz}}_{i,k} = 0$ for all $k$,
giving $\cos = 1$, $\sin = 0$ — identity rotation on dims 64..129.

### Channel 3 — Pass-through (dims 130..255)

$$q'_{i,d} = q_{i,d},\quad d \in \{130, \ldots, 255\}$$

### Attention score decomposition

After applying both rotations, the Q/K dot product splits additively:

$$
\langle q'_i,\, k'_j \rangle
= \underbrace{\sum_{d=0}^{63} q'_{i,d}\, k'_{j,d}}_{\text{Qwen M-RoPE: } \Delta(t,h,w)}
+ \underbrace{\sum_{d=64}^{129} q'_{i,d}\, k'_{j,d}}_{\text{new XYZ RoPE: } \Delta(x,y,z)}
+ \underbrace{\sum_{d=130}^{255} q'_{i,d}\, k'_{j,d}}_{\text{pure content}}
$$

Each RoPE band's pair-wise contribution reduces (via standard RoPE identity) to a
function of **phase difference** only:

$$
q'_{i,k} k'_{j,k} + q'_{i,k+N/2} k'_{j,k+N/2}
= A_{ij,k}\, \cos(\phi_i - \phi_j) + B_{ij,k}\, \sin(\phi_i - \phi_j)
$$

where $A_{ij,k},\ B_{ij,k}$ are bilinear in the unrotated $(q_i, k_j)$.
Thus:
- Channel 1 encodes $\Delta(t, h, w) = (t_i - t_j,\ h_i - h_j,\ w_i - w_j)$
- Channel 2 encodes $\Delta(x, y, z) = (x_i - x_j,\ y_i - y_j,\ z_i - z_j)$

The two channels are mathematically **orthogonal** — they share no dims.

### Text-text invariance (formal)

For any two text tokens $i, j$: $(x_i, y_i, z_i) = (x_j, y_j, z_j) = (0, 0, 0)$,
so $\phi^{\text{xyz}}_{i,k} - \phi^{\text{xyz}}_{j,k} = 0$ for all $k$. The XYZ channel becomes:

$$
\sum_{d=64}^{129} q'_{i,d}\, k'_{j,d} \;=\; \sum_{d=64}^{129} q_{i,d}\, k_{j,d}
\qquad (\text{un-rotated content dot-product})
$$

Combining with the identical Channel 1 (since $t=h=w=p$ for text, and $\beta_k$ unchanged):

$$
\langle q'_i,\, k'_j \rangle_{\text{decouple}}
\;=\;
\langle q'_i,\, k'_j \rangle_{\text{Qwen}}
\;+\; \underbrace{\sum_{d=64}^{129} q_{i,d}\, k_{j,d}}_{\text{extra content channel}}
$$

The added term is **position-independent**, so text-text relative attention
patterns are exactly preserved — the model just has 66 more content-only
dims for text reasoning.

### Relative-phase form (why only $\Delta$ matters)

Letting $p_i = p^{(k)}_i,\ p_j = p^{(k)}_j$ (or their xyz equivalent), the rotated-pair dot product:

$$
q'_{i,k} k'_{j,k} + q'_{i,k+N/2} k'_{j,k+N/2}
\;=\;
\operatorname{Re}\Big( (q_{i,k} + i\, q_{i,k+N/2})\, \overline{(k_{j,k} + i\, k_{j,k+N/2})}\cdot e^{i\, \beta_k (p_i - p_j)} \Big)
$$

The attention score depends on position only through $(p_i - p_j)$ — the defining
property of RoPE. In our case this means Channel 2 contributes a signal that
depends purely on $(\xi^{(k)}_i - \xi^{(k)}_j) \cdot \beta'_{m(k)}$ — the xyz
separation of the two patches, scaled by the band's frequency.

## 5. Mutual Exclusivity

`--decouple` is mutually exclusive with `--vanilla`, `--polar`, `--relative`:
- `--vanilla`: keeps original 3D M-RoPE only (no xyz at all). Conflicts because decouple ALSO wants original 3D M-RoPE (in 64 dims) but ADDs xyz.
- **`--polar`**: now also uses the decoupled architecture but feeds log-spherical input into the XYZ RoPE and uses `rope_theta=1000`. Since both modes select the same `SpaDecForConditionalGeneration` class, specifying both is redundant — mutex at the CLI prevents ambiguity. Pick one: `--decouple` for Cartesian, `--polar` for log-spherical. See [`polar_rope.md`](polar_rope.md).
- `--relative`: per-frame xyz rotation, requires modified mrope_section. Conflicts.

`--interleave_vision`: silently no-op under `--decouple` and `--polar` (interleave_vision affects the SpaTextRotaryEmbedding's mrope_section layout, which is unused when we don't touch the rotary 64 dims).

## 6. Comparison Across All Modes

| Mode | rotary 64 dims | new dims 64..129 | image_xyz | Dataset |
|---|---|---|---|---|
| `default` (4D) | repurposed `[2,10,10,10]` xyz | (pass-through) | Cartesian | `MindCube_Train_Dataset` |
| `--vanilla` | original `[11,11,10]` (no xyz) | (pass-through) | ignored | same |
| `--relative` | `[2,10,10,10]` per-frame xyz | (pass-through) | per-frame Cartesian | `MindCube_Train_Dataset_Relative` |
| **`--decouple`** | **original `[11,11,10]` UNCHANGED** | **NEW: XYZ rotary (cs=100, θ=10000)** | **Cartesian** | `MindCube_Train_Dataset` |
| **`--polar`** | **original `[11,11,10]` UNCHANGED** | **NEW: XYZ rotary (cs=100, θ=1000)** | **log-spherical (log r, θ, α)** | `MindCube_Train_Dataset` |

## 7. Coord Scale Plumbing

`AnswerOnlyModel` / `AnswerRelativeModel` now accept `coord_scale` at `__init__` (default 100). `train_correspondence.py:build_model` passes `coord_scale = 100.0` for all modes (decouple has no special value — uses the same 100 as the others). At forward, the value flows:

```
AnswerOnlyModel(coord_scale=100)
    → spa.forward(coord_scale=100)
    → SpaDecForConditionalGeneration.forward(coord_scale=100)
    → SpaDecModel.forward(coord_scale=100)
    → self.language_model._coord_scale = 100
    → SpaDecTextModel.forward
        → self.xyz_rotary_emb(xyz_pos, coord_scale=100)
```

Runtime override is supported at every level.

## 8. Wavelength Range Tuning Reference

If you ever need to retune (`--decouple` does not yet expose CLI flags for these):

| Goal | (cs, θ) | min wavelength | max wavelength | span |
|---|---|---|---|---|
| ±5 m indoor only | (100, 1000) | 0.063 m | 33.5 m | 530× |
| **default — broad scenes** | **(100, 10000)** | **0.063 m** | **272 m** | **4330×** |
| ±100 m outdoor | (100, 100000) | 0.063 m | 3540 m | 56000× |
| Tabletop ±0.5 m | (100, 100) | 0.063 m | 6.28 m | 100× |

Larger θ → broader range but coarser per-band intervals. The default (10000) is a compromise that covers cm precision through hundred-meter scenes.

## 9. Usage

```bash
# Train with decouple
bash scripts/train_correspondence.sh 2 --decouple
# Run name: correspondence_mindcube_decouple
# WandB run: corr_mindcube_r16_ep6_decouple

# Combine: --decouple is mutually exclusive with --polar/--vanilla/--relative
# but compatible with everything else (e.g. --max_samples N for smoke tests)
bash scripts/train_correspondence.sh 1 --decouple --max_samples 6
```

The shell script auto-suffixes `_decouple` to `RUN_NAME` and `WANDB_RUN_NAME`, and prints:
```
[INFO] Mode                 = decoupled (3D + new XYZ RoPE)
[INFO] Decouple position    = --decouple
```

## 10. Discussion Trail (key design decisions)

1. **Why touch pass-through at all?**  
   Qwen reserves 75% of head_dim for content (no position modulation). That's a lot of "wasted" dims for our 3D-spatial task. Carving 66 / 192 still leaves 126 pure-content dims — a 65% / 35% split between content and (text + xyz) position.

2. **Why 66 dims (= 33 bands)?**  
   Divisible by 6: gives 11 bands per axis, each band paired (split-half) within a 22-dim slice. Round numbers, leaves majority of pass-through intact.

3. **Why symmetric x/y/z spectrum?**  
   Initial draft used 33 sequential bands across the three axes (x got bands 0..10, y 11..21, z 22..32). This made x see only short wavelengths and z only long wavelengths — asymmetric for an isotropic 3D space. Fix: 11 bands per axis, all axes share the same spectrum.

4. **Why rope_theta = 10000 (not Qwen's 1e7)?**  
   Qwen's 1e7 covers context lengths up to 128K. For 3D scene coordinates the relevant scale is meters, not millions of tokens. 10000 at cs=100 gives wavelengths up to 272 m — covers indoor + city-scale scenes without wasting bands on kilometer wavelengths.

5. **Why coord_scale = 100 (not 10)?**  
   100 matches the existing pipeline's convention (cm-equivalent units after rounding). Going to 10 would shift the wavelength range to 0.63 m – 2720 m — too coarse on the short end (not enough cm-precision), too wide on the long end. cs=100 + θ=10000 hits a sweet spot.

6. **Why text → xyz=(0,0,0) → identity?**  
   Cleanest invariant: text-token attention is exactly unchanged from Qwen pretraining (since the new XYZ RoPE acts as identity on text). All "pretrained" semantics in text-text and text-image attention are preserved — the new channel only adds to image-image and image-text relations.

7. **Why side-channel `_xyz_pos` instead of new forward arg everywhere?**  
   Threading `xyz_position_embeddings` through every layer's forward signature would require subclassing Qwen3_5DecoderLayer. The side-channel pattern (set on language_model before forward, read inside forward) is the same trick used by `SpaRelativeModel._pf_cache` and is grad-checkpoint compatible.

8. **Cross-dataset generalization via log-spherical (now implemented as `--polar`)**  
   For true scale invariance, `(x,y,z) → (log r, θ, α)` is ideal (RoPE pos-diff = `log(r_i / r_j)` is invariant to global scaling). **This is now the `--polar` mode** — same decoupled architecture, but the XYZ RoPE consumes log-spherical input and uses `rope_theta = 1000` (tighter than Cartesian's 10000 to match polar's narrower dynamic range). The text=identity invariant is preserved via `zero_mask` (text tokens with xyz=0 stay at log_r=0, θ=0, α=0 → cos=1, sin=0). See [`polar_rope.md`](polar_rope.md).
