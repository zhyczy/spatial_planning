# `train_atten.py`: Per-Layer Geometric Attention Bias for Qwen3.5-VL

## 1. Purpose

Inject 3-D scene geometry into Qwen3.5-VL **without touching the position
embedding**. The original 3-D M-RoPE `[11, 11, 10]` is preserved verbatim;
spatial information enters the model only through an **additive per-head
attention bias** on vision-vision pairs.

For each LLM transformer layer, a tiny per-pair MLP turns the geometric
relationship between any two vision patches into a per-head bias scalar:

```
attention = softmax( QK^T / √d  +  causal_mask  +  B_spatial ) V
```

where `B_spatial[h, i, j]` is non-zero only when both `i` and `j` are vision
tokens.

Relation to sibling scripts:

| Script | Spatial signal lives in | Trainable extras |
|---|---|---|
| `train_correspondence.py` | 4D-RoPE positions | LoRA only |
| `train_coordinate.py`     | 4D-RoPE + per-patch xyz regression head | LoRA + coord head |
| **`train_atten.py`**      | **Per-layer additive attention bias** | **LoRA + per-layer MLP** |

## 2. Architecture

```
AnswerOnlyModel                                    ← LM-loss-only training shell
└── Qwen3_5ForConditionalGeneration  [LoRA-wrapped]
     ├── Qwen3_5VisionModel  (ViT, frozen)
     └── SpatialAttnVanillaModel              ← swapped in inline (build_model)
          └── SpatialAttnVanillaTextModel     ← punches V↔V hole in causal_mask
               └── decoder layers
                    │  (linear-attention layers are NOT wrapped — see §3)
                    └── self_attn = SpatialAttnWrapper(orig_attn, num_heads)
                         ├── orig attn (q/k/v/o + LoRA)
                         └── bias_module = SpatialAttentionBias
                              └── 2-layer MLP : (n_x, n_y, n_z, d) → num_heads
```

Why the swap recipe instead of subclassing the conditional-generation class?
`SpatialAttnVanillaModel` adds **zero new parameters** to its base
`Qwen3_5Model` — `from_pretrained` loads a stock backbone, we replace `.model`
in-place, `tie_weights()` re-binds `lm_head ↔ embed_tokens`. State dict keys
stay aligned so a strict load carries pretrained weights over with no remap.
See [build_model](../../train_atten.py).

## 3. Layer Coverage in Qwen3.5's Hybrid Stack

> Background: see [qwen3.5.md §1](qwen3.5.md#1-hybrid-attention-stack--full-attn--gated-deltanet)
> for the full base-model finding. This section summarises only the parts
> relevant to where the spatial bias actually fires.

Qwen3.5-4B's text decoder is a **hybrid**: 24 of its 32 layers are linear
attention (`Qwen3_5GatedDeltaNet`, a Mamba2/DeltaNet variant), and only 8
are standard softmax attention (`Qwen3_5Attention`). The pattern from
`config.text_config.layer_types` is `[L,L,L,F] × 8`, with full-attention
sitting at indices `{3, 7, 11, 15, 19, 23, 27, 31}`.

This matters because **the spatial bias mechanism is fundamentally an
additive logit modification**, which only the standard-attention path
supports.

### 3.1 Why linear-attention layers cannot consume the bias

| Property | Full attention (8 layers) | Linear attention (24 layers) |
|---|---|---|
| `forward(... position_embeddings: tuple, ...)` accepted? | Yes | **No** — signature has no such arg |
| Applies RoPE to Q/K? | Yes — `apply_rotary_pos_emb(q, k, cos, sin)` | **No** — RoPE is never invoked |
| Materialises `(L × L)` score matrix `S = QK^T`? | Yes | **No** — replaced by recurrent state of shape `(d × d)` |
| Uses `attention_mask` as additive pre-softmax bias? | Yes | **No** — `attention_mask` is only for `apply_mask_to_padding_states` (a padding *gate*, not a logit *bias*) |
| Positional info source | RoPE rotation in feature space | 1D Causal Conv + Mamba-style decay (`A_log`, `dt_bias`) + recurrence order |
| Where geometric prior could plug in | Add to logit matrix → standard | Would need to modulate decay / kernel features / recurrent state — different mechanism, not implemented |

The bias is `(B, H, L, L)` shaped and meant to be **added pre-softmax**. In
linear attention there is no pre-softmax — `softmax(QK^T)` was algebraically
decomposed away to avoid the L² cost. The bias has no place to land.
`Qwen3_5GatedDeltaNet.forward` doesn't even accept `position_embeddings` /
`attention_mask` in the additive sense; if we hand it our `new_mask` it gets
swallowed by `**kwargs` and discarded.

### 3.2 What `patch_attention_layers_spatial` actually does

```
For each decoder_layer in language_model.layers:
    if layer.layer_type == "linear_attention":
        skip                          # leave self_attn untouched
    else:
        layer.self_attn = SpatialAttnWrapper(layer.self_attn, num_heads=H)
        # → owns its own SpatialAttentionBias (one MLP per wrapped layer)
```

Result on Qwen3.5-4B: **8 wrappers** at layer indices {3, 7, 11, 15, 19, 23,
27, 31}, **24 layers untouched**. The trainable bias-module parameter count
scales with 8, not 32 — about 38K parameters total (4·128 + 128·H + bias
overheads × 8 layers, with H = 16 attention heads on 4B).

This is also why the previous draft of the patcher caused a DDP
`find_unused_parameters` failure: it wrapped all 32 layers, the 24 linear
ones consumed the wrapped output but discarded the bias, those bias-module
params never touched the loss, and DDP refused the next bucket rebuild.

### 3.3 The bias is in lockstep with RoPE

Both spatial bias and 3D M-RoPE are mechanisms that modify Q/K behaviour, and
**both only fire on the same 8 full-attention layers**. There is no asymmetry
to exploit: any positional / geometric prior that uses RoPE-style or additive-
logit interfaces inherits this 8/32 coverage. (The same caveat applies to the
4D / decoupled / polar RoPE variants used by
[decouple_xyz_rope.md](decouple_xyz_rope.md) and
[polar_rope.md](polar_rope.md) — they all live on 8 layers.)

### 3.4 Why this still works — serial signal propagation

Hidden states flow through every layer in sequence. Geometric structure
injected at layer 3 propagates into the input of layer 4 (linear), which
mixes it via its convolutional / recurrent kernel and hands an enriched
hidden state to layer 5 (linear), etc., until layer 7 (full) injects again.

```
layers 0..2 (L)   →  no direct injection, propagate vanilla hidden
layer 3      (F)  →  ★ inject bias #1, RoPE applied to Q/K
layers 4..6 (L)   →  carry the spatially-aware hidden state forward
layer 7      (F)  →  ★ inject bias #2, on top of accumulated signal
...
layer 31     (F)  →  ★ inject bias #8
```

So the bias module functions as **8 sparse injection points**, with the
24 linear layers acting as inter-injection mixers. It is sparser than a
"every-layer" prior would be, but it's the only mechanism the architecture
admits without redesigning the bias for DeltaNet kernels (deferred — see §11).

## 4. Spatial Attention Bias

### 4.1 Per-pair edge feature

For each ordered pair `(i, j)` of vision tokens with 3-D coords `p_i, p_j`:

```
Δp_ij  = p_j - p_i                          ∈ ℝ³        (Query i → Key j)
d_ij   = ‖Δp_ij‖₂                                       (raw scalar)
n_ij   = Δp_ij / d_ij                       ∈ ℝ³        (unit direction)
feat_ij = (n_x, n_y, n_z, d) ∈ ℝ⁴
```

**Why scale-decouple direction from magnitude.** A single Linear over
`(Δx, Δy, Δz, d)` entangles orientation with absolute scale: a vector twice
as long produces twice the activation on its direction channels. Normalizing
direction lets the MLP specialize the first three channels on geometry while
the distance channel carries scale separately.

**Numerical stability.** The diagonal (`i == j`) has `Δp = 0`, where `‖·‖`
has undefined gradient. We add `ε = 1e-8` inside the sqrt:

```
dist = sqrt((Δp ⊙ Δp).sum() + ε)
n    = Δp / dist   # at i==j: n ≈ 0, dist ≈ 1e-4 → harmless constant feature
```

This protects against NaN under autograd / gradient-checkpointing recompute,
even when `xyz` doesn't itself require grad.

### 4.2 The 2-layer MLP

```python
self.mlp = nn.Sequential(
    nn.Linear(4, hidden_dim),         # W₁ ∈ ℝ^{hidden × 4},  default hidden = 128
    nn.GELU(),                        # point-wise non-linearity
    nn.Linear(hidden_dim, num_heads), # W₂ ∈ ℝ^{H × hidden}
)
```

Same MLP weights are reused at every `(i, j)` pair location and across all
samples in the batch — i.e. **structurally a 1×1 convolution** over the N×N
"edge image" with 4 input channels and H output channels.

**Per-layer params.** `(4 + H) · hidden + (1 + H)`. With `H = 32, hidden = 128`
that's ~4.7K per layer; ~170K total on a 36-layer model — < 1% of a rank-16
LoRA. Going to `hidden = 256` doubles params/FLOPs but typically yields
marginal gains: input is only 4-D, so 128 GELU basis functions is already
plenty of non-linear capacity.

**Why per-head output (instead of head-shared scalar).** Different heads
specialize on different geometric patterns (vertical neighbors vs. far-distance
pairs vs. cross-image alignments). A per-head output gives each head its own
learnable spatial template; the per-pair 4-D input is shared.

### 4.3 Initialization

**Only the OUTPUT layer (W₂, b₂) is zero-initialized.** That alone forces
`B ≡ 0` at step 0 → pretrained attention is undisturbed. `W₁` keeps PyTorch's
default Kaiming-uniform init so the gradient path stays healthy from the
start. Zeroing both layers traps W₁ at zero (`∂L/∂W₁ ∝ W₂ᵀ · grad ≡ 0`).

### 4.4 Why this looks like a 1×1 conv (and what NOT to do)

The MLP applied per-pair with shared weights is mathematically equivalent to
`Conv2d(4, H, kernel_size=1)`. Tempting upgrade: `kernel_size=3` to give each
pair some "neighborhood context."

**Don't.** The neighbors of pair `(i, j)` in the matrix are `(i±1, j±1)` —
these are *index* neighbors, not *3-D geometric* neighbors. In a normal CNN,
pixel-index neighbors and spatial neighbors coincide; here they don't.
A 3×3 kernel on the N×N edge image would mix in geometrically arbitrary pairs.

If true neighborhood context is needed, the right path is graph-style kNN
aggregation in 3-D space (or augmenting `feat_ij` with sinusoidal grid-position
encodings of `i` and `j`). Both are larger redesigns; not currently done.

## 5. Three-Mask Safety System

Three orthogonal mask operations cooperate to make the bias safe and useful.
Each lives in a different module and serves a different concern.

### 5.1 `vision_mask` — spatial scoping (in `SpatialAttentionBias.forward`)

Built from `mm_token_type_ids != 0`. Inside the bias module, `bias` is
initialized to zeros and the MLP output is scattered **only at
`(vis_idx × vis_idx)` cells**:

```python
bias = torch.zeros(B, H, seq_len, seq_len, ...)
for b in range(B):
    vis_idx = vision_mask[b].nonzero(as_tuple=True)[0]   # (N_b,)
    bias[b, :, vis_idx[:, None], vis_idx[None, :]] = M_all[b, :, :N_b, :N_b]
```

Guarantee: `bias` is **identically zero** on V→T, T→V, T→T, and any padding
cell. The geometric prior cannot leak into cross-domain attention.

### 5.2 Prefix-mask hole — bidirectional V↔V (in `SpatialAttnVanillaTextModel.forward`)

Stock `create_causal_mask` is strict lower-triangular: a vision query at
position `i` can attend to vision keys `j` only if `j ≤ i`. That wastes half
the geometric signal — `B[i, j>i]` would be drowned by the `-inf` mask (and
actively pinned back to `-inf` by the wrapper's defense layer below).

We zero out `causal_mask` at every `(i, j)` where both positions are vision
tokens, **only during prefill**:

```python
is_vv_pair = (
    vision_mask.unsqueeze(1).unsqueeze(3)       # (B, 1, L, 1)  query axis
    & vision_mask.unsqueeze(1).unsqueeze(2)     # (B, 1, 1, L)  key   axis
)                                               # (B, 1, L, L)
causal_mask = torch.where(
    is_vv_pair,
    torch.zeros((), dtype=causal_mask.dtype, device=causal_mask.device),
    causal_mask,
)
```

Decode short-circuit: skipped when KV cache is non-empty (query is a single
new text token; V↔V hole is irrelevant and `vision_mask` wouldn't cover it).

### 5.3 Defensive re-mask — barrier preservation (in `SpatialAttnWrapper.forward`)

After `new_mask = attention_mask + bias`, we re-mask any cell whose original
mask was a "hard barrier":

```python
is_masked = attention_mask < -1e4
new_mask = torch.where(is_masked, attention_mask, new_mask)
```

Threshold `-1e4` is comfortably above any value real attention scores reach
(O(1)~O(10) post-softmax-pre-bias) and below any HF mask convention
(`-1e4` for fp16, `-65504` for bf16, `-3.4e38` / `-inf` for fp32). Catches
runaway bias from bf16 noise, training divergence, or upstream off-by-ones —
defends causal continuity for text and integrity of padding masks.

### 5.4 Combined truth table

| Cell type | original `causal_mask` | after V↔V hole | `bias` (vision_mask) | final `new_mask` |
|---|---|---|---|---|
| V→V, lower triangle | 0 | 0 | learned geometry | bias ✓ |
| V→V, upper triangle | -inf | **0 (hole)** | learned geometry | **bias ✓ (newly enabled)** |
| V→V, diagonal | 0 | 0 | learned geometry | bias ✓ |
| T→T, past | 0 | 0 | 0 | 0 ✓ |
| T→T, future | -inf | -inf | 0 | -inf (preserved by where) ✓ |
| V→T, T→V (cross-domain) | unchanged | unchanged | **0** (vision_mask) | unchanged ✓ |
| Padding row/col | -inf | -inf | **0** (vision_mask) | -inf ✓ |

The three mask layers are **orthogonal**: vision_mask cuts the *spatial*
domain of the bias; the prefix-hole opens the *temporal* domain on V×V
specifically; the wrapper's `torch.where` enforces the *temporal* contract
elsewhere. Each concern owns exactly one place in the code.

## 6. Batch Handling

### 6.1 Batch payload contract (7 fields)

Every training / eval step extracts these 7 fields from a batch dict.
Concrete shapes shown for one MindCube sample (4 images, 877-token sequence).
`tb["..."]` = required (KeyError if missing); `tb.get("...")` = optional
(returns None, downstream branches handle absence).

| Variable | Source | Shape | Dtype | Producer | Role |
|---|---|---|---|---|---|
| `t_ids`  | `tb["input_ids"]`         | `(1, 877)`            | int64   | tokenizer            | Token id sequence (`<|image_pad|>` placeholders for 4 image blocks of 192 each + text on either side). |
| `t_mask` | `tb["attention_mask"]`    | `(1, 877)`            | int64   | tokenizer            | Padding mask: `1` = real token, `0` = right-pad filler. With B=1 always all 1s. |
| `t_pv`   | `tb.get("pixel_values")`  | `(3072, 1536)`        | float32 | image processor      | 4 imgs × 768 ViT patches × 1536-dim patch features, concatenated along the patch axis. |
| `t_thw`  | `tb.get("image_grid_thw")`| `(4, 3)`              | int64   | image processor      | Per-image `(T, H, W)` grid, e.g. `[1, 32, 24]`. ViT uses it to reshape `t_pv`; M-RoPE uses it for (h,w) axes. |
| `t_mm`   | `tb.get("mm_token_type_ids")` | `(1, 877)`        | int64   | multimodal processor | Modality label per token: `0=text`, `1=image`, `2=video`. **REQUIRED for spatial bias** — see [qwen3.5.md §3](qwen3.5.md#3-mm_token_type_ids--modality-labeling-contract). |
| `t_lbl`  | `tb.get("labels")`        | `(1, 877)`            | int64   | dataset (custom)     | LM-loss target. `-100` at non-answer positions (system prompt + image_pads + question), real ids at answer + `<|im_end|>` + `\n`. |
| `t_xyz`  | `tb.get("image_xyz")`     | `list[(H,W,3)] × 4`   | float32 | dataset (custom)     | Per-image patch-level world-frame 3-D coords, derived from `pts3d.npy`. Inner shape `(16, 12, 3)` per image (= `H/sms × W/sms × 3`). All images share the view_0000 camera frame. |

`t_pv` / `t_thw` / `t_mm` are marked optional but in practice MindCube always
emits them. `t_xyz` is the only field that's genuinely conditional on whether
3-D ground truth exists for the sample (which is universally true for
MindCube but not for some other eval datasets).

### 6.2 Validity invariants

A correctly-assembled batch satisfies a chain of equalities — useful for
sanity-asserting in dev mode:

```
sum_k (t_xyz[k].numel() / 3)             ≡  Σ (16·12) × 4 imgs  =  768
                  ║  must equal
(t_mm == 1).sum()                         =  768   (image-token positions)
                  ║  must equal
sum_k (t_thw[k].prod() / sms²)            =  Σ (1·32·24/4) × 4  =  768
                  ║  must equal
count of <|image_pad|> in t_ids           =  768
                  ║  related (× sms²)
t_pv.shape[0]                             =  3072  (= 768 × sms²)
```

Where `sms = spatial_merge_size` (= 2 for Qwen3.5-VL default config). Any
mismatch indicates an indexing bug somewhere upstream — the spatial bias
will silently scatter into the wrong cells without an obvious error. The
runtime assertion in `SpatialAttnVanillaModel.forward` catches the most
critical one (`vision_mask.sum() == flat_xyz.shape[1]`).

### 6.3 `image_xyz` padding pipeline (B → 1 today, ready for B > 1)

`SpatialAttnVanillaModel.forward` accepts two input shapes for `image_xyz`
and normalizes both to a common form before dispatch:

| Input form | When | Normalization |
|---|---|---|
| `list[Tensor]` (B=1 shorthand) | default collate | wrap → `[image_xyz]` |
| `list[list[Tensor]]` | true batched collate (B>1) | already normalized |

Then per-sample flatten + `pad_sequence`:

```python
per_sample = [
    torch.cat([x.reshape(-1, 3) for x in sample], dim=0)
    for sample in image_xyz
]                                                     # list[(N_b, 3)]
flat_xyz = torch.nn.utils.rnn.pad_sequence(
    per_sample, batch_first=True, padding_value=0.0,
)                                                     # (B, N_max, 3)
```

Per-sample sanity (one `.tolist()` sync, B small): each sample's
`vision_mask[b].sum()` must equal that sample's `N_b`.

Downstream `SpatialAttentionBias.forward` runs the MLP **vectorized** over
the full padded grid `(B, N_max, N_max, 4)` → `(B, H, N_max, N_max)`, then
**scatters only the valid `[:N_b, :N_b]` sub-block per sample** into the
final bias. Padded cells cost FLOPs but contribute zero gradient (no path
to loss) and never reach the attention mask.

### 6.4 Note on collate

[train_atten.py](../../train_atten.py)'s `collate_fn` currently asserts
`len(batch) == 1`. The model side is now ready for
`per_device_train_batch_size > 1` but the collate still needs an upgrade
(text-side padding for `t_ids` / `t_mask` / `t_mm` / `t_lbl` + multi-sample
`image_xyz` collection into list-of-lists) before B>1 can actually be enabled.

## 7. Decode-Phase Short-Circuit

During autoregressive generation the query is a single newly-generated text
token; its row of `B` is zero by definition. `SpatialAttnWrapper` detects
`past_key_values.get_seq_length() > 0` and **falls through to vanilla
attention** with no extra cost — vision-aware reasoning was already baked
into the KV cache during prefill.

The prefix-mask hole is also gated to prefill only; decode keeps the stock
causal mask.

## 8. Data Flow (one training step)

```
batch (input_ids / attention_mask / pixel_values / image_grid_thw / image_xyz / labels)
  ↓
AnswerOnlyModel.forward                       ← passes image_xyz down
  ↓
Qwen3_5ForConditionalGeneration (PEFT-wrapped)
  ├── ViT(pixel_values) → image features → embed at <|image_pad|> positions
  └── SpatialAttnVanillaModel.forward
       ├── normalize image_xyz → list[list[Tensor]]
       ├── per-sample flatten → pad_sequence → (B, N_max, 3)
       ├── vision_mask = (mm_token_type_ids != 0)
       ├── per-sample length sanity check
       ├── stash on language_model._spatial_cache
       └── super().forward(...)
            ↓
            SpatialAttnVanillaTextModel.forward
              ├── embed_tokens → causal_mask (Qwen create_causal_mask)
              ├── ★ punch V×V hole in causal_mask (prefill only)
              ├── 3-D M-RoPE position_embeddings (UNCHANGED from Qwen)
              ├── read _spatial_cache → extra_attn_kwargs={flat_xyz, vision_mask}
              └── for each decoder_layer:
                   layer(hidden, attention_mask=causal_mask, **extra_attn_kwargs)
                     ↓
                     SpatialAttnWrapper.forward
                       ├── if decode or no spatial info → pass-through to attn
                       └── prefill:
                            ├── bias = bias_module(flat_xyz, vision_mask)  (B, H, L, L)
                            ├── new_mask = attention_mask + bias
                            ├── ★ defensive re-mask (preserve hard barriers)
                            └── attn(..., attention_mask=new_mask, ...)
                                 ← Q/K/V/O are LoRA-augmented, SDPA kernel runs
  ↓
logits → AnswerOnlyModel: shift, mask out -100, cross-entropy on answer tokens
```

★ = the two new mask operations that distinguish this stack from a stock VLM.

## 9. Checkpoint Format

```
step_<N>/
  adapter_*           ← PEFT LoRA adapter (saved via save_pretrained)
  spatial_bias.pt     ← {fully-qualified module name → state_dict} for every
                        SpatialAttentionBias instance.
                        Keys per module:  mlp.0.weight  mlp.0.bias
                                          mlp.2.weight  mlp.2.bias
```

Loaded by `load_spatial_bias_modules(model, ckpt_dir)` in
[train_atten.py](../../train_atten.py) — instantiate the model, run
`patch_attention_layers_spatial`, then call the loader.

**Migration note**: pre-MLP checkpoints had keys `proj.weight / proj.bias`
(single Linear). They are **not loadable** by the current 2-layer MLP
structure — re-train from the LoRA adapter or write a one-off remap.

## 10. Code Map

| Concern | File | Symbol |
|---|---|---|
| Edge feature + per-pair MLP | [src/models/spatial_attention_block.py](../../src/models/spatial_attention_block.py) | `SpatialAttentionBias` |
| Self-attn wrapper, defensive re-mask | [src/models/spatial_attention_llm.py](../../src/models/spatial_attention_llm.py) | `SpatialAttnWrapper` |
| TextModel + V×V prefix-hole + spatial-cache injection | [src/models/spatial_attention_llm.py](../../src/models/spatial_attention_llm.py) | `SpatialAttnVanillaTextModel` |
| Outer Model + image_xyz → pad_sequence → cache | [src/models/spatial_attention_llm.py](../../src/models/spatial_attention_llm.py) | `SpatialAttnVanillaModel` |
| Post-LoRA attention patching | [src/models/spatial_attention_llm.py](../../src/models/spatial_attention_llm.py) | `patch_attention_layers_spatial` |
| Build, train loop, checkpoints | [train_atten.py](../../train_atten.py) | `build_model`, `train`, `_save_checkpoint`, `load_spatial_bias_modules` |
| LM-loss training shell | [src/models/answer_llm.py](../../src/models/answer_llm.py) | `AnswerOnlyModel` |

## 11. Open Questions / Future Work

* **Distance Fourier features.** `d` is currently a raw scalar; MLPs have
  spectral bias, so high-frequency dependence on small distances may be
  hard to learn. Adding `[sin(2^k π d), cos(2^k π d)]` would give the MLP
  a high-frequency basis at low cost. Deferred — try once a baseline run is in.
* **Wider MLP.** Default `hidden_dim = 128` is comfortable but conservative.
  Bumping to 256 doubles params/FLOPs but stays well under 2% of LoRA;
  expected gain modest, decide from training curves.
* **Multi-batch collate.** Model side ready; `collate_fn` in
  [train_atten.py](../../train_atten.py) still asserts `B == 1`. Needs a
  text-side padding implementation before `per_device_train_batch_size > 1`.
* ~~**Linear-attention layers.**~~ **Resolved 2026-04-27.** Qwen3.5 mixes
  standard and linear attention; the linear-attn kernel ignores the additive
  `attention_mask`, so any bias passed through there was silently dropped and
  `bias_module.mlp.*` params on those layers never received gradient — DDP
  surfaced this as a `find_unused_parameters` failure.
  `patch_attention_layers_spatial` now skips layers tagged
  `layer_type == "linear_attention"` and only wraps the standard-attn ones.
  Logged as `wrapped N standard-attn layers, skipped K linear-attn layers`.
* **Geometric-neighborhood context.** A 1×1 MLP on each pair has no notion
  of "nearby pairs in 3-D." Real upgrade path is kNN-graph aggregation in 3-D
  or sinusoidal grid-position concat — both are larger redesigns.
* **First-token accuracy assumes single-letter answers.** The eval loop in
  [train_atten.py](../../train_atten.py) reports `*_acc` by comparing the
  argmax of the FIRST non-`-100` label position against the GT token at that
  position. This is exactly answer-accuracy because all three current datasets
  (MindCube train/tinybench, SpinBench test) consist of single ASCII letters
  A/B/C/D as answers — verified `100%` (1050 / 2739 / 10000 samples) — and
  Qwen tokenizes those as single tokens. **If a future dataset adds
  multi-token answers** (free-form text, numbers, parenthesized `(A)/(B)`,
  or any tokenizer that introduces a leading-space variant), this metric
  silently degrades into "first-token accuracy" and over-reports. To stay
  general, switch to either (a) all-position-match using the full
  non-`-100` slice or (b) `.generate()` + exact-match against the answer
  string.


32 层 decoder 里，只有第 3、7、11、15、19、23、27、31 层（共 8 层）会用到几何 bias。


Qwen3.5-4B 的 32 层 decoder（按从底到顶顺序）

层号:  0   1   2   3   4   5   6   7   8   9  10  11  ...  28  29  30  31
类型:  L   L   L   F   L   L   L   F   L   L   L   F        L   L   L   F
                    ↑               ↑               ↑                       ↑
              注入 bias        注入 bias        注入 bias              注入 bias
              SpatialAttn      SpatialAttn      SpatialAttn            SpatialAttn
              Wrapper          Wrapper          Wrapper                Wrapper
              + bias_module    + bias_module    + bias_module          + bias_module
              (独立 MLP)       (独立 MLP)       (独立 MLP)              (独立 MLP)

L = linear_attention（24 层，纯走原生 Qwen，bias 不动）
F = full_attention（8 层，wrap 之后 attention 分数里加 bias）
「应用在哪」三个层次具体讲
1. 模块所在位置
8 个 SpatialAttentionBias 实例，分别挂在 8 个 full-attention 层的 self_attn.bias_module 字段下。每个实例有自己独立的 2 层 MLP 权重（不共享），即整套结构有：

8 个 mlp.0.weight (shape: [128, 4])
8 个 mlp.0.bias (shape: [128])
8 个 mlp.2.weight (shape: [num_heads, 128])
8 个 mlp.2.bias (shape: [num_heads])
2. 计算上的应用点
在这 8 个层里，attention 的标准公式被改写成：


原本 (linear-attn 24 层 + full-attn 在修复前):
    softmax(QK^T / √d + causal_mask) · V

修复后的 8 个 full-attn 层:
    softmax(QK^T / √d + causal_mask + B_spatial) · V
                                       ↑
                                       │
                              这一项就是几何 bias
                              来自  bias_module(flat_xyz, vision_mask)
                              shape (1, num_heads, L, L)
                              非零仅在 V×V cells

剩下 24 个 linear-attn 层:
    完全不变，没有几何 bias
3. 信号传播路径

input_ids  →  embed
                ↓
Layer 0  (L)    hidden ← linear_attn(hidden)               ── 无 bias
                ↓
Layer 1  (L)    hidden ← linear_attn(hidden)               ── 无 bias
                ↓
Layer 2  (L)    hidden ← linear_attn(hidden)               ── 无 bias
                ↓
Layer 3  (F)    hidden ← full_attn(hidden, +bias_3)        ── ★ bias 注入 #1
                ↓
Layer 4  (L)    hidden ← linear_attn(hidden)               ── 用上面注入的 hidden
                ↓
        ... 重复 4 个一组 ...
                ↓
Layer 7  (F)    hidden ← full_attn(hidden, +bias_7)        ── ★ bias 注入 #2
                ↓
        ...
                ↓
Layer 31 (F)    hidden ← full_attn(hidden, +bias_31)       ── ★ bias 注入 #8
                ↓
              norm  →  lm_head  →  logits


spatial bias 设计联系起来看就很清楚了

信息层次:                               注入机制:           注入到的层数:

┌────────────────────────────────┐    ┌──────────────┐
│ 文本时序 (causal, autoregressive)│ → │ causal_mask   │ →  全部 32 层
└────────────────────────────────┘    └──────────────┘    （但 linear 用自己的 padding mask 形式）

┌────────────────────────────────┐    ┌──────────────┐
│ 多模态 token 类型 + 1D 位置      │ → │ 3D M-RoPE     │ →  仅 8 个 full-attn 层
│ (text_seq, t, h, w)             │    │ (Q/K rotation)│    （linear 完全不用 RoPE）
└────────────────────────────────┘    └──────────────┘

┌────────────────────────────────┐    ┌──────────────┐
│ 3D 几何先验                      │ → │ Spatial bias  │ →  仅 8 个 full-attn 层
│ (n_x, n_y, n_z, d) per pair     │    │ (additive)    │    （linear 注入不进去）
└────────────────────────────────┘    └──────────────┘

┌────────────────────────────────┐    ┌──────────────┐
│ 局部时序 + 递推顺序              │ → │ 1D Conv +     │ →  仅 24 个 linear 层
│                                  │    │ Mamba decay   │    （full-attn 没有这个）
└────────────────────────────────┘    └──────────────┘