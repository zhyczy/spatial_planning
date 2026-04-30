# Qwen3.5-VL — Base Model Findings

**Scope:** Notes about the **stock** Qwen3.5-VL backbone (`checkpoints/Qwen3.5-4B`)
that we discovered while building modifications on top of it. Treat this as
the running knowledge base of "things you should know about the substrate
before you change it." Each finding lists the source file/line in the
installed `transformers` package so it can be re-verified after a library
upgrade.

Cross-references:
- Spatial-attention bias design: [spatial_attention.md](spatial_attention.md)
- 4D RoPE variants: [decouple_xyz_rope.md](decouple_xyz_rope.md), [polar_rope.md](polar_rope.md)

---

## 1. Hybrid attention stack — Full-attn + Gated DeltaNet

Qwen3.5-4B's text decoder is **not pure transformer attention**. It is a
**hybrid**: most layers are linear-attention (Gated DeltaNet, a Mamba2/DeltaNet
variant), and only a sparse subset are standard softmax attention.

### 1.1 Layer count and pattern

```
config.text_config.num_hidden_layers = 32
config.text_config.layer_types       = list[32]
distribution: {'linear_attention': 24, 'full_attention': 8}
pattern:      [L, L, L, F]  ×  8        # every 4th layer is full-attention
indices F:    [3, 7, 11, 15, 19, 23, 27, 31]
```

`config.text_config.layer_types` is the source of truth — read it directly
rather than hardcoding the pattern (could change in future Qwen3.5 variants).
The TextModel forward dispatches per layer:

```python
# transformers/models/qwen3_5/modeling_qwen3_5.py: in TextModel.forward
layer_mask = (
    linear_attn_mask
    if decoder_layer.layer_type == "linear_attention"
    else causal_mask
)
```

### 1.2 What each layer type IS

|  | `Qwen3_5Attention` (full, 8 layers) | `Qwen3_5GatedDeltaNet` (linear, 24 layers) |
|---|---|---|
| Defined at | `modeling_qwen3_5.py:714` | `modeling_qwen3_5.py:446` |
| Has `q_proj` / `k_proj` / `v_proj` / `o_proj` ? | **Yes** — standard nn.Linears | **No** — uses fused `in_proj_qkv` + 1D causal Conv1d |
| Receives `position_embeddings` arg? | **Yes** — `forward(..., position_embeddings: tuple[Tensor, Tensor], ...)` | **No** — `forward(self, hidden_states, cache_params, cache_position, attention_mask)` |
| Applies RoPE? | **Yes** — `apply_rotary_pos_emb(q, k, cos, sin)` | **No** — RoPE is never invoked |
| Uses additive `attention_mask`? | **Yes** — added to QK^T pre-softmax | **No** — `attention_mask` is passed only for `apply_mask_to_padding_states` (padding mask, not logit bias) |
| Pair-wise score matrix `(L × L)` materialised? | **Yes** | **No** — replaced by recurrent state |
| Positional info comes from | RoPE rotation of Q/K | Causal Conv1d kernel + Mamba-style decay (`A_log`, `dt_bias`) + recurrence order |

### 1.3 Implications for any per-layer modification

Anything that lives in the standard-attention API (additive logit bias, RoPE,
explicit pair-wise scores) only fires on the **8 full-attention layers**. The
24 linear-attention layers will silently ignore parameters they don't have a
slot for in their forward signature — `**kwargs` swallows them.

Concrete consequences this codebase has hit:

1. **`SpatialAttentionBias` (additive logit bias on V×V)** — only meaningful at
   full-attention layers. Wrapping a DeltaNet layer creates `bias_module`
   parameters that never receive gradient → DDP raises
   `find_unused_parameters` failure. **Fix:** `patch_attention_layers_spatial`
   now skips layers with `layer_type == "linear_attention"`. See
   [spatial_attention.md §10](spatial_attention.md#10-open-questions--future-work).

2. **3D / 4D M-RoPE variants** (4D-cartesian / decoupled / polar in
   `train_correspondence.py`, etc.) — same coverage. Position encoding
   modifications only affect 8 / 32 layers; the remaining 24 layers run with
   the convolutional + recurrent positional structure unchanged.

3. **PEFT / LoRA target_modules** — `q_proj/k_proj/v_proj/o_proj` only exist
   on full-attention layers; `gate_proj/up_proj/down_proj` exist on every
   layer's MLP. So a typical LoRA config like
   `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
   actually attaches **7 LoRA pairs to each of 8 full-attn layers** and
   **3 LoRA pairs to each of 24 linear-attn layers** — not uniform across the
   stack. PEFT silently picks the modules that match the name; no error.

### 1.4 Information still propagates through all 32 layers

Even though only 8 layers directly process RoPE / spatial bias, hidden states
flow through every layer serially:

```
Layer 0  (L)  hidden ← linear_attn(hidden)              ← no RoPE / bias here
Layer 1  (L)  hidden ← linear_attn(hidden)
Layer 2  (L)  hidden ← linear_attn(hidden)
Layer 3  (F)  hidden ← full_attn(hidden, +RoPE +bias)   ← ★ injection #1
Layer 4  (L)  hidden ← linear_attn(hidden_with_signal)  ← carries signal forward
...
Layer 31 (F)  hidden ← full_attn(hidden, +RoPE +bias)   ← ★ injection #8
```

So linear-attention layers are not "dead" w.r.t. the geometric / multimodal
positional signal — they just don't directly *consume* it; they receive
already-encoded hidden states from upstream full-attention layers and apply
their own convolutional + recurrent processing on top.

---

## 2. Vision side — `Qwen3_5VisionModel` and image input pipeline

Qwen3.5 uses **dynamic resolution**: any input image is admitted, the
processor independently rounds H and W to multiples of
`factor = patch_size × merge_size = 14 × 2 = 28`, bounded by
`[min_pixels, max_pixels]`, then bicubic-resizes (no crop, no padding). The
number of vision tokens varies per image based on the resulting (H, W).

### 2.1 Vision-side configuration constants

- `config.vision_config.spatial_merge_size` controls the patch merge ratio
  (typically 2). Used everywhere in this codebase via
  `int(json.load(...)["vision_config"].get("spatial_merge_size", 2))`.
- The vision encoder uses its own `Qwen3_5VisionAttention` with separate
  `apply_rotary_pos_emb_vision` (different from text RoPE — see §4).
- Output gets re-embedded at `<|image_pad|>` positions in the input_ids
  (`mm_token_type_ids == 1` cells), driven by §3's modality contract.

### 2.2 The `smart_resize` formula

The image processor (Qwen2-VL family, used by Qwen3.5) computes the resize
target via:

```python
def smart_resize(height, width, factor=28,
                 min_pixels=56*56, max_pixels=14*14*4*1280):  # ≈ [3136, 1003520]
    h_bar = round(height / factor) * factor
    w_bar = round(width  / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor(height / beta / factor) * factor)
        w_bar = max(factor, floor(width  / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = sqrt(min_pixels / (height * width))
        h_bar = ceil(height * beta / factor) * factor
        w_bar = ceil(width  * beta / factor) * factor
    return h_bar, w_bar
```

Defaults:

| Constant | Value | Meaning |
|---|---|---|
| `patch_size` | 14 | DINOv2-style ViT patch |
| `merge_size` (`sms`) | 2 | Spatial merging into LLM tokens |
| `factor` | **28** | `= patch_size × merge_size` — the rounding granularity |
| `min_pixels` | 3,136 (= 56²) | Don't go below ~56×56 |
| `max_pixels` | **1,003,520** (= 14² × 4 × 1280) | Cap area, scaling down preserves AR via `beta` |

> Round semantics: Python's `round()` uses **banker's rounding** (round-half-to-even).
> E.g. `round(518/28) = round(18.5) = 18 → 504`. Inputs whose dimensions land near
> `.5` boundaries get tiny aspect-ratio drift (~1-2%).

### 2.3 What the processor returns

Per image:

```python
proc_out["pixel_values"]      # (T*H_bar*W_bar/14² , 3, 14, 14)   flattened patch tokens
proc_out["image_grid_thw"]    # (T, H_bar/14, W_bar/14)           per image
```

Vision-token count per image, before/after spatial merge:

| Stage | Token count |
|---|---|
| Pre-merge (raw patches, what ViT outputs) | `H_bar/14 × W_bar/14` |
| Post-merge (`sms=2`, what the LLM consumes) | `H_bar/28 × W_bar/28` |

The LLM patch grid is therefore `(image_grid_thw[1] // sms, image_grid_thw[2] // sms)`.

### 2.4 Worked examples

**480×640 input** (4:3 landscape, MindCube parquet):

```python
smart_resize(H=480, W=640, factor=28, max_pixels=1003520)
  h_bar = round(480/28)*28 = 17*28 = 476
  w_bar = round(640/28)*28 = 23*28 = 644
  area = 476*644 = 306,544  < 1,003,520 ✓ no scaling needed
  → (476, 644)
```

→ LLM patch grid `(17, 23)` = 391 patches. AR drift: 1.333 → 1.353 (+1.5%).

**1024×768 input** (4:3 landscape, VST si_distance):

```python
smart_resize(H=768, W=1024)
  h_bar = round(768/28)*28 = 27*28 = 756
  w_bar = round(1024/28)*28 = 37*28 = 1036
  area = 756*1036 = 783,216  < 1,003,520 ✓
  → (756, 1036)
```

→ LLM patch grid `(27, 37)` = 999 patches. AR drift: 1.333 → 1.370 (+2.8%).

**1920×1440 input** (4:3 landscape, VST si_depth_comparison high-res):

```python
smart_resize(H=1440, W=1920)
  h_bar = round(1440/28)*28 = 51*28 = 1428
  w_bar = round(1920/28)*28 = 69*28 = 1932
  area = 1428*1932 = 2,758,896  > 1,003,520 ✗ → scale down
  beta = sqrt(2,764,800 / 1,003,520) = 1.660
  h_bar = floor(1440/1.660/28)*28 = 30*28 = 840
  w_bar = floor(1920/1.660/28)*28 = 41*28 = 1148
  → (840, 1148)
```

→ LLM patch grid `(30, 41)` = 1,230 patches. Effective downscale ×0.6 (Qwen will
not let any input exceed ~1M pixels).

### 2.5 No crop, no pad — bicubic resize only

Aspect ratio is **not** preserved exactly (H and W round independently to 28-multiples,
introducing ~1-2% non-uniform stretch on most images), but the **full field of view
is preserved** — no scene content is cropped or padded.

For 4:3 (1.333) inputs, common landings:

| Input (W × H) | smart_resize (W × H) | New AR | AR drift |
|---|---|---|---|
| 640 × 480 | 644 × 476 | 1.353 | +1.5% |
| 1024 × 768 | 1036 × 756 | 1.370 | +2.8% |
| 1920 × 1440 | 1148 × 840 | 1.367 | +2.5% |

This drift is small but non-zero. For tasks where pixel-level alignment with another
representation that **does** preserve AR exactly (e.g. `pts3d.npy` from a fixed-resolution
reconstruction model — see [mapanything.md](mapanything.md)), the discrepancy needs
explicit handling at the dataloader.

### 2.6 Practical implications for dataloaders

The processor handles smart_resize internally:

```python
proc_out = self.processor(
    text=[prompt_text],
    images=images,            # any list of PIL Image — no manual resize required
    return_tensors="pt",
    padding=False,
)
thw_per_image = proc_out["image_grid_thw"]   # (N, 3) tensor — actual post-resize grid
```

You **do not** manually round to 28-multiples or apply your own resize before the
processor — doing so doubles the work and may introduce additional artifacts. The
dataloader can read images from any source (parquet bytes, on-disk PNG/JPG of
arbitrary size) and feed them directly. `image_grid_thw` reports the post-resize
patch grid the LLM's vision tokens describe.

### 2.7 Tuning `min_pixels` / `max_pixels`

The defaults `[3136, 1003520]` cap GPU memory at a reasonable size while keeping
enough resolution for typical VLM tasks. Three scenarios where overriding helps:

| Scenario | Adjustment |
|---|---|
| Low-VRAM training, high token-count regret | Lower `max_pixels` to e.g. 256² × 28² = 200,704 |
| Documents / OCR / text-heavy inputs | Raise `max_pixels` to keep small text legible |
| Very tiny thumbnails (< 56²) | Raise `min_pixels` so they get upsampled, not collapsed to a single token |

Pass via `processor(..., min_pixels=..., max_pixels=...)` or set on
`processor.image_processor` once at init.

### 2.8 Comparison with fixed-resolution preprocessors

Fixed-resolution preprocessors (CLIP at 224², our reconstruction models — see
[mapanything.md](mapanything.md)) require a single shape and either crop or pad to fit.
Qwen3.5 sidesteps this:

| Aspect | Qwen3.5 dynamic | Fixed-resolution (CLIP / MapAnything / VGGT) |
|---|---|---|
| Output shape | Variable per image | Single shape |
| Aspect-ratio preservation | Not exact (~1-2% drift) | Either exact (with crop) or distorted (with stretch) |
| Field of view | **Always full** (no crop, no pad) | Cropped or padded |
| Token count | Variable per image | Constant |
| Caller responsibility | None — handled internally | Must manually invoke preprocessor |

For VLM tasks, "always full FOV with small AR drift" is generally preferred: no
scene content is lost (even text on the edge stays visible) and the model's vision
tokens adapt to any input shape. The cost — patch-grid variability — is exactly
what `image_grid_thw` exposes for downstream consumers.

---

## 3. `mm_token_type_ids` — modality labeling contract

A `(B, L)` int tensor that labels every token in the sequence with which
**modality** it represents. **AutoProcessor emits it**; the **caller must
forward it down to the model** — Qwen3.5 does **not** internally derive it
from `input_ids`.

### 3.1 Value semantics

| Value | Meaning |
|---|---|
| `0` | text token (regular text + special tokens like `<|im_start|>`, `<|vision_start|>`, `<|vision_end|>`) |
| `1` | image patch token (`<|image_pad|>` placeholder, replaced by ViT output later) |
| `2` | video frame token (`<|video_pad|>`, not used in this project) |

### 3.2 Concrete shape (one MindCube training sample, 4 images)

```
sequence length L = 877
mm_token_type_ids unique values = {0: 109, 1: 768}
                                  text   image patches (4 imgs × 192 patches = 768)

structure:
  pos    0   <|im_start|>     mm=0   ← text prelude
  pos    4   <|image_pad|>    mm=1   ← image 1 (192 patches)
  pos  196   <|vision_end|>   mm=0   ← back to text
  pos  198   <|image_pad|>    mm=1   ← image 2
  pos  390   <|vision_end|>   mm=0
  pos  392   <|image_pad|>    mm=1   ← image 3
  pos  584   <|vision_end|>   mm=0
  pos  586   <|image_pad|>    mm=1   ← image 4
  pos  778   <|vision_end|>   mm=0   ← text question + answer suffix
```

### 3.3 Where it gets consumed inside Qwen3.5

Four downstream consumers — each one **silently degrades** if `mm_token_type_ids`
arrives as `None`:

| Consumer | Failure mode when mm_token_type_ids=None |
|---|---|
| `get_rope_index` / `compute_3d_position_ids` | 3D M-RoPE axis dispatch fails — `can_compute_mrope = False` ([modeling_qwen3_5.py:1683](file)). Falls back to 1D positions, image patches lose their (h, w) spatial encoding. |
| `get_placeholder_mask` + `masked_scatter` (in `Qwen3_5Model.forward`) | Image embedding scatter target positions can't be located. Recent transformers versions handle this via `image_mask` from input_ids, but historically depended on this tensor. |
| `compute_3d_position_ids` (full position_ids construction) | Uses `mm_token_type_ids` together with `image_grid_thw` to assemble the (text, t, h, w) position_ids. |
| **Our `SpatialAttnVanillaModel.forward`** | `vision_mask = (mm_token_type_ids != 0)`. Without it, `_spatial_cache` stays None → wrappers bypass → bias_module dead → DDP `find_unused_parameters` failure. |

### 3.4 Why externally provided (not internally derived)

Qwen3.5 deliberately doesn't reconstruct `mm_token_type_ids` from `input_ids`
inside the model. Reasons:

- **Insufficient signal** — `input_ids` alone tells you where `<|image_pad|>`
  tokens are, but per-image patch counts depend on `image_grid_thw` (which
  itself depends on input image resolution). The processor knows this when
  it builds the sequence; the model would have to re-derive it.
- **Efficiency** — processor already has all sequence-level metadata at
  hand; emitting it once is cheaper than recomputing.
- **Decoupling** — keeps the model forward signature agnostic to tokenizer-
  vocab specifics.

The cost: **the caller must remember to pass it**. Forgetting it (which we
did in [train_atten.py](../../train_atten.py) for several iterations) means
every downstream consumer silently goes "modality-blind" without a clear
error — until DDP eventually catches the dead bias_module params.

### 3.5 Distinction from `attention_mask`

Both are `(B, L)` int tensors — easy to confuse.

|  | `attention_mask` | `mm_token_type_ids` |
|---|---|---|
| Question it answers | Is this token **valid or padding**? | What **modality** is this token? |
| Value `0` | padding → ignore | text → regular processing |
| Value `1` | real token → attend | image patch → ViT-replace |
| Value `2`+ | n/a | video / other modalities |
| Source | tokenizer (text padding) | processor (multimodal sequence assembly) |
| Consumer | causal mask construction, padding masking | M-RoPE axis dispatch, image scatter, our vision_mask |

### 3.6 Practical contract for adding new training scripts

If you write a new `train_*.py` that uses Qwen3.5-VL with image inputs:

1. The dataset's `__getitem__` should already emit `mm_token_type_ids` via
   the `**proc_out` unpack (verified for `MindCube_Train_Dataset`).
2. The training loop **must** extract and forward it:
   ```python
   mm_token_type_ids = batch.get("mm_token_type_ids")
   if mm_token_type_ids is not None:
       mm_token_type_ids = mm_token_type_ids.to(device)
   model(input_ids=..., mm_token_type_ids=mm_token_type_ids, ...)
   ```
3. Any custom wrapper between caller and `Qwen3_5ForConditionalGeneration`
   (like `AnswerOnlyModel`) must accept it as a kwarg and pass through.
4. Verify with one sample: `mm_token_type_ids[0].unique()` should contain
   both `0` (text) and `1` (image) for any multi-modal sample.

---

## 4. M-RoPE (multi-axis rotary position embedding) for text

```
mrope_section = [11, 11, 10]                # text_config
position_ids shape (incl. text axis): [4, B, L]
                                       ↑
                                       (text_seq, t, h, w)  — 4 axes
```

The first axis (`text_position_ids = position_ids[0]`) is used by
`create_causal_mask`. The remaining three (`(t, h, w)`) are passed to
`rotary_emb` and produce the `(cos, sin)` consumed by full-attention layers.

This is the lever for any custom positional scheme: replacing `position_ids`
on the (t, h, w) axes lets you inject arbitrary 3-axis positional information
without touching attention weights — but again, **only the 8 full-attention
layers will see the change**.

The axis dispatch (which row of `position_ids` each token reads) is gated by
`mm_token_type_ids` — see §3 for the full contract. Without it,
`get_rope_index` falls back to 1D positions and the (h, w) image-patch
encoding is silently lost.

---

## 5. Things to add when discovered

Use this section as a TODO / future-finding log. When you discover a
non-obvious property of the base model that surprised the team, write it up
here with the verifying snippet and a brief "why this matters" line.

- [ ] How `Qwen3_5DynamicCache` handles the hybrid layer mix during generation
      (does it allocate per-layer KV slots only for full-attn? where does
      DeltaNet state live?)
- [ ] The exact tied-weights set after backbone swap — `tie_weights()` only
      re-binds `lm_head ↔ embed_tokens`, but is there anything else that needs
      manual re-tying after `spa.model = ...` replacement?
- [ ] Whether `Qwen3_5VisionAttention` is also part of a hybrid stack or pure
      transformer attention.
- [ ] `_update_linear_attn_mask` semantics — what shape does it produce and
      how does Gated DeltaNet consume it (we know it's NOT additive)?
