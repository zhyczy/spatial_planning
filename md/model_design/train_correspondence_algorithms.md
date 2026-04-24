# `train_correspondence.py` — Algorithm Overview

LoRA fine-tuning of Qwen3.5-VL on MindCube with **LM answer loss only**
(no coord head, no pose regression). The script supports five
mutually-exclusive position-embedding regimes plus one orthogonal modifier.

Cross-references:
- [`decouple_xyz_rope.md`](decouple_xyz_rope.md) — full math for the decoupled architecture (`--decouple`, `--polar`)
- [`polar_rope.md`](polar_rope.md) — `--polar` specifics (log-spherical transform, θ=1000 rationale, band utilization)

## 1. Mode Overview

| flag | Model class | Rotary layout | Data xyz | Prompt form |
|---|---|---|---|---|
| *(none)* | `SpaForConditionalGeneration` | 4D M-RoPE `[2,10,10,10]` | Cartesian `image_xyz` | images + question |
| `--vanilla` | `Qwen3_5ForConditionalGeneration` (stock) | Original 3D `[11,11,10]` | — (ignored) | images + question |
| `--decouple` | `SpaDecForConditionalGeneration` | Qwen `[11,11,10]` UNCHANGED + new XYZ RoPE (dims 64..129, θ=10000) | Cartesian `image_xyz` → XYZ RoPE | images + question |
| `--polar` | `SpaDecForConditionalGeneration` | Qwen `[11,11,10]` UNCHANGED + new XYZ RoPE (dims 64..129, **θ=1000**) | Cartesian `image_xyz` → converted to `(log r, θ, α)` inside XYZ RoPE | images + question |
| `--relative` | `SpaRelativeForConditionalGeneration` | 4D `[2,10,10,10]` + per-frame attention | `image_xyz_relative (N_frames,H,W,3)` | images + question |

Orthogonal modifier: `--interleave_vision` — changes band-to-axis layout
inside the 4D M-RoPE (only active for default / `--relative`; no effect on
`--vanilla` / `--decouple` / `--polar`).

## 2. Mode Details

### 2.1 Default (4D M-RoPE, Cartesian)

The baseline "add spatial xyz via 4D M-RoPE" design. mrope_section is
repurposed from Qwen's `[11,11,10]` (t,h,w) to `[2,10,10,10]` (t,x,y,z).

- **Config change**: `config.text_config.rope_scaling["mrope_section"] = [2, 10, 10, 10]`
- **Model**: `SpaForConditionalGeneration` (4D M-RoPE backbone)
- **Wrapper**: `AnswerOnlyModel(spa, use_xyz=True, polar=False)`
- **Dataset**: `MindCube_Train_Dataset` → returns `image_xyz` (per-patch Cartesian)
- **Position per token**:
  - Text at seq pos $p$: $(t, x, y, z) = (p, p, p, p)$
  - Image patch $(i, j)$ of image $k$: $(t, x, y, z) = (s_k,\ x_{\text{patch}},\ y_{\text{patch}},\ z_{\text{patch}})$
- **Rationale**: All 4 axes get dedicated band sections; model learns to
  encode sequential token position (via t) and 3D scene coords (via x, y, z)
  separately within the same 64 rotary dims.

### 2.2 `--vanilla` (Ablation: LoRA only)

Pure baseline to isolate LoRA's contribution from any spatial signal.

- **Config**: unchanged (`mrope_section = [11, 11, 10]`)
- **Model**: stock `Qwen3_5ForConditionalGeneration`
- **Wrapper**: `AnswerOnlyModel(spa, use_xyz=False)` — xyz not forwarded
- **Dataset**: `MindCube_Train_Dataset` (xyz computed but silently ignored)
- **Position per token**: Qwen original — text `(p, p, p)`, image patch `(s_k, s_k+i, s_k+j)`
- **Rationale**: Answers the question "does fine-tuning LoRA on MindCube
  help at all, without any 3D position information?"

### 2.3 `--polar` (decoupled XYZ RoPE + log-spherical input)

**`--polar` uses the same decoupled architecture as `--decouple`** (Qwen
original 3D M-RoPE preserved in rotary dims 0..63, new XYZ RoPE in
pass-through dims 64..129), but feeds **log-spherical** coords into the
XYZ RoPE instead of Cartesian, and uses a tighter `rope_theta`.

- **Config**: `mrope_section = [11, 11, 10]` (unchanged from Qwen; same as `--decouple`)
- **Model**: `SpaDecForConditionalGeneration` + `patch_attention_layers_dec`
- **Wrapper**: `AnswerOnlyModel(spa, use_xyz=True, polar=True, coord_scale=100)`
- **Dataset**: `MindCube_Train_Dataset` (Cartesian; conversion inside `SpaXYZRotaryEmbedding.forward`)
- **rope_theta swap**: after `from_pretrained` the script replaces
  `xyz_rotary_emb` with `SpaXYZRotaryEmbedding(rope_theta=1000.0)` (down
  from `--decouple`'s 10000, because polar's angles have narrower natural range)
- **Transformation** (applied per patch inside `SpaXYZRotaryEmbedding` when `polar=True`):

$$
\begin{aligned}
r &= \sqrt{x^2 + y^2 + z^2} \\
\log r &= \log(\max(r, 10^{-8})) \\
\theta &= \operatorname{atan2}(y, x) \in [-\pi, \pi]\text{ (azimuth)} \\
\alpha &= \operatorname{atan2}(\sqrt{x^2 + y^2},\ z) \in [0, \pi]\text{ (inclination)}
\end{aligned}
$$

Text tokens with $(x, y, z) = (0, 0, 0) \Rightarrow r = 0$ are special-cased to
`(log r, θ, α) = (0, 0, 0)` → identity rotation on dims 64..129.

- **Mutex**: `--polar` is mutually exclusive with `--decouple` (both imply
  the decouple architecture) and with `--vanilla` / `--relative`.
- **Rationale**: $\log r$ is scale-invariant — RoPE relative phase
  $\log r_i - \log r_j = \log(r_i / r_j)$ is unchanged by global scene
  rescaling $r \to k \cdot r$. This is expected to generalize across
  datasets with different scale conventions (MindCube ≈ meters, SAT ≈ meters,
  RoboSpatial ≈ tabletop). See [`polar_rope.md`](polar_rope.md) for the θ=1000
  rationale and per-band frequency analysis.

### 2.4 `--relative` (per-query-frame coords)

Every query token sees key tokens' 3D positions transformed into the query
token's own camera frame. Encodes true **relative** spatial relations.

- **Config**: `mrope_section = [2, 10, 10, 10]`
- **Model**: `SpaRelativeForConditionalGeneration`
- **Wrapper**: `AnswerRelativeModel(spa, polar=True)` (polar on by default for relative)
- **Dataset**: `MindCube_Train_Dataset_Relative` → returns `image_xyz_relative`
  of shape `(N_frames, llm_H, llm_W, 3)` per image
- **Coordinate transform** (computed in dataset):

$$
P^{\text{frame-}f}_i = R^{-1}_f \cdot (P^{\text{world}}_i - t_f)
$$

where $(R_f, t_f)$ is camera-to-world pose for frame $f$.

- **Attention modification**: `SpaRelativeAttentionWrapper` runs $N$
  attention passes (one per reference frame $f$), each with K positions
  in frame $f$'s camera coordinates. The output for query tokens from
  frame $f$ is taken from the $f$-th pass.
- **Constraint**: Incompatible with KV cache at prefill time (wrapper
  auto-falls-back to standard attention when `past_key_values is not None`).
- **Rationale**: "What's to the left of the chair" is inherently viewpoint-
  dependent; world-frame xyz can't disambiguate. With per-frame coords,
  relative spatial queries have a consistent answer per viewpoint.

### 2.5 `--decouple` (Decoupled XYZ RoPE, Cartesian input)

Keeps Qwen's original 3D M-RoPE **UNCHANGED** in the rotary 64 dims, and
adds a **new** XYZ RoPE in 66 of the 192 pass-through dims, consuming raw
Cartesian $(x, y, z)$.

- **Config**: unchanged (`mrope_section = [11, 11, 10]`)
- **Model**: `SpaDecForConditionalGeneration` (+ `patch_attention_layers_dec`)
- **Wrapper**: `AnswerOnlyModel(spa, use_xyz=True, polar=False, coord_scale=100)`
- **Dataset**: `MindCube_Train_Dataset` (Cartesian `image_xyz`)
- **Head-dim layout**:

```
dims   0..63  : Qwen 3D M-RoPE (UNCHANGED)
dims  64..129 : NEW XYZ RoPE (66d, 11 bands/axis, symmetric, θ_xyz = 10^4)
dims 130..255 : pass-through (unchanged)
```

- **Per-token xyz**: Text → $(0, 0, 0)$ → identity rotation; Image patch
  → mean valid-pixel xyz of the LLM patch.
- **Key invariant**: Text-text attention is **mathematically equivalent**
  to Qwen pretraining + an extra content similarity term.
- **Relationship to `--polar`**: `--polar` uses the same architecture but
  with log-spherical input and θ=1000. Mutually exclusive.
- **Full formulation**: See [`decouple_xyz_rope.md`](decouple_xyz_rope.md).

### 2.6 `--interleave_vision` (orthogonal modifier)

Changes the band-to-axis layout inside 4D M-RoPE.

**Sequential (default)**:
```
bands 0..1   → t
bands 2..11  → x   (all ≈ low-freq end for x)
bands 12..21 → y
bands 22..31 → z   (all ≈ high-freq end for z)
```

**Interleaved (`--interleave_vision`)**:
```
bands 0..1   → t   (high-freq end; text position critical)
bands 2..31  → x, y, z round-robin  (each axis spans full freq range)
```

- **Effect**: x, y, z each cover the full low-to-high freq spectrum.
  Without interleave, each axis lives in a sub-band → x/y/z have very
  different effective scales, possibly requiring per-axis `coord_scale`.
- **Applies to**: default, `--polar`, `--relative` (all 4D modes)
- **No effect on**: `--vanilla` (3D M-RoPE), `--decouple` (uses Qwen
  original rotary layout)

Implementation: `SpaTextRotaryEmbedding.apply_interleaved_mrope`, toggled
via `language_model.rotary_emb.visual_interleave = True`.

## 3. Mutual Compatibility

| | `--vanilla` | `--polar` | `--relative` | `--decouple` | `--interleave_vision` |
|---|:---:|:---:|:---:|:---:|:---:|
| `--vanilla` | — | ✗ | ✗ | ✗ | ✗ (no-op) |
| `--polar` | | — | ✗ | ✗ | ✗ (no-op) |
| `--relative` | | | — | ✗ | ✓ |
| `--decouple` | | | | — | ✗ (no-op) |
| `--interleave_vision` | | | | | — |

Raised `ValueError`:
- `--decouple` with any of `--vanilla` / `--relative` / `--polar`
- `--polar` with any of `--vanilla` / `--relative` / `--decouple`
  (polar already implies the decouple architecture)

`--decouple` and `--polar` keep Qwen's original `mrope_section = [11,11,10]`
intact, so `--interleave_vision` is a no-op in those modes (warning logged).

## 4. Shared Training Infrastructure

### 4.1 Backbone freezing

All modes: ViT frozen (`freeze_vision=True` default); only LoRA adapters
+ (optionally) vision encoder (with `--train_vision`) are trainable.

### 4.2 LoRA config

All modes apply the same LoRA:
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Rank `r = 16` (default; `--lora_rank` flag)
- `lora_alpha = 2r`, `lora_dropout = 0.05`, `bias = "none"`
- TaskType: `CAUSAL_LM`

### 4.3 Gradient checkpointing

Enabled on the language model via `gradient_checkpointing_enable` with
`use_reentrant=False` for all modes (trades ~20% speed for ~60% memory).

### 4.4 Training loss

LM answer cross-entropy only:

$$\mathcal{L} = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log p(y_t \mid y_{<t})$$

where $\mathcal{A}$ is the answer-token positions (rest masked with `-100`).
`AnswerOnlyModel` / `AnswerRelativeModel` extract logits at valid answer
positions and compute this loss.

### 4.5 Optimization

- AdamW, `lr = 2e-4` (default), `weight_decay = 0.01`
- Cosine annealing over `epochs × len(loader) / grad_accum` steps
- Gradient clipping: `max_norm = 1.0`
- Batch size per rank: **1** (gradient accumulation for effective batch size)
- DDP: `find_unused_parameters=False` (all trainable LoRA params active each forward)

### 4.6 Coord scale convention

All modes use `coord_scale = 100` (cm-equivalent units). For `--decouple`,
this scales the Cartesian xyz before the new XYZ RoPE
(→ wavelength range 0.063m .. 272m per axis).

### 4.7 Dataset coordinate-frame convention

`pts3d.npy` is stored in the **first camera's frame** (`view_0000` camera
pose ≈ identity). So for `--relative` mode, `image_xyz_relative[k][0]` is
identical to the world-frame `image_xyz[k]`.

## 5. Eval Protocol (Periodic)

Every `eval_steps` training steps, on held-out test sets
(MindCube tinybench + SpinBench by default):

- **Data**: `Eval_Dataset_Coord` (loads pts3d + optionally camera poses if
  `relative=True`) → batches include `image_xyz` (and `image_xyz_relative` in
  relative mode)
- **Forward**: matches training input distribution per mode:

| Mode | eval kwargs to `_spa(...)` |
|---|---|
| `--vanilla` | `input_ids, attention_mask, pixel_values, image_grid_thw` |
| default / `--decouple` | `+ image_xyz` (Cartesian) |
| `--polar` | `+ image_xyz, polar=True` (xyz → log-spherical inside `SpaXYZRotaryEmbedding`) |
| `--relative` | `+ image_xyz_relative` (per-frame coords) |

- **Metrics**:
  - `lm_loss`: cross-entropy over the answer tokens (ignore_index=-100)
  - `acc`: top-1 prediction accuracy on the **first answer token** (matches
    the convention in `coordinate_llm.py`). For multi-choice QA this is the
    letter pick (A/B/C/D).
- **Aggregation**: sum across ranks via `dist.all_reduce`, then `/ total_count`.
- **Logging**: `log.info` + `wandb.log({f"eval/{ds_name}_lm_loss": ..., f"eval/{ds_name}_acc": ...}, step=global_step)`.

### 5.1 Eval-train consistency (recent fix)

Previously `Eval_Dataset` didn't load pts3d at all, so all xyz-using modes
fell back to zeros at eval time (image patches saw `xyz=(0,0,0)`, same as
text tokens). This was fixed by switching to `Eval_Dataset_Coord` which
loads per-patch xyz (and per-frame transforms when `relative=True`), and by
dispatching the correct forward kwargs per mode.

## 6. Script Entry Point

`scripts/train_correspondence.sh` wraps `torchrun --nproc_per_node N
train_correspondence.py` with auto-detected GPU count, parses the flags
above, and suffixes `RUN_NAME`:

```
correspondence_mindcube[_relative][_polar][_interleave][_decouple][_vanilla]
```

Examples:
```bash
bash scripts/train_correspondence.sh                        # default 4D M-RoPE, all GPUs
bash scripts/train_correspondence.sh 2 --polar              # 2 GPUs, log-spherical
bash scripts/train_correspondence.sh 2 --relative           # 2 GPUs, per-frame
bash scripts/train_correspondence.sh 2 --decouple           # 2 GPUs, decoupled
bash scripts/train_correspondence.sh 2 --vanilla            # 2 GPUs, LoRA-only baseline
bash scripts/train_correspondence.sh 1 --max_samples 6      # 1 GPU, 6-sample smoke test
```

## 7. File Map

| Concern | File |
|---|---|
| Training entry point | `train_correspondence.py` |
| Shell launcher | `scripts/train_correspondence.sh` |
| Model: 4D M-RoPE, polar, interleave | `src/models/spa_emb.py` |
| Model: per-frame relative | `src/models/spa_emb_relative.py` |
| Model: decoupled XYZ RoPE | `src/models/spa_emb_dec.py` |
| Wrapper (LM loss only) | `src/models/answer_llm.py` |
| Dataset: training | `src/dataset/train_dataset.py` |
| Dataset: eval | `src/dataset/eval_dataset.py` |
| Decouple math details | [`decouple_xyz_rope.md`](decouple_xyz_rope.md) |
| Polar specifics (log-r, θ=1000) | [`polar_rope.md`](polar_rope.md) |
