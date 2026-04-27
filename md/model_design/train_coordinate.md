# `train_coordinate.py`: Dual-Supervision Fine-Tuning (LM + Per-Patch 3D)

## 1. Purpose

`train_coordinate.py` adds an auxiliary **per-patch 3D coordinate regression loss** on top of the standard LM answer loss. The goal: push the LM's last-layer hidden states to carry explicit xyz information, hoping this regularizes spatial reasoning and improves QA accuracy.

It extends `train_correspondence.py` (LM-only) by wrapping the backbone in a `CoordinateModel` that also runs a **coord head** reading vision-token hidden states.

Relation to other scripts:

| Script | Supervision | Extra head | Position embedding |
|---|---|---|---|
| `train_correspondence.py` | LM only | none | `--decouple` / `--polar` / `--vanilla` / default 4D |
| `train_coordinate.py` | LM + per-patch xyz regression | `DepthPredictionTransformer` | same set (`--decouple` / `--polar` / default 4D) |

## 2. Loss

```
total_loss = answer_weight · lm_loss + coord_weight · coord_loss
```

| Term | Source | Target |
|---|---|---|
| `lm_loss` | logits at answer tokens | causal CE on `labels` |
| `coord_loss` | `coord_head(hidden[vis_pos])` → per sub-pixel (x,y,z) | L1 against `image_xyz_hires` (dataset-provided) |

In `--polar` mode `coord_loss` is a **weighted** L1 `[1, 1/π, 1/π]` on channels `(log r, θ, α)` so a π-radian angle error matches a 1-unit log-r error; otherwise plain L1.

## 3. Architecture

```
 ┌─────────────────────────────── CoordinateModel ────────────────────────────────┐
 │                                                                                │
 │   SpaForConditionalGeneration  /  SpaDecForConditionalGeneration  + LoRA        │
 │   (selected by --decouple / --polar flags; position-embedding path docs:        │
 │    decouple_xyz_rope.md, polar_rope.md)                                         │
 │                                                                                │
 │   ┌──────────────────────────── spa_model ───────────────────────────────┐     │
 │   │   Vision ViT  →  LLM (32 Qwen3.5-4B layers + M-RoPE + [XYZ RoPE])    │     │
 │   │     │                                                                 │     │
 │   │     │     ┌── lm_head pre-hook (skip_layers=[-1])                    │     │
 │   │     │     │     captures post-norm last hidden state                 │     │
 │   │     │     │                                                           │     │
 │   │     │     └── forward_hook on layers[-(k-1)] (skip_layers=[-k], k≥2) │     │
 │   │     │           captures pre-norm output of layer -k                 │     │
 │   │     ▼                                                                 │     │
 │   │   logits  →  lm_loss  (cross-entropy on answer tokens only)          │     │
 │   └──────────────────────────────────────────────────────────────────────┘     │
 │                                                                                │
 │   hidden_coord = _lm_head_input[vis_pos per image]                              │
 │         │                                                                       │
 │         ▼                                                                       │
 │   DepthPredictionTransformer (input_proj → 2-layer transformer encoder          │
 │                                → output_proj → PixelShuffle)                    │
 │         │                                                                       │
 │         ▼                                                                       │
 │   predicted xyz at sub-pixel grid (H·upscale × W·upscale × 3)                   │
 │         │                                                                       │
 │         ▼                                                                       │
 │   coord_loss  =  L1( pred, image_xyz_hires )     (weighted if --polar)         │
 └────────────────────────────────────────────────────────────────────────────────┘
```

### Coord head (`DepthPredictionTransformer`)

```
hidden (h·w, 2560)
  → input_proj Linear(2560 → 512)
  → + 2D sinusoidal PE (regenerated for any h, w)
  → 2× TransformerEncoderLayer (pre-norm, 8 heads, FF=2048, d_model=512)
  → output_proj Linear(512 → 3 · upscale²)
  → view (1, h, w, 3·up²) → permute → PixelShuffle(upscale)
  → (h·up, w·up, 3)
```

No camera-conditioning token (`cam_dim=0` for this script). ~4M params.

### Hidden state capture (no `output_hidden_states=True`)

`CoordinateModel.__init__` registers **one** hook at build time:

- `skip_layers=[-1]` → `lm_head.register_forward_pre_hook` on the LM input (post-norm, last hidden state)
- `skip_layers=[-k]` (k≥2) → `layers[-(k-1)].register_forward_hook` on that decoder layer's output (pre-norm)

Both paths store the captured tensor in `self._lm_head_input`; `forward()` just reads it after running `spa_model(..., output_hidden_states=False)`.

**Why a hook, not `output_hidden_states=True`?**
`SpaDecTextModel` + gradient checkpointing produce NaN loss when `output_hidden_states=True` (empirically confirmed). The hook path avoids that code path entirely and also saves activation memory.

## 4. Flags

### Position-embedding selection (mutually exclusive; see linked design docs)

| Flag | Effect | Link |
|---|---|---|
| *(default)* | 4D M-RoPE: mrope_section → `[2,10,10,10]` | — |
| `--decouple` | Cartesian XYZ RoPE in pass-through 66 dims, θ=10000 | [decouple_xyz_rope.md](decouple_xyz_rope.md) |
| `--polar` | log-spherical XYZ RoPE in pass-through 66 dims, θ=1000 | [polar_rope.md](polar_rope.md) |

### Coord-head controls

| Flag | Default | Meaning |
|---|---|---|
| `--coord_weight` | 1.0 | Scalar on `coord_loss` in the combined objective |
| `--answer_weight` | 1.0 | Scalar on `lm_loss` |
| `--coord_upscale` | 4 | PixelShuffle factor; each patch predicts up² sub-pixel (x,y,z) |
| `--skip_layers` | `[-8, -4, -1]` (CLI accepts `+`); only `[0]` used | Which LLM layer to probe for coord regression |
| `--coord_scale` | 100.0 | Multiplier on xyz **before** it enters the position embedding (cm-equivalent when xyz is in meters) |

### Layout / training

| Flag | Default | |
|---|---|---|
| `--interleave_vision` | off | t at high-freq end, x/y/z round-robin (4D M-RoPE only; no effect on decouple/polar) |
| `--lora_rank` | 16 | |
| `--max_images` | 4 | |
| `--grad_accum` | 8 | |
| `--epochs` | 3 (script sets 6) | |
| `--lr` | 2e-4 | |
| `--train_vision` | off | Unfreeze ViT |

Run-name suffixes stamped by `scripts/train_coordinate.sh` (only for non-default values):
`_polar`, `_decouple`, `_r{rank}`, `_cw{w}`, `_sl{k}`.

## 5. Data flow — train and eval parity

Both train and eval go through `CoordinateModel.forward` with an identical signature:

```python
_, loss, loss_dict = model(
    input_ids, attention_mask, pixel_values, image_grid_thw,
    image_xyz,            # (list of (llm_H_k, llm_W_k, 3)) — Cartesian, first-cam frame
    image_xyz_hires,      # (list of (H_hires, W_hires, 3))  — Cartesian or polar (see below)
    coord_scale,          # scalar, default 100.0
    labels,
)
```

### Dataset choice

|  | Train | Eval (both mindcube & spinbench) |
|---|---|---|
| Default | `MindCube_Train_Dataset_Coord` | `Eval_Dataset_Coord` |
| `--polar` | `MindCube_Train_Dataset_Coord_Polar` | `Eval_Dataset_Coord` + manual `xyz_to_polar(t_xyz_hires)` in loop |

`image_xyz` stays **Cartesian** in both modes (the RoPE converts internally when `polar=True`).
`image_xyz_hires` is **log-spherical** when `--polar`, so the coord head's regression target matches the `[1, 1/π, 1/π]` weighting.

### Evaluation metric

Per-dataset, wandb shows:
```
eval/{ds}_lm_loss
eval/{ds}_coord_loss
eval/{ds}_acc           ← top-1 on first answer token
```

## 6. Key design decisions and lessons

### 6.1 The coord head is helpful only with the right probe layer

From our recent ablation (log in `md/findings/...` and wandb `coord_mindcube_*` runs):

| Config | mindcube acc @ step ~950 | spinbench acc |
|---|---|---|
| `corr_decouple` (no coord head) | ~0.85 | ~0.57 |
| `coord_decouple`, `sl=-1`, `cw=1.0`, `r=16` (original) | ~0.83 | ~0.60 |
| `coord_decouple`, `sl=-1`, `cw=0.1`, `r=32` | 0.82 | 0.61 |
| **`coord_decouple`, `sl=-8`, `cw=0.1`, `r=16`** | **0.95** | **0.64** |

Takeaway: coord-head **at the last layer** pollutes the answer-oriented representation and hurts mindcube acc. Probing 8 layers earlier (`--skip_layers -8`) releases the last-layer capacity for LM answer prediction while still giving the LM a geometric regularization signal from middle layers. `lora_rank` alone doesn't rescue the configuration; **probe position** does.

The mechanism (hypothesis): in a decoupled backbone the XYZ RoPE already encodes geometry in attention phases; demanding that **content dims** at the last layer also regress xyz duplicates the signal and competes with the LM head for the same content.

### 6.2 Gradient checkpointing + `output_hidden_states=True` is broken on `SpaDecTextModel`

Triggered persistent NaN loss from step 1 when `--skip_layers -8` was naïvely implemented via `outputs.hidden_states[-8]`. Solution: switch to hook-based capture for **all** single-layer probes; never set `output_hidden_states=True` in this script.

### 6.3 No per-axis `coord_scale`

Earlier drafts had `--coord_scale_xyz SX SY SZ`. Removed — the XYZ RoPE uses a **symmetric** per-axis spectrum (same `inv_freq_axis` for x / y / z), so per-axis scales have no physical justification once you move to `--decouple` / `--polar`. A single scalar is sufficient.

## 7. Checkpoint layout

```
{output_dir}/
  step_{global_step}/                 # every --save_steps
      adapter_model.safetensors       # LoRA adapter from peft.save_pretrained
      adapter_config.json
      coord_head.pt                   # DepthPredictionTransformer state_dict
      …tokenizer files
  step_{last}_final/                  # end of training
  train.log                           # rank-0 aggregate log
  train_rank{k}.log                   # per-rank log
  wandb/                              # wandb run artifacts
```

## 8. Typical invocations

```bash
# Best known config (as of 2026-04-24): decouple + cw 0.1 + sl -8
bash scripts/train_coordinate.sh 6 --decouple --coord_weight 0.1 --skip_layers -8

# Log-spherical variant
bash scripts/train_coordinate.sh 6 --polar --coord_weight 0.1 --skip_layers -8

# Sanity / smoke test
bash scripts/train_coordinate.sh 1 --decouple --max_samples 6
```

## 9. Related docs

- Position-embedding path: [decouple_xyz_rope.md](decouple_xyz_rope.md), [polar_rope.md](polar_rope.md)
- Motivation for moving beyond coord regression: [../discussion/learning_real_3d_geometry.md](../discussion/learning_real_3d_geometry.md) — argues the coord head is **vision-driven** (it cheats), and proposes multi-view contrastive / xyz-match as shortcut-resistant alternatives.
