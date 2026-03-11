# Image Generation — `image_generation.py`

Uses **FLUX2-klein-4B** (`Flux2KleinPipeline`) to generate one image per instruction produced by a planning model. Samples are sharded evenly across all available GPUs; each GPU worker loads its own copy of the pipeline.

## Prerequisites

```bash
conda activate SPR
# diffusers ≥ 0.37.0.dev0 required for Flux2KleinPipeline
pip install git+https://github.com/huggingface/diffusers.git
```

Checkpoint must exist at `checkpoints/flux2-klein-4B` (or pass a custom path via `--flux_ckpt`).

---

## Input

| Path | Description |
|------|-------------|
| `predicted_instructions/{planning_model}/results.jsonl` | One JSON object per line. Each record must contain `id`, `dataset`, `image_paths`, and `instructions`. |
| `datasets/evaluation/.../images/` | Source conditioning images referenced in `image_paths`. |

---

## Output Layout

```
generated_images/
  {dataset}/
    {planning_model}/
      {generation_model}/       ← e.g. flux2-klein-4B
        {question_id}/
          img_0.png
          img_1.png
          ...
        stats.json              ← per-run statistics
        run.log                 ← copy of the full run log
  run.log                       ← global log (all datasets/models)
```

### `stats.json` structure

```json
{
  "summary": {
    "number of 0 generation": 12,
    "ratio of 0 generation":  0.012,
    "number of 1 generation": 988,
    "ratio of 1 generation":  0.988
  },
  "0 generation": [2, 17, ...],
  "1 generation": [0, 1, 3, ...]
}
```

Keys are automatically created for every distinct image count (0, 1, 2, …).

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--planning_model NAME` | — | Process a single planning model (folder name under `predicted_instructions/`). **Mutually exclusive** with `--all_planning_models`. |
| `--all_planning_models` | — | Process every folder found under `predicted_instructions/`. |
| `--flux_ckpt PATH` | `checkpoints/flux2-klein-4B` | Path to the Flux2Klein checkpoint directory. |
| `--predicted_instructions_root PATH` | `predicted_instructions` | Root folder that contains planning-model subfolders. |
| `--output_root PATH` | `generated_images` | Root folder for all outputs. |
| `--num_inference_steps N` | `28` | Number of diffusion denoising steps. |
| `--height H` | model default | Output image height in pixels. |
| `--width W` | model default | Output image width in pixels. |
| `--num_gpus N` | `-1` (all) | Number of GPUs to use. Respects `CUDA_VISIBLE_DEVICES`. |
| `--no_skip_existing` | skip | Re-generate images that already exist on disk. |
| `--max_samples N` | `-1` (all) | Truncate to first N samples — useful for quick testing. |

---

## Usage Examples

### Quick test (1 GPU, 4 steps, 6 samples)

```bash
cd spatial_planning
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n SPR python image_generation.py \
  --planning_model Qwen3-VL-4B-Instruct \
  --max_samples 6 \
  --num_gpus 1 \
  --num_inference_steps 4
```

### Full run — single planning model (all GPUs)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n SPR bash scripts/run_image_generation.sh Qwen3-VL-4B-Instruct
```

### Full run — all planning models

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n SPR bash scripts/run_image_generation.sh all
```

### Specific GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n SPR bash scripts/run_image_generation.sh Qwen3-VL-4B-Instruct
```

### Re-generate everything (ignore existing outputs)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n SPR bash scripts/run_image_generation.sh Qwen3-VL-4B-Instruct --no_skip_existing
```

### Custom checkpoint

```bash
conda run -n SPR python image_generation.py \
  --planning_model Qwen3-VL-4B-Instruct \
  --flux_ckpt /path/to/other/checkpoint
```

---

## Notes

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is recommended when running multiple GPU workers simultaneously to avoid memory fragmentation OOM errors.
- Questions with an empty `instructions` list produce an empty output folder and are reported under `"0 generation"` in `stats.json`.
- Image paths in `results.jsonl` are automatically remapped from `dataset/` → `datasets/` if the original path does not exist.
- `stats.json` and `run.log` are written **after** all GPU workers finish, so they always reflect the complete run.
