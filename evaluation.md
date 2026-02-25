# evaluation.py

Dual-method reasoning evaluation on **MMSIBench** using a QwenVL model.

For every question, runs inference with **two methods in a single pass** and compares their results:

- **Baseline** — original dataset images + question only.
- **Augmented** — original dataset images **plus** generated images from `--gen_dir` + question. If no generated images exist for a sample (empty or missing sub-folder), the augmented method falls back to the same input as baseline.

Both methods share the same loaded model instance per GPU, so the model is loaded only once per GPU.

---

## How it works

1. Loads `datasets/evaluation/MMSIBench/data/test_data_final.json`.
2. For each sample, looks up `{gen_dir}/{sample_id}/` for generated images (`img_*.png`).
3. Runs **Method A (baseline)**: original images → model → predicted letter.
4. Runs **Method B (augmented)**: original + generated images → model → predicted letter.
5. Shards the dataset evenly across all visible GPUs; each GPU worker runs both methods independently.
6. Merges all shard results, computes per-category accuracy for each method, compares them sample-by-sample, and saves all outputs.

---

## Supported models

| `--model_type` | Model family |
|---|---|
| `qwen2.5-vl` | Qwen2.5-VL (e.g. 3B, 7B, 72B) |
| `qwen3-vl` | Qwen3-VL (e.g. 4B, 8B) |

Both use `AutoModelForImageTextToText` + `AutoProcessor`, so the correct architecture is selected automatically from the checkpoint config.

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `qwen2.5-vl` | Model family (`qwen2.5-vl` or `qwen3-vl`) |
| `--model_path` | *(required)* | Path to the model checkpoint directory |
| `--data_dir` | `datasets/evaluation/MMSIBench` | Root directory of the MMSIBench dataset |
| `--gen_dir` | `None` | Root directory of generated images. Sub-folder per sample index (e.g. `{gen_dir}/0/img_0.png`). If `None` or folder missing, augmented = baseline. |
| `--limit` | `None` | Cap the number of samples (useful for smoke tests) |
| `--batch_size` | `1` | Inference batch size (forced to 1 for variable-length multi-image inputs) |
| `--max_new_tokens` | `128` | Maximum tokens to generate per sample |
| `--output_dir` | `results/mmsibench` | Root directory for saving results |
| `--model_name` | `None` | Sub-folder name under `output_dir` (defaults to `model_type`) |
| `--gen_model_name` | `None` | Display name for the image generation model logged in the config header (defaults to the last component of `--gen_dir`) |

---

## Quick start

```bash
conda activate SPR
bash scripts/run_evaluation.sh
```

This uses all default values: model `qwen3-vl`, checkpoint `checkpoints/Qwen3-VL-4B-Instruct`, generated images from `generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B`, output to `results/mmsibench`.

---

## Shell script (`scripts/run_evaluation.sh`)

A convenience wrapper around `evaluation.py`. It resolves relative paths against the project root and prints a summary header before running.

### Usage

```
bash scripts/run_evaluation.sh [MODEL_TYPE] [MODEL_PATH] [GEN_DIR] [EXTRA_ARGS...]
```

All three positional arguments are optional — defaults are used when omitted.

| Position | Variable | Default |
|---|---|---|
| 1 | `MODEL_TYPE` | `qwen3-vl` |
| 2 | `MODEL_PATH` | `checkpoints/Qwen3-VL-4B-Instruct` |
| 3 | `GEN_DIR` | `generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B` |

Any arguments after position 3 are forwarded directly to `evaluation.py` (e.g. `--limit`, `--max_new_tokens`).

### Shell script examples

```bash
# All defaults
bash scripts/run_evaluation.sh

# Custom model, default gen_dir
bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct

# Full positional args
bash scripts/run_evaluation.sh \
    qwen3-vl \
    checkpoints/Qwen3-VL-4B-Instruct \
    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B

# Smoke test (12 samples)
bash scripts/run_evaluation.sh \
    qwen3-vl \
    checkpoints/Qwen3-VL-4B-Instruct \
    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B \
    --limit 12

# Restrict to specific GPUs
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_evaluation.sh \
    qwen3-vl \
    checkpoints/Qwen3-VL-4B-Instruct \
    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
```

---

## Python examples

### Smoke test (12 samples)

```bash
conda run -n SPR python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir   datasets/evaluation/MMSIBench \
    --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B \
    --limit 12
```

### Full evaluation, all 6 GPUs

```bash
conda run -n SPR python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir   datasets/evaluation/MMSIBench \
    --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
```

### Restrict to specific GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n SPR python evaluation.py \
    --model_type qwen2.5-vl \
    --model_path checkpoints/Qwen2.5-VL-3B-Instruct \
    --data_dir   datasets/evaluation/MMSIBench \
    --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
```

### Custom output folder name

```bash
conda run -n SPR python evaluation.py \
    --model_type  qwen3-vl \
    --model_path  checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir    datasets/evaluation/MMSIBench \
    --gen_dir     generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B \
    --output_dir  results/mmsibench \
    --model_name  Qwen3-VL-4B-flux2-klein-4B
```

### Baseline only (no generated images)

Omit `--gen_dir`; augmented method will be identical to baseline.

```bash
conda run -n SPR python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir   datasets/evaluation/MMSIBench
```

---

## Generated images directory structure

`--gen_dir` must follow this layout (produced by `image_generation.py`):

```
gen_dir/
├── 0/
│   └── img_0.png          # one or more generated images for sample id=0
├── 1/
│   ├── img_0.png
│   └── img_1.png
├── 2/                     # empty folder → no generation needed → falls back to baseline
└── ...
```

The sub-folder name is the integer sample `id` from the dataset JSON. Images are loaded in sorted filename order and appended after the original dataset images.

---

## Output structure

Results are saved under:

```
results/mmsibench/<model_name>/<model_name>_<timestamp>/
├── run.log                       # Full run log (planning model, gen model, params, metrics)
├── configuration.json            # All config params saved as JSON
│
├── eval_result_baseline.json     # Per-sample results — baseline method
├── eval_result_augmented.json    # Per-sample results — augmented method
│
├── metrics_baseline.json         # Accuracy metrics — baseline
├── metrics_augmented.json        # Accuracy metrics — augmented
├── metrics_comparison.json       # Side-by-side + delta
│
├── analysis_changes.json         # Per-sample change analysis (5 categories)
│
├── baseline_gpu0.json            # Per-GPU shard outputs (merged into eval_result_*.json)
├── augmented_gpu0.json
└── ...
```

### `run.log` structure

The log file is saved inside the results folder. It begins with a structured config header:

```
EVALUATION CONFIGURATION
============================================================

  [Planning model — VQA inference]
    model_name     : Qwen3-VL-4B-Instruct
    model_type     : qwen3-vl
    model_path     : /abs/path/checkpoints/Qwen3-VL-4B-Instruct
    max_new_tokens : 128
    batch_size     : 1

  [Generation model — augmented images]
    gen_model_name : flux2-klein-4B
    gen_dir        : /abs/path/generated_images/mmsibench/.../flux2-klein-4B

  [Dataset]
    data_dir       : /abs/path/datasets/evaluation/MMSIBench
    samples        : 985
    w/ gen images  : 985 / 985

  [Runtime]
    GPUs           : ['0', '1', '2', '3', '4', '5']
    output_dir     : /abs/path/results/mmsibench/qwen3-vl/qwen3-vl_20260220_010610
    log_file       : /abs/path/results/mmsibench/qwen3-vl/qwen3-vl_20260220_010610/run.log
============================================================
```

`gen_model_name` is auto-derived from the last path component of `--gen_dir` unless overridden with `--gen_model_name`.

---

## Output file schemas

### `configuration.json`

Saved immediately after the config header is logged, before any inference starts:

```json
{
  "planning_model": {
    "model_name": "Qwen3-VL-4B-Instruct",
    "model_type": "qwen3-vl",
    "model_path": "/abs/path/checkpoints/Qwen3-VL-4B-Instruct",
    "max_new_tokens": 128,
    "batch_size": 1
  },
  "reasoning_model": {
    "model_name": "Qwen3-VL-4B-Instruct",
    "model_type": "qwen3-vl",
    "model_path": "/abs/path/checkpoints/Qwen3-VL-4B-Instruct",
    "max_new_tokens": 128,
    "batch_size": 1
  },
  "generation_model": {
    "gen_model_name": "flux2-klein-4B",
    "gen_dir": "/abs/path/generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B"
  },
  "dataset": {
    "data_dir": "/abs/path/datasets/evaluation/MMSIBench",
    "limit": null,
    "total_samples": 985,
    "samples_with_gen_images": 985
  },
  "runtime": {
    "gpus": ["0", "1", "2", "3", "4", "5"],
    "output_dir": "/abs/path/results/mmsibench/qwen3-vl/qwen3-vl_20260220_010610",
    "log_file": "/abs/path/results/mmsibench/qwen3-vl/qwen3-vl_20260220_010610/run.log",
    "timestamp": "20260220_010610"
  }
}
```

---

### `eval_result_baseline.json` / `eval_result_augmented.json`

One entry per sample:

```json
{
  "method": "baseline",
  "index": 0,
  "category": "Motion (Cam.)",
  "question": "...",
  "answer": "C",
  "prediction": "C",
  "output": "``C``",
  "thought_gt": "...",
  "original_images": ["/abs/path/img_0.jpg", "/abs/path/img_1.jpg"],
  "generated_images": ["/abs/path/gen/0/img_0.png"],
  "prompt": "<|im_start|>user\n..."
}
```

`generated_images` is `[]` for baseline entries and for augmented entries where no generation was produced.

---

### `metrics_baseline.json` / `metrics_augmented.json`

```json
{
  "overall_accuracy": 0.3333,
  "total_samples": 12,
  "correct_samples": 4,
  "category_accuracy": {
    "Motion (Cam.)": 0.3333,
    "Positional Relationship (Cam.\u2013Obj.)": 0.3333
  },
  "category_counts": {
    "Motion (Cam.)": 3,
    "Positional Relationship (Cam.\u2013Obj.)": 3
  }
}
```

---

### `metrics_comparison.json`

```json
{
  "baseline":  { ... },
  "augmented": { ... },
  "delta_overall_accuracy": 0.1667
}
```

`delta_overall_accuracy` = augmented accuracy − baseline accuracy. Positive means generated images improved overall performance.

---

### `analysis_changes.json`

Per-sample breakdown into **5 mutually exclusive groups**:

| Group | Condition | Interpretation |
|---|---|---|
| `degraded` | Baseline ✓ → Augmented ✗ | Generated images hurt this sample |
| `improved` | Baseline ✗ → Augmented ✓ | Generated images helped this sample |
| `correct_no_gen` | Baseline ✓, no generated images, Augmented ✓ | Correct without any generation |
| `correct_with_gen` | Baseline ✓, has generated images, Augmented ✓ | Correct, generation didn't break it |
| `always_wrong` | Baseline ✗ & Augmented ✗ | Both methods wrong |

```json
{
  "total": 12,
  "counts": {
    "degraded": 0,
    "improved": 2,
    "correct_no_gen": 1,
    "correct_with_gen": 3,
    "always_wrong": 6
  },
  "proportions": {
    "degraded": 0.0,
    "improved": 0.167,
    "correct_no_gen": 0.083,
    "correct_with_gen": 0.25,
    "always_wrong": 0.5
  },
  "descriptions": {
    "degraded": "Baseline \u2713 \u2192 Augmented \u2717  (gen images hurt)",
    ...
  },
  "samples": {
    "improved": [
      {
        "index": 1,
        "category": "Motion (Cam.)",
        "question": "...",
        "answer": "B",
        "baseline_prediction": "A",
        "augmented_prediction": "B",
        "baseline_output": "...",
        "augmented_output": "...",
        "generated_images": ["/abs/path/gen/1/img_0.png"]
      }
    ],
    "degraded": [],
    "correct_no_gen": [...],
    "correct_with_gen": [...],
    "always_wrong": [...]
  }
}
```

---

## Multi-GPU behaviour

The dataset is automatically split into `N` equal shards where `N` = number of visible CUDA devices. Each shard is processed in a subprocess; each subprocess loads its own model copy on its assigned GPU. Both methods are run within the same subprocess (single model load per GPU). Results are merged by the main process after all workers finish.

```bash
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir   datasets/evaluation/MMSIBench \
    --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
```
