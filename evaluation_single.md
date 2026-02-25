# evaluation_single.py

Baseline reasoning evaluation on **MMSIBench** using a QwenVL model.

The script answers every question by feeding **all input images** together with the question directly into the model (no spatial reasoning pipeline, no image generation). This serves as the simplest possible baseline to compare against more sophisticated approaches.

---

## How it works

1. Loads `datasets/evaluation/MMSIBench/data/test_data_final.json`.
2. For each sample, builds a Qwen-VL chat message with all images prepended before the question text.
3. Appends the answer instruction: *"Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."*
4. Runs greedy decoding (`do_sample=False`) and extracts the predicted letter (A/B/C/D).
5. Automatically **shards the dataset across all visible GPUs** — each GPU runs its own model instance in a separate subprocess.
6. Merges results and saves `eval_result.json` + `metrics.json` under the output directory.

---

## Supported models

| `--model_type` | Model family |
|---|---|
| `qwen2.5-vl` | Qwen2.5-VL (e.g. 3B, 7B, 72B) |
| `qwen3-vl` | Qwen3-VL (e.g. 4B, 8B) |

Both are loaded via `AutoModelForImageTextToText` + `AutoProcessor`, so the correct architecture is selected automatically from the checkpoint's config.

---

## Quick start

```bash
# Activate environment first
conda activate SPR

# Single model type, all available GPUs
python evaluation_single.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir  datasets/evaluation/MMSIBench
```

Or use the convenience wrapper:

```bash
bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `qwen2.5-vl` | Model family (`qwen2.5-vl` or `qwen3-vl`) |
| `--model_path` | *(required)* | Path to the model checkpoint directory |
| `--data_dir` | `datasets/evaluation/MMSIBench` | Root directory of the MMSIBench dataset |
| `--limit` | `None` | Cap number of samples (useful for smoke tests) |
| `--batch_size` | `1` | Inference batch size (forced to 1 for variable-length multi-image inputs) |
| `--max_new_tokens` | `128` | Maximum tokens to generate per sample |
| `--output_dir` | `results_single/mmsibench` | Root directory for saving results |
| `--model_name` | `None` | Sub-folder name under `output_dir` (defaults to `model_type`) |

---

## Examples

### Smoke test (12 samples, 6 GPUs)

```bash
conda run -n SPR python evaluation_single.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir  datasets/evaluation/MMSIBench \
    --limit 12
```

### Full evaluation on all samples

```bash
conda run -n SPR bash scripts/run_evaluation.sh qwen3-vl checkpoints/Qwen3-VL-4B-Instruct
```

### Restrict to specific GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n SPR bash scripts/run_evaluation.sh \
    qwen2.5-vl checkpoints/Qwen2.5-VL-3B-Instruct
```

### Use a custom output folder name

```bash
conda run -n SPR python evaluation_single.py \
    --model_type  qwen3-vl \
    --model_path  checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir    datasets/evaluation/MMSIBench \
    --output_dir  results_single/mmsibench \
    --model_name  Qwen3-VL-4B-baseline
```

---

## Output structure

Results are saved under:

```
results_single/mmsibench/<model_name>/<model_name>_<timestamp>/
├── eval_result.json      # Per-sample predictions and raw model outputs
├── metrics.json          # Overall and per-category accuracy
├── run.log               # Full run log
├── results_gpu0.json     # Per-GPU shard (merged into eval_result.json)
├── results_gpu1.json
└── ...
```

### `eval_result.json` — one entry per sample

```json
{
  "index": 0,
  "category": "Motion (Cam.)",
  "question": "The images are taken continuously from a first-person perspective. In which direction are you moving?\nOptions: A: Left while moving backward, ...",
  "answer": "C",
  "prediction": "C",
  "output": "``C``",
  "thought_gt": "...",
  "prompt": "<|im_start|>user\n..."
}
```

### `metrics.json` — aggregated accuracy

```json
{
  "overall_accuracy": 0.3333,
  "total_samples": 12,
  "correct_samples": 4,
  "category_accuracy": {
    "Motion (Cam.)": 0.3333,
    "Positional Relationship (Cam.–Obj.)": 0.3333,
    ...
  },
  "category_counts": {
    "Motion (Cam.)": 3,
    ...
  }
}
```

---

## Multi-GPU behaviour

The dataset is automatically split into `N` equal shards where `N` = number of visible CUDA devices. Each shard is processed in a separate subprocess, each subprocess loading its own copy of the model on its assigned GPU. Results are merged by the main process after all workers finish.

To control which GPUs are used, set `CUDA_VISIBLE_DEVICES` before running:

```bash
# Use only GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python evaluation_single.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir  datasets/evaluation/MMSIBench
```
