# Spatial Planning — Visual Instruction Generation

This module uses a vision-language model (Qwen3-VL / Qwen2.5-VL) as a **planning model**: given the existing images and a spatial reasoning question, it decides whether a video clip needs to be generated, and if so, outputs **exactly one** video-generation description — the one the model judges will best help the downstream reasoning model.

---

## Overview

```
spatial_planning/
├── instruction_generation.py        # main pipeline (multi-GPU)
├── scripts/
│   └── run_generate_instructions.sh # convenience bash wrapper
├── checkpoints/
│   └── Qwen3-VL-4B-Instruct/        # local model weights
└── datasets/
    └── evaluation/
        ├── MindCube/
        ├── SAT/
        ├── vsibench/
        └── MMSIBench/
```

---

## How the Planning Model Works

For each sample the model receives:
- One or more scene images
- A spatial reasoning question

It responds in two structured sections:

```xml
<think>
  Step-by-step reasoning about what spatial information is visible,
  what is missing, whether a video would help, and which single video
  description would best aid the downstream reasoning model.
</think>
<instructions>
  <instruction>A slow forward dolly shot starting from directly behind the sofa, moving toward the center of the living room, revealing all objects between the sofa and the far wall.</instruction>
</instructions>
```

**Rules enforced by the prompt:**
| Case | Output |
|---|---|
| Existing images are sufficient | `<instructions></instructions>` → `"instructions": []` |
| Extra visual information needed | **Exactly 1** `<instruction>` — the single video-generation prompt the model judges most helpful |

The instruction describes a video that captures missing spatial information, e.g.:
- **Camera movement** — dolly, pan, or flythrough to reveal hidden geometry
- **Occlusion removal** — a trajectory where the object is unblocked
- **Spatial traversal** — slow pan or dolly that clarifies distances and positions
- **Sweeping context** — wide shot revealing overall spatial relationships

---

## Setup

```bash
conda activate spi
cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning
```

Supported model checkpoints (in `checkpoints/`):
| Path | Notes |
|---|---|
| `checkpoints/Qwen3-VL-4B-Instruct` | recommended, fast |
| `checkpoints/Qwen3-VL-8B-Instruct` | higher quality |
| `checkpoints/Qwen2.5-VL-3B-Instruct` | smaller, less capable |

---

## Running the Pipeline

### Option 1 — Bash script (recommended)

```bash
bash scripts/run_generate_instructions.sh <DATASET> <MODEL_PATH> [OPTIONS...]
```

**Examples:**

```bash
# MMSIBench — all GPUs
bash scripts/run_generate_instructions.sh mmsibench checkpoints/Qwen3-VL-4B-Instruct

# MindCube — specific GPUs
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_generate_instructions.sh mindcube checkpoints/Qwen3-VL-4B-Instruct

# SAT — quick test with 50 samples
bash scripts/run_generate_instructions.sh sat checkpoints/Qwen3-VL-4B-Instruct --max_samples 50

# VSIBench — greedy decoding
bash scripts/run_generate_instructions.sh vsibench checkpoints/Qwen3-VL-4B-Instruct --no-do_sample
```

Supported `DATASET` values: `mindcube` | `sat` | `vsibench` | `mmsibench`

---

### Option 2 — Direct Python

```bash
python instruction_generation.py \
  --dataset      mmsibench \
  --data_path    datasets/evaluation/MMSIBench/data/test_data_final.json \
  --image_root   datasets/evaluation/MMSIBench \
  --model_path   checkpoints/Qwen3-VL-4B-Instruct \
  --output_path  results/mmsibench_instructions.jsonl \
  --max_samples  100 \
  --max_new_tokens 512 \
  --no-do_sample
```

**All CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mindcube` | Dataset name |
| `--data_path` | *(per dataset)* | Path to annotation file |
| `--image_root` | *(per dataset)* | Root dir for image paths |
| `--model_path` | `Qwen/Qwen2.5-VL-7B-Instruct` | Local or HF model path |
| `--model_type` | `qwen-vl` | Model family tag (informational) |
| `--max_new_tokens` | `512` | Max output tokens |
| `--temperature` | `0.7` | Sampling temperature |
| `--top_p` | `0.9` | Top-p sampling |
| `--do_sample` / `--no-do_sample` | `True` | Sampling vs greedy |
| `--max_samples` | `-1` (all) | Limit number of samples |
| `--start_idx` | `0` | Start from this index |
| `--num_gpus` | `-1` (all) | Limit GPU count |
| `--output_path` | `results/image_instructions.jsonl` | Output base path |
| `--verbose` / `--no-verbose` | `True` | Per-sample logging |

---

## Multi-GPU Behavior

The pipeline automatically uses **all available GPUs** (or those listed in `CUDA_VISIBLE_DEVICES`). Data is sharded evenly across GPUs — each GPU runs a separate model copy in its own subprocess. Results are merged after all workers finish.

```bash
# Use GPUs 0,1,2,3 only
CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_image_instructions.py ...

# Force single GPU
CUDA_VISIBLE_DEVICES=0 python generate_image_instructions.py ...

# Limit via flag (takes first N GPUs from available/visible list)
python generate_image_instructions.py --num_gpus 2 ...
```

---

## Output Format

Results are saved to a **timestamped directory** under `results/`:

```
results/
└── mmsibench_image_instructions_20260219_213819/
    ├── run.log                # full log from all workers
    ├── shard_0_gpu0.jsonl     # per-GPU shard
    ├── shard_1_gpu1.jsonl
    └── results.jsonl          # merged final output
```

Each line of `results.jsonl` is a JSON object:

```json
{
  "id": 0,
  "dataset": "mmsibench",
  "question": "In which direction are you moving?",
  "image_paths": ["/abs/path/sample_0_img_0.jpg", "..."],
  "gt_answer": "C",
  "raw_output": "<think>...</think>\n<instructions>...</instructions>",
  "instructions": [
    "A slow forward dolly shot starting from behind the hallway entrance, moving toward the far end, clearly showing the direction of travel."
  ],
  "num_instructions": 1,
  "meta": {}
}
```

- `instructions` — list of length 0 or 1: empty = no video needed; one entry = the video-generation prompt
- `num_instructions` — 0 or 1
- `raw_output` — full model output including `<think>` reasoning

---

## Quick Test

```bash
conda activate spi
cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning

CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_generate_instructions.sh \
  mmsibench \
  checkpoints/Qwen3-VL-4B-Instruct \
  --max_samples 4 \
  --max_new_tokens 512 \
  --no-do_sample
```

Expected output (4 samples, ~15s total):
```
[1/2] id=0  (3.4s)
  Instructions : 1
  [1] A slow forward dolly shot starting from behind the hallway entrance ...

[2/2] id=1  (2.0s)
  Instructions : 0
  (none — existing images sufficient)
```
