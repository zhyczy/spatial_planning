# MindCube Dataset Evaluation

This directory contains scripts for evaluating models on the MindCube cognitive map benchmark dataset.

## Dataset Description

MindCube is a benchmark for evaluating spatial reasoning and cognitive map capabilities in vision-language models. It tests understanding of spatial relationships, perspective changes, and navigation in 3D environments.

## Files

- `eval_mindcube.py`: Main evaluation script for MindCube dataset
- `dataset_utils.py`: Utility functions for MindCube dataset processing and metrics calculation
- `__init__.py`: Module initialization

## Dataset Format

The MindCube dataset uses JSONL format with the following structure:

```json
{
  "id": "among_group002_q0_1_1",
  "category": ["perpendicular", "P-P", "meanwhile", "self"],
  "type": "0_frame",
  "meta_info": [["black waist bag", "plush toy", "black sofa", "window", "display shelves"], [null, null, "face", null, null]],
  "question": "Based on these two views showing the same scene: in which direction did I move from the first view to the second view? A. Directly left B. Diagonally forward and right C. Diagonally forward and left D. Directly right",
  "images": ["other_all_image/among/backpack_002/front_039.jpg", "other_all_image/among/backpack_002/left_123.jpg"],
  "gt_answer": "C"
}
```

## Available Splits

- **tinybench**: ~200 samples (for quick testing)
- **train**: ~2000 samples (training set)
- **full**: ~4500 samples (complete benchmark)

## Usage

### Basic Usage

```bash
# Evaluate on tinybench (quick test)
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate_mindcube.sh -m qwen2.5-vl -s tinybench

# Evaluate on full dataset
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate_mindcube.sh -m spatial-VACE -s full
```

### With Sample Limit

```bash
# Test with only 10 samples
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate_mindcube.sh -m qwen2.5-vl -s tinybench -l 10
```

### Different Models

```bash
# Qwen2.5-VL (baseline)
bash scripts/evaluation/evaluate_mindcube.sh -m qwen2.5-vl -s tinybench

# Spatial-MLLM
bash scripts/evaluation/evaluate_mindcube.sh -m spatial-mllm -s full

# Spatial-VACE (recommended)
bash scripts/evaluation/evaluate_mindcube.sh -m spatial-VACE -s full
```

### Arguments

- `-m MODEL_TYPE`: Type of model to use
  - `qwen2.5-vl`: Qwen2.5-VL base model
  - `spatial-mllm`: Spatial-MLLM model
  - `spatial-VACE`: Spatial-MLLM with VACE integration
  - `qwen-vace`: Qwen with VACE integration
- `-s SPLIT`: Dataset split to use
  - `tinybench`: Small subset (~200 samples)
  - `train`: Training set (~2000 samples)
  - `full`: Complete benchmark (~4500 samples)
- `-l LIMIT`: Limit number of samples (optional)
- `-p MODEL_PATH`: Custom model path (optional)
- `-v`: Enable VACE visualization (optional)

## Output Structure

Results are saved in the following structure:

```
results/
└── MindCube/
    └── {model_name}/
        └── {model_name}_{timestamp}/
            ├── run.log
            ├── eval_result.json
            ├── metrics.json
            └── results_{model_type}_{gpu_id}.json
```

- `run.log`: Execution logs
- `eval_result.json`: Detailed results for each sample
- `metrics.json`: Overall metrics including per-category and per-type accuracy

## Metrics

The evaluation computes:
- **Overall accuracy**: Exact match across all samples
- **Per-category accuracy**: Breakdown by spatial relation category (perpendicular, parallel, etc.)
- **Per-type accuracy**: Breakdown by question type (0_frame, multi_frame, etc.)

## Multi-GPU Support

The script automatically uses all available GPUs. To use specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/evaluation/evaluate_mindcube.sh -m spatial-VACE -s full
```

## Example Commands

```bash
# Quick test on tinybench with 10 samples
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate_mindcube.sh \
    -m qwen2.5-vl -s tinybench -l 10

# Full evaluation with Spatial-VACE
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/evaluate_mindcube.sh \
    -m spatial-VACE -s full

# Training set evaluation
CUDA_VISIBLE_DEVICES=0,1 bash scripts/evaluation/evaluate_mindcube.sh \
    -m spatial-mllm -s train
```

## Direct Python Usage

You can also run the evaluation script directly:

```bash
python src/evaluation/MindCube/eval_mindcube.py \
    --model_path checkpoints/Spatial-MLLM-v1.1-Instruct-135K \
    --model_type spatial-VACE \
    --data_jsonl datasets/evaluation/MindCube/MindCube_tinybench.jsonl \
    --data_dir datasets/evaluation/MindCube \
    --output_dir results \
    --batch_size 1 \
    --limit 10
```

## Notes

- The dataset uses multiple images per question (typically 2 views of the same scene)
- Questions are multiple choice with options A, B, C, D
- The script forces batch size to 1 for optimal performance
- Images are loaded from paths specified in the JSONL file, resolved relative to `--data_dir`
- Image directory is symlinked to the original MindCube dataset to save space

## Troubleshooting

### Image Loading Issues

If the evaluation script cannot find images, verify:
1. The symlink to `other_all_image` exists in `datasets/evaluation/MindCube/`
2. The original MindCube dataset has images in `MindCube/data/other_all_image/`

### Memory Issues

For large datasets, consider:
1. Using a smaller split first (tinybench)
2. Using the `--limit` parameter to test with fewer samples
3. Ensure GPU has sufficient memory (at least 24GB recommended for larger models)
