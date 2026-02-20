# SAT Dataset Evaluation

This directory contains scripts for evaluating models on the SAT (Spatial Abilities Test) dataset.

## Files

- `eval_SAT.py`: Main evaluation script for SAT dataset
- `dataset_utils.py`: Utility functions for SAT dataset processing and metrics calculation
- `__init__.py`: Module initialization

## Dataset Format

The SAT dataset uses a JSON format with the following structure:

```json
[
    {
        "database_idx": 0,
        "question_type": "obj_movement",
        "question": "were any of the objects in the initial frame that you can still see in the second frame moved from their original positions?",
        "answer_choices": [
            "chair was moved left and away from the camera",
            "chair was moved right and away from the camera"
        ],
        "correct_answer": "chair was moved left and away from the camera",
        "img_paths": [
            "./data/test/image_0_0.png",
            "./data/test/image_0_1.png"
        ]
    }
]
```

## Dataset Setup

Before running evaluation, you need to setup the SAT dataset images. See [datasets/evaluation/SAT/README.md](../../../datasets/evaluation/SAT/README.md) for instructions.

**Quick setup:**

```bash
cd datasets/evaluation/SAT
pip install datasets Pillow
python data_process.py --output_dir . --split test
python data_process.py --output_dir . --split val
```

## Usage

### Basic Usage

```bash
python src/evaluation/SAT/eval_SAT.py \
    --model_path /path/to/model \
    --model_type spatial-mllm \
    --data_json datasets/evaluation/SAT/test.json \
    --data_dir datasets/evaluation/SAT \
    --output_dir results
```

### With Custom Model Name

```bash
python src/evaluation/SAT/eval_SAT.py \
    --model_path checkpoints/Spatial-MLLM-v1.1-Instruct-135K \
    --model_type spatial-VACE \
    --model_name spatial-vace-v1.1 \
    --data_json datasets/evaluation/SAT/test.json \
    --data_dir datasets/evaluation/SAT \
    --output_dir results \
    --batch_size 1
```

### Arguments

- `--model_path`: Path to the model checkpoint
- `--model_type`: Type of model to use. Options:
  - `qwen2.5-vl`: Qwen2.5-VL base model
  - `spatial-mllm`: Spatial-MLLM model
  - `spatial-VACE`: Spatial-MLLM with VACE integration
  - `qwen-vace`: Qwen with VACE integration
- `--data_json`: Path to SAT JSON file (e.g., test.json or val.json)
- `--data_dir`: Directory containing the SAT image files
- `--output_dir`: Root directory to save evaluation results (default: results)
- `--batch_size`: Batch size for evaluation (default: 1)
- `--model_name`: Optional custom name for organizing results
- `--visualize_vace_videos`: Whether to visualize VACE decoded videos (default: False)

## Output Structure

Results are saved in the following structure:

```
results/
└── SAT/
    └── {model_name}/
        └── {model_name}_{timestamp}/
            ├── run.log
            ├── eval_result.json
            ├── metrics.json
            └── results_{model_type}_{gpu_id}.json
```

- `run.log`: Execution logs
- `eval_result.json`: Detailed results for each sample
- `metrics.json`: Overall metrics including per-question-type accuracy
- `results_{model_type}_{gpu_id}.json`: Per-GPU intermediate results

## Metrics

The evaluation computes:
- Overall accuracy across all samples
- Per-question-type accuracy and sample counts
- Question types include:
  - `obj_movement`: Object movement detection
  - `ego_movement`: Ego-centric camera movement
  - `obj_position`: Object position reasoning
  - `spatial_reasoning`: General spatial reasoning

## Multi-GPU Support

The script automatically uses all available GPUs. To use specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/evaluation/SAT/eval_SAT.py ...
```

## Example Command

```bash
# First, setup the dataset (if not already done)
cd datasets/evaluation/SAT
python data_process.py --output_dir . --split test
cd ../../..

# Then evaluate on SAT test set with Spatial-VACE model
CUDA_VISIBLE_DEVICES=0 python src/evaluation/SAT/eval_SAT.py \
    --model_path checkpoints/Spatial-MLLM-v1.1-Instruct-135K \
    --model_type spatial-VACE \
    --model_name spatial-vace-135k \
    --data_json datasets/evaluation/SAT/test.json \
    --data_dir datasets/evaluation/SAT \
    --output_dir results \
    --batch_size 1
```

## Notes

- The script forces batch size to 1 for optimal performance
- Images are loaded from paths specified in the JSON file, resolved relative to `--data_dir`
- The evaluation supports multi-image inputs (e.g., comparing two frames)
- Answer choices are automatically formatted as multiple choice options (A, B, C, D...)
