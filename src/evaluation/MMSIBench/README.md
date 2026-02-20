# MMSI-Bench Evaluation

MMSI-Bench is a comprehensive VQA benchmark for multi-image spatial intelligence, containing 1,000 challenging multiple-choice questions across 11 fundamental spatial reasoning tasks.

## Overview

**Paper**: [MMSI-Bench: Benchmarking Multi-image Spatial Intelligence](https://arxiv.org/abs/2505.23764)  
**Dataset**: [HuggingFace - RunsenXu/MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)  
**Leaderboard**: [OpenCompass Spatial Leaderboard](https://huggingface.co/spaces/opencompass/openlmm_spatial_leaderboard)

### Key Features

- **Multi-image**: Each task involves 2+ images requiring spatial reasoning
- **11 Task Types**: Covers camera-camera, camera-object, camera-region, object-object, object-region, region-region relationships, plus measurement, appearance, motion, and multi-step reasoning
- **High Quality**: Fully human-designed questions with step-by-step reasoning annotations
- **Real-world Scenarios**: Images from autonomous driving, robotic manipulation, and scene scanning
- **Challenging**: Largest model-human performance gap (~57% difference)

### Task Categories

1. **Camera-Camera**: Spatial relationships between different camera viewpoints
2. **Camera-Object**: Relationships between camera and objects in the scene
3. **Camera-Region**: Relationships between camera and semantic regions
4. **Object-Object**: Spatial relationships between objects
5. **Object-Region**: Relationships between objects and regions
6. **Region-Region**: Spatial relationships between semantic regions
7. **Measurement**: Quantitative spatial measurements
8. **Appearance**: Visual appearance understanding
9. **Camera Motion**: Understanding camera movement
10. **Object Motion**: Understanding object movement
11. **Multi-step Reasoning**: Complex reasoning requiring multiple steps

## Dataset Structure

```
datasets/evaluation/MMSIBench/
├── MMSI_bench.tsv           # Main dataset file (1000 samples)
├── images/                  # Image directory
│   ├── 0_0.jpg             # First image of sample 0
│   ├── 0_1.jpg             # Second image of sample 0
│   ├── 1_0.jpg
│   └── ...
└── setup_dataset.sh         # Dataset download script
```

### Data Format

TSV file with the following fields:

- `index`: Unique sample identifier
- `image`: JSON array of image paths (e.g., `["images/0_0.jpg", "images/0_1.jpg"]`)
- `question`: Question text with multiple-choice options (A/B/C/D)
- `answer`: Correct answer letter (A/B/C/D)
- `category`: Task category (one of 11 types)
- `thought`: Step-by-step reasoning process (human-annotated)
- `difficulty`: Difficulty level
- `mean_normed_duration_seconds`: Normalized human response time

## Setup

### 1. Download Dataset

```bash
cd datasets/evaluation/MMSIBench
bash setup_dataset.sh
```

This will:
- Download MMSI_bench.tsv from HuggingFace
- Extract images from base64 encoding
- Organize images in the `images/` directory

### 2. Verify Data

```bash
# Check data files
ls -lh MMSI_bench.tsv
ls -lh images/ | head -20

# Count samples
wc -l MMSI_bench.tsv

# Preview data
head -5 MMSI_bench.tsv | column -t -s $'\t'
```

## Running Evaluation

### Basic Usage

```bash
# Using evaluation script
bash scripts/evaluation/evaluate_mmsibench.sh \\
    -m qwen2.5-vl \\
    -p /path/to/Qwen2.5-VL-7B-Instruct \\
    -o results/mmsibench/qwen2.5-vl-7b

# Direct Python call
python src/evaluation/MMSIBench/eval_mmsibench.py \\
    --model_type qwen2.5-vl \\
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \\
    --data_dir datasets/evaluation/MMSIBench \\
    --output_dir results/mmsibench/test
```

### Multi-GPU Evaluation

```bash
# Use multiple GPUs for faster evaluation
bash scripts/evaluation/evaluate_mmsibench.sh \\
    -m spatial-mllm \\
    -p checkpoints/Spatial-MLLM-v1.1-Instruct-135K \\
    -o results/mmsibench/spatial-mllm \\
    -g 4  # Use 4 GPUs
```

### Test on Limited Samples

```bash
# Quick test with 10 samples
bash scripts/evaluation/evaluate_mmsibench.sh \\
    -m qwen2.5-vl \\
    -p /path/to/model \\
    -o results/mmsibench/test \\
    -l 10
```

### Model Types

Supported model types:
- `qwen2.5-vl`: Qwen2.5-VL base models
- `spatial-mllm`: Spatial-MLLM models  
- `spatial-VACE`: Spatial-MLLM with VACE encoder
- `qwen-vace`: Qwen-VL with VACE encoder

## Output Format

### Predictions File (`predictions.json`)

```json
[
  {
    "index": "0",
    "category": "Camera-Camera",
    "question": "Given two images taken from different viewpoints...",
    "answer": "B",
    "prediction": "B",
    "output": "The answer is `B`.",
    "thought": "Step 1: Identify camera positions...",
    "difficulty": "medium",
    "prompt": "..."
  },
  ...
]
```

### Metrics File (`predictions_metrics.json`)

```json
{
  "overall_accuracy": 0.287,
  "category_accuracy": {
    "Camera-Camera": 0.31,
    "Camera-Object": 0.28,
    "Camera-Region": 0.26,
    ...
  },
  "total_samples": 1000,
  "correct_samples": 287,
  "category_counts": {
    "Camera-Camera": 120,
    "Camera-Object": 95,
    ...
  }
}
```

## Expected Performance

Based on MMSI-Bench leaderboard (as of Feb 2026):

| Model | Accuracy | Type |
|-------|----------|------|
| Human | 97.2% | Baseline |
| Gemini-3-pro | 49.2% | Proprietary |
| SenseNova-SI-1.2 | 42.6% | Open-source |
| o3 | 41.0% | Proprietary |
| Qwen2.5-VL-72B | 30.7% | Open-source |
| Qwen2.5-VL-7B | 25.9% | Open-source |
| Random Guessing | 25.0% | Baseline |

Note: Significant model-human gap indicates challenging spatial reasoning requirements.

## Answer Extraction

MMSI-Bench uses specific post-prompts and regex patterns:

**Post-prompt**:
```
Answer with the option's letter from the given choices directly. 
Enclose the option's letter within ``.
```

**Extraction Logic**:
1. Extract content between double backticks: `` ``answer`` ``
2. Extract content between single backticks: `` `answer` ``
3. Find isolated uppercase letter (A/B/C/D) with word boundaries
4. Exclude indefinite articles like "A bike"

This follows the official MMSI-Bench evaluation protocol.

## Integration with Spatial_RAI

MMSI-Bench evaluation is fully integrated with the Spatial_RAI framework:

```python
from src.evaluation.MMSIBench import evaluate_mmsibench, load_mmsibench_dataset

# Load dataset
data_dir = Path("datasets/evaluation/MMSIBench")
dataset = load_mmsibench_dataset(data_dir / "MMSI_bench.tsv")

# Run evaluation  
results, metrics = evaluate_mmsibench(
    mmsibench_data=dataset,
    model_type="qwen2.5-vl",
    model_path="/path/to/model",
    batch_size=1,
    data_dir=data_dir,
    output_path=Path("results/predictions.json"),
)

# Print results
print(f"Accuracy: {metrics['overall_accuracy']:.2%}")
for category, acc in metrics['category_accuracy'].items():
    print(f"  {category}: {acc:.2%}")
```

## Troubleshooting

### Issue: Data file not found

```bash
# Ensure you're in the correct directory
cd datasets/evaluation/MMSIBench

# Check if TSV exists
ls -lh MMSI_bench.tsv

# Re-download if needed
bash setup_dataset.sh
```

### Issue: Images not loading

```bash
# Verify images directory exists
ls -lh images/ | head

# Check image paths in TSV match actual files
python -c "
import pandas as pd
import json
df = pd.read_csv('MMSI_bench.tsv', sep='\t')
img_paths = json.loads(df.iloc[0]['image'])
print(img_paths)
"
```

### Issue: Low accuracy (~25%)

This is expected for base models without spatial reasoning training:
- MMSI-Bench is extremely challenging
- Random guessing accuracy is 25% (4 choices)
- Spatial-MLLM fine-tuned models achieve ~35-40%
- Top proprietary models reach ~40-50%

## Citation

If you use MMSI-Bench in your research, please cite:

```bibtex
@inproceedings{yang2026mmsi,
  title={MMSI-Bench: A Comprehensive Benchmark for Assessing Multi-Image Spatial Intelligence in Vision Language Models},
  author={Yang, Sihan and Xu, Runsen and others},
  booktitle={ICLR},
  year={2026}
}
```

## References

- [MMSI-Bench Paper](https://arxiv.org/abs/2505.23764)
- [HuggingFace Dataset](https://huggingface.co/datasets/RunsenXu/MMSI-Bench)
- [Official Repository](https://github.com/OpenRobotLab/MMSI-Bench)
- [EASI Leaderboard](https://huggingface.co/spaces/lmms-lab-si/easi-leaderboard)
- [OpenCompass Spatial Leaderboard](https://huggingface.co/spaces/opencompass/openlmm_spatial_leaderboard)
