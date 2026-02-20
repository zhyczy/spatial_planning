#!/bin/bash

# Example script to run SAT evaluation
# Usage: bash run_eval_sat_example.sh

# Note: Before running this script, make sure to setup SAT dataset images:
#   cd datasets/evaluation/SAT
#   python data_process.py --output_dir . --split test
#   python data_process.py --output_dir . --split val

# Set paths
MODEL_PATH="checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
MODEL_TYPE="spatial-VACE"
MODEL_NAME="spatial-vace-135k"

# SAT dataset paths (now in Spatial_RAI)
DATA_JSON="datasets/evaluation/SAT/test.json"
DATA_DIR="datasets/evaluation/SAT"

# Output directory
OUTPUT_DIR="results"

# Batch size (forced to 1 for best performance)
BATCH_SIZE=1

# Select GPUs to use (optional)
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
python src/evaluation/SAT/eval_SAT.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --model_name ${MODEL_NAME} \
    --data_json ${DATA_JSON} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --visualize_vace_videos False

echo "Evaluation completed! Results saved to ${OUTPUT_DIR}/SAT/${MODEL_NAME}/"
