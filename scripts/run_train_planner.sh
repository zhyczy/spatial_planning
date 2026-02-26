#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  DPO training for the spatial Planning Model.
#
#  Prerequisite: run scripts/run_generate_dpo_data.sh first to produce
#  dpo_train.json under results/dpo_data/<dataset>_<timestamp>/.
#
#  Usage:
#    # LoRA fine-tune (default, fast, less memory)
#    bash scripts/run_train_planner.sh
#
#    # Full fine-tune
#    bash scripts/run_train_planner.sh full
#
#    # Specify data path explicitly
#    DPO_DATA=results/dpo_data/mmsibench_20260225/dpo_train.json \
#      bash scripts/run_train_planner.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────
MODE="${1:-lora}"   # "lora" or "full"

MODEL_NAME="checkpoints/Qwen3-VL-4B-Instruct"

# Auto-detect latest dpo_train.json if not specified
if [[ -z "${DPO_DATA:-}" ]]; then
    DPO_DATA=$(find results/dpo_data -name "dpo_train.json" -type f | sort -r | head -1)
    if [[ -z "$DPO_DATA" ]]; then
        echo "ERROR: No dpo_train.json found. Run run_generate_dpo_data.sh first."
        exit 1
    fi
fi

# DeepSpeed configs are copied from Qwen-VL-Series-Finetune
FINETUNE_SCRIPTS="../Qwen-VL-Series-Finetune/scripts"

echo "========================================"
echo " DPO Training — Planning Model"
echo "========================================"
echo " Mode          : $MODE"
echo " Model         : $MODEL_NAME"
echo " DPO data      : $DPO_DATA"
echo "========================================"

if [[ "$MODE" == "lora" ]]; then
    # ── LoRA fine-tune ────────────────────────────────────────────────────
    BATCH_PER_DEVICE=2
    GRAD_ACCUM=8
    OUTPUT_DIR="output/planner_dpo_lora"

    deepspeed train_planner.py \
        --dpo_loss "sigmoid" \
        --precompute_ref_log_probs False \
        --beta 0.1 \
        --use_liger_loss True \
        --deepspeed "$FINETUNE_SCRIPTS/zero2.json" \
        --model_id "$MODEL_NAME" \
        --data_path "$DPO_DATA" \
        --image_folder "" \
        --remove_unused_columns False \
        --lora_enable True \
        --lora_rank 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger True \
        --bf16 True \
        --fp16 False \
        --disable_flash_attn2 False \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 3 \
        --per_device_train_batch_size $BATCH_PER_DEVICE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --image_min_pixels $((512 * 28 * 28)) \
        --image_max_pixels $((1280 * 28 * 28)) \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --gradient_checkpointing True \
        --report_to tensorboard \
        --lazy_preprocess True \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 5 \
        --dataloader_num_workers 4

elif [[ "$MODE" == "full" ]]; then
    # ── Full fine-tune ────────────────────────────────────────────────────
    BATCH_PER_DEVICE=1
    GRAD_ACCUM=16
    OUTPUT_DIR="output/planner_dpo_full"

    deepspeed train_planner.py \
        --dpo_loss "sigmoid" \
        --precompute_ref_log_probs False \
        --beta 0.1 \
        --use_liger_loss True \
        --deepspeed "$FINETUNE_SCRIPTS/zero3_offload.json" \
        --model_id "$MODEL_NAME" \
        --data_path "$DPO_DATA" \
        --image_folder "" \
        --remove_unused_columns False \
        --freeze_vision_tower False \
        --freeze_llm False \
        --freeze_merger False \
        --bf16 True \
        --fp16 False \
        --disable_flash_attn2 False \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 3 \
        --per_device_train_batch_size $BATCH_PER_DEVICE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --image_min_pixels $((512 * 28 * 28)) \
        --image_max_pixels $((1280 * 28 * 28)) \
        --learning_rate 5e-6 \
        --vision_lr 2e-6 \
        --merger_lr 5e-6 \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --gradient_checkpointing True \
        --report_to tensorboard \
        --lazy_preprocess True \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 5 \
        --dataloader_num_workers 4
else
    echo "Unknown mode: $MODE. Use 'lora' or 'full'."
    exit 1
fi

echo "Training complete. Output: $OUTPUT_DIR"
