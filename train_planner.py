"""
DPO training for the spatial Planning Model.

This script fine-tunes a Qwen VL model (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) using
Direct Preference Optimization (DPO) so it learns to generate better
image-generation instructions for spatial reasoning.

It reuses components from Qwen-VL-Series-Finetune (dataset, trainer, params,
monkey patches) while being a self-contained entry point under spatial_planning/.

Prerequisite:
  Run generate_dpo_data.py first to produce dpo_train.json.

Usage:
  # LoRA fine-tune (recommended for starting)
  deepspeed train_planner.py \
    --deepspeed scripts/zero2.json \
    --model_id checkpoints/Qwen3-VL-4B-Instruct \
    --data_path results/dpo_data/mmsibench_xxx/dpo_train.json \
    --image_folder "" \
    --output_dir output/planner_dpo_lora \
    --lora_enable True \
    --freeze_llm True \
    --freeze_vision_tower True \
    --freeze_merger True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --bf16 True

  # Full fine-tune (needs more GPU memory)
  deepspeed train_planner.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id checkpoints/Qwen3-VL-4B-Instruct \
    --data_path results/dpo_data/mmsibench_xxx/dpo_train.json \
    --image_folder "" \
    --output_dir output/planner_dpo_full \
    --freeze_llm False \
    --freeze_vision_tower False \
    --freeze_merger False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --vision_lr 2e-6 \
    --merger_lr 5e-6 \
    --bf16 True
"""

import ast
import os
import pathlib
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

# ── Import Qwen3 classes (may not exist in older transformers) ────────────────
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLMoeForConditionalGeneration = None

# ── Add Qwen-VL-Series-Finetune to path for reuse ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNE_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "Qwen-VL-Series-Finetune")
sys.path.insert(0, os.path.join(FINETUNE_ROOT, "src"))
sys.path.insert(0, FINETUNE_ROOT)

from src.dataset import make_dpo_data_module
from src.params import DataArguments, DPOArguments, ModelArguments
from src.trainer import QwenDPOTrainer
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)

# Monkey patches for mixed-modality forward (required for DPO training)
from train.monkey_patch_forward import (
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
)

try:
    from train.monkey_patch_forward import replace_qwen3_with_mixed_modality_forward
except ImportError:
    replace_qwen3_with_mixed_modality_forward = None

try:
    from train.monkey_patch_forward import replace_qwen3_vl_moe_with_mixed_modality_forward
except ImportError:
    replace_qwen3_vl_moe_with_mixed_modality_forward = None

try:
    from train.monkey_patch_vision import replace_qwen2_5_vision
except ImportError:
    replace_qwen2_5_vision = None

# ── Helpers ──────────────────────────────────────────────────────────────────

local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None,
                             verbose=True):
    """Find all linear / embedding module names for LoRA targeting."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    set_requires_grad(model.visual.parameters(),
                      not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(),
                      not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        set_requires_grad(model.visual.deepstack_merger_list.parameters(),
                          not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(),
                      not training_args.freeze_llm)
    set_requires_grad(model.language_model.parameters(),
                      not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True
    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, DPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Validation ────────────────────────────────────────────────────────────
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("Cannot set both `nframes` and `fps`.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("`lora_enable=True` requires `freeze_llm=True`.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "vision_lora requires lora_enable."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("`vision_lora=True` requires `freeze_vision_tower=True`.")

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(
            training_args.lora_namespan_exclude
        )
    else:
        training_args.lora_namespan_exclude = []

    if training_args.lora_enable and not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ── Quantization config ───────────────────────────────────────────────────
    bnb_kwargs = {}
    if training_args.bits in [4, 8]:
        bnb_kwargs["device_map"] = {"": training_args.device}
        bnb_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(training_args.bits == 4),
            load_in_8bit=(training_args.bits == 8),
            llm_int8_skip_modules=["visual", "lm_head"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
        )

    # ── Load model ────────────────────────────────────────────────────────────
    ref_model = None
    config = AutoConfig.from_pretrained(model_args.model_id)
    model_type = config.model_type

    rank0_print(f"Detected model_type: {model_type}")

    common_kwargs = dict(
        dtype=compute_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
        ),
        **bnb_kwargs,
    )

    if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration is not None:
        if replace_qwen3_vl_moe_with_mixed_modality_forward:
            replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
        if not training_args.lora_enable:
            ref_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)

    elif model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration is not None:
        if replace_qwen3_with_mixed_modality_forward:
            replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
        if not training_args.lora_enable:
            ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)

    elif model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        if replace_qwen2_5_vision:
            replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
        if not training_args.lora_enable:
            ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)

    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
        if not training_args.lora_enable:
            ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)

    # ── Configure model ───────────────────────────────────────────────────────
    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    # ── Quantization preparation ──────────────────────────────────────────────
    if training_args.bits in [4, 8]:
        model.config.dtype = (
            torch.float32 if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Re-enable vision tower / merger if they should be trainable
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # ── Reference model ───────────────────────────────────────────────────────
    if ref_model is not None:
        ref_model.eval()
        ref_model.config.use_cache = False

    # ── Fix dtypes for quantized models ───────────────────────────────────────
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ── Log trainable parameters ──────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters      : {total_params:,}")
    rank0_print(f"Trainable parameters  : {trainable_params:,}")
    rank0_print(f"Trainable ratio       : {trainable_params / total_params:.4%}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_module = make_dpo_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    training_args.padding_value = processor.tokenizer.pad_token_id

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = QwenDPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        data_collator=dataset_module["data_collator"],
        processing_class=processor,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    rank0_print("Training complete!")
    rank0_print(f"Output saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()
