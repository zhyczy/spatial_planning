"""
GRPO training for the spatial Planning Model.

Implements Group Relative Policy Optimization (GRPO) with advantage-weighted
policy gradient loss plus optional KL penalty from a frozen reference model.

This script is a drop-in replacement for train_planner.py when using GRPO
instead of DPO.  It is called by grpo.py via torchrun, just as train_planner.py
is called by iterative_dpo.py.

Data format (produced by grpo.py's build_grpo_dataset):
  JSON list where each element is one (prompt, completion, advantage) triple:
  [
    {
      "image": ["abs/path/img1.jpg", ...],   // original scene images
      "conversations": [
        {"from": "system", "value": "<SYSTEM_PROMPT>"},
        {"from": "human",  "value": "<image>...<image>\\nQuestion: ..."},
        {"from": "gpt",    "value": "<think>...</think>\\n<instructions>...</instructions>"}
      ],
      "advantage": 1.5,          // pre-computed GRPO advantage (float)
      "group_id": "sample_001"   // optional, for logging
    },
    ...
  ]

GRPO Loss:
  L = E_i [ A_i * NLL(completion_i | prompt) ] + beta * E_i [ KL(pi || pi_ref)_i ]
  
  where:
    A_i   = (r_i - mean_r) / std_r   (pre-computed advantage, stored in data)
    NLL   = mean negative log-prob over completion tokens
    KL    = mean (log_pi - log_pi_ref) over completion tokens
    beta  = KL penalty weight (0 = pure policy gradient REINFORCE)

Usage:
  # Single-GPU (development / smoke test)
  python train_planner_grpo.py \\
    --model_id checkpoints/Qwen3-VL-4B-Instruct \\
    --data_path results/grpo/sat_xxx/iter_0/grpo_accumulated.json \\
    --output_dir output/planner_grpo_lora \\
    --lora_enable True --freeze_llm True --freeze_vision_tower True \\
    --beta 0.04 --learning_rate 1e-5 --num_train_epochs 1 --bf16 True

  # Multi-GPU via torchrun (called by grpo.py)
  torchrun --nproc_per_node 4 train_planner_grpo.py \\
    --model_id checkpoints/Qwen3-VL-4B-Instruct \\
    --data_path ...accumulated.json \\
    --output_dir ... \\
    --lora_enable True --freeze_llm True --freeze_vision_tower True \\
    --bf16 True --beta 0.04
"""

import ast
import copy
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import ujson as json
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ── Import Qwen3 classes (may not exist in older transformers) ────────────────
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLMoeForConditionalGeneration = None

# ── Add Qwen-VL-Series-Finetune to path ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNE_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "Qwen-VL-Series-Finetune")
sys.path.insert(0, os.path.join(FINETUNE_ROOT, "src"))
sys.path.insert(0, FINETUNE_ROOT)

from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    LLAVA_IMAGE_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
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


def find_target_linear_names(model, num_lora_modules=-1,
                             lora_namespan_exclude=None, verbose=True):
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            names.append(name)
    if num_lora_modules > 0:
        names = names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(names)} LoRA target modules")
    return names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    model.visual.to(dtype=compute_dtype, device=device)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)
    if hasattr(model.visual, "deepstack_merger_list"):
        set_requires_grad(
            model.visual.deepstack_merger_list.parameters(),
            not training_args.freeze_merger,
        )


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.language_model.parameters(), not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm=0, k_vis=0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True
    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO Training Arguments
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GRPOTrainingArguments(TrainingArguments):
    """Extends TrainingArguments with GRPO-specific hyperparameters."""
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL penalty coefficient for GRPO. "
                    "0 = pure policy gradient (REINFORCE with baseline). "
                    "Larger values keep the policy closer to the reference model."
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _replace_image_tokens(text: str) -> str:
    """Convert <image> (LLaVA-style) to Qwen2-VL vision tokens."""
    import re
    pattern = r"\n?" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
    replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, text)


class GRPODataset(Dataset):
    """Dataset for GRPO training with pre-computed advantages.

    Reads JSON files produced by grpo.py's build_grpo_dataset().  Each record
    contains a (prompt, completion, advantage) triple where:
      - prompt      = system turn + user turn (with image tokens)
      - completion  = one on-policy rollout output from the planner
      - advantage   = pre-normalised A_i = (r_i - mean_r) / std_r

    The __getitem__ method returns tokenized input_ids, labels (completion tokens
    only; prompt tokens masked to IGNORE_INDEX = -100), pixel_values (processed
    images), and the scalar advantage.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        data_args: DataArguments,
        model_id: str,
    ):
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = data_path

        self.processor = processor
        self.data_args = data_args
        self.model_id = model_id

        # Image processing parameters
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

        # Qwen3 uses 16×16 patches; Qwen2 uses 14×14
        if "Qwen3" in model_id:
            self.image_patch_size = 16
        else:
            self.image_patch_size = 14

    def __len__(self):
        return len(self.records)

    def _get_image_info(self, image_path: str):
        """Load and preprocess a single image using qwen_vl_utils."""
        from src.dataset.data_utils import get_image_info
        return get_image_info(
            image_path,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_resized_w,
            self.image_resized_h,
            self.image_patch_size,
        )

    def __getitem__(self, i) -> Dict:
        rec = self.records[i]
        advantage = float(rec.get("advantage", 0.0))

        # ── Load images ───────────────────────────────────────────────────
        image_files = rec.get("image", [])
        if isinstance(image_files, str):
            image_files = [image_files]
        image_folder = self.data_args.image_folder or ""

        images = []
        for img_file in image_files:
            if not os.path.exists(img_file) and not img_file.startswith("http"):
                img_file = os.path.join(image_folder, img_file)
            try:
                images.append(self._get_image_info(img_file))
            except Exception as e:
                rank0_print(f"[GRPODataset] Warning: failed to load image {img_file}: {e}")

        # ── Build conversation text ───────────────────────────────────────
        # Conversation format: system + human + gpt
        # (fallback: if no system turn, prepend SYSTEM_MESSAGE)
        conversations = rec.get("conversations", [])
        system_text = None
        human_text = None
        gpt_text = None

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "system":
                system_text = value
            elif role in ("human", "user"):
                human_text = value
            elif role in ("gpt", "assistant"):
                gpt_text = value

        if human_text is None:
            rank0_print(f"[GRPODataset] Warning: record {i} has no human turn, skipping.")
            human_text = ""

        if gpt_text is None:
            gpt_text = ""

        # ── Tokenize via processor's apply_chat_template ──────────────────
        # We build two token sequences:
        #   prompt_ids   = system + human (no response)
        #   full_ids     = system + human + response
        # labels[len(prompt):] = completion token ids
        # labels[:len(prompt)]  = IGNORE_INDEX

        processor = self.processor
        is_qwen3 = "Qwen3" in self.model_id

        # Build message dicts for apply_chat_template
        messages_prompt = []
        if system_text:
            messages_prompt.append({"role": "system", "content": system_text})
        messages_prompt.append(
            {"role": "user", "content": _replace_image_tokens(human_text)}
        )

        messages_full = messages_prompt + [
            {"role": "assistant", "content": gpt_text}
        ]

        try:
            # Prompt only (used to locate the start of the completion)
            prompt_text = processor.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Full conversation (prompt + completion)
            full_text = processor.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            rank0_print(f"[GRPODataset] apply_chat_template failed for record {i}: {e}")
            prompt_text = ""
            full_text = gpt_text

        # Tokenize
        has_images = len(images) > 0

        # For image-conditioned tokenization we need to pass pixel_values through
        # the processor so it can insert the correct number of image patches.
        # We use the "images" argument which the processor will handle via
        # process_vision_info under the hood.
        try:
            prompt_enc = processor(
                text=[prompt_text],
                images=images if has_images else None,
                return_tensors="pt",
                add_special_tokens=False,
            )
            full_enc = processor(
                text=[full_text],
                images=images if has_images else None,
                return_tensors="pt",
                add_special_tokens=False,
            )
        except Exception as e:
            rank0_print(f"[GRPODataset] Processor failed for record {i}: {e}")
            return self.__getitem__(0)

        prompt_ids = prompt_enc["input_ids"][0]          # [prompt_len]
        full_ids = full_enc["input_ids"][0]              # [prompt_len + comp_len]

        prompt_len = len(prompt_ids)

        # Labels: mask prompt tokens with IGNORE_INDEX so loss is only on completion
        labels = full_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        data = {
            "input_ids": full_ids,
            "labels": labels,
            "advantage": torch.tensor(advantage, dtype=torch.float32),
        }

        # ── Pixel values & image grid ─────────────────────────────────────
        if has_images and "pixel_values" in full_enc:
            data["pixel_values"] = full_enc["pixel_values"]
        if has_images and "image_grid_thw" in full_enc:
            data["image_grid_thw"] = full_enc["image_grid_thw"]
        if has_images and "pixel_values_videos" in full_enc:
            data["pixel_values_videos"] = full_enc["pixel_values_videos"]
        if has_images and "video_grid_thw" in full_enc:
            data["video_grid_thw"] = full_enc["video_grid_thw"]

        return data


def grpo_collate_fn(features: list):
    """Collate GRPO examples into a padded batch.

    Handles variable-length input_ids and labels by left-padding (attention mask
    is derived from non-pad positions).  Pixel values are concatenated since the
    model expects a flat list of image patches.
    """
    if not features:
        return {}

    # Determine maximum sequence length in the batch
    max_len = max(f["input_ids"].size(0) for f in features)
    pad_id = 0  # Qwen tokenizers use 0 as pad token

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_advantages = []
    pixel_values_list = []
    image_grid_thw_list = []

    for f in features:
        ids = f["input_ids"]
        lbls = f["labels"]
        seq_len = ids.size(0)
        pad_len = max_len - seq_len

        # Left-pad so the computation graph stays consistent with flash-attn
        padded_ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
        padded_lbls = torch.cat(
            [torch.full((pad_len,), IGNORE_INDEX, dtype=lbls.dtype), lbls]
        )
        attn_mask = torch.cat(
            [torch.zeros(pad_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)]
        )

        batch_input_ids.append(padded_ids)
        batch_labels.append(padded_lbls)
        batch_attention_mask.append(attn_mask)
        batch_advantages.append(f["advantage"])

        if "pixel_values" in f:
            pixel_values_list.append(f["pixel_values"])
        if "image_grid_thw" in f:
            image_grid_thw_list.append(f["image_grid_thw"])

    result = {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
        "advantage": torch.stack(batch_advantages),
    }

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO Trainer
# ══════════════════════════════════════════════════════════════════════════════

class GRPOPlannerTrainer(Trainer):
    """HuggingFace Trainer subclass that implements the GRPO loss.

    GRPO Loss:
        L = mean_i [A_i * NLL(c_i | x)] + beta * mean_i [KL(pi_theta || pi_ref)_i]

    where:
        A_i       = advantage stored in each training example (pre-normalised)
        NLL(c|x)  = mean negative log-prob over completion tokens
        KL_i      = mean token-level (log_pi - log_pi_ref) over completion tokens
        beta      = KL penalty coefficient (training_args.beta)

    When beta=0 the loss reduces to REINFORCE with advantage baseline (no ref model
    is loaded, saving memory).

    The reference model is always the frozen base model (identical to the one used
    to initialise the policy LoRA adapter).  This mirrors the DPO setup where the
    reference is the base model.
    """

    def __init__(self, ref_model=None, **kwargs):
        super().__init__(**kwargs)
        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        self.ref_model = ref_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the GRPO loss: advantage-weighted NLL + optional KL penalty."""

        # Pop the advantage tensor (not a standard model input)
        advantages = inputs.pop("advantage", None)  # [B]

        # ── Forward pass through policy model ────────────────────────────
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        labels = inputs["labels"]  # [B, T]  (-100 for prompt/pad positions)

        # Autoregressive: predict token t+1 from token t
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()       # [B, T-1]

        # Mask to completion tokens only
        comp_mask = (shift_labels != IGNORE_INDEX).float()  # [B, T-1]

        # Per-token NLL (CrossEntropy with reduction="none")
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.clamp(min=0).view(-1),  # replace -100 to avoid index OOB
        ).view(shift_logits.size(0), -1)  # [B, T-1]

        # Zero out non-completion tokens
        per_token_nll = per_token_nll * comp_mask

        # Mean NLL per sample (over completion tokens)
        n_comp_tokens = comp_mask.sum(dim=1).clamp(min=1)  # [B]
        mean_nll = per_token_nll.sum(dim=1) / n_comp_tokens  # [B]  (= -mean_log_prob)

        # ── Advantage-weighted policy gradient loss ───────────────────────
        if advantages is not None:
            adv = advantages.to(mean_nll.device).float()  # [B]
            # L = mean_i [A_i * NLL_i]
            # Minimising this drives NLL down for positive-advantage completions
            # and up for negative-advantage completions.
            pg_loss = (adv * mean_nll).mean()
        else:
            # Fallback: standard SFT loss (should not happen during GRPO training)
            pg_loss = mean_nll.mean()

        loss = pg_loss

        # ── KL penalty from reference model ──────────────────────────────
        if self.ref_model is not None and self.args.beta > 0:
            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits[:, :-1, :].contiguous()  # [B, T-1, V]

            # Token-level log-probs
            policy_log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)       # [B, T-1, V]

            # Gather log-probs along the actual token dimension
            comp_token_ids = shift_labels.clamp(min=0).unsqueeze(-1)  # [B, T-1, 1]
            policy_lp = policy_log_probs.gather(-1, comp_token_ids).squeeze(-1)  # [B, T-1]
            ref_lp = ref_log_probs.gather(-1, comp_token_ids).squeeze(-1)        # [B, T-1]

            # KL approximation: E_t[log_pi(y_t) - log_pi_ref(y_t)] over completion tokens
            kl_per_token = (policy_lp - ref_lp) * comp_mask  # [B, T-1]
            kl_mean = kl_per_token.sum(dim=1) / n_comp_tokens  # [B]
            kl_loss = kl_mean.mean()

            loss = pg_loss + self.args.beta * kl_loss

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, **kwargs):
        """Save model, honouring LoRA adapter extraction for ZeRO-3."""
        if output_dir is None:
            output_dir = self.args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=True
            )
            if self.args.local_rank in (0, -1):
                self.model.config.save_pretrained(output_dir)
                self.model.save_pretrained(output_dir, state_dict=state_dict)
                self.processing_class.save_pretrained(output_dir)
                torch.save(
                    non_lora_state_dict,
                    os.path.join(output_dir, "non_lora_state_dict.bin"),
                )
        else:
            safe_save_model_for_hf_trainer(self, output_dir=output_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ── Validation ────────────────────────────────────────────────────────────
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("Cannot set both `nframes` and `fps`.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("`lora_enable=True` requires `freeze_llm=True`.")

    if not training_args.lora_enable:
        assert not getattr(training_args, "vision_lora", False), \
            "vision_lora requires lora_enable."

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(
            training_args.lora_namespan_exclude
        )
    else:
        training_args.lora_namespan_exclude = []

    if training_args.lora_enable and not getattr(training_args, "vision_lora", False):
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ── Quantization ──────────────────────────────────────────────────────────
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

    # ── Load policy model ─────────────────────────────────────────────────────
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
    elif model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration is not None:
        if replace_qwen3_with_mixed_modality_forward:
            replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
    elif model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        if replace_qwen2_5_vision:
            replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id, **common_kwargs)

    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.lora_enable or getattr(training_args, "vision_lora", False):
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    # ── Quantization preparation ──────────────────────────────────────────────
    if training_args.bits in [4, 8]:
        model.config.dtype = compute_dtype
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
            model.to(torch.bfloat16 if training_args.bf16 else torch.float16)
        rank0_print("Adding LoRA adapters to the model...")
        model = get_peft_model(model, peft_config)

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    # ── Reference model (for KL penalty) ─────────────────────────────────────
    # Only load the reference model when KL penalty is enabled (beta > 0).
    # For LoRA mode the unfinetuned base model is always the reference; we
    # reload it fresh without LoRA adapters.
    ref_model = None
    if training_args.beta > 0:
        rank0_print(f"Loading reference model (beta={training_args.beta}) ...")
        if model_type == "qwen3_vl_moe" and Qwen3VLMoeForConditionalGeneration is not None:
            ref_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)
        elif model_type == "qwen3_vl" and Qwen3VLForConditionalGeneration is not None:
            ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)
        elif model_type == "qwen2_5_vl":
            ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)
        else:
            ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_id, **common_kwargs)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        rank0_print("Reference model loaded and frozen.")

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_args.model_id)
    processor.image_processor.do_resize = False

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_args.model_id,
    )
    rank0_print(f"Loaded {len(train_dataset)} GRPO training examples.")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOPlannerTrainer(
        ref_model=ref_model,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=grpo_collate_fn,
        processing_class=processor,
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
        if local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    rank0_print("GRPO training complete!")
    rank0_print(f"Output saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()
