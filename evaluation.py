"""
evaluation.py

Two-method QA evaluation:

  Method A — baseline
      Model : Qwen3.5-VL loaded as AutoModelForImageTextToText
      Input : original dataset images + question (standard VLM inference)

  Method B — correspondence
      Model : SpaForConditionalGeneration (4D M-RoPE) + LoRA adapter from
              --correspondence_ckpt, evaluated WITHOUT the PoseRegressionHead
      Input : original dataset images + question
      3D pos: CoordEstimator (MapAnything) estimates per-pixel XYZ for each
              image, aligns all frames to the first-frame camera coordinate
              system, and passes image_xyz to the 4D M-RoPE position embedding.

Run a single method or both:
  --method baseline          → only Method A
  --method correspondence    → only Method B
  --method both              → both (default)

Usage
-----
# baseline only
python evaluation.py \\
    --method baseline \\
    --model_path checkpoints/Qwen3.5-4B \\
    --data_dir  datasets/evaluation/MMSIBench

# correspondence only
python evaluation.py \\
    --method correspondence \\
    --model_path            checkpoints/Qwen3.5-4B \\
    --correspondence_ckpt   train_records/correspondence/final \\
    --data_dir              datasets/evaluation/MMSIBench

# both
python evaluation.py \\
    --model_path            checkpoints/Qwen3.5-4B \\
    --correspondence_ckpt   train_records/correspondence/final \\
    --data_dir              datasets/evaluation/MMSIBench

Multi-GPU (auto-shards across all visible GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --method both ...

Smoke test:
    python evaluation.py --limit 6 --method both ...
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ── sys.path: ensure spatial_planning/ root is importable ──────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
QUESTION_TEMPLATE = "{Question}"

# ── Non-thinking mode: answer first, then reasoning ──────────────────────────
ANSWER_INSTRUCTION = (
    "First output your answer as <answer>X</answer> where X is the option letter, "
    "then explain your reasoning."
)
EVAL_SYSTEM_PROMPT = (
    "You are a spatial reasoning expert. "
    "IMPORTANT: Begin your response by outputting your answer in the format "
    "<answer>X</answer> where X is the option letter (e.g. <answer>A</answer>). "
    "Then provide your step-by-step reasoning."
)

# ── Thinking mode: model reasons first, then outputs answer ──────────────────
ANSWER_INSTRUCTION_THINKING = (
    "After your reasoning, output your final answer as <answer>X</answer> "
    "where X is the option letter (e.g. <answer>A</answer>)."
)
EVAL_SYSTEM_PROMPT_THINKING = (
    "You are a spatial reasoning expert. "
    "Think step by step about the question. "
    "After your reasoning, output your final answer in the format "
    "<answer>X</answer> where X is the option letter (e.g. <answer>A</answer>)."
)


# ===========================================================================
# Dataset utilities  (unchanged from previous version)
# ===========================================================================

def load_dataset(
    data_dir: Path,
    limit: Optional[int] = None,
    dataset: str = "mmsibench",
) -> List[Dict[str, Any]]:
    """Load evaluation dataset.

    Supports: mmsibench | mindcube | sat | vsibench
    Image paths are resolved to absolute paths.
    """
    samples: List[Dict[str, Any]] = []

    if dataset == "mmsibench":
        json_file = data_dir / "data" / "test_data_final.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {json_file}\n"
                "Run datasets/evaluation/MMSIBench/download.py first."
            )
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            local_images = item.get("local_images", [])
            image_paths = [str((data_dir / p).resolve()) for p in local_images]
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("type", "unknown"),
                "thought": item.get("thought_gt", ""),
                "data_dir": str(data_dir),
            })

    elif dataset == "mindcube":
        jsonl_file = data_dir / "MindCube_tinybench.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            image_paths = [str((data_dir / p).resolve()) for p in item.get("images", [])]
            category = item.get("category", [])
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("gt_answer", ""),
                "category": category[0] if category else "unknown",
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset in ("sat", "sat_real"):
        # SAT (Spatial Awareness Tasks) — real image dataset.
        # Keys: database_idx, question_type, question, answer_choices, correct_answer, img_paths
        json_file = data_dir / "test.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        _letters = "ABCDEFGHIJ"
        for item in raw:
            img_paths = item.get("img_paths", item.get("images", []))
            image_paths = [str((data_dir / p).resolve()) for p in img_paths]
            choices = item.get("answer_choices", [])
            correct = item.get("correct_answer", item.get("answer", ""))
            # Format as "A. choice\nB. choice\n..." and store letter as answer
            if choices:
                formatted = "\n".join(
                    f"{_letters[i]}. {c}" for i, c in enumerate(choices)
                )
                question_text = item.get("question", "") + "\n" + formatted
                try:
                    answer_letter = _letters[choices.index(correct)]
                except ValueError:
                    answer_letter = correct  # fallback: store raw text
            else:
                question_text = item.get("question", "")
                answer_letter = correct
            samples.append({
                "index": item.get("database_idx", item.get("id", len(samples))),
                "image": image_paths,
                "question": question_text,
                "answer": answer_letter,
                "category": item.get("question_type", item.get("type", "unknown")),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset == "vsibench":
        jsonl_file = data_dir / "test.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            image_paths = [str((data_dir / p).resolve()) for p in item.get("images", [])]
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("gt_answer", "")),
                "category": item.get("type", "unknown"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset in ("sparbench_multi_view", "sparbench_single_view", "sparbench_mv"):
        # SPARBench — images are base64-encoded JPEG strings embedded in the JSON.
        # Keys: id, img_type, format_type, task, source, question, answer, images
        import base64, tempfile
        if dataset == "sparbench_mv":
            suffix = "mv"
        elif dataset == "sparbench_multi_view":
            suffix = "multi_view"
        else:
            suffix = "single_view"
        json_file = data_dir / f"sparbench_{suffix}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        # Create a single temp dir per dataset load; images persist for the process lifetime.
        _tmp_dir = Path(tempfile.mkdtemp(prefix=f"sparbench_{suffix}_"))
        for item in raw:
            b64_images = item.get("images", [])
            image_paths = []
            item_id = item.get("id", len(samples))
            for img_idx, b64 in enumerate(b64_images):
                img_bytes = base64.b64decode(b64)
                img_path = _tmp_dir / f"{item_id}_{img_idx}.jpg"
                img_path.write_bytes(img_bytes)
                image_paths.append(str(img_path))
            samples.append({
                "index": item_id,
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("task", "unknown"),
                "format_type": item.get("format_type", "select"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Choose: mmsibench | mindcube | sat | sat_real | vsibench | "
            "sparbench_multi_view | sparbench_single_view | sparbench_mv"
        )

    return samples


def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [dataset[s : s + chunk_size] for s in range(0, len(dataset), chunk_size)]


# ===========================================================================
# Answer extraction
# ===========================================================================

def extract_answer_letter(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    m = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(
        r"(?:the\s+)?(?:answer|option|choice)\s+(?:is\s+)?[:\s]*([A-Za-z])\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return ""


def extract_answer_number(text: str) -> str:
    """Extract a numeric answer from <answer>...</answer> tags (for fill-format)."""
    if not text or not isinstance(text, str):
        return ""
    m = re.search(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: last standalone number in text
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


# ===========================================================================
# Model loading
# ===========================================================================

def load_baseline_model(
    model_path: str,
    device: str = "cuda:0",
) -> Tuple[Any, Any]:
    """Load standard Qwen3.5-VL for baseline evaluation."""
    logger = logging.getLogger(__name__)
    logger.info(f"[baseline] Loading from '{model_path}' onto {device}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    attn_impls = ["sdpa"]
    try:
        import fla  # noqa: F401
        logger.info("flash-linear-attention available (linear-attn fast path active)")
    except Exception:
        pass

    last_exc = None
    model = None
    for attn_impl in attn_impls:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation=attn_impl,
                local_files_only=True,
            )
            logger.info(f"[baseline] attn_implementation={attn_impl}")
            break
        except Exception as exc:
            last_exc = exc
            continue

    if model is None:
        raise RuntimeError(f"Failed to load baseline model from {model_path}") from last_exc

    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    logger.info(f"[baseline] Model ready on {next(model.parameters()).device}")
    return model, processor


def load_spa_model(
    base_model_path: str,
    ckpt_path: str,
    device: str = "cuda:0",
) -> Tuple[Any, Any]:
    """Load SpaForConditionalGeneration with LoRA adapter for correspondence evaluation.

    Steps:
      1. Load config and set mrope_section to 4 equal parts (4D M-RoPE).
      2. Load base SpaForConditionalGeneration from base_model_path.
      3. Load processor/tokenizer from ckpt_path (has <pose> in vocab).
      4. Resize embedding table to match the saved tokenizer.
      5. Load PEFT LoRA adapter from ckpt_path, then merge into base weights.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[correspondence] Loading SPA model: base={base_model_path}  ckpt={ckpt_path}")

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from peft import PeftModel
    from models.spa_emb import SpaForConditionalGeneration

    # 1. Config: switch to 4D mrope_section
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    section_size = sum(orig_section) // 4
    config.text_config.rope_scaling["mrope_section"] = [section_size] * 4
    logger.info(f"[correspondence] mrope_section: {orig_section} → {[section_size]*4}")

    # 2. Load base model
    spa = SpaForConditionalGeneration.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # 3. Processor from base model; swap in the checkpoint tokenizer (<pose> token added during training)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    processor.tokenizer = tokenizer

    # 4. Resize embedding table to match saved tokenizer vocab
    new_vocab = len(tokenizer)
    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    if new_vocab > old_vocab:
        spa.resize_token_embeddings(new_vocab)
        logger.info(f"[correspondence] Embedding: {old_vocab} → {new_vocab}")

    # 5. Load LoRA adapter and merge
    spa = PeftModel.from_pretrained(spa, ckpt_path, is_trainable=False)
    spa = spa.merge_and_unload()
    logger.info("[correspondence] LoRA adapter merged.")

    spa = spa.to(device).eval()
    logger.info(f"[correspondence] Model ready on {next(spa.parameters()).device}")
    return spa, processor


# ===========================================================================
# 3D coordinate estimation helpers
# ===========================================================================

def _resize_xyz(
    xyz: np.ndarray,
    target_h: int,
    target_w: int,
    valid: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Average-pool per-pixel XYZ map to (target_h, target_w, 3).

    Mirrors resize_xyz() in train_correspondence.py exactly so that the
    image_xyz format matches what the model was trained with.
    """
    H, W = xyz.shape[:2]
    xyz_f = xyz.astype(np.float32)

    if valid is None:
        valid = np.ones((H, W), dtype=bool)

    stride_h = H // target_h
    stride_w = W // target_w
    H_crop = target_h * stride_h
    W_crop = target_w * stride_w
    xyz_f = xyz_f[:H_crop, :W_crop]
    valid_f = valid[:H_crop, :W_crop].astype(np.float32)

    xyz_blocks   = xyz_f.reshape(target_h, stride_h, target_w, stride_w, 3)
    valid_blocks = valid_f.reshape(target_h, stride_h, target_w, stride_w)

    xyz_sum   = (xyz_blocks * valid_blocks[..., None]).sum(axis=(1, 3))  # (th, tw, 3)
    valid_cnt = valid_blocks.sum(axis=(1, 3))                            # (th, tw)
    denom     = np.maximum(valid_cnt, 1)[..., None]
    xyz_mean  = xyz_sum / denom
    xyz_mean[valid_cnt == 0] = 0.0

    return torch.from_numpy(xyz_mean)  # (target_h, target_w, 3)


def load_precomputed_coords(item: Dict[str, Any]) -> Optional[List[Dict]]:
    """Load precomputed per-pixel 3D coordinates from 3d_results/<index>/view_XXXX/.

    Returns a list of dicts (one per image view) with keys:
        pts3d       : np.ndarray (H, W, 3)
        camera_pose : np.ndarray (4, 4)
        mask        : np.ndarray (H, W) bool
    Returns None if the 3d_results directory for this sample does not exist.
    """
    data_dir = item.get("data_dir")
    index = item.get("index")
    if data_dir is None or index is None:
        return None

    sample_dir = Path(data_dir) / "3d_results" / str(index)
    if not sample_dir.exists():
        return None

    view_dirs = sorted(sample_dir.glob("view_*"))
    if not view_dirs:
        return None

    results = []
    for vd in view_dirs:
        pts3d_path = vd / "pts3d.npy"
        mask_path  = vd / "mask.npy"
        pose_path  = vd / "camera_pose.npy"
        if not pts3d_path.exists():
            continue
        pts3d = np.load(str(pts3d_path))          # (H, W, 3)
        mask  = np.load(str(mask_path)).astype(bool) if mask_path.exists() else np.ones(pts3d.shape[:2], dtype=bool)
        pose  = np.load(str(pose_path)) if pose_path.exists() else np.eye(4, dtype=np.float64)
        results.append({"pts3d": pts3d, "camera_pose": pose, "mask": mask})

    return results if results else None


def build_image_xyz(
    coord_results: List[Dict],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> List[torch.Tensor]:
    """Convert CoordEstimator output to image_xyz list for SpaForConditionalGeneration.

    1. Aligns all frames to the first-frame camera coordinate system
       (T0_inv = inv(camera_pose[0])).
    2. Resizes each per-pixel XYZ map to the LLM patch grid resolution
       (grid_thw[i, 1:] // spatial_merge_size).

    Parameters
    ----------
    coord_results : list of dicts from CoordEstimator.estimate(), one per image.
        Each dict has: pts3d (H,W,3), camera_pose (4,4), mask (H,W).
    image_grid_thw : (N, 3) int tensor from the processor output.
    spatial_merge_size : from vision_config (typically 2).

    Returns
    -------
    List of N tensors, each (llm_H_i, llm_W_i, 3) float32.
    """
    N = image_grid_thw.shape[0]

    # World-to-first-frame transform
    T0_inv = np.linalg.inv(coord_results[0]["camera_pose"].astype(np.float64))  # (4,4)

    xyz_list: List[torch.Tensor] = []
    for k in range(N):
        thw_k = image_grid_thw[k]                      # (T, H_patches, W_patches)
        llm_h = int(thw_k[1]) // spatial_merge_size
        llm_w = int(thw_k[2]) // spatial_merge_size

        if k < len(coord_results):
            r    = coord_results[k]
            pts  = r["pts3d"].astype(np.float64)        # (H, W, 3)
            mask = r["mask"]                            # (H, W) bool
            H, W = pts.shape[:2]

            # Transform pts to first-frame coords
            pts_flat = pts.reshape(-1, 3)
            ones     = np.ones((H * W, 1), dtype=np.float64)
            pts_hom  = np.concatenate([pts_flat, ones], axis=1)  # (H*W, 4)
            pts_ff   = (T0_inv @ pts_hom.T).T[:, :3].reshape(H, W, 3).astype(np.float32)

            xyz_list.append(_resize_xyz(pts_ff, llm_h, llm_w, valid=mask))
        else:
            # Fallback: zero coords (no coord estimate available for this frame)
            xyz_list.append(torch.zeros(llm_h, llm_w, 3))

    return xyz_list


# ===========================================================================
# Inference helpers — baseline
# ===========================================================================

def _build_user_message(item: Dict[str, Any], thinking: bool = False) -> Dict:
    image_contents = [{"type": "image", "image": p} for p in item["image"]]
    instruction = ANSWER_INSTRUCTION_THINKING if thinking else ANSWER_INSTRUCTION
    text = (
        f"{QUESTION_TEMPLATE.format(Question=item['question'])}\n"
        f"{instruction}"
    )
    return {"role": "user", "content": image_contents + [{"type": "text", "text": text}]}


def prepare_batch_baseline(
    batch_data: List[Dict],
    processor: Any,
    thinking: bool = False,
) -> Tuple[Dict, List[str]]:
    """Tokenise a batch for the standard Qwen3.5-VL baseline."""
    from qwen_vl_utils import process_vision_info

    system_prompt = EVAL_SYSTEM_PROMPT_THINKING if thinking else EVAL_SYSTEM_PROMPT
    batch_messages = [
        [{"role": "system", "content": system_prompt},
         _build_user_message(item, thinking=thinking)]
        for item in batch_data
    ]

    if thinking:
        # enable_thinking=True lets the model emit <think>...</think> before answering.
        # Do NOT append "<answer>" — that would suppress the thinking block.
        prompts_text = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            for msgs in batch_messages
        ]
    else:
        # Append "<answer>" to force the model to start its output with the answer.
        prompts_text = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ) + "<answer>"
            for msgs in batch_messages
        ]

    all_image_inputs, all_video_inputs = [], []
    for msgs in batch_messages:
        imgs, vids = process_vision_info(msgs)
        all_image_inputs.extend(imgs or [])
        all_video_inputs.extend(vids or [])

    batch_inputs = processor(
        text=prompts_text,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    return batch_inputs, prompts_text


def run_inference_baseline(
    batch_inputs: Dict,
    model: Any,
    processor: Any,
    max_new_tokens: int = 512,
) -> List[str]:
    batch_inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    trimmed = [
        out[len(inp):]
        for inp, out in zip(batch_inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


# ===========================================================================
# Inference helpers — correspondence (SpaForConditionalGeneration)
# ===========================================================================

def prepare_batch_spa(
    item: Dict[str, Any],
    processor: Any,
    spatial_merge_size: int,
    use_coord: bool,
    coord_scale: float,
    thinking: bool = False,
) -> Tuple[Dict, str, List[torch.Tensor]]:
    """Tokenise one sample and build image_xyz for SPA model inference.

    Returns
    -------
    inputs     : processor output dict (input_ids, attention_mask, pixel_values, image_grid_thw)
    prompt_str : prompt text (used for logging)
    image_xyz  : list of (llm_H, llm_W, 3) tensors, or None if use_coord=False
    """
    from qwen_vl_utils import process_vision_info

    system_prompt = EVAL_SYSTEM_PROMPT_THINKING if thinking else EVAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        _build_user_message(item, thinking=thinking),
    ]
    if thinking:
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        prompt_text = (
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ) + "<answer>"
        )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=False,
    )

    # Build image_xyz from precomputed 3d_results
    image_xyz: Optional[List[torch.Tensor]] = None
    if use_coord and item["image"]:
        try:
            coord_results = load_precomputed_coords(item)
            image_grid_thw = inputs.get("image_grid_thw")
            if coord_results is not None and image_grid_thw is not None and len(coord_results) > 0:
                image_xyz = build_image_xyz(
                    coord_results,
                    image_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )
            else:
                logging.getLogger(__name__).warning(
                    f"No precomputed 3D coords for sample {item.get('index')} "
                    f"(data_dir={item.get('data_dir')}). Falling back to zero xyz."
                )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"Failed to load precomputed coords for sample {item.get('index')}: {exc}. "
                "Falling back to zero xyz."
            )

    return inputs, prompt_text, image_xyz


def run_inference_spa(
    inputs: Dict,
    image_xyz: Optional[List[torch.Tensor]],
    model: Any,
    processor: Any,
    max_new_tokens: int = 512,
    coord_scale: float = 100.0,
) -> str:
    """Run generation with SpaForConditionalGeneration.

    image_xyz (if provided) is passed as a kwarg to model.generate() which
    forwards it to forward() → get_rope_index() for 4D M-RoPE.
    The PoseRegressionHead is NOT used; we only call generate() for QA.
    """
    device = next(model.parameters()).device
    inputs_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Move image_xyz to device
    xyz_on_device = None
    if image_xyz is not None:
        xyz_on_device = [xyz.to(device) for xyz in image_xyz]

    # Pre-compute 5D position_ids (seq, t, x, y, z) so that generate()'s
    # _prepare_position_ids_for_generation is bypassed.  Without this,
    # SpaForConditionalGeneration.forward uses *args/**kwargs and
    # inspect.signature can't see "position_ids", so accepts_position_ids=False
    # and the model falls back to compute_3d_position_ids() → 3D → shape
    # mismatch in Spa4DRotaryEmbedding ("4 vs 3" error).
    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids=inputs_dev["input_ids"],
            mm_token_type_ids=inputs_dev["mm_token_type_ids"],
            image_grid_thw=inputs_dev.get("image_grid_thw"),
            video_grid_thw=inputs_dev.get("video_grid_thw"),
            attention_mask=inputs_dev.get("attention_mask"),
            image_xyz=xyz_on_device,
            coord_scale=coord_scale,
        )

    gen_kwargs: Dict[str, Any] = dict(
        **inputs_dev,
        position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
        coord_scale=coord_scale,
    )
    # mm_token_type_ids was already consumed to pre-compute position_ids above.
    # HuggingFace's _validate_model_kwargs would reject it because it's not in
    # SpaForConditionalGeneration.prepare_inputs_for_generation's signature.
    gen_kwargs.pop("mm_token_type_ids", None)
    if xyz_on_device is not None:
        gen_kwargs["image_xyz"] = xyz_on_device

    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    trimmed = generated_ids[0][inputs_dev["input_ids"].shape[1]:]
    return processor.decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )


# ===========================================================================
# Result packaging
# ===========================================================================

def _make_result(
    item: Dict,
    output: str,
    prompt: str,
    method: str,
    thinking: bool = False,
) -> Dict:
    # Non-thinking: model output starts right after the "<answer>" prefix we
    # injected into the prompt, so we prepend it back for a complete tag.
    # Thinking: model generates the full response (including <think>...</think>
    # and <answer>X</answer>) — no prefix needed.
    full_output = output if thinking else "<answer>" + output
    fmt = item.get("format_type", "select")
    if fmt == "fill":
        prediction = extract_answer_number(full_output)
    else:
        prediction = extract_answer_letter(full_output)
    return {
        "method": method,
        "index": item.get("index", ""),
        "category": item.get("category", "unknown"),
        "format_type": fmt,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "prediction": prediction,
        "output": full_output,
        "thought_gt": item.get("thought", ""),
        "image_paths": item["image"],
        "prompt": prompt,
    }


def _error_result(item: Dict, exc: Exception, method: str) -> Dict:
    return {
        "method": method,
        "index": item.get("index", ""),
        "category": item.get("category", "unknown"),
        "format_type": item.get("format_type", "select"),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "prediction": "",
        "output": f"ERROR: {exc}",
        "thought_gt": item.get("thought", ""),
        "image_paths": item.get("image", []),
        "prompt": "",
    }


# ===========================================================================
# Metrics
# ===========================================================================

def _mra_score(pred_str: str, gt_str: str) -> float:
    """Mean Relative Accuracy for a single numeric prediction.
    MRA = max(0, 1 - |pred - gt| / |gt|).  Returns 0 if unparseable.
    """
    try:
        pred = float(pred_str)
        gt   = float(gt_str)
        if gt == 0:
            return 1.0 if pred == 0 else 0.0
        return max(0.0, 1.0 - abs(pred - gt) / abs(gt))
    except (ValueError, TypeError):
        return 0.0


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    total = len(results)
    correct = 0
    cat_correct: dict = defaultdict(float)
    cat_total: dict = defaultdict(int)
    for r in results:
        pred = r.get("prediction", "")
        gt   = r.get("answer", "")
        cat  = r.get("category", "unknown")
        fmt  = r.get("format_type", "select")
        cat_total[cat] += 1
        if fmt == "fill":
            score = _mra_score(pred, gt)
            correct += score
            cat_correct[cat] += score
        else:
            if pred.lower().strip() == gt.lower().strip():
                correct += 1
                cat_correct[cat] += 1
    cat_accuracy = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": correct,
        "category_accuracy": cat_accuracy,
        "category_counts": dict(cat_total),
    }


def log_metrics(metrics: Dict, label: str, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info(f"RESULTS — {label}")
    logger.info("=" * 60)
    logger.info(f"  Total   : {metrics['total_samples']}")
    logger.info(f"  Correct : {metrics['correct_samples']}")
    logger.info(f"  Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info("  Per-category accuracy:")
    for cat, acc in sorted(metrics["category_accuracy"].items()):
        n = metrics["category_counts"].get(cat, 0)
        logger.info(f"    {cat:35s}: {acc:6.2%}  ({n} samples)")
    logger.info("=" * 60)


# ===========================================================================
# Core evaluation loop (runs on one GPU)
# ===========================================================================

def evaluate(
    data: List[Dict],
    method: str,
    # baseline args
    baseline_model_path: str,
    # correspondence args
    spa_base_model_path: str,
    correspondence_ckpt: Optional[str],
    use_coord: bool,
    coord_scale: float,
    # common
    max_new_tokens: int,
    output_dir: Path,
    device: str = "cuda:0",
    thinking: bool = False,
) -> Dict[str, List[Dict]]:
    """Run evaluation for the requested method(s) on *data*.

    Returns dict mapping method name → list of result dicts.
    """
    logger = logging.getLogger(__name__)

    run_baseline = method in ("baseline", "both")
    run_correspondence = method in ("correspondence", "both")

    # Lazy-load only what we need
    baseline_model = baseline_proc = None
    spa_model = spa_proc = None

    if run_baseline:
        baseline_model, baseline_proc = load_baseline_model(baseline_model_path, device)

    if run_correspondence:
        if correspondence_ckpt is None:
            raise ValueError("--correspondence_ckpt is required for method='correspondence'/'both'")
        spa_model, spa_proc = load_spa_model(spa_base_model_path, correspondence_ckpt, device)

        # Resolve spatial_merge_size from base model config
        cfg_path = Path(spa_base_model_path) / "config.json"
        with open(cfg_path) as f:
            _vcfg = json.load(f).get("vision_config", {})
        spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
        logger.info(f"[correspondence] spatial_merge_size={spatial_merge_size}")

        if not use_coord:
            logger.info("[correspondence] --no_coord set: using zero image_xyz")

    results_map: Dict[str, List[Dict]] = {
        "baseline": [],
        "correspondence": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(data, desc=f"[{device}]"):

        # ---- Baseline ----
        if run_baseline:
            try:
                inputs, prompt = prepare_batch_baseline([item], baseline_proc,
                                                         thinking=thinking)
                outputs = run_inference_baseline(inputs, baseline_model, baseline_proc,
                                                 max_new_tokens)
                results_map["baseline"].append(
                    _make_result(item, outputs[0], prompt[0], "baseline",
                                 thinking=thinking)
                )
            except Exception as exc:
                logger.error(f"[baseline] idx={item.get('index')}: {exc}", exc_info=True)
                results_map["baseline"].append(_error_result(item, exc, "baseline"))

        # ---- Correspondence ----
        if run_correspondence:
            try:
                inputs, prompt, image_xyz = prepare_batch_spa(
                    item, spa_proc,
                    spatial_merge_size, use_coord, coord_scale,
                    thinking=thinking,
                )
                output = run_inference_spa(
                    inputs, image_xyz, spa_model, spa_proc,
                    max_new_tokens, coord_scale,
                )
                results_map["correspondence"].append(
                    _make_result(item, output, prompt, "correspondence",
                                 thinking=thinking)
                )
            except Exception as exc:
                logger.error(f"[correspondence] idx={item.get('index')}: {exc}", exc_info=True)
                results_map["correspondence"].append(_error_result(item, exc, "correspondence"))

    # Save per-worker partial results
    for mname, mresults in results_map.items():
        if mresults:
            partial_path = output_dir / f"{mname}_{device.replace(':', '')}.json"
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(mresults, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(mresults)} {mname} results → {partial_path}")

    return results_map


# ===========================================================================
# Multi-GPU worker
# ===========================================================================

def _worker(
    gpu_id: str,
    data_shard: List[Dict],
    method: str,
    baseline_model_path: str,
    spa_base_model_path: str,
    correspondence_ckpt: Optional[str],
    use_coord: bool,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: str,
    log_file: Optional[str],
    thinking: bool = False,
) -> None:
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )
    logger = logging.getLogger(__name__)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))
    logger.info(f"[Worker {gpu_id}] Starting — {len(data_shard)} samples on {device}")

    evaluate(
        data=data_shard,
        method=method,
        baseline_model_path=baseline_model_path,
        spa_base_model_path=spa_base_model_path,
        correspondence_ckpt=correspondence_ckpt,
        use_coord=use_coord,
        coord_scale=coord_scale,
        max_new_tokens=max_new_tokens,
        output_dir=Path(output_dir),
        device=device,
        thinking=thinking,
    )
    logger.info(f"[Worker {gpu_id}] Done.")


# ===========================================================================
# Change analysis (baseline vs correspondence)
# ===========================================================================

def analyze_changes(
    baseline_results: List[Dict],
    correspondence_results: List[Dict],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Categorise per-sample changes between baseline and correspondence."""
    corr_by_idx = {r["index"]: r for r in correspondence_results}

    groups: Dict[str, List] = {
        "improved":    [],   # baseline ✗, correspondence ✓
        "degraded":    [],   # baseline ✓, correspondence ✗
        "both_correct": [],  # both ✓
        "both_wrong":  [],   # both ✗
    }

    for b in baseline_results:
        idx = b["index"]
        c = corr_by_idx.get(idx)
        if c is None:
            logger.warning(f"No correspondence result for index {idx}, skipping.")
            continue

        b_ok = b.get("prediction", "").lower() == b.get("answer", "").lower()
        c_ok = c.get("prediction", "").lower() == c.get("answer", "").lower()

        entry = {
            "index": idx,
            "category": b.get("category", ""),
            "question": b.get("question", ""),
            "answer": b.get("answer", ""),
            "baseline_prediction": b.get("prediction", ""),
            "correspondence_prediction": c.get("prediction", ""),
        }

        if not b_ok and c_ok:
            groups["improved"].append(entry)
        elif b_ok and not c_ok:
            groups["degraded"].append(entry)
        elif b_ok and c_ok:
            groups["both_correct"].append(entry)
        else:
            groups["both_wrong"].append(entry)

    total = sum(len(v) for v in groups.values())
    counts = {k: len(v) for k, v in groups.items()}
    proportions = {k: len(v) / total if total else 0.0 for k, v in groups.items()}

    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE vs CORRESPONDENCE — change analysis")
    logger.info("=" * 60)
    descs = {
        "improved":     "Baseline ✗ → Correspondence ✓  (3D helps)",
        "degraded":     "Baseline ✓ → Correspondence ✗  (3D hurts)",
        "both_correct": "Both correct",
        "both_wrong":   "Both wrong",
    }
    for k, desc in descs.items():
        logger.info(f"  {desc:<50s}: {counts[k]:4d}  ({proportions[k]:.1%})")
    logger.info(f"  {'Total':<50s}: {total:4d}")
    logger.info("=" * 60)

    return {
        "total": total,
        "counts": counts,
        "proportions": proportions,
        "samples": {k: v for k, v in groups.items()},
    }


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation: Qwen3.5-VL baseline vs SPA correspondence model."
    )

    # ── method ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["baseline", "correspondence", "both"],
        help="Which method(s) to run.",
    )

    # ── model paths ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to Qwen3.5-VL base model (used as baseline model AND as "
             "base for loading the SPA correspondence model).",
    )
    parser.add_argument(
        "--correspondence_ckpt", type=str, default=None,
        help="Path to the LoRA checkpoint saved by train_correspondence.py "
             "(contains adapter_model.safetensors + tokenizer). "
             "Required when --method is 'correspondence' or 'both'.",
    )

    # ── 3D coordinate estimation ──────────────────────────────────────────────
    parser.add_argument(
        "--no_coord", action="store_true", default=False,
        help="Skip CoordEstimator; pass zero image_xyz to the SPA model. "
             "Useful for ablation or debugging without MapAnything.",
    )
    parser.add_argument(
        "--coord_scale", type=float, default=100.0,
        help="Scale applied to XYZ values before discretisation in M-RoPE "
             "(must match the value used during training, default: 100.0).",
    )

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="mmsibench",
        choices=[
            "mmsibench", "mindcube",
            "sat", "sat_real",
            "sparbench_multi_view", "sparbench_single_view", "sparbench_mv",
            "vsibench",
        ],
    )
    parser.add_argument("--data_dir", type=str, default="datasets/evaluation/MMSIBench")
    parser.add_argument("--limit", type=int, default=None)

    # ── inference ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--thinking", action="store_true", default=False,
        help="Enable Qwen3 thinking mode (enable_thinking=True in chat template). "
             "The model reasons in a <think>...</think> block before outputting "
             "<answer>X</answer>. Increases generation length significantly; "
             "raise --max_new_tokens to at least 4096 (recommend 8192).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation. "
                             "Use ≥4096 (recommend 8192) with --thinking.")

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Sub-folder name (default: auto-generated timestamp).")

    args = parser.parse_args()

    # ── setup ─────────────────────────────────────────────────────────────────
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.method}_{timestamp}"
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # ── load dataset ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir).resolve()
    dataset = load_dataset(data_dir, limit=args.limit, dataset=args.dataset)

    # ── GPU setup ─────────────────────────────────────────────────────────────
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("At least one CUDA device is required.")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = (
        [x.strip() for x in cuda_visible.split(",") if x.strip()]
        if cuda_visible
        else [str(i) for i in range(n_gpu)]
    )

    # ── log config ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  method              : {args.method}")
    logger.info(f"  thinking            : {args.thinking}")
    logger.info(f"  model_path          : {args.model_path}")
    logger.info(f"  correspondence_ckpt : {args.correspondence_ckpt}")
    logger.info(f"  use_coord           : {not args.no_coord}")
    logger.info(f"  coord_scale         : {args.coord_scale}")
    logger.info(f"  dataset             : {args.dataset}  ({len(dataset)} samples)")
    logger.info(f"  max_new_tokens      : {args.max_new_tokens}")
    logger.info(f"  GPUs                : {gpu_ids}")
    logger.info(f"  output_dir          : {output_dir}")
    logger.info("=" * 60)

    if args.thinking and args.max_new_tokens < 4096:
        logger.warning(
            f"--thinking is enabled but --max_new_tokens={args.max_new_tokens} "
            "is likely too small (recommend ≥4096, ideally 8192). "
            "The model may truncate mid-thought and produce no <answer> tag."
        )

    # ── save config ───────────────────────────────────────────────────────────
    with open(output_dir / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"output_dir": str(output_dir), "gpus": gpu_ids,
                                 "timestamp": timestamp}, f, indent=2, ensure_ascii=False)

    # ── launch workers ────────────────────────────────────────────────────────
    shards = chunk_dataset(dataset, len(gpu_ids))
    processes: List[mp.Process] = []

    for gpu_id, shard in zip(gpu_ids, shards):
        p = mp.Process(
            target=_worker,
            args=(
                gpu_id, shard,
                args.method,
                args.model_path,     # baseline + spa base
                args.model_path,     # spa_base_model_path (same base)
                args.correspondence_ckpt,
                not args.no_coord,
                args.coord_scale,
                args.max_new_tokens,
                str(output_dir),
                str(log_file),
                args.thinking,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ── merge results ─────────────────────────────────────────────────────────
    logger.info("Merging results from all workers…")

    def merge(prefix: str) -> List[Dict]:
        merged = []
        for gpu_id in gpu_ids:
            path = output_dir / f"{prefix}_cuda{gpu_id}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
            else:
                logger.warning(f"Missing worker output: {path}")
        merged.sort(key=lambda r: r.get("index", 0))
        return merged

    all_results: Dict[str, List[Dict]] = {}

    run_baseline = args.method in ("baseline", "both")
    run_correspondence = args.method in ("correspondence", "both")

    if run_baseline:
        all_results["baseline"] = merge("baseline")
    if run_correspondence:
        all_results["correspondence"] = merge("correspondence")

    # ── metrics ───────────────────────────────────────────────────────────────
    for mname, mresults in all_results.items():
        if mresults:
            m = compute_metrics(mresults)
            label = "BASELINE  (Qwen3.5-VL)" if mname == "baseline" else "CORRESPONDENCE  (SPA + CoordEst)"
            log_metrics(m, label, logger)
            with open(output_dir / f"metrics_{mname}.json", "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
            with open(output_dir / f"results_{mname}.json", "w", encoding="utf-8") as f:
                json.dump(mresults, f, ensure_ascii=False, indent=2)

    # ── side-by-side summary + change analysis ────────────────────────────────
    if run_baseline and run_correspondence and all_results.get("baseline") and all_results.get("correspondence"):
        mb = compute_metrics(all_results["baseline"])
        mc = compute_metrics(all_results["correspondence"])

        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY COMPARISON")
        logger.info("=" * 60)
        logger.info(f"  {'Method':<45} {'Accuracy':>8}  {'Correct':>8} / Total")
        logger.info(f"  {'-'*45}  {'-'*8}  {'-'*14}")
        logger.info(
            f"  {'Baseline  (Qwen3.5-VL)':<45} "
            f"{mb['overall_accuracy']:>8.2%}  "
            f"{mb['correct_samples']:>8} / {mb['total_samples']}"
        )
        logger.info(
            f"  {'Correspondence  (SPA + CoordEst)':<45} "
            f"{mc['overall_accuracy']:>8.2%}  "
            f"{mc['correct_samples']:>8} / {mc['total_samples']}"
        )
        logger.info(f"  {'Delta':<45} {mc['overall_accuracy'] - mb['overall_accuracy']:>+8.2%}")
        logger.info("=" * 60)

        comparison = {
            "baseline": mb,
            "correspondence": mc,
            "delta_overall_accuracy": mc["overall_accuracy"] - mb["overall_accuracy"],
        }
        with open(output_dir / "metrics_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        analysis = analyze_changes(all_results["baseline"], all_results["correspondence"], logger)
        with open(output_dir / "analysis_changes.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
