"""
evaluation.py

Multi-method QA evaluation:

  baseline
      Model : Qwen3.5-VL (AutoModelForImageTextToText)
      Input : images + question  (standard VLM inference)

  vanilla
      Model : Qwen3.5-VL + LoRA, original 3D M-RoPE (ablation: LoRA only)
      Input : images + pose_sentences + question  (no <coord> tokens)

  position_embedding
      Model : SpaForConditionalGeneration (4D M-RoPE) + LoRA
      Input : images + pose_sentences + question  (no <coord> tokens)
      3D pos: precomputed XYZ → 4D M-RoPE on image patches

  coordinate
      Model : SpaForConditionalGeneration (4D M-RoPE) + LoRA
      Input : images + <coord>-token sentences + question  (no_cam variant)
      3D pos: precomputed XYZ → 4D M-RoPE on image patches AND <coord> tokens

  coordinate_pose
      Model : SpaForConditionalGeneration (4D M-RoPE) + LoRA
      Input : images + pose_sentences + <coord>-token sentences + question
      3D pos: precomputed XYZ → 4D M-RoPE on image patches AND <coord> tokens

  both   → baseline + coordinate  (primary comparison)

Usage
-----
# baseline only
python evaluation.py \\
    --method baseline \\
    --model_path checkpoints/Qwen3.5-4B \\
    --data_dir  datasets/evaluation/MMSIBench

# coordinate (full SPA method)
python evaluation.py \\
    --method coordinate \\
    --model_path            checkpoints/Qwen3.5-4B \\
    --correspondence_ckpt   train_records/correspondence/final \\
    --data_dir              datasets/evaluation/MMSIBench

# both (baseline + coordinate)
python evaluation.py \\
    --model_path            checkpoints/Qwen3.5-4B \\
    --correspondence_ckpt   train_records/correspondence/final \\
    --data_dir              datasets/evaluation/MMSIBench

# vanilla ablation
python evaluation.py \\
    --method vanilla \\
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
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from peft import PeftModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from src.models import SpaForConditionalGeneration, CoordinateRegressionHead
from src.dataset import load_testing_dataset, chunk_dataset

# ── sys.path: ensure spatial_planning/ root is importable ──────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── sys.path: RoboSpatial-Eval mask-based evaluation ──────────────────────
_ROBOSPATIAL_EVAL_ROOT = _ROOT.parent / "RoboSpatial-Eval"
if _ROBOSPATIAL_EVAL_ROOT.exists() and str(_ROBOSPATIAL_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOSPATIAL_EVAL_ROOT))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
QUESTION_TEMPLATE = "{Question}"

# ── RoboSpatial: open-ended, no multiple-choice letter ───────────────────────
ROBOSPATIAL_SYSTEM_PROMPT = "You are a spatial reasoning expert helping with robot navigation tasks."

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
    vanilla: bool = False,
) -> Tuple[Any, Any]:
    """Load SPA model with LoRA adapter.

    Steps:
      1. Load config and set mrope_section to 4 equal parts (4D M-RoPE).
         For vanilla: keep original 3D mrope_section and use stock
         Qwen3_5ForConditionalGeneration instead of SpaForConditionalGeneration.
      2. Load base model from base_model_path.
      3. Load processor/tokenizer from ckpt_path (has <pose> in vocab).
      4. Resize embedding table to match the saved tokenizer.
      5. Load PEFT LoRA adapter from ckpt_path, then merge into base weights.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[spa] Loading SPA model: base={base_model_path}  ckpt={ckpt_path}  vanilla={vanilla}")

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])

    if vanilla:
        # vanilla ablation: keep original 3D M-RoPE, use stock Qwen3.5 model
        logger.info(f"[spa] mrope_section: {orig_section} (original 3D M-RoPE, vanilla)")
        spa = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        # 4D M-RoPE for normal / plus / no_cam
        total = sum(orig_section)
        xyz_size = (total - 2) // 3
        new_section = [2, xyz_size, xyz_size, xyz_size]
        config.text_config.rope_scaling["mrope_section"] = new_section
        logger.info(f"[spa] mrope_section: {orig_section} → {new_section}")
        spa = SpaForConditionalGeneration.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    # 3. Processor from base model; swap in the checkpoint tokenizer
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    processor.tokenizer = tokenizer

    # 4. Resize embedding table to match the LoRA checkpoint's embed_tokens size.
    #
    # We read the target vocab size directly from the saved adapter weights because
    # len(tokenizer) can differ from config.text_config.vocab_size (Qwen3.5-4B has
    # vocab_size=248320 in config but the tokenizer only contains 248077 entries).
    # During training, resize_token_embeddings uses embed_tokens.shape[0] (248320)
    # as the base, producing a saved embed of 248321.  But here len(tokenizer)=248078
    # < 248320, so the naive `if new_vocab > old_vocab` guard never fires.
    # Reading the shape from safetensors is the only reliable way to stay in sync.
    _adapter_path = Path(ckpt_path) / "adapter_model.safetensors"
    _target_vocab: Optional[int] = None
    if _adapter_path.exists():
        try:
            from safetensors import safe_open as _safe_open
            with _safe_open(str(_adapter_path), framework="pt", device="cpu") as _f:
                _embed_keys = [k for k in _f.keys() if "embed_tokens" in k and k.endswith(".weight")]
                if _embed_keys:
                    _target_vocab = _f.get_tensor(_embed_keys[0]).shape[0]
        except Exception as _exc:
            logger.warning(f"[spa] Could not read embed size from safetensors: {_exc}")

    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    if _target_vocab is not None and _target_vocab != old_vocab:
        spa.resize_token_embeddings(_target_vocab)
        logger.info(f"[spa] Embedding: {old_vocab} → {_target_vocab} (from adapter checkpoint)")
    elif _target_vocab is None:
        # Fallback: use tokenizer length (original logic)
        new_vocab = len(tokenizer)
        if new_vocab > old_vocab:
            spa.resize_token_embeddings(new_vocab)
            logger.info(f"[spa] Embedding: {old_vocab} → {new_vocab} (from tokenizer)")

    # 5. Load LoRA adapter and merge
    spa = PeftModel.from_pretrained(spa, ckpt_path, is_trainable=False)
    spa = spa.merge_and_unload()
    logger.info("[spa] LoRA adapter merged.")

    spa = spa.to(device).eval()
    logger.info(f"[spa] Model ready on {next(spa.parameters()).device}")
    return spa, processor


def _load_coord_head(
    ckpt_path: str,
    device: str,
) -> Optional[CoordinateRegressionHead]:
    """Load CoordinateRegressionHead from coord_head.pt in the checkpoint directory.

    Infers hidden_dim and upscale_factor from the saved weight shape so no
    extra config is needed.  Returns None if coord_head.pt is not present.
    """
    logger = logging.getLogger(__name__)
    coord_head_path = Path(ckpt_path) / "coord_head.pt"
    if not coord_head_path.exists():
        logger.info(f"[coordinate] coord_head.pt not found in {ckpt_path} — skipping coord head.")
        return None

    state = torch.load(str(coord_head_path), map_location="cpu", weights_only=True)
    # linear_proj.weight shape: (3 * upscale^2, hidden_dim)
    proj_out, hidden_dim = state["linear_proj.weight"].shape
    upscale_factor = int(round((proj_out / 3) ** 0.5))

    coord_head = CoordinateRegressionHead(hidden_dim=hidden_dim, upscale_factor=upscale_factor)
    coord_head.load_state_dict(state)
    coord_head = coord_head.to(device).to(torch.bfloat16).eval()
    logger.info(
        f"[coordinate] CoordinateRegressionHead loaded from {coord_head_path} "
        f"(hidden_dim={hidden_dim}, upscale={upscale_factor})"
    )
    return coord_head


def _get_coord_predictions(
    model: Any,
    inputs: Dict[str, Any],
    coord_token_id: int,
    coord_head: CoordinateRegressionHead,
    spatial_merge_size: int,
    image_xyz: List[torch.Tensor],   # llm-resolution GT, used as xyz for 4D RoPE
    coord_scale: float,
) -> Optional[List[torch.Tensor]]:
    """Single forward pass (no generation) → coord head predictions at <coord> tokens.

    Uses an lm_head pre-hook to capture the post-norm last hidden state, matching
    exactly what CoordinatePlusModel / CoordinateModel do during training.

    Returns a list of (llm_H * upscale, llm_W * upscale, 3) float32 tensors on CPU,
    one per image.  Returns None if no <coord> tokens are found or the hook fails.
    """
    device = next(model.parameters()).device
    inputs_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    image_grid_thw = inputs_dev.get("image_grid_thw")
    if image_grid_thw is None:
        return None

    xyz_on_device = [x.to(device) for x in image_xyz]

    # Pre-compute 4D position_ids (same logic as run_inference_spa)
    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids=inputs_dev["input_ids"],
            mm_token_type_ids=inputs_dev["mm_token_type_ids"],
            image_grid_thw=image_grid_thw,
            video_grid_thw=inputs_dev.get("video_grid_thw"),
            attention_mask=inputs_dev.get("attention_mask"),
            image_xyz=xyz_on_device,
            coord_scale=coord_scale,
            coord_token_id=coord_token_id,
        )

    # Register lm_head pre-hook to capture post-norm last hidden state
    captured: Dict[str, Any] = {}

    def _hook(_module, args):
        captured["h"] = args[0].detach()

    hook_handle = None
    for name, mod in model.named_modules():
        if name.endswith("lm_head"):
            hook_handle = mod.register_forward_pre_hook(_hook)
            break

    try:
        gen_inputs = {k: v for k, v in inputs_dev.items() if k != "mm_token_type_ids"}
        with torch.no_grad():
            model(
                **gen_inputs,
                position_ids=position_ids,
                return_dict=True,
                image_xyz=xyz_on_device,
                coord_scale=coord_scale,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if "h" not in captured:
        return None

    last_hidden = captured["h"][0]  # (seq_len, hidden_dim)
    coord_positions = (
        inputs_dev["input_ids"][0] == coord_token_id
    ).nonzero(as_tuple=True)[0]
    if len(coord_positions) == 0:
        return None

    sms = spatial_merge_size
    N = image_grid_thw.shape[0]
    preds: List[torch.Tensor] = []
    start = 0
    dtype = coord_head.linear_proj.weight.dtype

    with torch.no_grad():
        for k in range(N):
            thw_k = image_grid_thw[k]
            llm_h = int(thw_k[1]) // sms
            llm_w = int(thw_k[2]) // sms
            n_tok = llm_h * llm_w
            if start + n_tok > len(coord_positions):
                break
            h_k = last_hidden[coord_positions[start: start + n_tok]].to(dtype)
            pred_k = coord_head(h_k, llm_h, llm_w)    # (llm_h*up, llm_w*up, 3)
            preds.append(pred_k.cpu().float())
            start += n_tok

    return preds if preds else None


def _compute_coord_mae(
    preds: List[torch.Tensor],    # (llm_h*up, llm_w*up, 3) per image
    gt_list: List[torch.Tensor],  # (llm_h, llm_w, 3) per image (RoPE-resolution GT)
) -> float:
    """Mean L1 error between coord head predictions and GT xyz, averaged over images.

    The prediction is at (llm_h * upscale, llm_w * upscale) resolution; GT is at
    (llm_h, llm_w).  Predictions are average-pooled to GT resolution before comparison.
    """
    maes: List[float] = []
    for pred, gt in zip(preds, gt_list):
        gt_f = gt.float()
        if pred.shape == gt_f.shape:
            maes.append((pred - gt_f).abs().mean().item())
        else:
            # Pool pred to GT resolution: (llm_h*up, llm_w*up, 3) → (llm_h, llm_w, 3)
            pred_t = pred.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H*up, W*up)
            pred_ds = F.adaptive_avg_pool2d(pred_t, gt_f.shape[:2])  # (1, 3, H, W)
            gt_t = gt_f.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)
            maes.append((pred_ds - gt_t).abs().mean().item())
    return float(np.mean(maes)) if maes else 0.0


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

def _build_user_message(item: Dict[str, Any], thinking: bool = False,
                        train_template: bool = False) -> Dict:
    image_contents = [{"type": "image", "image": p} for p in item["image"]]
    if train_template or item.get("format_type") == "robospatial":
        text = QUESTION_TEMPLATE.format(Question=item["question"])
    else:
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

    prompts_text = []
    batch_messages = []
    for item in batch_data:
        is_robospatial = item.get("format_type") == "robospatial"
        sys_prompt = ROBOSPATIAL_SYSTEM_PROMPT if is_robospatial else (
            EVAL_SYSTEM_PROMPT_THINKING if thinking else EVAL_SYSTEM_PROMPT
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            _build_user_message(item, thinking=thinking),
        ]
        batch_messages.append(msgs)
        if is_robospatial or thinking:
            prompts_text.append(
                processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    **({"enable_thinking": True} if thinking and not is_robospatial else {}),
                )
            )
        else:
            prompts_text.append(
                processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                ) + "<answer>"
            )

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
    use_pose: bool = True,
) -> Tuple[Dict, str, List[torch.Tensor]]:
    """Tokenise one sample and build image_xyz for SPA model inference.

    The prompt matches training format:
      [images] + (pose_sentences if use_pose) + (coord_sentences if use_coord) + question

    use_pose=False corresponds to the --no_cam training variant (coordinate method):
      no <pose> tokens, no pose sentences; only <coord> tokens + question.
    use_pose=True corresponds to the full training variant (coordinate_pose method):
      pose sentences with <pose> tokens + <coord> tokens + question.

    No system prompt, no answer instruction — identical to train_dataset.py.

    Returns
    -------
    inputs     : processor output dict (input_ids, attention_mask, pixel_values, image_grid_thw)
    prompt_str : prompt text (used for logging)
    image_xyz  : list of (llm_H, llm_W, 3) tensors, or None if use_coord=False
    """
    from qwen_vl_utils import process_vision_info

    POSE_TOKEN = "<pose>"
    COORD_TOKEN = "<coord>"

    image_paths = item["image"]
    N = len(image_paths)
    question = item.get("question", "")

    # ── image content ────────────────────────────────────────────────────────
    content: list = [{"type": "image", "image": p} for p in image_paths]

    # ── pose sentences (like training; skipped when use_pose=False / no_cam) ─
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    pose_sentences = (
        [
            f"The camera pose of image {j + 1} relative to image {i + 1} is "
            f"{POSE_TOKEN}."
            for (i, j) in pairs
        ]
        if (use_pose and N >= 2) else []
    )

    if use_coord and N >= 2:
        # ── probe step: get image_grid_thw to know patch counts per image ──
        probe_content = list(content)
        probe_text = " ".join(pose_sentences) + " " + question if pose_sentences else question
        probe_content.append({"type": "text", "text": probe_text})
        probe_messages = [{"role": "user", "content": probe_content}]
        probe_prompt = processor.apply_chat_template(
            probe_messages, tokenize=False, add_generation_prompt=False,
        )
        probe_images, _ = process_vision_info(probe_messages)
        probe_out = processor(
            text=[probe_prompt],
            images=probe_images if probe_images else None,
            return_tensors="pt", padding=False,
        )
        thw_all = probe_out["image_grid_thw"]  # (N, 3)
        sms = spatial_merge_size

        # ── coord sentences (one <coord> token per LLM patch) ──────────────
        coord_sentences = []
        for k in range(N):
            llm_h = int(thw_all[k][1]) // sms
            llm_w = int(thw_all[k][2]) // sms
            n_tok = llm_h * llm_w
            coord_tokens = "".join([COORD_TOKEN] * n_tok)
            coord_sentences.append(
                f"Image {k + 1} 3D spatial coordinates: {coord_tokens}."
            )

        # ── final text: (pose +) coord + question ─────────────────────────
        parts = []
        if pose_sentences:
            parts.append(" ".join(pose_sentences))
        parts.append(" ".join(coord_sentences))
        parts.append(question)
        final_text = " ".join(parts)
    else:
        # ── final text: (pose +) question (no coord) ──────────────────────
        if pose_sentences:
            final_text = " ".join(pose_sentences) + " " + question
        else:
            final_text = question

    content.append({"type": "text", "text": final_text})

    # ── build messages (no system prompt — matches training) ─────────────────
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
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
    vanilla: bool = False,
) -> str:
    """Run generation with SPA model (or stock Qwen3.5 for vanilla ablation).

    image_xyz (if provided) is passed as a kwarg to model.generate() which
    forwards it to forward() → get_rope_index() for 4D M-RoPE.
    The PoseRegressionHead is NOT used; we only call generate() for QA.

    When vanilla=True, the model is a stock Qwen3_5ForConditionalGeneration
    with original 3D M-RoPE — no custom get_rope_index or image_xyz needed.
    """
    device = next(model.parameters()).device
    inputs_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Resolve <coord> token id for 3D-RoPE on coord tokens
    coord_token_id = None
    _coord_id = processor.tokenizer.convert_tokens_to_ids("<coord>")
    if isinstance(_coord_id, int) and _coord_id != processor.tokenizer.unk_token_id:
        coord_token_id = _coord_id

    if vanilla:
        # Stock Qwen3.5: let the model compute its own 3D position_ids
        gen_kwargs: Dict[str, Any] = dict(
            **inputs_dev,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    else:
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
                coord_token_id=coord_token_id,
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
        if coord_token_id is not None:
            gen_kwargs["coord_token_id"] = coord_token_id

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
    fmt = item.get("format_type", "select")
    if fmt == "robospatial":
        # Open-ended output: store raw for mask-based evaluation
        full_output = output
        prediction = output
    else:
        # Non-thinking: model output starts right after the "<answer>" prefix we
        # injected into the prompt, so we prepend it back for a complete tag.
        # Thinking: model generates the full response (including <think>...</think>
        # and <answer>X</answer>) — no prefix needed.
        full_output = output if thinking else "<answer>" + output
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
        "mask": item.get("mask"),
        "data_dir": item.get("data_dir"),
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


def _failed_case(r: Dict) -> Dict:
    return {
        "index":      r.get("index", ""),
        "question":   r.get("question", ""),
        "answer":     r.get("answer", ""),
        "prediction": r.get("prediction", ""),
        "output":     r.get("output", ""),
        "images":     r.get("image_paths", []),
    }


def _save_failure_cases(failures_dir: Path, failures: Dict[str, List[Dict]]) -> None:
    """Save failure cases as a folder tree with images and qa.txt per case.

    Structure:
        failures_{mname}/
            {category}/
                case_{i}/
                    image_0.jpg [image_1.jpg ...]
                    qa.txt
    """
    failures_dir.mkdir(parents=True, exist_ok=True)
    for cat, cases in failures.items():
        cat_dir = failures_dir / cat
        cat_dir.mkdir(exist_ok=True)
        for i, case in enumerate(cases):
            case_dir = cat_dir / f"case_{i:02d}"
            case_dir.mkdir(exist_ok=True)

            # Copy images
            for j, img_path in enumerate(case.get("images", [])):
                src = Path(img_path)
                if src.exists():
                    suffix = src.suffix or ".jpg"
                    shutil.copy2(src, case_dir / f"image_{j}{suffix}")

            # Write QA text
            lines = [
                f"Index     : {case.get('index', '')}",
                f"Question  : {case.get('question', '')}",
                f"Expected  : {case.get('answer', '')}",
                f"Prediction: {case.get('prediction', '')}",
            ]
            if "parsed_prediction" in case:
                lines.append(f"Parsed    : {case.get('parsed_prediction', '')}")
            if "mask" in case:
                lines.append(f"Mask      : {case.get('mask', '')}")
            lines += [
                "",
                "--- Full output ---",
                case.get("output", ""),
            ]
            (case_dir / "qa.txt").write_text("\n".join(lines), encoding="utf-8")


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    total = len(results)
    correct = 0
    cat_correct: dict = defaultdict(float)
    cat_total: dict = defaultdict(int)
    cat_failures: dict = defaultdict(list)
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
            if score < 1.0 and len(cat_failures[cat]) < 10:
                cat_failures[cat].append(_failed_case(r))
        else:
            if pred.lower().strip() == gt.lower().strip():
                correct += 1
                cat_correct[cat] += 1
            elif len(cat_failures[cat]) < 10:
                cat_failures[cat].append(_failed_case(r))
    cat_accuracy = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}
    metrics: Dict[str, Any] = {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": correct,
        "category_accuracy": cat_accuracy,
        "category_counts": dict(cat_total),
        "category_failures": dict(cat_failures),
    }
    # Aggregate coord_mae when present (coordinate method)
    coord_maes = [r["coord_mae"] for r in results if r.get("coord_mae") is not None]
    if coord_maes:
        metrics["coord_mae_mean"] = float(np.mean(coord_maes))
        metrics["coord_mae_std"]  = float(np.std(coord_maes))
        metrics["coord_mae_n"]    = len(coord_maes)
    return metrics


def compute_metrics_robospatial(results: List[Dict]) -> Dict[str, Any]:
    """Compute metrics for RoboSpatial using mask-based evaluation."""
    from evaluation import evaluate_answer  # from RoboSpatial-Eval/evaluation.py

    total = len(results)
    correct = 0
    illformed = 0
    cat_correct: dict = defaultdict(float)
    cat_total: dict = defaultdict(int)
    cat_failures: dict = defaultdict(list)

    for r in results:
        gt = r.get("answer", "")
        pred = r.get("prediction", "")
        cat = r.get("category", "unknown")
        mask_rel = r.get("mask")
        data_dir = r.get("data_dir")
        cat_total[cat] += 1

        is_correct, _, parsed, is_parsable = evaluate_answer(
            gt, pred,
            mask_path=mask_rel,
            data_dir=data_dir,
            category=cat,
        )
        if not is_parsable:
            illformed += 1
        if is_correct:
            correct += 1
            cat_correct[cat] += 1
        elif len(cat_failures[cat]) < 10:
            case = _failed_case(r)
            case["parsed_prediction"] = str(parsed) if parsed is not None else None
            case["mask"] = mask_rel
            cat_failures[cat].append(case)

    cat_accuracy = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": correct,
        "illformed_responses": illformed,
        "category_accuracy": cat_accuracy,
        "category_counts": dict(cat_total),
        "category_failures": dict(cat_failures),
    }


def log_metrics(metrics: Dict, label: str, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info(f"RESULTS — {label}")
    logger.info("=" * 60)
    logger.info(f"  Total   : {metrics['total_samples']}")
    logger.info(f"  Correct : {metrics['correct_samples']}")
    logger.info(f"  Accuracy: {metrics['overall_accuracy']:.2%}")
    if "coord_mae_mean" in metrics:
        logger.info(
            f"  Coord MAE: {metrics['coord_mae_mean']:.4f} ± {metrics['coord_mae_std']:.4f}"
            f"  (n={metrics['coord_mae_n']})"
        )
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
    # spa args
    spa_base_model_path: str,
    correspondence_ckpt: Optional[str],
    coord_scale: float,
    # common
    max_new_tokens: int,
    output_dir: Path,
    device: str = "cuda:0",
    thinking: bool = False,
) -> Dict[str, List[Dict]]:
    """Run evaluation for the requested method(s) on *data*.

    method choices:
      baseline           — stock Qwen3.5-VL
      vanilla            — SPA LoRA + 3D M-RoPE (no <coord>)
      position_embedding — SPA LoRA + 4D M-RoPE (no <coord>)
      coordinate         — SPA LoRA + 4D M-RoPE + <coord> tokens, no_cam variant (no pose)
      coordinate_pose    — SPA LoRA + 4D M-RoPE + <coord> tokens, full variant (with pose)
      both               — baseline + coordinate

    Returns dict mapping method name → list of result dicts.
    """
    logger = logging.getLogger(__name__)

    run_baseline = method in ("baseline", "both")
    run_vanilla = method == "vanilla"
    run_position_embedding = method == "position_embedding"
    run_coordinate = method in ("coordinate", "both")
    run_coordinate_pose = method == "coordinate_pose"
    run_spa = run_vanilla or run_position_embedding or run_coordinate or run_coordinate_pose

    # Lazy-load only what we need
    baseline_model = baseline_proc = None
    spa_model = spa_proc = None
    spatial_merge_size = 2

    if run_baseline:
        baseline_model, baseline_proc = load_baseline_model(baseline_model_path, device)

    coord_token_id_val: Optional[int] = None
    spa_coord_head: Optional[CoordinateRegressionHead] = None

    if run_spa:
        if correspondence_ckpt is None:
            raise ValueError(
                f"--correspondence_ckpt is required for method='{method}'"
            )
        use_vanilla_arch = run_vanilla
        spa_model, spa_proc = load_spa_model(
            spa_base_model_path, correspondence_ckpt, device, vanilla=use_vanilla_arch
        )

        # Resolve spatial_merge_size from base model config
        cfg_path = Path(spa_base_model_path) / "config.json"
        with open(cfg_path) as f:
            _vcfg = json.load(f).get("vision_config", {})
        spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
        logger.info(f"[spa] spatial_merge_size={spatial_merge_size}")

        # Resolve <coord> token id once
        _cid = spa_proc.tokenizer.convert_tokens_to_ids("<coord>")
        if isinstance(_cid, int) and _cid != spa_proc.tokenizer.unk_token_id:
            coord_token_id_val = _cid

        # Load CoordinateRegressionHead if this is a coordinate checkpoint
        if run_coordinate or run_coordinate_pose:
            spa_coord_head = _load_coord_head(correspondence_ckpt, device)

    # Determine which SPA variants to run.
    # Tuple: (method_name, use_coord, use_pose)
    # vanilla / position_embedding: no <coord> tokens, but include pose sentences
    # coordinate       (no_cam): <coord> tokens, NO pose sentences
    # coordinate_pose  (full):   <coord> tokens, WITH pose sentences
    spa_variants: List[Tuple[str, bool, bool]] = []
    if run_vanilla:
        spa_variants.append(("vanilla", False, True))
    if run_position_embedding:
        spa_variants.append(("position_embedding", False, True))
    if run_coordinate:
        spa_variants.append(("coordinate", True, False))
    if run_coordinate_pose:
        spa_variants.append(("coordinate_pose", True, True))

    active_methods = (
        (["baseline"] if run_baseline else [])
        + [name for name, _, __ in spa_variants]
    )
    results_map: Dict[str, List[Dict]] = {m: [] for m in active_methods}

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

        # ---- SPA variants ----
        for spa_method_name, use_coord, use_pose in spa_variants:
            try:
                inputs, prompt, image_xyz = prepare_batch_spa(
                    item, spa_proc,
                    spatial_merge_size, use_coord, coord_scale,
                    thinking=thinking,
                    use_pose=use_pose,
                )
                output = run_inference_spa(
                    inputs, image_xyz, spa_model, spa_proc,
                    max_new_tokens, coord_scale, vanilla=use_vanilla_arch,
                )
                result = _make_result(item, output, prompt, spa_method_name,
                                      thinking=thinking)

                # ---- CoordinateRegressionHead evaluation (coordinate method only) ----
                if (
                    use_coord
                    and spa_coord_head is not None
                    and coord_token_id_val is not None
                    and image_xyz is not None
                ):
                    try:
                        preds = _get_coord_predictions(
                            spa_model, inputs,
                            coord_token_id_val, spa_coord_head,
                            spatial_merge_size, image_xyz, coord_scale,
                        )
                        if preds is not None:
                            result["coord_mae"] = _compute_coord_mae(preds, image_xyz)
                    except Exception as ce:
                        logger.warning(
                            f"[coordinate] coord_head failed for idx="
                            f"{item.get('index')}: {ce}"
                        )

                results_map[spa_method_name].append(result)
            except Exception as exc:
                logger.error(
                    f"[{spa_method_name}] idx={item.get('index')}: {exc}",
                    exc_info=True,
                )
                results_map[spa_method_name].append(
                    _error_result(item, exc, spa_method_name)
                )

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
    method_a_results: List[Dict],
    method_b_results: List[Dict],
    logger: logging.Logger,
    method_a_name: str = "baseline",
    method_b_name: str = "coordinate",
) -> Dict[str, Any]:
    """Categorise per-sample changes between two methods (A vs B)."""
    b_by_idx = {r["index"]: r for r in method_b_results}

    groups: Dict[str, List] = {
        "improved":    [],   # A ✗, B ✓
        "degraded":    [],   # A ✓, B ✗
        "both_correct": [],  # both ✓
        "both_wrong":  [],   # both ✗
    }

    for a in method_a_results:
        idx = a["index"]
        b = b_by_idx.get(idx)
        if b is None:
            logger.warning(f"No {method_b_name} result for index {idx}, skipping.")
            continue

        a_ok = a.get("prediction", "").lower() == a.get("answer", "").lower()
        b_ok = b.get("prediction", "").lower() == b.get("answer", "").lower()

        entry = {
            "index": idx,
            "category": a.get("category", ""),
            "question": a.get("question", ""),
            "answer": a.get("answer", ""),
            f"{method_a_name}_prediction": a.get("prediction", ""),
            f"{method_b_name}_prediction": b.get("prediction", ""),
        }

        if not a_ok and b_ok:
            groups["improved"].append(entry)
        elif a_ok and not b_ok:
            groups["degraded"].append(entry)
        elif a_ok and b_ok:
            groups["both_correct"].append(entry)
        else:
            groups["both_wrong"].append(entry)

    total = sum(len(v) for v in groups.values())
    counts = {k: len(v) for k, v in groups.items()}
    proportions = {k: len(v) / total if total else 0.0 for k, v in groups.items()}

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{method_a_name.upper()} vs {method_b_name.upper()} — change analysis")
    logger.info("=" * 60)
    descs = {
        "improved":     f"{method_a_name} ✗ → {method_b_name} ✓  (3D helps)",
        "degraded":     f"{method_a_name} ✓ → {method_b_name} ✗  (3D hurts)",
        "both_correct": "Both correct",
        "both_wrong":   "Both wrong",
    }
    for k, desc in descs.items():
        logger.info(f"  {desc:<55s}: {counts[k]:4d}  ({proportions[k]:.1%})")
    logger.info(f"  {'Total':<55s}: {total:4d}")
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
        choices=["baseline", "vanilla", "position_embedding", "coordinate", "coordinate_pose", "both"],
        help=(
            "Which method(s) to run. "
            "baseline=stock Qwen3.5-VL; "
            "vanilla=SPA LoRA + 3D M-RoPE (no <coord>); "
            "position_embedding=SPA LoRA + 4D M-RoPE (no <coord>); "
            "coordinate=SPA LoRA + 4D M-RoPE + <coord> tokens (no_cam, no pose); "
            "coordinate_pose=SPA LoRA + 4D M-RoPE + <coord> tokens (full, with pose); "
            "both=baseline + coordinate."
        ),
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
            "vsibench", "spinbench", "robospatial",
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
    dataset = load_testing_dataset(data_dir, limit=args.limit, dataset=args.dataset)

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
                args.model_path,     # baseline model path
                args.model_path,     # spa_base_model_path (same checkpoint)
                args.correspondence_ckpt,
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

    # Determine which method names were actually run by workers
    _method_names = {
        "baseline":           args.method in ("baseline", "both"),
        "vanilla":            args.method == "vanilla",
        "position_embedding": args.method == "position_embedding",
        "coordinate":         args.method in ("coordinate", "both"),
        "coordinate_pose":    args.method == "coordinate_pose",
    }
    all_results: Dict[str, List[Dict]] = {}
    for mname, active in _method_names.items():
        if active:
            all_results[mname] = merge(mname)

    # ── metrics ───────────────────────────────────────────────────────────────
    _METHOD_LABELS = {
        "baseline":           "baseline           (Qwen3.5-VL)",
        "vanilla":            "vanilla            (SPA LoRA + 3D M-RoPE)",
        "position_embedding": "position_embedding (SPA LoRA + 4D M-RoPE)",
        "coordinate":         "coordinate         (SPA LoRA + 4D M-RoPE + <coord>, no_cam)",
        "coordinate_pose":    "coordinate_pose    (SPA LoRA + 4D M-RoPE + <coord> + pose)",
    }
    _metrics_fn = compute_metrics_robospatial if args.dataset == "robospatial" else compute_metrics
    for mname, mresults in all_results.items():
        if mresults:
            m = _metrics_fn(mresults)
            log_metrics(m, _METHOD_LABELS.get(mname, mname), logger)
            failures = m.pop("category_failures", {})
            with open(output_dir / f"metrics_{mname}.json", "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
            with open(output_dir / f"results_{mname}.json", "w", encoding="utf-8") as f:
                json.dump(mresults, f, ensure_ascii=False, indent=2)
            if failures:
                _save_failure_cases(output_dir / f"failures_{mname}", failures)

    # ── side-by-side summary + change analysis ────────────────────────────────
    # Compare any two methods that are both present; primary comparison is
    # baseline vs coordinate (the "both" mode).
    _compare_pairs = [
        ("baseline", "coordinate"),
        ("baseline", "coordinate_pose"),
        ("baseline", "position_embedding"),
        ("baseline", "vanilla"),
    ]
    for method_a, method_b in _compare_pairs:
        if not (all_results.get(method_a) and all_results.get(method_b)):
            continue
        ma = _metrics_fn(all_results[method_a])
        mb = _metrics_fn(all_results[method_b])

        logger.info("")
        logger.info("=" * 65)
        logger.info("SUMMARY COMPARISON")
        logger.info("=" * 65)
        logger.info(f"  {'Method':<50} {'Accuracy':>8}  {'Correct':>8} / Total")
        logger.info(f"  {'-'*50}  {'-'*8}  {'-'*14}")
        for mname, mx in [(method_a, ma), (method_b, mb)]:
            label = _METHOD_LABELS.get(mname, mname)
            logger.info(
                f"  {label:<50} "
                f"{mx['overall_accuracy']:>8.2%}  "
                f"{mx['correct_samples']:>8} / {mx['total_samples']}"
            )
        delta = mb["overall_accuracy"] - ma["overall_accuracy"]
        logger.info(f"  {'Delta':<50} {delta:>+8.2%}")
        logger.info("=" * 65)

        comparison = {
            method_a: ma,
            method_b: mb,
            "delta_overall_accuracy": delta,
        }
        suffix = f"{method_a}_vs_{method_b}"
        with open(output_dir / f"metrics_comparison_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        analysis = analyze_changes(
            all_results[method_a], all_results[method_b], logger,
            method_a_name=method_a, method_b_name=method_b,
        )
        with open(output_dir / f"analysis_changes_{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
