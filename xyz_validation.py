"""
xyz_validation.py

Per-method ablation: does the trained checkpoint actually consume the per-patch
3D coordinates that get fed into its xyz pathway?

For the same checkpoint, run inference twice on the same dataset:
  • PASS 1 — normal xyz   : image_xyz loaded from <data_dir>/3d_results/<id>/.
  • PASS 2 — image_xyz=0  : image_xyz replaced with torch.zeros_like(...) right
                            before any downstream consumer (RoPE,
                            coord_head, SpatialAttentionBias).

If the model genuinely depends on xyz, we expect overall_accuracy and
coord_mae (when applicable) to clearly degrade in PASS 2. A near-zero delta
means the xyz pathway is being ignored at inference.

Supported methods (must touch xyz at inference time):
  position_embedding, coordinate, decouple, atten

(`baseline` / `vanilla` don't ingest xyz — there's nothing to ablate, so the
CLI rejects them.)

This script is fully self-contained — it does NOT import from evaluation.py.
That isolation is deliberate: a previous run was contaminated when evaluation.py
was edited mid-run between the normal and zero passes.

Usage:
    python xyz_validation.py \\
        --method      coordinate \\
        --ckpt        train_records/coordinate_no_cam_mindcube/step_1000 \\
        --model_path  checkpoints/Qwen3.5-4B \\
        --dataset     mindcube \\
        --data_dir    datasets/evaluation/MindCube \\
        --output_dir  vis_results/xyz_val_coord_mc

Multi-GPU is automatic via CUDA_VISIBLE_DEVICES; both passes shard the same
data across visible GPUs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from peft import PeftModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models import (
    SpaForConditionalGeneration,
    DepthPredictionTransformer,
    SpatialAttentionBias,
    SpatialAttnVanillaModel,
    patch_attention_layers_spatial,
)
from src.dataset import (
    load_testing_dataset, chunk_dataset, _qwen_align_view,
    build_interleaved_content,
    extract_answer_letter as _extract_answer_letter,
    extract_answer_number as _extract_answer_number,
    extract_answer_content as _extract_answer_content,
)


# ===========================================================================
# Method registry
# ===========================================================================

_XYZ_METHODS = (
    "position_embedding", "coordinate",
    "decouple",
    "atten",
)


def _method_flags(method: str) -> Dict[str, bool]:
    """Per-method routing flags into _load_spa_model / _run_inference_spa."""
    return {
        "vanilla":  False,
        "decouple": method == "decouple",
        "atten":    method == "atten",
    }


# Answer extractors — re-exported from src.dataset.answer_format above
# (`_extract_answer_letter`, `_extract_answer_number`, `_extract_answer_content`).
# Single source of truth: see [src/dataset/answer_format.py](src/dataset/answer_format.py).


# ===========================================================================
# Checkpoint resolution + model loader
# ===========================================================================

def _resolve_spa_ckpt_dir(ckpt_path: str) -> Path:
    """Accepts either a step directory or a run root containing step_* subdirs."""
    p = Path(ckpt_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    def _has_adapter(d: Path) -> bool:
        return (d / "adapter_model.safetensors").exists()

    def _usable(d: Path) -> bool:
        return d.is_dir() and _has_adapter(d)

    if _usable(p):
        return p

    step_pat = re.compile(r"^step_(\d+)(?:_final)?$")
    candidates: List[Tuple[int, int, Path]] = []
    if p.is_dir():
        for d in p.iterdir():
            if not d.is_dir():
                continue
            m = step_pat.match(d.name)
            if m is None or not _usable(d):
                continue
            candidates.append((int(m.group(1)),
                               1 if d.name.endswith("_final") else 0,
                               d))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[-1][2]

    raise FileNotFoundError(
        f"Could not resolve a valid checkpoint from {p}. "
        f"Expected adapter_model.safetensors."
    )


def _load_spatial_bias_modules(model: Any, ckpt_dir: Path) -> int:
    """Reload spatial_bias.pt; match by suffix `language_model.layers.*` so the
    PEFT-wrapped saved keys line up with the merged eval model's keys."""
    path = Path(ckpt_dir) / "spatial_bias.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"[atten] spatial_bias.pt missing in checkpoint dir: {path}."
        )
    bias_state = torch.load(str(path), map_location="cpu")

    def _suffix(name: str) -> str:
        idx = name.find("language_model.layers.")
        return name[idx:] if idx >= 0 else name

    saved = {_suffix(k): v for k, v in bias_state.items()}
    n_loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, SpatialAttentionBias):
            suf = _suffix(name)
            if suf in saved:
                mod.load_state_dict(saved[suf])
                n_loaded += 1
    if n_loaded == 0:
        raise RuntimeError(
            f"[atten] No SpatialAttentionBias modules matched between "
            f"{path} (n={len(bias_state)}) and runtime model."
        )
    return n_loaded


def _load_spa_model(
    base_model_path: str,
    ckpt_path: str,
    device: str,
    decouple: bool,
    atten: bool,
    xyz_rope_dim: int = 66,
) -> Tuple[Any, Any]:
    """Load LoRA-merged SPA model. Mirrors evaluation.py:load_spa_model but no
    `vanilla=True` branch (this script never ablates a method that ignores xyz)."""
    logger = logging.getLogger(__name__)
    ckpt_dir = _resolve_spa_ckpt_dir(ckpt_path)
    logger.info(
        f"[spa] base={base_model_path}  ckpt={ckpt_dir}  "
        f"decouple={decouple} atten={atten}"
    )

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])

    if atten and decouple:
        raise ValueError("[spa] --atten is mutually exclusive with --decouple.")

    if atten:
        logger.info(f"[spa] mrope_section: {orig_section} (UNCHANGED — atten)")
        spa = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_path, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        new_inner = SpatialAttnVanillaModel(spa.config)
        new_inner.load_state_dict(spa.model.state_dict(), strict=True)
        spa.model = new_inner.to(dtype=torch.bfloat16)
        spa.tie_weights()
    elif decouple:
        from src.models.spa_emb_dec import (
            SpaDecForConditionalGeneration, SpaXYZRotaryEmbedding,
        )
        if xyz_rope_dim % 6 != 0 or xyz_rope_dim <= 0 or xyz_rope_dim > 192:
            raise ValueError(
                f"[spa] xyz_rope_dim must be a positive multiple of 6 ≤ 192; "
                f"got {xyz_rope_dim}."
            )
        _xyz_theta = 10000.0
        logger.info(
            f"[spa] mrope_section: {orig_section} (UNCHANGED — decouple) "
            f"+ XYZ RoPE ({xyz_rope_dim} dims, theta={_xyz_theta:g}) [Cartesian (x, y, z)]"
        )
        spa = SpaDecForConditionalGeneration.from_pretrained(
            base_model_path, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        if xyz_rope_dim != 66:
            _lm = spa.model.language_model
            _old = _lm.xyz_rotary_emb
            _new = SpaXYZRotaryEmbedding(
                xyz_dim=xyz_rope_dim, rope_theta=_xyz_theta,
                default_coord_scale=_old.default_coord_scale,
            )
            _lm.xyz_rotary_emb = _new.to(next(_lm.parameters()).device)
    else:
        # 4D M-RoPE for position_embedding / coordinate
        total = sum(orig_section)
        xyz_size = (total - 2) // 3
        new_section = [2, xyz_size, xyz_size, xyz_size]
        config.text_config.rope_scaling["mrope_section"] = new_section
        logger.info(f"[spa] mrope_section: {orig_section} → {new_section}")
        spa = SpaForConditionalGeneration.from_pretrained(
            base_model_path, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )

    # Processor, with the checkpoint's tokenizer
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    processor.tokenizer = tokenizer

    # Resize embedding table to match adapter
    _adapter_path = ckpt_dir / "adapter_model.safetensors"
    _target_vocab: Optional[int] = None
    if _adapter_path.exists():
        try:
            from safetensors import safe_open as _safe_open
            with _safe_open(str(_adapter_path), framework="pt", device="cpu") as _f:
                _embed_keys = [k for k in _f.keys()
                               if "embed_tokens" in k and k.endswith(".weight")]
                if _embed_keys:
                    _target_vocab = _f.get_tensor(_embed_keys[0]).shape[0]
        except Exception as _exc:
            logger.warning(f"[spa] could not read embed size: {_exc}")

    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    if _target_vocab is not None and _target_vocab != old_vocab:
        spa.resize_token_embeddings(_target_vocab)
        logger.info(f"[spa] Embedding: {old_vocab} → {_target_vocab}")
    elif _target_vocab is None:
        new_vocab = len(tokenizer)
        if new_vocab > old_vocab:
            spa.resize_token_embeddings(new_vocab)

    # Wrap BEFORE PEFT load — saved LoRA keys carry wrapper prefix in atten/decouple.
    if atten:
        n_wrapped = patch_attention_layers_spatial(spa)
        for module in spa.modules():
            if isinstance(module, SpatialAttentionBias):
                module.to(dtype=torch.bfloat16)
        logger.info(f"[spa] atten: pre-wrapped {n_wrapped} self_attn layers")
    if decouple:
        from src.models.spa_emb_dec import patch_attention_layers_dec
        patch_attention_layers_dec(spa)

    spa = PeftModel.from_pretrained(spa, str(ckpt_dir), is_trainable=False)
    spa = spa.merge_and_unload()
    logger.info("[spa] LoRA adapter merged.")

    if atten:
        n_loaded = _load_spatial_bias_modules(spa, ckpt_dir)
        logger.info(f"[spa] atten: loaded {n_loaded} SpatialAttentionBias modules")
        # HF generate() strips image_xyz; stash on spa.model and inject via shim.
        _orig_inner_forward = spa.model.forward
        def _atten_eval_forward(*args, image_xyz=None, mm_token_type_ids=None, **kwargs):
            if image_xyz is None:
                image_xyz = getattr(spa.model, "_eval_image_xyz", None)
            return _orig_inner_forward(
                *args,
                image_xyz=image_xyz,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
        spa.model.forward = _atten_eval_forward

    spa = spa.to(device).eval()
    logger.info(f"[spa] Model ready on {next(spa.parameters()).device}")
    return spa, processor


# ===========================================================================
# Coord-head loader
# ===========================================================================

def _load_coord_head(ckpt_path: str, device: str) -> Optional[DepthPredictionTransformer]:
    logger = logging.getLogger(__name__)
    ckpt_dir = _resolve_spa_ckpt_dir(ckpt_path)
    coord_head_path = None
    for _name in ("coord_head.pt", "dpt_head.pt"):
        if (ckpt_dir / _name).exists():
            coord_head_path = ckpt_dir / _name
            break
    if coord_head_path is None:
        logger.info(f"[coord] no coord_head.pt / dpt_head.pt in {ckpt_dir}")
        return None
    state = torch.load(str(coord_head_path), map_location="cpu", weights_only=True)
    d_model, hidden_dim = state["input_proj.weight"].shape
    proj_out = state["output_proj.weight"].shape[0]
    upscale_factor = int(round((proj_out / 3) ** 0.5))
    coord_head = DepthPredictionTransformer(
        hidden_dim=hidden_dim, d_model=d_model, upscale_factor=upscale_factor,
    )
    coord_head.load_state_dict(state)
    coord_head = coord_head.to(device).to(torch.bfloat16).eval()
    logger.info(
        f"[coord] coord_head loaded "
        f"(hidden_dim={hidden_dim}, d_model={d_model}, upscale={upscale_factor})"
    )
    return coord_head


def _resize_xyz(
    xyz: np.ndarray, target_h: int, target_w: int,
    valid: Optional[np.ndarray] = None,
) -> torch.Tensor:
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
    xyz_sum   = (xyz_blocks * valid_blocks[..., None]).sum(axis=(1, 3))
    valid_cnt = valid_blocks.sum(axis=(1, 3))
    denom     = np.maximum(valid_cnt, 1)[..., None]
    xyz_mean  = xyz_sum / denom
    xyz_mean[valid_cnt == 0] = 0.0
    return torch.from_numpy(xyz_mean)


def _load_precomputed_coords(item: Dict[str, Any]) -> Optional[List[Dict]]:
    """Load per-view (pts3d, mask, camera_pose, image) from 3d_results/<index>/.

    `image` comes from `view_XXXX/image.png` — coord_esti.py saved it at the
    SAME shape as `pts3d.npy`, so feeding it to the processor (instead of the
    raw benchmark image) makes pixel resolution == geometric resolution and
    matches `_load_and_align_views` in train_dataset_qwen35.py.
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
        image_path = vd / "image.png"
        if not pts3d_path.exists():
            continue
        pts3d = np.load(str(pts3d_path))
        mask  = (np.load(str(mask_path)).astype(bool)
                 if mask_path.exists() else np.ones(pts3d.shape[:2], dtype=bool))
        pose  = (np.load(str(pose_path))
                 if pose_path.exists() else np.eye(4, dtype=np.float64))
        image = Image.open(str(image_path)).convert("RGB") if image_path.exists() else None
        results.append({"pts3d": pts3d, "camera_pose": pose, "mask": mask, "image": image})
    return results if results else None


def _build_image_xyz(
    coord_results: List[Dict],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> List[torch.Tensor]:
    """Block-mean per-pixel pts3d to the LLM patch grid. Uses pts3d as-is to
    match training (no T0_inv pose normalization)."""
    N = image_grid_thw.shape[0]
    xyz_list: List[torch.Tensor] = []
    for k in range(N):
        thw_k = image_grid_thw[k]
        llm_h = int(thw_k[1]) // spatial_merge_size
        llm_w = int(thw_k[2]) // spatial_merge_size
        if k < len(coord_results):
            r    = coord_results[k]
            pts  = r["pts3d"].astype(np.float32)
            mask = r["mask"]
            xyz_list.append(_resize_xyz(pts, llm_h, llm_w, valid=mask))
        else:
            xyz_list.append(torch.zeros(llm_h, llm_w, 3))
    return xyz_list


# ===========================================================================
# Prepare batch + run inference (SPA path; matches train_dataset.py prompt)
# ===========================================================================

def _prepare_batch_spa(
    item: Dict[str, Any],
    processor: Any,
    spatial_merge_size: int,
) -> Tuple[Dict, str, Optional[List[torch.Tensor]]]:
    """Tokenise one sample and load image_xyz from 3d_results. Strict: raises if
    xyz unavailable (matches the original `strict_xyz=True` behaviour).
    """
    from qwen_vl_utils import process_vision_info

    image_paths = item["image"]
    question = item.get("question", "")

    # Interleave each `<image>` placeholder with the corresponding image (see
    # answer_format.build_interleaved_content for the policy + the SpinBench
    # `<image>X</image>` pollution it fixes).
    content = build_interleaved_content(question, image_paths)

    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Pre-load pts3d/mask so we can Qwen-align in lockstep with images.
    # Without this, image goes to /28 via Qwen's BICUBIC smart_resize but
    # pts3d stays at MapAny shape — _build_image_xyz's integer-stride
    # block-mean would then truncate edge pixels (see train_dataset_qwen35.py).
    coord_results: Optional[List[Dict]] = None
    if item.get("image"):
        try:
            coord_results = _load_precomputed_coords(item)
        except Exception as exc:
            raise RuntimeError(
                f"[xyz-required] sample idx={item.get('index')}: "
                f"_load_precomputed_coords raised: {exc!r}"
            ) from exc

    # Qwen-align each view (image + pts3d + mask). When 3d_results is
    # available, swap the raw benchmark image for `view_XXXX/image.png` —
    # coord_esti.py saved both pts3d and image.png at the same MapAny native
    # shape, so pixel resolution == geometric resolution. Without this swap,
    # the raw benchmark image and pts3d have different shapes and
    # _qwen_align_view's pixel-alignment assertion fires. Mirrors training
    # (_load_and_align_views in train_dataset_qwen35.py) so train and eval
    # feed the model bit-equivalent (image, pts3d, mask) tuples.
    if image_inputs:
        aligned_imgs: list = []
        for k, img in enumerate(image_inputs):
            r = coord_results[k] if (coord_results is not None and k < len(coord_results)) else None
            if r is not None:
                src_img = r.get("image") or img
                img_q, pts_q, mask_q = _qwen_align_view(src_img, r["pts3d"], r["mask"])
                r["pts3d"] = pts_q
                r["mask"]  = mask_q
            else:
                img_q = _qwen_align_view(img, None, None)[0]
            aligned_imgs.append(img_q)
        image_inputs = aligned_imgs

    inputs = processor(
        text=[prompt_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt", padding=False,
    )

    image_xyz: Optional[List[torch.Tensor]] = None
    if item.get("image"):
        image_grid_thw = inputs.get("image_grid_thw")
        if (coord_results is not None
                and image_grid_thw is not None
                and len(coord_results) > 0):
            image_xyz = _build_image_xyz(
                coord_results, image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
        else:
            raise RuntimeError(
                f"[xyz-required] sample idx={item.get('index')}: "
                f"no usable 3d_results under "
                f"{Path(item.get('data_dir', '?')) / '3d_results' / str(item.get('index', '?'))}"
            )

    return inputs, prompt_text, image_xyz


def _run_inference_spa(
    inputs: Dict,
    image_xyz: Optional[List[torch.Tensor]],
    model: Any,
    processor: Any,
    method: str,
    flags: Dict[str, bool],
    max_new_tokens: int,
    coord_scale: float,
) -> str:
    """Generate one response. Routes per method exactly like
    evaluation.py:run_inference_spa, minus the vanilla branch (excluded by CLI)."""
    device = next(model.parameters()).device
    inputs_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    if flags["atten"]:
        if image_xyz is not None:
            model.model._eval_image_xyz = [xyz.to(device) for xyz in image_xyz]
        else:
            model.model._eval_image_xyz = None
        gen_kwargs: Dict[str, Any] = dict(
            **inputs_dev,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    elif flags["decouple"]:
        xyz_on_device = (
            [xyz.to(device) for xyz in image_xyz] if image_xyz is not None else None
        )
        with torch.no_grad():
            xyz_pos = model.model._compute_xyz_pos(
                input_ids         = inputs_dev["input_ids"],
                mm_token_type_ids = inputs_dev["mm_token_type_ids"],
                image_grid_thw    = inputs_dev.get("image_grid_thw"),
                attention_mask    = inputs_dev.get("attention_mask"),
                image_xyz         = xyz_on_device,
            )
        model.model.language_model._xyz_pos     = xyz_pos
        model.model.language_model._coord_scale = float(coord_scale)
        gen_kwargs = dict(
            **inputs_dev,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            coord_scale=coord_scale,
        )
        gen_kwargs.pop("mm_token_type_ids", None)
    else:
        # 4D M-RoPE branch (position_embedding / coordinate / rotation*)
        xyz_on_device = (
            [xyz.to(device) for xyz in image_xyz] if image_xyz is not None else None
        )
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
        gen_kwargs = dict(
            **inputs_dev,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            coord_scale=coord_scale,
        )
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
# Rotation R + coord-head readout
# ===========================================================================

def _get_coord_predictions(
    model: Any,
    inputs: Dict[str, Any],
    image_token_id: int,
    coord_head: DepthPredictionTransformer,
    spatial_merge_size: int,
    image_xyz: List[torch.Tensor],
    coord_scale: float,
) -> Optional[List[torch.Tensor]]:
    device = next(model.parameters()).device
    inputs_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    image_grid_thw = inputs_dev.get("image_grid_thw")
    if image_grid_thw is None:
        return None
    xyz_on_device = [x.to(device) for x in image_xyz]
    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids=inputs_dev["input_ids"],
            mm_token_type_ids=inputs_dev["mm_token_type_ids"],
            image_grid_thw=image_grid_thw,
            video_grid_thw=inputs_dev.get("video_grid_thw"),
            attention_mask=inputs_dev.get("attention_mask"),
            image_xyz=xyz_on_device,
            coord_scale=coord_scale,
        )

    captured: Dict[str, Any] = {}
    def _hook(_m, args):
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

    last_hidden = captured["h"][0]
    vis_positions = (
        inputs_dev["input_ids"][0] == image_token_id
    ).nonzero(as_tuple=True)[0]
    if len(vis_positions) == 0:
        return None

    sms = spatial_merge_size
    N = image_grid_thw.shape[0]
    preds: List[torch.Tensor] = []
    start = 0
    dtype = coord_head.input_proj.weight.dtype
    with torch.no_grad():
        for k in range(N):
            thw_k = image_grid_thw[k]
            llm_h = int(thw_k[1]) // sms
            llm_w = int(thw_k[2]) // sms
            n_tok = llm_h * llm_w
            if start + n_tok > len(vis_positions):
                break
            h_k = last_hidden[vis_positions[start: start + n_tok]].to(dtype)
            pred_k = coord_head(h_k, llm_h, llm_w)
            preds.append(pred_k.cpu().float())
            start += n_tok
    return preds if preds else None


def _compute_coord_mae(
    preds: List[torch.Tensor],
    gt_list: List[torch.Tensor],
) -> float:
    maes: List[float] = []
    for pred, gt in zip(preds, gt_list):
        gt_f = gt.float()
        if pred.shape == gt_f.shape:
            maes.append((pred - gt_f).abs().mean().item())
        else:
            pred_t = pred.permute(2, 0, 1).unsqueeze(0)
            pred_ds = F.adaptive_avg_pool2d(pred_t, gt_f.shape[:2])
            gt_t = gt_f.permute(2, 0, 1).unsqueeze(0)
            maes.append((pred_ds - gt_t).abs().mean().item())
    return float(np.mean(maes)) if maes else 0.0


# ===========================================================================
# Result packaging + metrics
# ===========================================================================

def _make_result(
    item: Dict, output: str, prompt: str, method: str,
) -> Dict:
    fmt = item.get("format_type", "select")
    if fmt == "robospatial":
        full_output = output
        prediction = output
    else:
        full_output = output
        prediction = (
            _extract_answer_number(full_output) if fmt == "fill"
            else _extract_answer_letter(full_output)
        )
    return {
        "method": method,
        "index": item.get("index", ""),
        "category": item.get("category", "unknown"),
        "format_type": fmt,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "prediction": prediction,
        "output": full_output,
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
        "image_paths": item.get("image", []),
        "prompt": "",
    }


def _mra_score(pred_str: str, gt_str: str) -> float:
    try:
        pred = float(pred_str); gt = float(gt_str)
        if gt == 0:
            return 1.0 if pred == 0 else 0.0
        return max(0.0, 1.0 - abs(pred - gt) / abs(gt))
    except (ValueError, TypeError):
        return 0.0


def _compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    total = len(results)
    correct = 0.0
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
    cat_accuracy = {c: cat_correct[c] / cat_total[c] for c in cat_total}
    metrics: Dict[str, Any] = {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": correct,
        "category_accuracy": cat_accuracy,
        "category_counts": dict(cat_total),
    }
    coord_maes = [r["coord_mae"] for r in results if r.get("coord_mae") is not None]
    if coord_maes:
        metrics["coord_mae_mean"] = float(np.mean(coord_maes))
        metrics["coord_mae_std"]  = float(np.std(coord_maes))
        metrics["coord_mae_n"]    = len(coord_maes)
    return metrics


def _log_metrics(metrics: Dict, label: str, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info(f"RESULTS — {label}")
    logger.info("=" * 60)
    logger.info(f"  Total   : {metrics['total_samples']}")
    logger.info(f"  Correct : {metrics['correct_samples']}")
    logger.info(f"  Accuracy: {metrics['overall_accuracy']:.2%}")
    if "coord_mae_mean" in metrics:
        logger.info(
            f"  Coord MAE: {metrics['coord_mae_mean']:.4f} "
            f"± {metrics['coord_mae_std']:.4f}  (n={metrics['coord_mae_n']})"
        )
    logger.info("  Per-category accuracy:")
    for cat, acc in sorted(metrics["category_accuracy"].items()):
        n = metrics["category_counts"].get(cat, 0)
        logger.info(f"    {cat:35s}: {acc:6.2%}  ({n} samples)")
    logger.info("=" * 60)


# ===========================================================================
# Per-GPU pass: run inference once over a data shard
# ===========================================================================

def _evaluate_one_pass(
    data: List[Dict],
    ckpt: str,
    model_path: str,
    method: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: Path,
    tag: str,                       # "normal" | "zero"
    zero_xyz: bool,
    device: str,
    xyz_rope_dim: int,
) -> List[Dict]:
    logger = logging.getLogger(__name__)
    flags = _method_flags(method)

    spa_model, spa_proc = _load_spa_model(
        model_path, ckpt, device,
        decouple=flags["decouple"],
        atten=flags["atten"], xyz_rope_dim=xyz_rope_dim,
    )

    cfg_path = Path(model_path) / "config.json"
    with open(cfg_path) as f:
        _vcfg = json.load(f).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    logger.info(f"[{tag}] spatial_merge_size={spatial_merge_size}")

    image_token_id_val: Optional[int] = None
    _iid = spa_proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if isinstance(_iid, int) and _iid != spa_proc.tokenizer.unk_token_id:
        image_token_id_val = _iid

    use_coord_head = method == "coordinate"
    spa_coord_head = _load_coord_head(ckpt, device) if use_coord_head else None

    if zero_xyz:
        logger.info(f"[{tag}] image_xyz will be ZEROED before each forward")

    results: List[Dict] = []
    for item in tqdm(data, desc=f"[{device}|{method}|{tag}]"):
        try:
            inputs, prompt, image_xyz = _prepare_batch_spa(
                item, spa_proc, spatial_merge_size,
            )

            if zero_xyz and image_xyz is not None:
                image_xyz = [torch.zeros_like(x) for x in image_xyz]

            output = _run_inference_spa(
                inputs, image_xyz, spa_model, spa_proc,
                method, flags, max_new_tokens, coord_scale,
            )
            res = _make_result(item, output, prompt, method)

            if (use_coord_head
                    and spa_coord_head is not None
                    and image_token_id_val is not None
                    and image_xyz is not None):
                preds = _get_coord_predictions(
                    spa_model, inputs,
                    image_token_id_val, spa_coord_head,
                    spatial_merge_size, image_xyz, coord_scale,
                )
                if preds is not None:
                    res["coord_mae"] = _compute_coord_mae(preds, image_xyz)

            results.append(res)
        except Exception as exc:
            logger.error(
                f"[{tag}] idx={item.get('index')}: {exc}", exc_info=True,
            )
            results.append(_error_result(item, exc, method))

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{method}_{tag}_{device.replace(':', '')}.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[{tag}] saved {len(results)} results → {partial_path}")
    return results


# ===========================================================================
# Multi-GPU launcher
# ===========================================================================

def _worker(
    gpu_id: str,
    data_shard: List[Dict],
    ckpt: str,
    model_path: str,
    method: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: str,
    log_file: str,
    tag: str,
    zero_xyz: bool,
    xyz_rope_dim: int,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))
    logger.info(
        f"[Worker {gpu_id}|{method}|{tag}] start — "
        f"{len(data_shard)} samples on {device}"
    )
    _evaluate_one_pass(
        data=data_shard, ckpt=ckpt, model_path=model_path, method=method,
        coord_scale=coord_scale, max_new_tokens=max_new_tokens,
        output_dir=Path(output_dir), tag=tag, zero_xyz=zero_xyz,
        device=device, xyz_rope_dim=xyz_rope_dim,
    )
    logger.info(f"[Worker {gpu_id}|{method}|{tag}] done.")


def _run_pass(
    dataset: List[Dict],
    gpu_ids: List[str],
    ckpt: str,
    model_path: str,
    method: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: Path,
    log_file: Path,
    tag: str,
    zero_xyz: bool,
    xyz_rope_dim: int,
) -> List[Dict]:
    logger = logging.getLogger(__name__)
    for p in output_dir.glob(f"{method}_{tag}_cuda*.json"):
        logger.info(f"removing stale partial {p}")
        p.unlink()

    shards = chunk_dataset(dataset, len(gpu_ids))
    procs: List[mp.Process] = []
    spawned: List[str] = []
    for gid, shard in zip(gpu_ids, shards):
        p = mp.Process(
            target=_worker,
            args=(
                gid, shard, ckpt, model_path, method,
                coord_scale, max_new_tokens, str(output_dir), str(log_file),
                tag, zero_xyz, xyz_rope_dim,
            ),
        )
        p.start()
        procs.append(p)
        spawned.append(gid)

    failed: List[Tuple[str, int]] = []
    for p, gid in zip(procs, spawned):
        p.join()
        if p.exitcode != 0:
            failed.append((gid, int(p.exitcode) if p.exitcode is not None else -1))
    if failed:
        details = ", ".join(f"GPU {g}(exit={ec})" for g, ec in failed)
        raise RuntimeError(f"[{tag}] worker(s) failed: {details}")

    merged: List[Dict] = []
    missing: List[str] = []
    for gid in spawned:
        path = output_dir / f"{method}_{tag}_cuda{gid}.json"
        if path.exists():
            with open(path) as f:
                merged.extend(json.load(f))
        else:
            missing.append(gid)
    if missing:
        raise RuntimeError(
            f"[{tag}] missing partials from GPU(s) {missing} despite clean exit"
        )
    if len(merged) != len(dataset):
        raise RuntimeError(
            f"[{tag}] merged {len(merged)} samples but dataset has {len(dataset)}"
        )
    merged.sort(key=lambda r: str(r.get("index", "")))

    with open(output_dir / f"results_{method}_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="xyz=0 vs normal-xyz ablation for any xyz-using SPA method.",
    )
    ap.add_argument("--method", required=True, choices=list(_XYZ_METHODS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument(
        "--dataset", default="mindcube",
        choices=[
            "mmsibench", "mindcube",
            "sat", "sat_real",
            "sparbench_multi_view", "sparbench_single_view", "sparbench_mv",
            "spinbench", "robospatial", "viewspatial",
            "omnispatial_pt", "embspatial",
        ],
    )
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--coord_scale", type=float, default=100.0)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument(
        "--xyz_rope_dim", type=int, default=66,
        help="Match training; only relevant for --method decouple.",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--only", choices=["normal", "zero", "both"], default="both",
        help="Which pass(es) to run. Default both.",
    )
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)

    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("at least one CUDA device is required.")
    gpu_ids = [str(i) for i in range(n_gpu)]

    dataset = load_testing_dataset(
        Path(args.data_dir).resolve(), limit=args.limit, dataset=args.dataset,
    )

    logger.info("=" * 65)
    logger.info(f"xyz=0 ablation  [method={args.method}]")
    logger.info("=" * 65)
    logger.info(f"  ckpt       : {args.ckpt}")
    logger.info(f"  model_path : {args.model_path}")
    logger.info(f"  dataset    : {args.dataset} ({len(dataset)} samples)")
    logger.info(f"  data_dir   : {args.data_dir}")
    logger.info(f"  GPUs       : {gpu_ids}")
    logger.info(f"  output_dir : {out_dir}")
    logger.info(f"  passes     : {args.only}")
    logger.info("=" * 65)

    with open(out_dir / "configuration.json", "w") as f:
        json.dump(
            vars(args) | {
                "gpus": gpu_ids,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_samples": len(dataset),
            },
            f, indent=2, ensure_ascii=False,
        )

    metrics: Dict[str, Dict] = {}

    if args.only in ("normal", "both"):
        logger.info("\n>>> PASS 1: normal xyz")
        results_normal = _run_pass(
            dataset, gpu_ids, args.ckpt, args.model_path, args.method,
            args.coord_scale, args.max_new_tokens,
            out_dir, log_file, tag="normal", zero_xyz=False,
            xyz_rope_dim=args.xyz_rope_dim,
        )
        m_normal = _compute_metrics(results_normal)
        _log_metrics(m_normal, f"{args.method} — normal xyz", logger)
        with open(out_dir / f"metrics_{args.method}_normal.json", "w") as f:
            json.dump(m_normal, f, ensure_ascii=False, indent=2)
        metrics["normal"] = m_normal

    if args.only in ("zero", "both"):
        logger.info("\n>>> PASS 2: image_xyz = 0")
        results_zero = _run_pass(
            dataset, gpu_ids, args.ckpt, args.model_path, args.method,
            args.coord_scale, args.max_new_tokens,
            out_dir, log_file, tag="zero", zero_xyz=True,
            xyz_rope_dim=args.xyz_rope_dim,
        )
        m_zero = _compute_metrics(results_zero)
        _log_metrics(m_zero, f"{args.method} — xyz = 0", logger)
        with open(out_dir / f"metrics_{args.method}_zero.json", "w") as f:
            json.dump(m_zero, f, ensure_ascii=False, indent=2)
        metrics["zero"] = m_zero

    if "normal" in metrics and "zero" in metrics:
        a = metrics["normal"]["overall_accuracy"]
        b = metrics["zero"]["overall_accuracy"]
        delta_acc = b - a
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"xyz=0 ABLATION SUMMARY  [{args.method}]")
        logger.info("=" * 70)
        logger.info(
            f"  {'normal xyz':<28s} {a:.2%}  "
            f"({metrics['normal']['correct_samples']}/{metrics['normal']['total_samples']})"
        )
        logger.info(
            f"  {'xyz = 0':<28s} {b:.2%}  "
            f"({metrics['zero']['correct_samples']}/{metrics['zero']['total_samples']})"
        )
        logger.info(f"  {'Δ accuracy (zero - normal)':<28s} {delta_acc:+.2%}")

        summary: Dict[str, Any] = {
            "method": args.method,
            "normal_accuracy": a,
            "zero_accuracy": b,
            "delta_accuracy_zero_minus_normal": delta_acc,
            "n_samples": metrics["normal"]["total_samples"],
        }
        a_mae = metrics["normal"].get("coord_mae_mean")
        b_mae = metrics["zero"].get("coord_mae_mean")
        if a_mae is not None and b_mae is not None:
            delta_mae = b_mae - a_mae
            logger.info(f"  {'normal coord_mae':<28s} {a_mae:.4f}")
            logger.info(f"  {'zero coord_mae':<28s} {b_mae:.4f}")
            logger.info(f"  {'Δ coord_mae (zero - normal)':<28s} {delta_mae:+.4f}")
            summary.update({
                "normal_coord_mae_mean": a_mae,
                "zero_coord_mae_mean": b_mae,
                "delta_coord_mae_zero_minus_normal": delta_mae,
            })
        logger.info("=" * 70)
        logger.info(
            "  Interpretation: a near-zero Δ means xyz is being ignored at "
            "inference; a clearly negative Δ accuracy / positive Δ coord_mae "
            "means xyz is actually being consumed."
        )
        with open(out_dir / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"all done → {out_dir}")


if __name__ == "__main__":
    main()
