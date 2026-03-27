#!/usr/bin/env python3
"""
Launcher for 3DRS training with two model presets:
- qwen2.5vl-3b
- qwen3.5-4b

This script delegates training to 3DRS/llava/train/train_3d.py so the
original training loss (including grounding + 3D distillation terms) remains
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ModelPreset:
    model_name_or_path: str
    prompt_version: str = "qwen_1_5"
    local_candidates: tuple[str, ...] = ()


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "qwen2.5vl-3b": ModelPreset(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        local_candidates=(
            "checkpoints/Qwen2.5-VL-3B-Instruct",
            "checkpoints/Qwen2_5_VL_3B_Instruct",
            "checkpoints/Qwen2.5-VL-3B",
        ),
    ),
    "qwen3.5-4b": ModelPreset(
        model_name_or_path="Qwen/Qwen3.5-4B-Instruct",
        local_candidates=(
            "checkpoints/Qwen3.5-4B-Instruct",
            "checkpoints/Qwen3.5-4B",
            "checkpoints/Qwen3_5_4B_Instruct",
        ),
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "qwen2.5": "qwen2.5vl-3b",
    "qwen25": "qwen2.5vl-3b",
    "qwen3.5": "qwen3.5-4b",
    "qwen35": "qwen3.5-4b",
}


def normalize_model_key(raw_model_key: str) -> str:
    key = raw_model_key.strip().lower()
    if key in MODEL_PRESETS:
        return key
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    choices = sorted(list(MODEL_PRESETS.keys()) + list(MODEL_ALIASES.keys()))
    raise ValueError(f"Unknown --model '{raw_model_key}'. Supported values: {', '.join(choices)}")


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    repo_root = Path(__file__).resolve().parent
    spatial_root = repo_root.parent
    threedrs_root = repo_root / "repo" / "3DRS"
    default_vggt_root = spatial_root / "external" / "vggt"
    default_data_path = spatial_root / "datasets" / "train" / "SPAR_7M" / "spar" / "train_10k.json"
    default_image_folder = spatial_root / "datasets" / "train" / "SPAR_7M" / "spar" / "scannetpp" / "images"

    parser = argparse.ArgumentParser(
        description=(
            "Train 3DRS with the original loss implementation and selectable "
            "Qwen model presets."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model preset. Canonical: qwen2.5vl-3b | qwen3.5-4b; "
            "aliases: qwen2.5/qwen25, qwen3.5/qwen35."
        ),
    )
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--allow-incompatible-qwen35",
        action="store_true",
        help=(
            "Allow running qwen3.5-4b preset even when checkpoint architecture "
            "is incompatible with 3DRS qwen2-style backbone."
        ),
    )

    parser.add_argument("--repo-3drs", type=Path, default=threedrs_root)
    parser.add_argument("--vggt-root", type=Path, default=default_vggt_root)

    parser.add_argument("--data-path", "--data-yaml", dest="data_path", default=str(default_data_path))
    parser.add_argument("--image-folder", default=str(default_image_folder))
    parser.add_argument("--video-folder", default="data")
    parser.add_argument("--embodiedscan-folder", default="data/embodiedscan/")
    parser.add_argument("--teacher-feature-dir", default=None)
    parser.add_argument(
        "--require-teacher-feature",
        dest="require_teacher_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require per-sample teacher 3D features for distillation.",
    )
    parser.add_argument(
        "--image-only",
        dest="image_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force image-only training. Video samples/assets are ignored.",
    )

    parser.add_argument("--vision-tower", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--deepspeed-config", default="scripts/zero3.json")
    parser.add_argument("--master-port", type=int, default=43000)

    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--frame-sampling-strategy", default="uniform")

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dry-run", action="store_true")

    args, passthrough = parser.parse_known_args()
    args.model = normalize_model_key(args.model)
    return args, passthrough


def resolve_model_name_or_path(args: argparse.Namespace, preset: ModelPreset, workspace_root: Path) -> str:
    if args.model_name_or_path:
        return args.model_name_or_path

    for rel_path in preset.local_candidates:
        candidate = (workspace_root / rel_path).resolve()
        if candidate.exists():
            return str(candidate)

    return preset.model_name_or_path


def compute_grad_accumulation(args: argparse.Namespace) -> tuple[int, int]:
    world_batch = max(1, args.num_gpus) * max(1, args.per_device_train_batch_size)
    grad_acc_steps = max(1, args.global_batch_size // world_batch)
    effective_global_batch = grad_acc_steps * world_batch
    return grad_acc_steps, effective_global_batch


def _resolve_path_from_repo(repo_3drs: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_3drs / candidate).resolve()


def _sample_list_contains_video(items: object) -> bool:
    if not isinstance(items, list):
        return False
    for sample in items[:256]:
        if isinstance(sample, dict) and "video" in sample:
            return True
    return False


def _json_has_video_samples(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _sample_list_contains_video(payload)


def _jsonl_has_video_samples(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 256:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "video" in obj:
                return True
    return False


def _expand_brace_json_paths(raw_path: str, repo_3drs: Path) -> List[Path]:
    match = re.match(r"^(.*)\{(.*)\}\.json$", raw_path)
    if not match:
        return []
    base_path, file_pattern = match.groups()
    file_names = [name.strip() for name in file_pattern.split(",") if name.strip()]
    out: List[Path] = []
    for name in file_names:
        expanded = f"{base_path}{name}.json"
        out.append(_resolve_path_from_repo(repo_3drs, expanded))
    return out


def data_path_requires_video_assets(raw_data_path: str, repo_3drs: Path) -> bool:
    expanded_paths = _expand_brace_json_paths(raw_data_path, repo_3drs)
    if expanded_paths:
        candidates = expanded_paths
    else:
        candidates = [_resolve_path_from_repo(repo_3drs, raw_data_path)]

    for path in candidates:
        suffix = path.suffix.lower()
        if suffix == ".json":
            if _json_has_video_samples(path):
                return True
        elif suffix == ".jsonl":
            if _jsonl_has_video_samples(path):
                return True
        else:
            # Conservative fallback for unsupported formats (e.g. yaml).
            return True
    return False


def validate_training_data_assets(args: argparse.Namespace, repo_3drs: Path) -> None:
    data_path = _resolve_path_from_repo(repo_3drs, args.data_path)
    image_folder = _resolve_path_from_repo(repo_3drs, args.image_folder)
    video_folder = _resolve_path_from_repo(repo_3drs, args.video_folder)
    embodiedscan_folder = _resolve_path_from_repo(repo_3drs, args.embodiedscan_folder)

    missing_files: List[Path] = []
    if not data_path.exists():
        missing_files.append(data_path)
    if not image_folder.exists():
        missing_files.append(image_folder)

    requires_video_assets = (not args.image_only) and data_path_requires_video_assets(args.data_path, repo_3drs)
    if requires_video_assets:
        if not video_folder.exists():
            missing_files.append(video_folder)
        if not embodiedscan_folder.exists():
            missing_files.append(embodiedscan_folder)

        for split in ("train", "val", "test"):
            pkl_file = embodiedscan_folder / f"embodiedscan_infos_{split}.pkl"
            if not pkl_file.exists():
                missing_files.append(pkl_file)

        metadata_dir = repo_3drs / "data" / "metadata"
        for name in ("scannet_train_gt_box.json", "scannet_val_pred_box.json"):
            metadata_file = metadata_dir / name
            if not metadata_file.exists():
                missing_files.append(metadata_file)

    if missing_files:
        unique_missing = list(dict.fromkeys(missing_files))
        docs_path = repo_3drs / "scripts" / "3d" / "preprocessing" / "README.md"
        missing_text = "\n".join(f"  - {p}" for p in unique_missing)
        raise FileNotFoundError(
            "Missing required 3DRS data assets:\n"
            f"{missing_text}\n"
            "Please prepare the dataset following:\n"
            f"  - {docs_path}\n"
            "Note: video samples require ScanNet metadata from repo_3drs/data/metadata (hardcoded).\n"
            "You can override paths, for example:\n"
            "  - --video-folder /path/to/data\n"
            "  - --embodiedscan-folder /path/to/embodiedscan\n"
            "Then rerun training."
        )

    args.data_path = str(data_path)
    args.image_folder = str(image_folder)
    args.video_folder = str(video_folder)
    args.embodiedscan_folder = str(embodiedscan_folder)


def build_command(args: argparse.Namespace, model_name_or_path: str, torchrun_bin: str) -> tuple[List[str], int, int]:
    preset = MODEL_PRESETS[args.model]
    run_name = args.run_name or f"3drs-{args.model}"
    output_dir = args.output_dir or f"./ckpt/{run_name}"

    grad_acc_steps, effective_global_batch = compute_grad_accumulation(args)

    cmd = [
        torchrun_bin,
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(args.num_gpus),
        "--master_port",
        str(args.master_port),
        "llava/train/train_3d.py",
        "--deepspeed",
        args.deepspeed_config,
        "--model_name_or_path",
        model_name_or_path,
        "--version",
        preset.prompt_version,
        "--data_path",
        args.data_path,
        "--image_folder",
        args.image_folder,
        "--image_only",
        "True" if args.image_only else "False",
        "--require_teacher_feature",
        "True" if args.require_teacher_feature else "False",
        "--video_folder",
        args.video_folder,
        "--embodiedscan_folder",
        args.embodiedscan_folder,
        # LoRA mode (instead of full-parameter mm_tunable_parts finetuning).
        "--lora_enable",
        "True",
        "--lora_r",
        "64",
        "--lora_alpha",
        "16",
        "--lora_dropout",
        "0.05",
        "--lora_bias",
        "none",
        "--vision_tower",
        args.vision_tower,
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "anyres_max_9",
        "--image_grid_pinpoints",
        "(1x1),...,(6x6)",
        "--mm_patch_merge_type",
        "spatial_unpad",
        "--bf16",
        "True",
        "--run_name",
        run_name,
        "--output_dir",
        output_dir,
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(grad_acc_steps),
        "--eval_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        "2000",
        "--save_total_limit",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        "0.0",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--tf32",
        "True",
        "--model_max_length",
        "32768",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "1",
        "--lazy_preprocess",
        "True",
        "--torch_compile",
        "True",
        "--torch_compile_backend",
        "inductor",
        "--dataloader_drop_last",
        "True",
        "--mm_newline_position",
        "grid",
        "--add_spatial_instruction",
        "True",
        "--force_sample",
        "True",
        "--mm_spatial_pool_stride",
        "2",
        "--frame_sampling_strategy",
        args.frame_sampling_strategy,
        "--frames_upbound",
        str(args.frames_upbound),
    ]
    if args.teacher_feature_dir:
        cmd += ["--teacher_feature_dir", args.teacher_feature_dir]
    return cmd, grad_acc_steps, effective_global_batch


def _load_local_model_config(model_name_or_path: str) -> dict | None:
    model_dir = Path(model_name_or_path)
    config_path = model_dir / "config.json"
    if not model_dir.exists() or not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_uses_nested_language_model_keys(model_name_or_path: str) -> bool:
    model_dir = Path(model_name_or_path)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    sample_keys = list(weight_map.keys())[:128]
    return any(key.startswith("model.language_model.") for key in sample_keys)


def validate_model_compatibility(args: argparse.Namespace, model_name_or_path: str) -> None:
    # train_3d.py routes all qwen checkpoints to llava_qwen (Qwen2-style).
    # Newer Qwen-VL/Qwen3.5 checkpoints can be structurally incompatible.
    cfg = _load_local_model_config(model_name_or_path)
    uses_nested_lm = _checkpoint_uses_nested_language_model_keys(model_name_or_path)

    if args.model != "qwen3.5-4b":
        return

    model_type = cfg.get("model_type") if isinstance(cfg, dict) else None
    architectures = cfg.get("architectures", []) if isinstance(cfg, dict) else []
    text_cfg = cfg.get("text_config", {}) if isinstance(cfg, dict) else {}
    layer_types = text_cfg.get("layer_types", []) if isinstance(text_cfg, dict) else []

    has_linear_attention_layers = isinstance(layer_types, list) and any(
        t == "linear_attention" for t in layer_types
    )
    is_conditional_generation = any(
        isinstance(a, str) and a.endswith("ForConditionalGeneration") for a in architectures
    )

    incompatible = uses_nested_lm or has_linear_attention_layers or is_conditional_generation
    if incompatible and not args.allow_incompatible_qwen35:
        raise RuntimeError(
            "Incompatible qwen3.5 checkpoint for current 3DRS backend. "
            "Detected a Qwen-VL/Qwen3.5 conditional-generation layout "
            "(e.g., model.language_model.* and/or linear_attention layers), "
            "while 3DRS llava_qwen expects Qwen2-style causal-lm weights. "
            "This would silently initialize most language weights from scratch. "
            "Please pass a compatible text-style Qwen checkpoint, or rerun with "
            "--allow-incompatible-qwen35 only if you intentionally want random-init training."
        )


def main() -> int:
    args, passthrough = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent

    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be >= 1")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be >= 1")
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be >= 1")

    repo_3drs = args.repo_3drs.resolve()
    if not repo_3drs.exists():
        raise FileNotFoundError(f"3DRS repo not found: {repo_3drs}")

    vggt_root = args.vggt_root.resolve()
    if not vggt_root.exists():
        raise FileNotFoundError(f"VGGT root not found: {vggt_root}")

    if not args.dry_run:
        validate_training_data_assets(args, repo_3drs)

    requires_video_assets = (not args.image_only) and data_path_requires_video_assets(args.data_path, repo_3drs)
    passthrough_flags = set(passthrough)
    if not requires_video_assets:
        # Image-only runs can hit backward recompute mismatches with the default
        # compile+checkpointing combo; add safe overrides unless user set them.
        if "--gradient_checkpointing" not in passthrough_flags and "--gradient-checkpointing" not in passthrough_flags:
            passthrough += ["--gradient_checkpointing", "False"]
        if "--torch_compile" not in passthrough_flags and "--torch-compile" not in passthrough_flags:
            passthrough += ["--torch_compile", "False"]

    if args.image_only and data_path_requires_video_assets(args.data_path, repo_3drs):
        print("[3drs] image-only mode enabled: video samples in dataset will be ignored.")

    preset = MODEL_PRESETS[args.model]
    model_name_or_path = resolve_model_name_or_path(args, preset, workspace_root)
    validate_model_compatibility(args, model_name_or_path)

    torchrun_bin = shutil.which("torchrun")
    if torchrun_bin is None:
        candidate = Path(sys.executable).resolve().parent / "torchrun"
        if candidate.exists():
            torchrun_bin = str(candidate)
        else:
            raise FileNotFoundError(
                "torchrun not found. Please activate your training environment "
                "or install pytorch distributed launcher in that environment."
            )

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Ensure dependencies are importable. We intentionally do not force the
    # repository's vendored transformers source to avoid tokenizers version
    # conflicts in the active environment.
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_items = [str(vggt_root), str(repo_3drs)]
    if existing_pythonpath:
        pythonpath_items.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)

    cmd, grad_acc_steps, effective_global_batch = build_command(args, model_name_or_path, torchrun_bin)
    cmd += passthrough
    printable_cmd = " ".join(shlex.quote(part) for part in cmd)

    print(f"[3drs] cwd: {repo_3drs}")
    print(f"[3drs] model preset: {args.model}")
    print(f"[3drs] model_name_or_path: {model_name_or_path}")
    print(f"[3drs] data_path: {args.data_path}")
    print(f"[3drs] image_folder: {args.image_folder}")
    print(f"[3drs] image_only: {args.image_only}")
    print(f"[3drs] teacher_feature_dir: {args.teacher_feature_dir}")
    print(f"[3drs] require_teacher_feature: {args.require_teacher_feature}")
    print(f"[3drs] vggt root: {vggt_root}")
    print(f"[3drs] per_device_train_batch_size: {args.per_device_train_batch_size}")
    print(f"[3drs] gradient_accumulation_steps: {grad_acc_steps}")
    print(f"[3drs] effective_global_batch_size: {effective_global_batch}")
    print(f"[3drs] command: {printable_cmd}")

    if effective_global_batch != args.global_batch_size:
        print(
            "[3drs][warning] Requested global batch size "
            f"{args.global_batch_size} is not divisible by "
            f"num_gpus * per_device_train_batch_size ({args.num_gpus * args.per_device_train_batch_size}). "
            f"Using effective_global_batch_size={effective_global_batch}."
        )

    # 3DRS train_3d.py uses local_files_only=True in the qwen branch. If this is a
    # remote HF id, make the caveat explicit to avoid confusing runtime failures.
    if "/" in model_name_or_path and not Path(model_name_or_path).exists():
        print(
            "[3drs][warning] model_name_or_path looks like a HF repo id. "
            "train_3d.py loads qwen models with local_files_only=True, so please "
            "ensure weights are already cached locally or pass a local checkpoint path."
        )

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(repo_3drs), env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
