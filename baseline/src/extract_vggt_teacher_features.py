#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-sample VGGT teacher features for SPAR multi-image training."
    )
    parser.add_argument(
        "--data-json",
        default="datasets/train/SPAR_7M/spar/train_10k.json",
        help="Path to training json file.",
    )
    parser.add_argument(
        "--spar-root",
        default="datasets/train/SPAR_7M/spar",
        help="SPAR root containing scannet/scannetpp/structured3d folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/train/SPAR_7M/spar/vggt_teacher_features",
        help="Output directory to save per-sample .npz teacher features.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/VGGT-1B/model.pt",
        help="Path to local VGGT checkpoint (.pt).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start sample index (inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="End sample index (exclusive). -1 means all samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Process at most this many samples after start-index. -1 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature files.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (cuda/cuda:0/cpu).",
    )
    return parser.parse_args()


def pick_dtype(device: str):
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    capability = torch.cuda.get_device_capability(torch.device(device))
    return torch.bfloat16 if capability[0] >= 8 else torch.float16


def resolve_image_path(image_rel: str, spar_root: Path):
    roots = [
        spar_root / "scannetpp" / "images",
        spar_root / "scannet" / "images",
        spar_root / "structured3d" / "images",
    ]
    for root in roots:
        candidate = root / image_rel
        if candidate.exists():
            return candidate
    return None


def load_vggt_model(repo_root: Path, checkpoint_path: Path, device: str):
    vggt_root = repo_root / "external" / "vggt"
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    model = VGGT()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    msg = model.load_state_dict(payload, strict=False)
    print(f"[VGGT] load_state_dict: {msg}")
    model = model.to(device).eval()
    return model, load_and_preprocess_images


def find_repo_root(script_path: Path) -> Path:
    # Resolve against the spatial_planning root regardless of where this script is located.
    for candidate in [script_path.parent] + list(script_path.parents):
        if (candidate / "external" / "vggt").exists() and (candidate / "datasets").exists():
            return candidate
    # Fallback keeps behavior predictable if expected folders are absent.
    return script_path.parent


def resolve_path(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def main():
    args = parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    data_json = resolve_path(args.data_json, repo_root)
    spar_root = resolve_path(args.spar_root, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    checkpoint_path = resolve_path(args.checkpoint, repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_json.exists():
        raise FileNotFoundError(f"data json not found: {data_json}")
    if not spar_root.exists():
        raise FileNotFoundError(f"spar root not found: {spar_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint_path}")

    with data_json.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    start = max(0, args.start_index)
    end = len(samples) if args.end_index < 0 else min(len(samples), args.end_index)
    if args.limit >= 0:
        end = min(end, start + args.limit)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable, falling back to CPU")
        device = "cpu"
    dtype = pick_dtype(device)

    model, load_and_preprocess_images = load_vggt_model(repo_root, checkpoint_path, device)

    processed = 0
    skipped_exist = 0
    skipped_invalid = 0
    failed = 0

    for idx in tqdm(range(start, end), desc="Extract VGGT teacher features"):
        sample = samples[idx]
        save_path = output_dir / f"{idx:06d}.npz"

        if save_path.exists() and not args.overwrite:
            skipped_exist += 1
            continue

        images = sample.get("image")
        if images is None:
            skipped_invalid += 1
            continue
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            skipped_invalid += 1
            continue

        image_paths = []
        missing = False
        for rel in images:
            resolved = resolve_image_path(str(rel), spar_root)
            if resolved is None:
                missing = True
                break
            image_paths.append(str(resolved))

        if missing:
            skipped_invalid += 1
            continue

        try:
            image_tensor = load_and_preprocess_images(image_paths)
            image_tensor = image_tensor.to(device)
            if dtype != torch.float32:
                image_tensor = image_tensor.to(dtype)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                autocast_enabled = device.startswith("cuda")
                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=dtype):
                    aggregated_tokens_list, ps_idx = model.aggregator(image_tensor)

            feature = aggregated_tokens_list[-1].detach().float().cpu().numpy()
            if feature.ndim == 4 and feature.shape[0] == 1:
                feature = feature[0]

            ps_idx_val = int(np.array(ps_idx.detach().cpu() if torch.is_tensor(ps_idx) else ps_idx).reshape(-1)[0])
            if feature.ndim != 3 or ps_idx_val < 0 or ps_idx_val >= feature.shape[1]:
                skipped_invalid += 1
                continue

            feature_3d = feature[:, ps_idx_val:, :]
            np.savez_compressed(
                save_path,
                feature_3d=feature_3d,
                feature=feature,
                ps_idx=np.array(ps_idx_val, dtype=np.int32),
                sample_id=str(sample.get("id", idx)),
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[warn] sample {idx} failed: {exc}")

    print("[done] VGGT teacher feature extraction summary")
    print(f"  processed      : {processed}")
    print(f"  skipped_exist  : {skipped_exist}")
    print(f"  skipped_invalid: {skipped_invalid}")
    print(f"  failed         : {failed}")
    print(f"  output_dir     : {output_dir}")


if __name__ == "__main__":
    main()
