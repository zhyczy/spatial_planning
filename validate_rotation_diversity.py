"""Measure case-by-case diversity of rotation_enc outputs on an eval set.

Purpose
-------
Quantify whether the SFT-trained rotation_enc (train_alternate.py) learns a
per-sample R or collapses to a near-constant "dataset canonical" R. This
settles the debate from the conversation: if R_i ≈ R_bar for all samples,
the RL approach over 24 anchors is a non-starter because per-sample signal
is absent by construction.

Usage (single-GPU):
    /egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python \\
        spatial_planning/validate_rotation_diversity.py \\
        --ckpt spatial_planning/train_records/rotation_alternate_mindcube/step_1600 \\
        --dataset mindcube \\
        --data_dir spatial_planning/datasets/evaluation/MindCube
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from evaluation import (                           # noqa: E402
    _compute_rotation_R,
    _load_rotation_enc,
    load_spa_model,
    prepare_batch_spa,
)
from src.dataset import load_testing_dataset        # noqa: E402


def _mean_rotation(R_stack: np.ndarray) -> np.ndarray:
    """Closest SO(3) element to the element-wise mean under Frobenius norm.

    Projects the mean matrix to SO(3) via SVD with det-correction.
    """
    M = R_stack.mean(axis=0)
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    S[2, 2] = float(np.sign(np.linalg.det(U @ Vt)))
    return U @ S @ Vt


def _angle_from(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    rel = R_a @ R_b.T
    tr = float(np.trace(rel))
    cos_t = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(cos_t))


def _euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) in degrees under ZYX Tait-Bryan convention."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) > 1e-6:
        yaw  = math.atan2(R[1, 0],  R[0, 0])
        roll = math.atan2(R[2, 1],  R[2, 2])
    else:
        yaw  = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(_ROOT / "checkpoints" / "Qwen3.5-4B"))
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--dataset",    default="mindcube",
                    choices=["mindcube", "spinbench", "sat_real"])
    ap.add_argument("--data_dir",   default=str(_ROOT / "datasets" / "evaluation" / "MindCube"))
    ap.add_argument("--coord_scale", type=float, default=100.0)
    ap.add_argument("--limit",       type=int, default=None)
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--output_json", default=None,
                    help="Optional path to dump per-sample R records.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    # ── load dataset ──
    data = load_testing_dataset(Path(args.data_dir), limit=args.limit,
                                dataset=args.dataset)
    log.info(f"Loaded {len(data)} samples from {args.dataset}")

    # ── load model ──
    spa_model, spa_proc = load_spa_model(args.model_path, args.ckpt,
                                         device=args.device, vanilla=False)
    spa_model.eval()

    rotation_enc = _load_rotation_enc(
        args.ckpt, spa_model, device=args.device, expect_rl=False,
    )
    if rotation_enc is None:
        raise RuntimeError("rotation_enc.pt not found in ckpt")
    rotation_enc.eval()

    # Resolve spatial_merge_size and image_token_id
    cfg_path = Path(args.model_path) / "config.json"
    with open(cfg_path) as f:
        spatial_merge_size = int(
            json.load(f).get("vision_config", {}).get("spatial_merge_size", 2)
        )
    image_token_id = spa_proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ── run rotation_enc on each sample, collect R ──
    R_list:       list[np.ndarray]     = []
    categories:   list[str]            = []
    indices:      list[str]            = []
    angle_from_I: list[float]          = []

    for i, item in enumerate(data):
        try:
            inputs, _prompt, image_xyz = prepare_batch_spa(
                item, spa_proc, spatial_merge_size,
                use_coord=False, coord_scale=args.coord_scale,
                thinking=False, load_xyz=True,
            )
            if image_xyz is None:
                continue
            R, _cam_feat = _compute_rotation_R(
                spa_model, rotation_enc, inputs, image_xyz,
                image_token_id, spatial_merge_size, args.coord_scale,
            )
            if R is None:
                continue
            R_np = R.detach().float().cpu().numpy()
            R_list.append(R_np)
            categories.append(str(item.get("category", "unknown")))
            indices.append(str(item.get("index", i)))
            angle_from_I.append(_angle_from(R_np, np.eye(3)))
        except Exception as exc:
            log.warning(f"sample {item.get('index')}: {exc}")
            continue

        if (i + 1) % 50 == 0:
            log.info(f"  processed {i + 1}/{len(data)}  (valid={len(R_list)})")

    if not R_list:
        log.error("No valid rotations collected — aborting.")
        return

    R_stack = np.stack(R_list, axis=0)                   # (N, 3, 3)
    N = R_stack.shape[0]
    log.info(f"\nCollected {N} rotation matrices.\n")

    # ── 1. mean rotation (SO(3) projection of element-wise mean) ──
    R_bar = _mean_rotation(R_stack)

    # ── 2. per-sample angle to R_bar (case-by-case variance)  ──
    angle_to_bar = np.array(
        [_angle_from(R, R_bar) for R in R_stack], dtype=np.float64,
    )

    # ── 3. Frobenius deviation from mean  ──
    fro = np.linalg.norm(R_stack - R_bar, axis=(1, 2))

    # ── 4. Euler angle statistics (yaw/pitch/roll) ──
    eulers = np.array([_euler_zyx(R) for R in R_stack])  # (N, 3)

    log.info("=" * 70)
    log.info(f"ROTATION DIVERSITY REPORT  (N={N} samples, ckpt={args.ckpt})")
    log.info("=" * 70)
    log.info("")
    log.info("[1] Angle to identity  (does R ≈ I?)")
    log.info(f"    angle_I_deg:  mean={np.mean(angle_from_I):6.2f}  "
             f"std={np.std(angle_from_I):6.2f}  "
             f"min={np.min(angle_from_I):6.2f}  max={np.max(angle_from_I):6.2f}")
    log.info("")
    log.info("[2] Angle to dataset-mean R  (per-sample variation around canonical)")
    log.info(f"    angle_bar_deg: mean={angle_to_bar.mean():6.2f}  "
             f"std={angle_to_bar.std():6.2f}  "
             f"min={angle_to_bar.min():6.2f}  max={angle_to_bar.max():6.2f}")
    log.info("    ↑ If ≈ 0, R is essentially constant across the eval set.")
    log.info("")
    log.info("[3] Frobenius ||R - R_bar||_F  (0 if all equal; max ≈ 2√2 ≈ 2.83)")
    log.info(f"    fro_norm:      mean={fro.mean():6.4f}  "
             f"std={fro.std():6.4f}  "
             f"min={fro.min():6.4f}  max={fro.max():6.4f}")
    log.info("")
    log.info("[4] Per-axis (ZYX Euler) std across samples  (deg)")
    log.info(f"    yaw   std={eulers[:, 0].std():6.2f}")
    log.info(f"    pitch std={eulers[:, 1].std():6.2f}")
    log.info(f"    roll  std={eulers[:, 2].std():6.2f}")
    log.info("")
    log.info("[5] Euler of R_bar  (the 'dataset canonical' the model collapsed to)")
    _yaw_b, _pitch_b, _roll_b = _euler_zyx(R_bar)
    log.info(f"    yaw={_yaw_b:6.2f}  pitch={_pitch_b:6.2f}  roll={_roll_b:6.2f}  deg")
    log.info("")

    # ── per-category breakdown ──
    if categories and len(set(categories)) > 1:
        log.info("[6] Per-category R_bar angle-to-global-bar  (does R depend on category?)")
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            if len(idx) < 2:
                continue
            R_cat_bar = _mean_rotation(R_stack[idx])
            dtheta = _angle_from(R_cat_bar, R_bar)
            per_sample_ang = np.array(
                [_angle_from(R_stack[i], R_cat_bar) for i in idx]
            )
            log.info(f"    {cat:30s}  n={len(idx):4d}  "
                     f"bar-angle-to-global={dtheta:6.2f}°  "
                     f"within-cat std={per_sample_ang.std():5.2f}°")
        log.info("")

    # ── optional JSON dump ──
    if args.output_json:
        records = [
            {
                "index":        indices[i],
                "category":     categories[i],
                "R":            R_stack[i].tolist(),
                "angle_I_deg":  float(angle_from_I[i]),
                "angle_bar_deg": float(angle_to_bar[i]),
                "euler_ypr_deg": [float(x) for x in eulers[i]],
            }
            for i in range(N)
        ]
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "ckpt":     args.ckpt,
                "dataset":  args.dataset,
                "n":        N,
                "R_bar":    R_bar.tolist(),
                "euler_bar_ypr_deg": [float(x) for x in _euler_zyx(R_bar)],
                "records":  records,
            }, f, indent=2)
        log.info(f"Dumped per-sample records → {out_path}")


if __name__ == "__main__":
    main()
