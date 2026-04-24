"""Rotation-robustness sweep for a coordinate-supervised (no-cam) checkpoint.

For each eval sample, rotate the precomputed first-camera-frame xyz by a set
of R matrices {I, Rx(θ), Ry(θ), Rz(θ)} (θ ∈ {30, 60, 90, 120, 150, 180, 270}),
feed the rotated xyz to the SPA model as 4D M-RoPE positions, run inference,
and record QA accuracy + coord_mae at each R.

The model under test (e.g. coordinate_no_cam_mindcube/step_1000) was trained
with xyz in the view_0000 frame (R = I). This script measures how QA accuracy
and coord_mae degrade (or improve) as we rotate the coordinate frame into
configurations the model never saw during training.

Multi-GPU: auto-shards samples across CUDA_VISIBLE_DEVICES (mirrors
evaluation.py's launcher).

Usage
-----
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    /egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/python \\
        spatial_planning/validate_coord_rotation_robustness.py \\
        --ckpt     spatial_planning/train_records/coordinate_no_cam_mindcube/step_1000 \\
        --model_path  spatial_planning/checkpoints/Qwen3.5-4B \\
        --data_dir spatial_planning/datasets/evaluation/MindCube \\
        --dataset  mindcube

Outputs
-------
    <ckpt>/R_sweep/results_cuda{gpu}.json    — per-worker (sample × R) rows
    <ckpt>/R_sweep/R_sweep.json              — merged per-(sample, R) rows
    <ckpt>/R_sweep/metrics_per_R.json        — summary accuracy + coord_mae per R
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from evaluation import (                             # noqa: E402
    _compute_coord_mae,
    _get_coord_predictions,
    _load_coord_head,
    _make_result,
    extract_answer_letter,
    load_spa_model,
    prepare_batch_spa,
    run_inference_spa,
)
from src.dataset import chunk_dataset, load_testing_dataset      # noqa: E402
from src.models.rotation_rope_llm import _apply_rotation_to_xyz  # noqa: E402


# ===========================================================================
# Rotation grid
# ===========================================================================

AXIS_ANGLES_DEG: Tuple[int, ...] = (30, 60, 90, 120, 150, 180, 270)


def _axis_rot(axis: str, deg: float) -> torch.Tensor:
    """Return the 3×3 rotation matrix for angle `deg` about principal `axis`."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    if axis == "x":
        R = [[1, 0, 0], [0, c, -s], [0, s,  c]]
    elif axis == "y":
        R = [[ c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == "z":
        R = [[c, -s, 0], [s,  c, 0], [0, 0, 1]]
    else:
        raise ValueError(f"axis must be x/y/z, got {axis!r}")
    return torch.tensor(R, dtype=torch.float32)


def build_rotation_grid() -> List[Tuple[str, torch.Tensor]]:
    """List of (name, R) pairs. `I` first, then Rx / Ry / Rz sweeps."""
    grid: List[Tuple[str, torch.Tensor]] = [("I", torch.eye(3, dtype=torch.float32))]
    for axis in ("x", "y", "z"):
        for deg in AXIS_ANGLES_DEG:
            grid.append((f"R{axis}_{deg:03d}", _axis_rot(axis, float(deg))))
    return grid


# ===========================================================================
# Per-sample rotation sweep
# ===========================================================================

def _sweep_one_sample(
    item: Dict,
    spa_model,
    spa_proc,
    coord_head,
    spatial_merge_size: int,
    image_token_id: int,
    coord_scale: float,
    max_new_tokens: int,
    rot_grid: List[Tuple[str, torch.Tensor]],
    device: str,
) -> List[Dict]:
    """Run the 22-R sweep on a single sample and return one result dict per R."""
    # Tokenize + load xyz ONCE; reuse inputs across all rotations.
    inputs, prompt, image_xyz = prepare_batch_spa(
        item, spa_proc,
        spatial_merge_size=spatial_merge_size,
        use_coord=False,
        coord_scale=coord_scale,
        thinking=False,
        load_xyz=True,         # need xyz for 4D M-RoPE
    )

    gt_letter = str(item.get("answer", "")).strip().upper()
    model_device = next(spa_model.parameters()).device

    results: List[Dict] = []
    for R_name, R in rot_grid:
        R_dev = R.to(model_device)

        # Rotate xyz (gracefully no-op if xyz wasn't loaded)
        if image_xyz is not None:
            xyz_rot = _apply_rotation_to_xyz(
                R_dev, [x.to(model_device) for x in image_xyz]
            )
        else:
            xyz_rot = None

        # Inference
        try:
            output = run_inference_spa(
                inputs, xyz_rot, spa_model, spa_proc,
                max_new_tokens, coord_scale,
                vanilla=False, polar=False,
            )
        except Exception as exc:
            output = f"[ERROR] {type(exc).__name__}: {exc}"

        result = _make_result(item, output, prompt, f"coordinate@{R_name}",
                              thinking=False)
        result["R_name"]     = R_name
        pred_upper = str(result.get("prediction", "")).upper()
        result["is_correct"] = (
            int(pred_upper == gt_letter) if gt_letter else None
        )

        # coord_mae: predictions in rotated frame are compared to rotated xyz
        if xyz_rot is not None and coord_head is not None:
            try:
                preds = _get_coord_predictions(
                    spa_model, inputs,
                    image_token_id, coord_head,
                    spatial_merge_size, xyz_rot, coord_scale,
                    cam_feat=None,
                )
                if preds is not None:
                    xyz_cpu = [r.detach().cpu() for r in xyz_rot]
                    result["coord_mae"] = _compute_coord_mae(preds, xyz_cpu)
            except Exception as ce:
                result["coord_mae_error"] = f"{type(ce).__name__}: {ce}"

        results.append(result)

    return results


# ===========================================================================
# Worker (one GPU)
# ===========================================================================

def _worker(
    gpu_label: str,          # logical label (from CUDA_VISIBLE_DEVICES) — for filenames
    local_index: int,        # PyTorch-visible device index (0..N-1)
    data_shard: List[Dict],
    model_path: str,
    ckpt: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: str,
    log_file: Optional[str],
    no_coord_head: bool = False,
    decouple: bool = False,
    polar: bool = False,
) -> None:
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )
    logger = logging.getLogger(f"worker-{gpu_label}")
    device = f"cuda:{local_index}"
    torch.cuda.set_device(local_index)
    logger.info(
        f"[Worker {gpu_label} (local={local_index})] "
        f"Starting — {len(data_shard)} samples on {device}"
    )

    # Load SPA model + coord head ONCE per worker
    spa_model, spa_proc = load_spa_model(
        model_path, ckpt, device, vanilla=False,
        decouple=decouple, polar=polar,
    )
    cfg_path = Path(model_path) / "config.json"
    with open(cfg_path) as f:
        _vcfg = json.load(f).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))

    image_token_id = spa_proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if no_coord_head:
        coord_head = None
        logger.info("[coord_head] skipped (--no_coord_head) — QA-only run")
    else:
        coord_head = _load_coord_head(ckpt, device, expect_relative=None)
    rot_grid = build_rotation_grid()

    all_rows: List[Dict] = []
    desc = f"[cuda:{gpu_label}]"
    for item in tqdm(data_shard, desc=desc):
        try:
            rows = _sweep_one_sample(
                item, spa_model, spa_proc, coord_head,
                spatial_merge_size, image_token_id, coord_scale,
                max_new_tokens, rot_grid, device,
            )
            all_rows.extend(rows)
        except Exception as exc:
            logger.error(
                f"[Worker {gpu_label}] idx={item.get('index')}: {exc}",
                exc_info=True,
            )
            # Emit error rows for every R so downstream merge is consistent
            for R_name, _ in rot_grid:
                all_rows.append({
                    "index":   item.get("index"),
                    "category": item.get("category", "unknown"),
                    "R_name":  R_name,
                    "error":   f"{type(exc).__name__}: {exc}",
                })

    out_path = Path(output_dir) / f"results_cuda{gpu_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    logger.info(f"[Worker {gpu_label}] Done — wrote {len(all_rows)} rows to {out_path}")


# ===========================================================================
# Aggregation
# ===========================================================================

def compute_per_R_metrics(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Accuracy + mean coord_mae per R_name."""
    by_R: Dict[str, List[Dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        by_R.setdefault(r["R_name"], []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for R_name, rs in by_R.items():
        correct = [r for r in rs if r.get("is_correct") == 1]
        maes    = [r["coord_mae"] for r in rs if "coord_mae" in r]
        out[R_name] = {
            "n":             len(rs),
            "accuracy":      len(correct) / max(len(rs), 1),
            "coord_mae":     float(np.mean(maes)) if maes else float("nan"),
            "coord_mae_n":   len(maes),
        }
    return out


def _pretty_log_metrics(per_R: Dict[str, Dict[str, float]],
                         logger: logging.Logger) -> None:
    logger.info("=" * 70)
    logger.info(f"{'R_name':<12}  {'n':>5}  {'acc':>8}  {'coord_mae':>10}")
    logger.info("-" * 70)
    # Order: I, then Rx, Ry, Rz with ascending angle
    ordered_names = ["I"] + [
        f"R{a}_{d:03d}" for a in "xyz" for d in AXIS_ANGLES_DEG
    ]
    for R_name in ordered_names:
        if R_name not in per_R:
            continue
        m = per_R[R_name]
        logger.info(
            f"{R_name:<12}  {m['n']:>5d}  {m['accuracy']:>7.4f}   "
            f"{m['coord_mae']:>9.4f}"
        )
    logger.info("=" * 70)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to correspondence_ckpt (e.g. .../step_1000).")
    ap.add_argument("--model_path", required=True,
                    help="Base Qwen3.5-VL checkpoint.")
    ap.add_argument("--data_dir",  required=True,
                    help="Evaluation dataset root (e.g. datasets/evaluation/MindCube).")
    ap.add_argument("--dataset", default="mindcube",
                    help="load_testing_dataset key (default: mindcube).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on #samples (debug).")
    ap.add_argument("--coord_scale", type=float, default=100.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output_dir", default=None,
                    help="Default: <ckpt>/R_sweep/")
    ap.add_argument("--no_coord_head", action="store_true",
                    help="Skip coord_head loading; only report QA accuracy "
                         "(use when ckpt has no coord_head.pt/dpt_head.pt).")
    ap.add_argument("--decouple", action="store_true",
                    help="Ckpt was trained with train_correspondence.py --decouple: "
                         "keep original 3D M-RoPE in rotary 64 dims, add XYZ RoPE "
                         "in pass-through dims 64..129.")
    ap.add_argument("--polar", action="store_true",
                    help="Ckpt was trained with --polar (log-spherical XYZ RoPE, "
                         "theta=1000). Implies decouple architecture.")
    args = ap.parse_args()
    if args.polar:
        args.decouple = True

    ckpt_dir = Path(args.ckpt).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir \
        else ckpt_dir / "R_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = load_testing_dataset(Path(args.data_dir).resolve(),
                                    limit=args.limit, dataset=args.dataset)

    # ── GPUs ──────────────────────────────────────────────────────────────────
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("At least one CUDA device is required.")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = (
        [x.strip() for x in cuda_visible.split(",") if x.strip()]
        if cuda_visible
        else [str(i) for i in range(n_gpu)]
    )

    logger.info("=" * 60)
    logger.info("COORD-MODEL ROTATION-ROBUSTNESS SWEEP")
    logger.info("=" * 60)
    logger.info(f"  ckpt         : {ckpt_dir}")
    logger.info(f"  model_path   : {args.model_path}")
    logger.info(f"  data_dir     : {args.data_dir}")
    logger.info(f"  dataset      : {args.dataset}  ({len(dataset)} samples)")
    logger.info(f"  GPUs         : {gpu_ids}")
    logger.info(f"  output_dir   : {output_dir}")
    logger.info(f"  R-grid       : 1 identity + 3 axes × "
                f"{len(AXIS_ANGLES_DEG)} angles = "
                f"{1 + 3 * len(AXIS_ANGLES_DEG)} R's")
    logger.info("=" * 60)

    # ── save config ───────────────────────────────────────────────────────────
    with open(output_dir / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"output_dir": str(output_dir),
                                 "gpus": gpu_ids,
                                 "timestamp": timestamp,
                                 "axis_angles_deg": list(AXIS_ANGLES_DEG)},
                   f, indent=2, ensure_ascii=False)

    # ── launch workers ────────────────────────────────────────────────────────
    # gpu_ids are the LOGICAL labels from CUDA_VISIBLE_DEVICES (e.g. ["4","5","6"])
    # Inside a subprocess that inherits the same env, PyTorch renumbers them to
    # 0..N-1 — so `cuda:<local_index>` is what torch accepts, while gpu_label is
    # kept only for output filenames and logging.
    shards = chunk_dataset(dataset, len(gpu_ids))
    procs: List[mp.Process] = []
    for local_index, (gpu_label, shard) in enumerate(zip(gpu_ids, shards)):
        p = mp.Process(
            target=_worker,
            args=(
                gpu_label, local_index, shard,
                args.model_path,
                str(ckpt_dir),
                args.coord_scale,
                args.max_new_tokens,
                str(output_dir),
                str(log_file),
                args.no_coord_head,
                args.decouple,
                args.polar,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # ── merge shards ──────────────────────────────────────────────────────────
    logger.info("Merging results from all workers…")
    merged: List[Dict] = []
    for gpu_label in gpu_ids:
        p = output_dir / f"results_cuda{gpu_label}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                merged.extend(json.load(f))
        else:
            logger.warning(f"Missing worker output: {p}")

    # Stable sort: index then R position in the rotation grid
    R_order = {name: i for i, (name, _) in enumerate(build_rotation_grid())}
    merged.sort(key=lambda r: (r.get("index", 0),
                                 R_order.get(r.get("R_name", ""), 1e9)))

    out_all = output_dir / "R_sweep.json"
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(merged)} merged rows to {out_all}")

    # ── metrics per R ─────────────────────────────────────────────────────────
    per_R = compute_per_R_metrics(merged)
    out_metrics = output_dir / "metrics_per_R.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(per_R, f, ensure_ascii=False, indent=2)
    _pretty_log_metrics(per_R, logger)
    logger.info(f"Per-R metrics → {out_metrics}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
