"""
validation.py

xyz=0 ablation for the atten checkpoint (train_atten.py output).

Hypothesis: if SpatialAttentionBias actually conditions on the per-patch
3D coords, replacing image_xyz with zeros at inference should hurt accuracy.
If accuracy is unchanged, the model has not learned to use 3D — it is
solving the task from text + 2D vision alone.

Reuses load_spa_model / prepare_batch_spa / run_inference_spa / metrics from
evaluation.py. Only difference: a single hook that replaces image_xyz with
torch.zeros_like(image_xyz) before run_inference_spa, on selected runs.

Usage
-----
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python validation.py \
    --ckpt        train_records/app/atten_mindcube/step_1000 \
    --model_path  checkpoints/Qwen3.5-4B \
    --dataset     mindcube \
    --data_dir    datasets/evaluation/MindCube \
    --output_dir  vis_results/atten_xyz0_step_1000

Runs both passes (normal xyz, then zero xyz) and prints the delta.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# Force spatial_planning/ to the FRONT of sys.path. evaluation.py (when
# imported below) inserts ../RoboSpatial-Eval/ at position 0 — it has its own
# evaluation.py that would otherwise shadow ours when a multiprocessing-spawn
# worker re-imports this file.
_ROOT = Path(__file__).resolve().parent
while str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

from evaluation import (
    load_spa_model,
    prepare_batch_spa,
    run_inference_spa,
    _make_result,
    _error_result,
    compute_metrics,
    log_metrics,
)
from src.dataset import load_testing_dataset, chunk_dataset


# ---------------------------------------------------------------------------
# Per-GPU evaluation loop (atten only)
# ---------------------------------------------------------------------------
def evaluate_atten_xyz(
    data: List[Dict],
    ckpt: str,
    model_path: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: Path,
    tag: str,                     # "normal" | "zero"
    zero_xyz: bool,
    device: str,
) -> List[Dict]:
    """Run atten-only inference on `data`, optionally zeroing image_xyz."""
    logger = logging.getLogger(__name__)

    spa_model, spa_proc = load_spa_model(
        model_path, ckpt, device,
        vanilla=False, decouple=False, polar=False, atten=True,
    )

    cfg_path = Path(model_path) / "config.json"
    with open(cfg_path) as f:
        _vcfg = json.load(f).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    logger.info(f"[{tag}] spatial_merge_size={spatial_merge_size}")

    if zero_xyz:
        logger.info(f"[{tag}] image_xyz will be ZEROED before each forward")

    results: List[Dict] = []
    for item in tqdm(data, desc=f"[{device}|{tag}]"):
        try:
            inputs, prompt, image_xyz = prepare_batch_spa(
                item, spa_proc,
                spatial_merge_size, use_coord=False, coord_scale=coord_scale,
                thinking=False, load_xyz=True, strict_xyz=True,
            )
            if zero_xyz and image_xyz is not None:
                image_xyz = [torch.zeros_like(x) for x in image_xyz]

            output = run_inference_spa(
                inputs, image_xyz, spa_model, spa_proc,
                max_new_tokens, coord_scale,
                vanilla=False, polar=False, decouple=False, atten=True,
            )
            results.append(_make_result(item, output, prompt, "atten"))
        except Exception as exc:
            logger.error(
                f"[{tag}] idx={item.get('index')}: {exc}", exc_info=True
            )
            results.append(_error_result(item, exc, "atten"))

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"atten_{tag}_{device.replace(':', '')}.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"[{tag}] saved {len(results)} results → {partial_path}")
    return results


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------
def _worker(
    gpu_id: str,
    data_shard: List[Dict],
    ckpt: str,
    model_path: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: str,
    log_file: str,
    tag: str,
    zero_xyz: bool,
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
        f"[Worker {gpu_id}|{tag}] start — {len(data_shard)} samples on {device}"
    )

    evaluate_atten_xyz(
        data=data_shard,
        ckpt=ckpt,
        model_path=model_path,
        coord_scale=coord_scale,
        max_new_tokens=max_new_tokens,
        output_dir=Path(output_dir),
        tag=tag,
        zero_xyz=zero_xyz,
        device=device,
    )
    logger.info(f"[Worker {gpu_id}|{tag}] done.")


# ---------------------------------------------------------------------------
# Run one full pass (across all GPUs) and merge
# ---------------------------------------------------------------------------
def _run_pass(
    dataset: List[Dict],
    gpu_ids: List[str],
    ckpt: str,
    model_path: str,
    coord_scale: float,
    max_new_tokens: int,
    output_dir: Path,
    log_file: Path,
    tag: str,
    zero_xyz: bool,
) -> List[Dict]:
    logger = logging.getLogger(__name__)

    # Wipe stale partials from a prior crashed run with this tag.
    for p in output_dir.glob(f"atten_{tag}_cuda*.json"):
        logger.info(f"removing stale partial {p}")
        p.unlink()

    shards = chunk_dataset(dataset, len(gpu_ids))
    procs: List[mp.Process] = []
    spawned: List[str] = []
    for gid, shard in zip(gpu_ids, shards):
        p = mp.Process(
            target=_worker,
            args=(
                gid, shard, ckpt, model_path,
                coord_scale, max_new_tokens, str(output_dir), str(log_file),
                tag, zero_xyz,
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

    # Merge
    merged: List[Dict] = []
    missing: List[str] = []
    for gid in spawned:
        path = output_dir / f"atten_{tag}_cuda{gid}.json"
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
    merged.sort(key=lambda r: r.get("index", 0))

    with open(output_dir / f"results_atten_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="atten ckpt: xyz=0 vs xyz=normal accuracy ablation."
    )
    ap.add_argument("--ckpt", required=True, help="atten checkpoint dir (step_*).")
    ap.add_argument("--model_path", required=True, help="Qwen3.5-VL base model dir.")
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
    ap.add_argument("--max_new_tokens", type=int, default=512)
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

    logger.info("=" * 60)
    logger.info("xyz=0 ablation (atten)")
    logger.info("=" * 60)
    logger.info(f"  ckpt           : {args.ckpt}")
    logger.info(f"  model_path     : {args.model_path}")
    logger.info(f"  dataset        : {args.dataset} ({len(dataset)} samples)")
    logger.info(f"  data_dir       : {args.data_dir}")
    logger.info(f"  GPUs           : {gpu_ids}")
    logger.info(f"  output_dir     : {out_dir}")
    logger.info(f"  passes         : {args.only}")
    logger.info("=" * 60)

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
            dataset, gpu_ids, args.ckpt, args.model_path,
            args.coord_scale, args.max_new_tokens,
            out_dir, log_file, tag="normal", zero_xyz=False,
        )
        m_normal = compute_metrics(results_normal)
        log_metrics(m_normal, "atten — normal xyz", logger)
        m_normal.pop("category_failures", None)
        with open(out_dir / "metrics_atten_normal.json", "w") as f:
            json.dump(m_normal, f, ensure_ascii=False, indent=2)
        metrics["normal"] = m_normal

    if args.only in ("zero", "both"):
        logger.info("\n>>> PASS 2: image_xyz = 0")
        results_zero = _run_pass(
            dataset, gpu_ids, args.ckpt, args.model_path,
            args.coord_scale, args.max_new_tokens,
            out_dir, log_file, tag="zero", zero_xyz=True,
        )
        m_zero = compute_metrics(results_zero)
        log_metrics(m_zero, "atten — xyz = 0", logger)
        m_zero.pop("category_failures", None)
        with open(out_dir / "metrics_atten_zero.json", "w") as f:
            json.dump(m_zero, f, ensure_ascii=False, indent=2)
        metrics["zero"] = m_zero

    if "normal" in metrics and "zero" in metrics:
        a = metrics["normal"]["overall_accuracy"]
        b = metrics["zero"]["overall_accuracy"]
        delta = b - a
        logger.info("")
        logger.info("=" * 65)
        logger.info("xyz=0 ABLATION SUMMARY")
        logger.info("=" * 65)
        logger.info(
            f"  {'normal xyz':<30s} {a:.2%}  "
            f"({metrics['normal']['correct_samples']}/{metrics['normal']['total_samples']})"
        )
        logger.info(
            f"  {'xyz = 0':<30s} {b:.2%}  "
            f"({metrics['zero']['correct_samples']}/{metrics['zero']['total_samples']})"
        )
        logger.info(f"  {'delta (zero - normal)':<30s} {delta:+.2%}")
        logger.info("=" * 65)
        logger.info(
            "  Interpretation: a near-zero delta means the bias module is "
            "ignoring image_xyz; a clearly negative delta means xyz is "
            "actually being used."
        )
        with open(out_dir / "ablation_summary.json", "w") as f:
            json.dump(
                {
                    "normal_accuracy": a,
                    "zero_accuracy": b,
                    "delta_zero_minus_normal": delta,
                    "n_samples": metrics["normal"]["total_samples"],
                },
                f, indent=2, ensure_ascii=False,
            )

    logger.info(f"all done → {out_dir}")


if __name__ == "__main__":
    main()
