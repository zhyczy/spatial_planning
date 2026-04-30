"""
curve_evaluation.py

Evaluate checkpoints at multiple training steps and plot accuracy-step curves
per dataset.

For a given checkpoint directory (e.g. train_records/coordinate_no_cam_mindcube),
this script iterates over step_N sub-directories (from --start to --end with
--step_size stride), runs evaluation.py on each, collects overall_accuracy per
dataset, and saves:
  • curve_results.json  — nested dict: {dataset: {step: accuracy}}
  • curve_{dataset}.png — accuracy vs. step plot per dataset
  • curve_all.png       — all datasets on one figure

Usage
-----
python curve_evaluation.py \\
    --ckpt_dir  train_records/coordinate_no_cam_mindcube \\
    --start     500 \\
    --end       1000 \\
    --step_size 50 \\
    --method    coordinate \\  # or polar / decouple / atten
    --datasets  mindcube,sat_real \\
    --gpus      0,1,2,3 \\
    --output_dir eval_results/curves/coordinate_no_cam_mindcube
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent

PYTHON = str(Path(sys.executable))

ALL_DATASETS = [
    "mindcube", "sat_real", "spinbench", "robospatial",
    "viewspatial", "omnispatial_pt", "embspatial",
]

DATASET_DIR = {
    "mindcube":              "datasets/evaluation/MindCube",
    "sat_real":              "datasets/evaluation/SAT",
    "spinbench":             "datasets/evaluation/spinbench_data",
    "robospatial":           "datasets/evaluation/RoboSpatial",
    "viewspatial":           "datasets/evaluation/ViewSpatial-Bench",
    "omnispatial_pt":        "datasets/evaluation/OmniSpatial",
    "embspatial":            "datasets/evaluation/EmbSpatial-Bench",
}

MODEL_PATH = str(_ROOT / "checkpoints" / "Qwen3.5-4B")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_metrics_file(run_dir: Path) -> Path | None:
    """Return the first metrics_*.json (not comparison) found in run_dir."""
    candidates = sorted(run_dir.glob("metrics_*.json"))
    for c in candidates:
        if "comparison" not in c.name:
            return c
    return None


def read_accuracy(metrics_path: Path) -> float | None:
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        return data.get("overall_accuracy")
    except Exception as exc:
        logger.warning(f"Could not read {metrics_path}: {exc}")
        return None


def read_extras(metrics_path: Path) -> dict:
    """Pull optional extras (coord_mae) from a metrics file."""
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not read extras from {metrics_path}: {exc}")
        return {}
    out: dict = {}
    if "coord_mae_mean" in data:
        out["coord_mae_mean"] = data["coord_mae_mean"]
        if "coord_mae_std" in data:
            out["coord_mae_std"] = data["coord_mae_std"]
        if "coord_mae_n" in data:
            out["coord_mae_n"] = data["coord_mae_n"]
    return out


def run_evaluation(
    step: int,
    dataset: str,
    ckpt_dir: Path,
    output_dir: Path,
    method: str,
    limit: int | None,
    gpus: str,
    max_new_tokens: int,
) -> Path | None:
    """Run evaluation.py for one (step, dataset) and return the output sub-dir."""
    ckpt_path = ckpt_dir / f"step_{step}"
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
        return None

    data_dir_rel = DATASET_DIR.get(dataset)
    if data_dir_rel is None:
        logger.warning(f"Unknown dataset '{dataset}', skipping.")
        return None

    data_dir = _ROOT / data_dir_rel
    if not data_dir.exists():
        logger.warning(f"Data directory not found for '{dataset}': {data_dir} — skipping.")
        return None

    run_name = f"step{step:06d}_{dataset}"
    eval_runs_dir = output_dir / "eval_runs"
    run_dir = eval_runs_dir / run_name

    # Skip if already done (resumable)
    metrics = find_metrics_file(run_dir)
    if metrics is not None:
        logger.info(f"[SKIP] step={step} dataset={dataset} — results already exist at {run_dir}")
        return run_dir

    cmd = [
        PYTHON, str(_ROOT / "evaluation.py"),
        "--model_path",          MODEL_PATH,
        "--method",              method,
        "--correspondence_ckpt", str(ckpt_path),
        "--dataset",             dataset,
        "--data_dir",            str(data_dir),
        "--output_dir",          str(eval_runs_dir),
        "--run_name",            run_name,
        "--max_new_tokens",      str(max_new_tokens),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus

    logger.info(f"[RUN] step={step} dataset={dataset}  ckpt={ckpt_path}")
    logger.info(f"      cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if result.returncode != 0:
        logger.error(f"evaluation.py exited with code {result.returncode} "
                     f"for step={step} dataset={dataset}")
        return None

    return run_dir


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curve(steps: list[int], accs: list[float], dataset: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, [a * 100 for a in accs], marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"{dataset} — accuracy vs. step", fontsize=14)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 100)
    for s, a in zip(steps, accs):
        ax.annotate(f"{a*100:.1f}", (s, a * 100), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved curve → {out_path}")


def plot_all_curves(
    curve_data: dict[str, dict[int, float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset, step_acc in curve_data.items():
        if not step_acc:
            continue
        xs = sorted(step_acc.keys())
        ys = [step_acc[x] * 100 for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=dataset)
    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs. training step (all datasets)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved combined curve → {out_path}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _save_summary(
    path: Path,
    curve_data: dict[str, dict[int, float]],
    extras_data: dict[str, dict[int, dict]],
    steps: list[int],
    datasets: list[str],
    args,
) -> None:
    """Save a structured summary JSON:

    {
      "config": { ... run parameters ... },
      "by_dataset":         { dataset: { step: accuracy } },
      "by_step":            { step:    { dataset: accuracy } },
      "by_dataset_extras":  { dataset: { step: {coord_mae_*} } }
    }
    """
    by_step: dict[str, dict[str, float]] = {}
    for step in steps:
        row = {}
        for ds in datasets:
            acc = curve_data[ds].get(step)
            if acc is not None:
                row[ds] = acc
        if row:
            by_step[str(step)] = row

    by_dataset_extras: dict[str, dict[str, dict]] = {}
    for ds, step_extras in extras_data.items():
        if not step_extras:
            continue
        by_dataset_extras[ds] = {str(s): e for s, e in step_extras.items() if e}

    summary = {
        "config": {
            "ckpt_dir":     args.ckpt_dir,
            "start":        args.start,
            "end":          args.end,
            "step_size":    args.step_size,
            "method":       args.method,
            "datasets":     datasets,
        },
        "by_dataset": {
            ds: {str(s): a for s, a in step_acc.items()}
            for ds, step_acc in curve_data.items()
        },
        "by_step": by_step,
        "by_dataset_extras": by_dataset_extras,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints at multiple steps and plot accuracy curves."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Directory containing step_N checkpoint sub-directories "
             "(e.g. train_records/coordinate_no_cam_mindcube).",
    )
    parser.add_argument("--start",     type=int, default=500,
                        help="First step to evaluate (default: 500).")
    parser.add_argument("--end",       type=int, default=1000,
                        help="Last step to evaluate (default: 1000).")
    parser.add_argument("--step_size", type=int, default=50,
                        help="Step stride (default: 50).")
    parser.add_argument(
        "--method", type=str, default="coordinate",
        choices=["baseline", "vanilla", "position_embedding", "coordinate", "polar",
                 "decouple", "atten"],
        help="Evaluation method (default: coordinate).",
    )
    parser.add_argument(
        "--datasets", type=str, default=",".join(ALL_DATASETS),
        help=f"Comma-separated list of datasets to evaluate "
             f"(default: {','.join(ALL_DATASETS)}).",
    )
    parser.add_argument(
        "--gpus", type=str, default="",
        help="Comma-separated GPU IDs (default: all visible).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Truncate each dataset to N samples (debug / smoke test).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Base output directory for eval results and plots "
             "(default: eval_results/curves/<ckpt_dir_basename>).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max new tokens for generation (default: 512).",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _ROOT / "curve_evaluation_results" / ckpt_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"curve_evaluation_{timestamp}.log"
    _setup_logging(log_path)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    steps = list(range(args.start, args.end + 1, args.step_size))

    logger.info("=" * 60)
    logger.info("[curve_evaluation] Starting")
    logger.info(f"  ckpt_dir     : {ckpt_dir}")
    logger.info(f"  steps        : {steps}")
    logger.info(f"  datasets     : {datasets}")
    logger.info(f"  method       : {args.method}")
    logger.info(f"  output_dir   : {output_dir}")
    logger.info("=" * 60)

    # curve_data[dataset][step] = accuracy
    curve_data: dict[str, dict[int, float]] = {ds: {} for ds in datasets}
    # extras_data[dataset][step] = {"coord_mae_mean": ..., "coord_mae_std": ...}
    extras_data: dict[str, dict[int, dict]] = {ds: {} for ds in datasets}

    # Try to load existing summary (for resumability)
    summary_path = output_dir / "curve_results.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                saved = json.load(f)
            for ds in datasets:
                if ds in saved.get("by_dataset", {}):
                    curve_data[ds] = {int(k): v for k, v in saved["by_dataset"][ds].items()}
                if ds in saved.get("by_dataset_extras", {}):
                    extras_data[ds] = {
                        int(k): v for k, v in saved["by_dataset_extras"][ds].items()
                    }
            logger.info(f"Loaded existing results from {summary_path}")
        except Exception as exc:
            logger.warning(f"Could not load existing summary: {exc}")

    # Main evaluation loop
    for step in steps:
        for dataset in datasets:
            if step in curve_data[dataset]:
                logger.info(f"[SKIP] step={step} dataset={dataset} — already in summary")
                continue

            run_dir = run_evaluation(
                step=step,
                dataset=dataset,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir,
                method=args.method,
                limit=args.limit,
                gpus=args.gpus,
                max_new_tokens=args.max_new_tokens,
            )

            if run_dir is None:
                continue

            metrics_file = find_metrics_file(run_dir)
            if metrics_file is None:
                logger.warning(f"No metrics file found in {run_dir}")
                continue

            acc = read_accuracy(metrics_file)
            if acc is None:
                logger.warning(f"Could not read accuracy from {metrics_file}")
                continue

            curve_data[dataset][step] = acc
            extras = read_extras(metrics_file)
            if extras:
                extras_data[dataset][step] = extras

            logger.info(f"[OK] step={step} dataset={dataset} accuracy={acc:.4f} ({acc*100:.2f}%)")

            # Save incrementally after each evaluation
            _save_summary(summary_path, curve_data, extras_data, steps, datasets, args)

    # Final save
    _save_summary(summary_path, curve_data, extras_data, steps, datasets, args)
    logger.info(f"Saved curve results → {summary_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("Summary (overall_accuracy)")
    header = f"{'step':>8}" + "".join(f"  {ds:>15}" for ds in datasets)
    logger.info(header)
    logger.info("-" * len(header))
    for step in steps:
        row = f"{step:>8}"
        for ds in datasets:
            acc = curve_data[ds].get(step)
            row += f"  {acc*100:>14.2f}%" if acc is not None else f"  {'N/A':>14}"
        logger.info(row)

    # Plotting
    for dataset in datasets:
        step_acc = curve_data[dataset]
        if not step_acc:
            logger.warning(f"No data to plot for dataset '{dataset}'")
            continue
        xs = sorted(step_acc.keys())
        ys = [step_acc[x] for x in xs]
        plot_curve(xs, ys, dataset, output_dir / f"curve_{dataset}.png")

    plot_all_curves(curve_data, output_dir / "curve_all.png")

    logger.info(f"\nAll done. Results and plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
