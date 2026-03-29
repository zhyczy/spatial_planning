"""
estimate_3D.py

Batch 3D estimation for evaluation datasets.
Loads CoordEstimator once, then processes every entry in each dataset.

Datasets
--------
  mindcube              MindCube_tinybench.jsonl    — images: relative paths
  mmsibench             test_data_final.json        — local_images: relative paths
  sat_real              test.json                   — img_paths: relative paths
  sparbench_multi_view  sparbench_multi_view.json   — images: base64
  sparbench_single_view sparbench_single_view.json  — images: base64
  sparbench_mv          sparbench_mv.json           — images: base64 (multi-view, video excluded)

Routing
-------
  1 image   → Depth Pro   (monocular metric depth)
  ≥2 images → MapAnything (multi-view stereo)

Output layout (under each dataset directory)
--------------------------------------------
  3d_results/
  └── <entry_id>/
      ├── view_0000/
      │   ├── pts3d.npy        (H, W, 3) float32 — world-frame 3-D coords
      │   ├── camera_pose.npy  (4, 4)    float32 — cam-to-world matrix
      │   ├── intrinsics.npy   (3, 3)    float32 — pinhole K
      │   ├── depth.npy        (H, W)    float32 — metric depth (m)
      │   └── mask.npy         (H, W)    bool    — valid pixels
      ├── view_0001/
      │   └── ...
      └── cameras.json

Usage
-----
  python estimate_3D.py                                   # all datasets
  python estimate_3D.py --datasets mindcube sat_real      # selected
  python estimate_3D.py --datasets mindcube --limit 10    # smoke test
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Resolve spatial_planning root and import CoordEstimator
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SPATIAL_PLANNING_DIR = _THIS_DIR.parent.parent  # data_process → src → spatial_planning

if str(_SPATIAL_PLANNING_DIR) not in sys.path:
    sys.path.insert(0, str(_SPATIAL_PLANNING_DIR))

from coord_esti import CoordEstimator, save_results  # noqa: E402

_EVAL_ROOT = _SPATIAL_PLANNING_DIR / "datasets" / "evaluation"
_TRAIN_ROOT = _SPATIAL_PLANNING_DIR / "datasets" / "train"


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_images_from_paths(root: Path, rel_paths: List[str]) -> List[np.ndarray]:
    """Load relative image paths (w.r.t. *root*) into uint8 RGB arrays."""
    arrays = []
    for rel in rel_paths:
        p = root / rel          # pathlib handles leading "./" transparently
        if p.exists():
            arrays.append(np.array(Image.open(p).convert("RGB"), dtype=np.uint8))
        else:
            print(f"[WARN] image not found: {p}")
    return arrays


def _decode_b64_images(b64_list: List[str]) -> List[np.ndarray]:
    """Decode base64-encoded image strings into uint8 RGB arrays."""
    arrays = []
    for b64 in b64_list:
        raw = base64.b64decode(b64)
        arrays.append(np.array(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8))
    return arrays


# ---------------------------------------------------------------------------
# Per-dataset iterators  →  (entry_id, [rgb_arrays])
# ---------------------------------------------------------------------------

def _iter_mindcube(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    root = _EVAL_ROOT / "MindCube"
    with open(root / "MindCube_tinybench.jsonl") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            entry = json.loads(line)
            imgs = _load_images_from_paths(root, entry.get("images", []))
            if imgs:
                yield str(entry.get("id", i)), imgs


def _iter_mindcube_train(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    img_root = _EVAL_ROOT / "MindCube"  # images shared with eval
    json_path = _TRAIN_ROOT / "MindCube" / "train_10k.json"
    with open(json_path) as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        imgs = _load_images_from_paths(img_root, entry.get("image", []))
        if imgs:
            yield str(entry.get("id", i)), imgs


def _iter_mmsibench(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    root = _EVAL_ROOT / "MMSIBench"
    with open(root / "data" / "test_data_final.json") as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        imgs = _load_images_from_paths(root, entry.get("local_images", []))
        if imgs:
            yield str(entry.get("id", i)), imgs


def _iter_sat(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    root = _EVAL_ROOT / "SAT"
    with open(root / "test.json") as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        imgs = _load_images_from_paths(root, entry.get("img_paths", []))
        if imgs:
            yield str(entry.get("database_idx", i)), imgs


def _iter_sparbench(json_path: Path, limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    with open(json_path) as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        imgs = _decode_b64_images(entry.get("images", []))
        if imgs:
            yield str(entry.get("id", i)), imgs


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "mindcube": {
        "iter": _iter_mindcube,
        "out_dir": _EVAL_ROOT / "MindCube" / "3d_results",
    },
    "mmsibench": {
        "iter": _iter_mmsibench,
        "out_dir": _EVAL_ROOT / "MMSIBench" / "3d_results",
    },
    "sat_real": {
        "iter": _iter_sat,
        "out_dir": _EVAL_ROOT / "SAT" / "3d_results",
    },
    "sparbench_multi_view": {
        "iter": lambda limit: _iter_sparbench(
            _EVAL_ROOT / "SPARBench" / "sparbench_multi_view.json", limit
        ),
        "out_dir": _EVAL_ROOT / "SPARBench" / "3d_results",
    },
    "sparbench_single_view": {
        "iter": lambda limit: _iter_sparbench(
            _EVAL_ROOT / "SPARBench" / "sparbench_single_view.json", limit
        ),
        "out_dir": _EVAL_ROOT / "SPARBench" / "3d_results",
    },
    "mindcube_train": {
        "iter": _iter_mindcube_train,
        "out_dir": _TRAIN_ROOT / "MindCube" / "3d_results",
    },
    "sparbench_mv": {
        "iter": lambda limit: _iter_sparbench(
            _EVAL_ROOT / "SPARBench" / "sparbench_mv.json", limit
        ),
        "out_dir": _EVAL_ROOT / "SPARBench" / "3d_results",
    },
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_dataset(
    name: str,
    estimator: CoordEstimator,
    limit: int = -1,
    skip_existing: bool = True,
) -> None:
    """Estimate 3D for every entry in *name* and save results to disk."""
    cfg = DATASETS[name]
    out_root: Path = cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_err = 0

    for entry_id, imgs in cfg["iter"](limit):
        entry_out = out_root / entry_id
        if skip_existing and (entry_out / "cameras.json").exists():
            n_skip += 1
            continue

        try:
            results = estimator.estimate(imgs, save_dir=None)
            save_results(results, save_dir=out_root, run_name=entry_id)
            n_ok += 1
        except Exception as exc:
            print(f"[WARN] {name}/{entry_id} failed: {exc}")
            n_err += 1

    print(f"[{name}] done — ok={n_ok}  skipped={n_skip}  errors={n_err}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _iter_custom_json(json_path: str, img_root: str, img_key: str,
                      id_key: str, limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """Generic iterator for any JSON/JSONL file with image paths."""
    p = Path(json_path)
    root = Path(img_root)
    if p.suffix == ".jsonl":
        with open(p) as f:
            data = [json.loads(line) for line in f]
    else:
        with open(p) as f:
            data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        paths = entry.get(img_key, [])
        if isinstance(paths, str):
            paths = [paths]
        imgs = _load_images_from_paths(root, paths)
        if imgs:
            yield str(entry.get(id_key, i)), imgs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch 3D estimation for evaluation datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(DATASETS.keys()),
        help="Predefined datasets to process.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to a custom JSON/JSONL file to process.",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default=None,
        help="Root directory for resolving image paths in --json_path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for 3d_results when using --json_path.",
    )
    parser.add_argument(
        "--img_key",
        type=str,
        default="image",
        help="JSON key for image paths (default: 'image').",
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="JSON key for entry id (default: 'id').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max entries per dataset (-1 = all).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cuda:0'. Auto-detected if not set.",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Reprocess entries even if output already exists.",
    )
    args = parser.parse_args()

    print("Loading CoordEstimator …")
    estimator = CoordEstimator(device=args.device)

    if args.json_path:
        # Custom JSON mode
        if not args.img_root or not args.out_dir:
            parser.error("--json_path requires --img_root and --out_dir")
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  JSON    : {args.json_path}")
        print(f"  Img root: {args.img_root}")
        print(f"  Out dir : {out_root}")
        print(f"{'='*60}")

        # Register as temporary dataset
        DATASETS["_custom"] = {
            "iter": lambda limit: _iter_custom_json(
                args.json_path, args.img_root, args.img_key, args.id_key, limit
            ),
            "out_dir": out_root,
        }
        process_dataset("_custom", estimator, limit=args.limit,
                        skip_existing=not args.no_skip)
    else:
        # Predefined datasets mode
        ds_list = args.datasets or list(DATASETS.keys())
        for ds in ds_list:
            print(f"\n{'='*60}")
            print(f"  Dataset : {ds}")
            print(f"  Out dir : {DATASETS[ds]['out_dir']}")
            print(f"{'='*60}")
            process_dataset(ds, estimator, limit=args.limit,
                            skip_existing=not args.no_skip)

    print("\nAll done.")


if __name__ == "__main__":
    main()
