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
  viewspatial           ViewSpatial-Bench.json      — image_path: rel paths (strip prefix)
  omnispatial_pt        OmniSpatial-test/data.json  — Perspective_Taking subset (561)
  embspatial            embspatial_bench.json       — image_path: pre-cached images/{qid}.jpg

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

Saved (H, W) lands on RESOLUTION_MAPPINGS[518] for ALL paths — Step A
(find_closest_aspect_ratio + crop_resize_if_necessary) is applied by
CoordEstimator: _views_from_arrays for multi-view, _estimate_single_view
for single-view. So pts3d shape distribution is uniform across training
(VST si_/mi_) and eval (multi-view + single-view).

The Qwen smart_resize alignment (H, W rounded to multiples of 28) is NOT
applied at save time — the dataloader handles it at load time so the same
3d_results/ can feed downstream models with different patch grids. Pass
--qwen_align to re-enable save-time alignment.

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
    json_path = _TRAIN_ROOT / "MindCube" / "MindCube_train.jsonl"
    with open(json_path) as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            entry = json.loads(line)
            imgs = _load_images_from_paths(img_root, entry.get("images", []))
            if imgs:
                yield str(entry.get("id", i)), imgs


def _iter_sat_train(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    img_root = _EVAL_ROOT / "SAT"  # images shared with eval (./data/train/...)
    json_path = _TRAIN_ROOT / "SAT" / "train_36k.json"
    with open(json_path) as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        imgs = _load_images_from_paths(img_root, entry.get("img_paths", []))
        if imgs:
            yield str(entry.get("database_idx", i)), imgs


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


def _iter_viewspatial(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """ViewSpatial-Bench: image_path entries are 'ViewSpatial-Bench/...' —
    strip the leading prefix so they resolve under the dataset root.
    Index matches eval_dataset.py's `idx` (sequential 0..5711)."""
    root = _EVAL_ROOT / "ViewSpatial-Bench"
    with open(root / "ViewSpatial-Bench.json") as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        rels = []
        for p in entry.get("image_path", []):
            rels.append(p.split("/", 1)[1] if p.startswith("ViewSpatial-Bench/") else p)
        imgs = _load_images_from_paths(root, rels)
        if imgs:
            yield str(i), imgs


def _iter_omnispatial_pt(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """OmniSpatial Perspective_Taking subset (561 Q). Each qid has its own
    image at Perspective_Taking/{image_number}.png. Entry id = qid (e.g. '0_0')
    matches eval_dataset.py's index."""
    root = _EVAL_ROOT / "OmniSpatial"
    with open(root / "OmniSpatial-test" / "data.json") as f:
        data = json.load(f)
    data = [d for d in data if d.get("task_type") == "Perspective_Taking"]
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        qid = entry.get("id", str(i))
        img_num = qid.split("_")[0]
        imgs = _load_images_from_paths(
            root, [f"OmniSpatial-test/Perspective_Taking/{img_num}.png"]
        )
        if imgs:
            yield qid, imgs


def _iter_vst_train(
    limit: int = -1,
    shard_idx: int = 0,
    n_shards: int = 1,
    subsets=None,
) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """VST SFT data: si_* (1-image) + mi_* (2-image), skipping video/.

    Step A (find_closest_aspect_ratio + crop_resize_if_necessary) is applied
    by CoordEstimator on both paths — _views_from_arrays for multi-view and
    _estimate_single_view for single-view — so iterators just hand over raw
    decoded images and pts3d uniformly lands on RESOLUTION_MAPPINGS[518].

    Reads images directly from parquet bytes (no parse_vst_500k step required).
    Internal sharding skips decode work for entries belonging to other shards,
    which matters at this scale (~563K entries).
    Entry id = '<subset>/<row_id>' so 3d_results stays organized by subset.
    """
    import pyarrow.parquet as pq
    from io import BytesIO as _BIO

    root = _TRAIN_ROOT / "VST"
    if subsets is None:
        subsets = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name.startswith(("si_", "mi_"))
        )
    else:
        subsets = list(subsets)

    global_idx = 0
    yielded = 0
    for subset in subsets:
        sub_dir = root / subset
        if not sub_dir.is_dir():
            print(f"[WARN] vst_train: subset dir missing: {sub_dir}")
            continue
        for parquet_file in sorted(sub_dir.glob("*.parquet")):
            try:
                pf = pq.ParquetFile(parquet_file)
            except Exception as e:
                print(f"[WARN] vst_train: cannot open {parquet_file}: {e}")
                continue
            for batch in pf.iter_batches(batch_size=64, columns=["id", "images"]):
                for row in batch.to_pylist():
                    if (global_idx % n_shards) != shard_idx:
                        global_idx += 1
                        continue
                    if 0 < limit <= yielded:
                        return
                    entry_id = f"{subset}/{row['id']}"
                    arrs: List[np.ndarray] = []
                    for img_obj in row["images"]:
                        b = img_obj.get("bytes") or img_obj.get("byres")
                        if not b:
                            continue
                        try:
                            arrs.append(np.array(
                                Image.open(_BIO(b)).convert("RGB"),
                                dtype=np.uint8,
                            ))
                        except Exception as e:
                            print(f"[WARN] vst_train decode failed {entry_id}: {e}")
                    if arrs:
                        yield entry_id, arrs
                        yielded += 1
                    global_idx += 1


def _iter_spinbench(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """SpinBench: 4-image multiple-choice tasks. Each entry's `images` field
    has 4 paths but the number of *unique* images is 1-5 (most tasks reuse the
    same view for several choices). Existing 3d_results follows that
    convention — view count on disk = number of unique images. We replicate
    it here by deduplicating paths in-order before estimation.

    Entry id = 12-char hex hash from test_idx_to_3d_id.json (so the layout
    matches what downstream eval/training scripts already index)."""
    root = _EVAL_ROOT / "spinbench_data"
    with open(root / "test_idx_to_3d_id.json") as f:
        idx_to_id = json.load(f)
    with open(root / "test.jsonl") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            entry = json.loads(line)
            entry_id = idx_to_id.get(str(i), str(i))
            uniq_paths = list(dict.fromkeys(entry.get("images", [])))
            imgs = _load_images_from_paths(root, uniq_paths)
            if imgs:
                yield entry_id, imgs


def _iter_embspatial(limit: int = -1) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """EmbSpatial-Bench: images are pre-extracted to data_dir/images/{qid}.jpg
    by the eval_dataset.py loader on first load. Entry id = question_id
    (e.g. 'mp3d_0') matches eval_dataset.py's index."""
    root = _EVAL_ROOT / "EmbSpatial-Bench"
    with open(root / "embspatial_bench.json") as f:
        data = json.load(f)
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        qid = entry.get("question_id", str(i))
        imgs = _load_images_from_paths(root, [f"images/{qid}.jpg"])
        if imgs:
            yield qid, imgs


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
    "sat_train": {
        "iter": _iter_sat_train,
        "out_dir": _TRAIN_ROOT / "SAT" / "3d_results",
    },
    "sparbench_mv": {
        "iter": lambda limit: _iter_sparbench(
            _EVAL_ROOT / "SPARBench" / "sparbench_mv.json", limit
        ),
        "out_dir": _EVAL_ROOT / "SPARBench" / "3d_results",
    },
    "viewspatial": {
        "iter": _iter_viewspatial,
        "out_dir": _EVAL_ROOT / "ViewSpatial-Bench" / "3d_results",
    },
    "omnispatial_pt": {
        "iter": _iter_omnispatial_pt,
        "out_dir": _EVAL_ROOT / "OmniSpatial" / "3d_results",
    },
    "embspatial": {
        "iter": _iter_embspatial,
        "out_dir": _EVAL_ROOT / "EmbSpatial-Bench" / "3d_results",
    },
    "spinbench": {
        "iter": _iter_spinbench,
        "out_dir": _EVAL_ROOT / "spinbench_data" / "3d_results",
    },
    "vst_train": {
        "iter": _iter_vst_train,
        "out_dir": _TRAIN_ROOT / "VST" / "3d_results",
        "shard_aware": True,
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
    shard_idx: int = 0,
    n_shards: int = 1,
    save_image: bool = True,
    qwen_align: bool = False,
) -> None:
    """Estimate 3D for every entry in *name* and save results to disk.

    Sharding: with n_shards > 1, only entries where (i % n_shards == shard_idx)
    are processed (i is the enumeration index from the iterator). For datasets
    with shard_aware=True, the iterator handles sharding internally so per-shard
    decode work is also avoided.

    qwen_align: if False (default), save at the reconstruction model's native
    output shape (MapAny /14 or parquet-native for single-view eval) and let
    the dataloader handle Qwen smart_resize alignment. Set True to bake
    /28 alignment into the saved files.
    """
    cfg = DATASETS[name]
    out_root: Path = cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    shard_aware = cfg.get("shard_aware", False)
    if shard_aware:
        iterator = cfg["iter"](limit, shard_idx=shard_idx, n_shards=n_shards)
    else:
        iterator = cfg["iter"](limit)

    n_ok = n_skip = n_err = 0
    tag = f"{name}[{shard_idx}/{n_shards}]" if n_shards > 1 else name

    for i, (entry_id, imgs) in enumerate(iterator):
        if not shard_aware and n_shards > 1 and (i % n_shards) != shard_idx:
            continue
        entry_out = out_root / entry_id
        if skip_existing and (entry_out / "cameras.json").exists():
            n_skip += 1
            continue

        try:
            results = estimator.estimate(imgs, save_dir=None,
                                          qwen_align=qwen_align)
            save_results(results, save_dir=out_root, run_name=entry_id,
                         save_image=save_image)
            n_ok += 1
        except Exception as exc:
            print(f"[WARN] {tag}/{entry_id} failed: {exc}")
            n_err += 1

    print(f"[{tag}] done — ok={n_ok}  skipped={n_skip}  errors={n_err}")


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
    parser.add_argument(
        "--no_save_image",
        action="store_true",
        help="Skip saving image.png in each view directory (saves disk).",
    )
    parser.add_argument(
        "--vst_subsets",
        nargs="+",
        default=None,
        help="VST subsets to process (e.g. 'si_distance mi_correspondence'). "
             "Default: all si_* and mi_* subsets.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Shard 'i/n' (e.g. '0/4'): process only entries where index%n==i. "
             "Use this to split one dataset across multiple GPUs in parallel.",
    )
    parser.add_argument(
        "--qwen_align",
        action="store_true",
        help="Save pts3d/depth/mask/image at Qwen smart_resize target shape "
             "(/28 multiples). Default: save at reconstruction model's native "
             "shape and defer alignment to the dataloader.",
    )
    args = parser.parse_args()

    shard_idx, n_shards = 0, 1
    if args.shard:
        try:
            shard_idx, n_shards = (int(x) for x in args.shard.split("/"))
        except Exception:
            parser.error("--shard must be 'i/n' (e.g. '0/4')")
        if not (0 <= shard_idx < n_shards):
            parser.error("--shard 'i/n' requires 0 <= i < n")

    # Apply --vst_subsets to vst_train iterator if provided
    if args.vst_subsets and "vst_train" in DATASETS:
        _vst_subs = list(args.vst_subsets)
        DATASETS["vst_train"]["iter"] = (
            lambda limit, shard_idx=0, n_shards=1, _s=_vst_subs:
                _iter_vst_train(limit, shard_idx, n_shards, subsets=_s)
        )

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
                        skip_existing=not args.no_skip,
                        shard_idx=shard_idx, n_shards=n_shards,
                        save_image=not args.no_save_image,
                        qwen_align=args.qwen_align)
    else:
        # Predefined datasets mode
        ds_list = args.datasets or list(DATASETS.keys())
        for ds in ds_list:
            print(f"\n{'='*60}")
            print(f"  Dataset : {ds}")
            print(f"  Out dir : {DATASETS[ds]['out_dir']}")
            if n_shards > 1:
                print(f"  Shard   : {shard_idx}/{n_shards}")
            print(f"  Qwen-align : {args.qwen_align}")
            print(f"{'='*60}")
            process_dataset(ds, estimator, limit=args.limit,
                            skip_existing=not args.no_skip,
                            shard_idx=shard_idx, n_shards=n_shards,
                            save_image=not args.no_save_image,
                            qwen_align=args.qwen_align)

    print("\nAll done.")


if __name__ == "__main__":
    main()
