"""
process_pointcloud.py

For each entry in train_10k.json, reconstructs per-pixel 3D world coordinates
(x, y, z) in the first-frame camera coordinate system and saves the results.

Output layout
-------------
pointcloud/
  {entry_id}.npz
    frames/         — one group per frame index (0, 1, 2, ...)
      {i}_xyz       (H, W, 3) float16  — 3-D world coords (metres), first-frame origin
      {i}_valid     (H, W) bool        — True where depth > 0
      {i}_frame_id  scalar str         — original frame identifier
    scene_id        scalar str
    dataset         scalar str

float16 is used by default to keep file sizes manageable (~1.8 MB per 640×480
frame vs ~3.7 MB for float32).  Pass --float32 to override.

Usage
-----
python process_pointcloud.py                  # test: first 10 entries
python process_pointcloud.py --max -1         # all entries
python process_pointcloud.py --max -1 --float32
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reconstruct_3d import reconstruct_entry

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SPAR_ROOT   = (
    "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/"
    "spatial_planning/datasets/train/SPAR_7M/spar"
)
_DEFAULT_IN  = os.path.join(_SPAR_ROOT, "train_10k.json")
_DEFAULT_OUT = os.path.join(_SPAR_ROOT, "3D_pos")


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    spar_root: str,
    dtype: np.dtype = np.float16,
) -> Optional[dict]:
    """Reconstruct per-pixel 3-D coords for every frame in one QA entry.

    Returns
    -------
    dict with keys:
        frames    : list of dicts, each with
                      'xyz'      (H, W, 3) dtype   — 3-D in first-frame world
                      'valid'    (H, W) bool
                      'frame_id' str
        scene_id  : str
        dataset   : str
    Returns None on failure.
    """
    # Some task types (position_matching, view_change_infer) omit the 'depth'
    # key even though the depth images exist on disk.  Derive paths from the
    # colour image paths: image_color/X.jpg → image_depth/X.png
    if "depth" not in entry:
        entry = dict(entry)  # shallow copy — don't mutate the original
        entry["depth"] = [
            img.replace("image_color", "image_depth").replace(".jpg", ".png")
            for img in entry["image"]
        ]

    try:
        frame_results = reconstruct_entry(entry, spar_root)
    except Exception as exc:
        print(f"  [ERR] {entry.get('id')}: {exc}")
        return None

    frames = []
    for fr in frame_results:
        xyz   = fr["points_world"].astype(dtype)   # (H, W, 3)
        valid = fr["valid_mask"]                    # (H, W) bool
        # Zero out invalid pixels to avoid garbage float16 Inf/NaN
        xyz[~valid] = 0.0
        frames.append({
            "xyz":      xyz,
            "valid":    valid,
            "frame_id": fr["frame_id"],
        })

    return {
        "frames":   frames,
        "scene_id": frame_results[0]["scene_id"],
        "dataset":  frame_results[0]["dataset"],
    }


def save_entry(result: dict, out_path: str) -> None:
    """Save one entry's point-cloud data to a compressed .npz file."""
    arrays = {}
    for i, fr in enumerate(result["frames"]):
        arrays[f"frame_{i}_xyz"]      = fr["xyz"]
        arrays[f"frame_{i}_valid"]    = fr["valid"]
        arrays[f"frame_{i}_frame_id"] = np.array(fr["frame_id"])

    arrays["scene_id"] = np.array(result["scene_id"])
    arrays["dataset"]  = np.array(result["dataset"])
    arrays["n_frames"] = np.array(len(result["frames"]))

    np.savez_compressed(out_path, **arrays)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def process_json(
    input_path: str,
    spar_root: str,
    out_dir: str,
    max_entries: int = 10,
    dtype: np.dtype = np.float16,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path) as fh:
        data = json.load(fh)

    total = len(data) if max_entries < 0 else min(max_entries, len(data))
    print(f"Processing {total} entries  dtype={dtype.__name__}  → {out_dir}")

    ok = skipped = 0

    for i, entry in enumerate(data[:total]):
        entry_id = entry.get("id", f"entry_{i:06d}")
        out_path = os.path.join(out_dir, f"{entry_id}.npz")

        result = process_entry(entry, spar_root, dtype=dtype)
        if result is None:
            skipped += 1
            print(f"  [{i:5d}] SKIP      {entry_id}")
            continue

        save_entry(result, out_path)

        shapes = " | ".join(
            f"{fr['frame_id']}:{fr['xyz'].shape}" for fr in result["frames"]
        )
        print(
            f"  [{i:5d}] OK  {entry_id:42s}  "
            f"ds={result['dataset']:12s}  {shapes}"
        )
        ok += 1

    print(
        f"\nDone.  ok={ok}  error={skipped}  total={total}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct per-pixel 3D point clouds from SPAR depth maps."
    )
    p.add_argument("--input",   default=_DEFAULT_IN,  help="Input JSON file")
    p.add_argument("--spar",    default=_SPAR_ROOT,   help="spar/ root dir")
    p.add_argument("--out",     default=_DEFAULT_OUT, help="Output directory")
    p.add_argument("--max",     type=int, default=10, help="Max entries (-1 = all)")
    p.add_argument("--float32", action="store_true",  help="Save as float32 (larger)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dtype = np.float32 if args.float32 else np.float16
    process_json(
        input_path=args.input,
        spar_root=args.spar,
        out_dir=args.out,
        max_entries=args.max,
        dtype=dtype,
    )
