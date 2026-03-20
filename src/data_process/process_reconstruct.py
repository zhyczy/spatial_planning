"""
process_reconstruct.py

Processes entries from train_10k.json (or any SPAR JSON file), computes per-frame
camera poses and pairwise relative transformation matrices, and saves the results to
a reconstruct/ folder.

For each entry, saves one .npz file:
    {out_dir}/{entry_id}.npz
    Keys:
        poses              (N, 4, 4) float64  camera-to-world in first-frame coords
        relative_transforms (N, N, 4, 4) float64
                           [i, j] = T_{i→j}: transforms 3D points from camera-i
                           frame to camera-j frame  (= pose_j^{-1} @ pose_i)
        frame_ids          (N,) str array — e.g. ["3148", "107"]
        scene_id           str — first scene id in entry
        dataset            str — "scannet" / "scannetpp" / "structured3d"

Usage
-----
python process_reconstruct.py                        # test: first 10 entries
python process_reconstruct.py --max 10
python process_reconstruct.py --max -1               # all entries
python process_reconstruct.py --input my.json --max 50
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# resolve imports whether run as script or module
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reconstruct_3d import (
    detect_dataset,
    load_extrinsics,
    load_intrinsics,
    load_pose,
    _normalise_pose,
    _scene_dir,
    _frame_id_from_path,
    _scene_id_from_path,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SPAR_ROOT = (
    "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/"
    "spatial_planning/datasets/train/SPAR_7M/spar"
)
_DEFAULT_INPUT = os.path.join(_SPAR_ROOT, "train_10k.json")
_DEFAULT_OUT   = os.path.join(_SPAR_ROOT, "reconstruct")


# ---------------------------------------------------------------------------
# Pose loading (no depth required)
# ---------------------------------------------------------------------------

def load_frame_pose(
    spar_root: str,
    dataset: str,
    scene_id: str,
    frame_id: str,
) -> np.ndarray:
    """Load and normalise (to metres) the 4×4 camera-to-world pose for one frame."""
    sd = _scene_dir(spar_root, dataset, scene_id)
    pose_raw = load_pose(sd, frame_id)
    pose = _normalise_pose(pose_raw, dataset)

    # ScanNet: compose with extrinsic_depth so depth-cam rays align with
    # the pose reference frame (no-op in practice since extrinsic = identity).
    if dataset == "scannet":
        ext_depth, _ = load_extrinsics(dataset, sd)
        pose = pose @ ext_depth

    return pose  # (4, 4)


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------

def process_entry(entry: dict, spar_root: str) -> Optional[dict]:
    """Compute camera poses and pairwise transforms for all frames in an entry.

    Works with or without a 'depth' key (pose loading doesn't need depth maps).

    Returns None if the entry cannot be processed (e.g. missing pose file).
    """
    image_paths: List[str] = entry.get("image", [])
    if not image_paths:
        return None

    # Detect dataset from the first image path
    scene_id_0 = _scene_id_from_path(image_paths[0])
    dataset = detect_dataset(scene_id_0)

    # ------------------------------------------------------------------
    # Load all poses
    # ------------------------------------------------------------------
    poses_world: List[np.ndarray] = []   # camera-to-world (dataset frame, metres)
    frame_ids: List[str] = []

    for img_rel in image_paths:
        scene_id = _scene_id_from_path(img_rel)
        frame_id  = _frame_id_from_path(img_rel)
        try:
            pose = load_frame_pose(spar_root, dataset, scene_id, frame_id)
        except FileNotFoundError as exc:
            print(f"  [SKIP] Missing file: {exc}")
            return None
        poses_world.append(pose)
        frame_ids.append(frame_id)

    N = len(poses_world)

    # ------------------------------------------------------------------
    # Normalise to first-frame origin
    #   pose_world[0] = camera-0-to-dataset-world
    #   T0_inv        = dataset-world-to-camera-0-frame
    #   pose_ff[i]    = camera-i-to-first-frame-world
    # ------------------------------------------------------------------
    T0_inv = np.linalg.inv(poses_world[0])
    poses_ff = np.stack(
        [T0_inv @ p for p in poses_world], axis=0
    )  # (N, 4, 4)

    # poses_ff[0] should be identity (camera 0 in its own frame)

    # ------------------------------------------------------------------
    # Pairwise relative transforms
    #   relative_transforms[i, j] = T_{i→j}
    #   = pose_j^{-1} @ pose_i    (in first-frame coords)
    #   Transforms a 3-D point from camera-i frame to camera-j frame.
    # ------------------------------------------------------------------
    rel = np.zeros((N, N, 4, 4), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            # pose_ff[i] = camera_i-to-first-frame
            # pose_ff[j] = camera_j-to-first-frame
            # T_{i→j} = inv(pose_ff[j]) @ pose_ff[i]
            rel[i, j] = np.linalg.inv(poses_ff[j]) @ poses_ff[i]

    return {
        "poses": poses_ff,                  # (N, 4, 4)
        "relative_transforms": rel,          # (N, N, 4, 4)
        "frame_ids": np.array(frame_ids),    # (N,) dtype=str
        "scene_id": scene_id_0,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------

def process_json(
    input_path: str,
    spar_root: str,
    out_dir: str,
    max_entries: int = 10,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path) as fh:
        data = json.load(fh)

    total = len(data) if max_entries < 0 else min(max_entries, len(data))
    print(f"Processing {total} entries → {out_dir}")

    success = skipped = 0

    for i, entry in enumerate(data[:total]):
        entry_id = entry.get("id", f"entry_{i:06d}")
        out_path = os.path.join(out_dir, f"{entry_id}.npz")

        result = process_entry(entry, spar_root)
        if result is None:
            print(f"  [{i:5d}] SKIP  {entry_id}")
            skipped += 1
            continue

        np.savez_compressed(
            out_path,
            poses=result["poses"],
            relative_transforms=result["relative_transforms"],
            frame_ids=result["frame_ids"],
            scene_id=np.array(result["scene_id"]),
            dataset=np.array(result["dataset"]),
        )
        n = len(result["frame_ids"])
        print(
            f"  [{i:5d}] OK    {entry_id:40s}  "
            f"dataset={result['dataset']:12s}  "
            f"frames={n}  "
            f"saved → {os.path.basename(out_path)}"
        )
        success += 1

    print(f"\nDone. success={success}  skipped={skipped}  total={total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Reconstruct camera poses from SPAR JSON.")
    p.add_argument("--input",  default=_DEFAULT_INPUT, help="Path to input JSON file")
    p.add_argument("--spar",   default=_SPAR_ROOT,     help="Path to spar/ root dir")
    p.add_argument("--out",    default=_DEFAULT_OUT,   help="Output directory")
    p.add_argument("--max",    type=int, default=10,
                   help="Max entries to process (-1 = all)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_json(
        input_path=args.input,
        spar_root=args.spar,
        out_dir=args.out,
        max_entries=args.max,
    )
