"""
reconstruct_3d.py

Given a QA JSONL entry (from SPAR_7M), reconstructs per-pixel 3D world coordinates
for each frame referenced in the entry. The first frame's camera pose is used as the
global world coordinate origin (all subsequent frames are expressed relative to it).

Supports scannet, scannetpp, and structured3d with their respective camera parameter
layouts:

  ScanNet      — intrinsic_color/depth (4×4), extrinsic_color/depth (4×4 identity),
                 per-frame pose (4×4 camera-to-world, meters).
                 Depth images: 640×480, 16-bit PNG, unit = mm → /1000 for metres.
                 Color images: 1296×968. Depth is already registered to the colour
                 camera frame (extrinsics are identity), so we unproject depth pixels
                 using intrinsic_depth and apply the per-frame pose directly.

  ScanNetPP    — intrinsic_color/depth (4×4, identical),
                 NO extrinsic files (single sensor, perfectly aligned),
                 per-frame pose (4×4 camera-to-world, meters).
                 Depth images: ~192×256, unit = mm → /1000.

  Structured3D — per-frame intrinsic (3×3, one file per image), NO extrinsics,
                 per-frame pose (4×4 camera-to-world, mm) → /1000 for metres.
                 Depth images: 720×1280, unit = mm → /1000.
                 Synthetic data: single virtual camera per frame.

Coordinate convention (all datasets after unprojection):
  X right, Y down, Z forward  (standard pinhole camera coords)
  Units: metres

Usage example
-------------
>>> from reconstruct_3d import reconstruct_entry
>>> results = reconstruct_entry(entry, spar_root="/path/to/spar")
>>> for frame in results:
...     pts = frame["points_world"]   # (H, W, 3) float32, metres, world frame
...     valid = frame["valid_mask"]   # (H, W) bool
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset detection
# ---------------------------------------------------------------------------

def detect_dataset(scene_id: str) -> str:
    """Infer dataset name from scene identifier.

    Returns one of: 'scannet', 'scannetpp', 'structured3d'.
    """
    if re.fullmatch(r"scene\d{4}_\d{2}", scene_id):
        return "scannet"
    if re.fullmatch(r"scene_\d{5}", scene_id):
        return "structured3d"
    # ScanNetPP uses hex scene IDs (e.g. "8a20d62ac0")
    return "scannetpp"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _scene_dir(spar_root: str, dataset: str, scene_id: str) -> str:
    return os.path.join(spar_root, dataset, "images", scene_id)


def _frame_id_from_path(image_rel: str) -> str:
    """Extract frame id (stem without extension) from a relative image path.

    e.g. "scene0012_01/image_color/3148.jpg"  -> "3148"
         "scene_02968/image_color/350_1.jpg"   -> "350_1"
         "8a20d62ac0/image_color/3130.jpg"     -> "3130"
    """
    return os.path.splitext(os.path.basename(image_rel))[0]


def _scene_id_from_path(image_rel: str) -> str:
    """Extract scene id from relative image path (first path component)."""
    return image_rel.split("/")[0]


# ---------------------------------------------------------------------------
# Camera parameter loaders
# ---------------------------------------------------------------------------

def _load_matrix(path: str) -> np.ndarray:
    """Load a whitespace-delimited matrix from a text file."""
    return np.loadtxt(path, dtype=np.float64)


def load_intrinsics(
    dataset: str, scene_dir: str, frame_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load depth and colour intrinsic matrices (both as 3×3).

    Returns
    -------
    K_depth : (3, 3)  intrinsic matrix for the depth image
    K_color : (3, 3)  intrinsic matrix for the colour image
    """
    intrinsic_dir = os.path.join(scene_dir, "intrinsic")

    if dataset == "structured3d":
        # One file per frame, stored as 3×3.
        path = os.path.join(intrinsic_dir, f"{frame_id}.txt")
        K = _load_matrix(path).astype(np.float64)  # already 3×3
        if K.shape != (3, 3):
            raise ValueError(f"Expected 3×3 intrinsic at {path}, got {K.shape}")
        return K, K  # single camera; depth == colour

    # ScanNet / ScanNetPP: 4×4 homogeneous matrix → extract upper-left 3×3
    def _to_3x3(mat: np.ndarray) -> np.ndarray:
        if mat.shape == (4, 4):
            return mat[:3, :3]
        if mat.shape == (3, 3):
            return mat
        raise ValueError(f"Unexpected intrinsic shape {mat.shape}")

    K_depth = _to_3x3(_load_matrix(os.path.join(intrinsic_dir, "intrinsic_depth.txt")))
    K_color = _to_3x3(_load_matrix(os.path.join(intrinsic_dir, "intrinsic_color.txt")))
    return K_depth, K_color


def load_extrinsics(dataset: str, scene_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load depth-to-reference and colour-to-reference extrinsic matrices (4×4).

    For ScanNet both are identity (sensors already co-registered).
    For ScanNetPP and Structured3D no extrinsic files exist → return identity.
    """
    identity = np.eye(4, dtype=np.float64)
    if dataset != "scannet":
        return identity, identity

    intrinsic_dir = os.path.join(scene_dir, "intrinsic")
    ext_depth = _load_matrix(os.path.join(intrinsic_dir, "extrinsic_depth.txt"))
    ext_color = _load_matrix(os.path.join(intrinsic_dir, "extrinsic_color.txt"))
    return ext_depth.astype(np.float64), ext_color.astype(np.float64)


def load_pose(scene_dir: str, frame_id: str) -> np.ndarray:
    """Load per-frame 4×4 camera-to-world transformation matrix."""
    path = os.path.join(scene_dir, "pose", f"{frame_id}.txt")
    return _load_matrix(path).astype(np.float64)


def load_depth(scene_dir: str, frame_id: str) -> np.ndarray:
    """Load depth image as float32 array (H, W), unit = mm (raw integer values)."""
    path = os.path.join(scene_dir, "image_depth", f"{frame_id}.png")
    img = Image.open(path)
    return np.array(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Depth scale helpers
# ---------------------------------------------------------------------------

# All three datasets store depth in millimetres; divide by 1000 → metres.
_DEPTH_SCALE_MM_TO_M = 1.0 / 1000.0

# Structured3D stores pose translations in mm as well.
_POSE_SCALE: Dict[str, float] = {
    "scannet": 1.0,       # metres
    "scannetpp": 1.0,     # metres
    "structured3d": 1e-3, # mm → metres
}


def _normalise_pose(pose: np.ndarray, dataset: str) -> np.ndarray:
    """Convert pose translation to metres if necessary."""
    scale = _POSE_SCALE[dataset]
    if scale == 1.0:
        return pose
    out = pose.copy()
    out[:3, 3] *= scale
    return out


# ---------------------------------------------------------------------------
# 3D unprojection
# ---------------------------------------------------------------------------

def unproject_depth(
    depth_m: np.ndarray,  # (H, W) float32, metres
    K_depth: np.ndarray,  # (3, 3)
    pose_c2w: np.ndarray, # (4, 4) camera-to-world, metres
    T0_inv: Optional[np.ndarray] = None,  # (4, 4) world-to-first-frame world
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map into 3D world coordinates.

    Pipeline
    --------
    pixel (u, v) + depth d  →  camera-frame point  →  world-frame point
                                                    →  (optionally) first-frame world

    Parameters
    ----------
    depth_m  : (H, W) depth in metres; 0 or NaN means invalid.
    K_depth  : 3×3 intrinsic of the depth sensor.
    pose_c2w : 4×4 camera-to-world for this frame (metres).
    T0_inv   : if provided, transforms world coords into the first-frame camera
               coordinate system (first-frame origin convention).

    Returns
    -------
    points_world : (H, W, 3) float32 – 3D points in world frame (metres).
    valid_mask   : (H, W) bool       – True where depth is valid (> 0).
    """
    H, W = depth_m.shape
    valid_mask = (depth_m > 0) & np.isfinite(depth_m)

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]

    # Pixel coordinate grids
    us = np.arange(W, dtype=np.float64)
    vs = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)  # (H, W)

    d = depth_m.astype(np.float64)

    # Unproject to camera-frame coordinates
    Xc = (uu - cx) * d / fx
    Yc = (vv - cy) * d / fy
    Zc = d

    # Stack to homogeneous (4, H*W)
    ones = np.ones_like(Zc)
    pts_cam = np.stack([Xc, Yc, Zc, ones], axis=0).reshape(4, -1)

    # Camera → world (dataset world frame)
    pts_world = pose_c2w @ pts_cam  # (4, H*W)

    # World → first-frame world (optional normalisation)
    if T0_inv is not None:
        pts_world = T0_inv @ pts_world  # (4, H*W)

    points_world = pts_world[:3, :].T.reshape(H, W, 3).astype(np.float32)
    return points_world, valid_mask


# ---------------------------------------------------------------------------
# Per-entry reconstruction
# ---------------------------------------------------------------------------

def reconstruct_entry(
    entry: dict,
    spar_root: str,
) -> List[Dict]:
    """Reconstruct 3D world coordinates for every frame in a QA entry.

    The first listed frame's camera position defines the world origin:
    all points are expressed in a coordinate system where that camera's
    optical centre is at (0, 0, 0).

    Parameters
    ----------
    entry     : one parsed line from a SPAR JSONL file.
    spar_root : absolute path to the ``spar/`` directory (contains
                ``scannet/``, ``scannetpp/``, ``structured3d/``).

    Returns
    -------
    List of per-frame dicts, one per image listed in ``entry["image"]``:
        {
          "frame_id"     : str,          # e.g. "3148"
          "scene_id"     : str,          # e.g. "scene0012_01"
          "dataset"      : str,          # "scannet" | "scannetpp" | "structured3d"
          "points_world" : (H, W, 3) float32,  # 3-D coords, metres, first-frame origin
          "valid_mask"   : (H, W) bool,         # True where depth is valid
          "pose_world"   : (4, 4) float64,      # camera-to-world in first-frame coords
          "K_depth"      : (3, 3) float64,      # depth intrinsic
        }
    """
    image_paths: List[str] = entry["image"]
    depth_paths: List[str] = entry.get("depth", [])

    if len(image_paths) != len(depth_paths):
        raise ValueError(
            f"Entry {entry.get('id')} has {len(image_paths)} images but "
            f"{len(depth_paths)} depth maps."
        )

    # ------------------------------------------------------------------
    # Step 1: infer scene and dataset from the first image path
    # ------------------------------------------------------------------
    scene_id_0 = _scene_id_from_path(image_paths[0])
    dataset = detect_dataset(scene_id_0)

    # ------------------------------------------------------------------
    # Step 2: compute T0_inv (first-frame camera-to-world, inverted)
    # ------------------------------------------------------------------
    frame_id_0 = _frame_id_from_path(image_paths[0])
    scene_dir_0 = _scene_dir(spar_root, dataset, scene_id_0)
    pose0_raw = load_pose(scene_dir_0, frame_id_0)
    pose0 = _normalise_pose(pose0_raw, dataset)   # metres
    T0_inv = np.linalg.inv(pose0)                  # world-to-first-frame

    # ------------------------------------------------------------------
    # Step 3: process each frame
    # ------------------------------------------------------------------
    results = []

    for img_rel, dep_rel in zip(image_paths, depth_paths):
        scene_id = _scene_id_from_path(img_rel)
        frame_id = _frame_id_from_path(img_rel)

        if scene_id != scene_id_0:
            raise ValueError(
                f"Entry {entry.get('id')} mixes scenes: "
                f"{scene_id_0} vs {scene_id}."
            )

        scene_dir = _scene_dir(spar_root, dataset, scene_id)

        # Camera parameters
        K_depth, _ = load_intrinsics(dataset, scene_dir, frame_id)
        pose_raw = load_pose(scene_dir, frame_id)
        pose = _normalise_pose(pose_raw, dataset)  # camera-to-world (metres)

        # For ScanNet, extrinsics document the depth↔colour alignment.
        # Both are identity here, but we apply ext_depth for completeness
        # so that depth-camera rays are expressed in the colour-camera frame
        # (which is what the pose references).
        if dataset == "scannet":
            ext_depth, _ = load_extrinsics(dataset, scene_dir)
            # pose_depth_to_world = pose_color_to_world @ ext_depth_to_color
            # ext_depth here is depth-to-color (identity in practice)
            pose = pose @ ext_depth

        # Depth map (raw mm integers → metres)
        depth_raw = load_depth(scene_dir, frame_id)
        depth_m = depth_raw * _DEPTH_SCALE_MM_TO_M

        # Unproject
        points_world, valid_mask = unproject_depth(
            depth_m, K_depth, pose, T0_inv=T0_inv
        )

        # Express camera pose itself in first-frame coordinates
        pose_in_first_frame = T0_inv @ pose

        results.append(
            {
                "frame_id": frame_id,
                "scene_id": scene_id,
                "dataset": dataset,
                "points_world": points_world,   # (H, W, 3) metres, first-frame origin
                "valid_mask": valid_mask,        # (H, W) bool
                "pose_world": pose_in_first_frame,  # (4, 4) in first-frame coords
                "K_depth": K_depth,              # (3, 3)
            }
        )

    return results


# ---------------------------------------------------------------------------
# Convenience: process a whole JSONL file
# ---------------------------------------------------------------------------

def reconstruct_jsonl(
    jsonl_path: str,
    spar_root: str,
    max_entries: Optional[int] = None,
) -> List[List[Dict]]:
    """Reconstruct 3D for every entry in a JSONL file.

    Parameters
    ----------
    jsonl_path  : path to the .jsonl file.
    spar_root   : path to the ``spar/`` root directory.
    max_entries : if set, stop after this many entries (useful for debugging).

    Returns
    -------
    List of per-entry reconstruction results (same structure as
    ``reconstruct_entry``).
    """
    import json

    all_results = []
    with open(jsonl_path, "r") as fh:
        for i, line in enumerate(fh):
            if max_entries is not None and i >= max_entries:
                break
            entry = json.loads(line)
            try:
                result = reconstruct_entry(entry, spar_root)
            except Exception as exc:
                print(f"[WARN] Entry {entry.get('id')} failed: {exc}")
                result = []
            all_results.append(result)
    return all_results


# ---------------------------------------------------------------------------
# Quick smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    SPAR_ROOT = (
        "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/"
        "spatial_planning/datasets/train/SPAR_7M/spar"
    )

    test_cases = {
        "scannet": (
            f"{SPAR_ROOT}/scannet/qa_jsonl/train/camera_motion_infer/fill/fill_91526.jsonl"
        ),
        "scannetpp": (
            f"{SPAR_ROOT}/scannetpp/qa_jsonl/train/camera_motion_infer/fill"
        ),
        "structured3d": (
            f"{SPAR_ROOT}/structured3d/qa_jsonl/train/camera_motion_infer/fill"
        ),
    }

    import glob

    for ds, path in test_cases.items():
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.jsonl"))
            if not files:
                continue
            path = files[0]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  |  file: {os.path.basename(path)}")
        print("=" * 60)

        with open(path) as fh:
            entry = json.loads(fh.readline())

        frames = reconstruct_entry(entry, SPAR_ROOT)
        for fr in frames:
            pts = fr["points_world"]
            vm  = fr["valid_mask"]
            print(
                f"  frame {fr['frame_id']:>10s}  "
                f"depth shape={pts.shape}  valid={vm.sum()}  "
                f"X∈[{pts[vm,0].min():.2f}, {pts[vm,0].max():.2f}]  "
                f"Y∈[{pts[vm,1].min():.2f}, {pts[vm,1].max():.2f}]  "
                f"Z∈[{pts[vm,2].min():.2f}, {pts[vm,2].max():.2f}]"
            )
        print(f"  pose of frame[0] in first-frame coords (should be identity):")
        print(frames[0]["pose_world"].round(4))
