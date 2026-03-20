"""
visualize_spar.py

Visualize precomputed 3D point clouds and camera poses for any entry in
train_10k.json.  Reads from:
  - spar/3D_pos/{entry_id}.npz        per-pixel xyz + valid mask
  - spar/reconstruct/{entry_id}.npz   camera poses + pairwise transforms

Combines with original RGB images (loaded from spar/) and produces an
interactive HTML plot identical in style to coord_esti.py's visualize_results():
  - Coloured point cloud (all views merged, down-sampled to max_pts)
  - Camera coordinate axes (red X / green Y / blue Z) with frame-id labels
  - Pairwise relative-transform arrows (cam-i origin → cam-j origin, optional)
  - World-origin axes (3× longer, 2× thicker)

Usage
-----
python visualize_spar.py                            # random entry
python visualize_spar.py --id scene0089_00_9        # specific entry
python visualize_spar.py --id scene0089_00_9 --out /tmp/vis
python visualize_spar.py --no-arrows                # hide transform arrows
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SPAR_ROOT = (
    "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/"
    "spatial_planning/datasets/train/SPAR_7M/spar"
)
_VIS_DIR = (
    "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/"
    "spatial_planning/vis_results"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_entry_data(entry_id: str, spar_root: str) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Load precomputed xyz, valid masks, poses, and relative transforms.

    Returns
    -------
    frames : list of dicts with keys 'xyz' (H,W,3) float32, 'valid' (H,W) bool, 'frame_id' str
    poses  : (N, 4, 4) float64  camera-to-first-frame-world poses
    rel    : (N, N, 4, 4) float64  relative transforms T_{i→j}
    """
    pos_path = os.path.join(spar_root, "3D_pos",    f"{entry_id}.npz")
    rec_path = os.path.join(spar_root, "reconstruct", f"{entry_id}.npz")

    pos = np.load(pos_path)
    rec = np.load(rec_path)

    n_frames = int(pos["n_frames"])
    frames = []
    for i in range(n_frames):
        frames.append({
            "xyz":      pos[f"frame_{i}_xyz"].astype(np.float32),  # (H, W, 3)
            "valid":    pos[f"frame_{i}_valid"],                     # (H, W) bool
            "frame_id": str(pos[f"frame_{i}_frame_id"]),
        })

    poses = rec["poses"].astype(np.float64)                          # (N, 4, 4)
    rel   = rec["relative_transforms"].astype(np.float64)            # (N, N, 4, 4)

    return frames, poses, rel


def load_images(entry: dict, spar_root: str, target_shapes: List[Tuple]) -> List[np.ndarray]:
    """Load original RGB images and resize to match xyz (H, W).

    Asserts that the number of images in the entry equals len(target_shapes),
    i.e. 3D_pos / reconstruct are strictly in sync with train_10k.json.
    """
    import PIL.Image
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "data_process"))
    from reconstruct_3d import detect_dataset, _scene_id_from_path

    n_img = len(entry["image"])
    n_frames = len(target_shapes)
    assert n_img == n_frames, (
        f"Entry '{entry['id']}': JSON has {n_img} images but npz has {n_frames} frames. "
        "Run the audit script to find and fix mismatches first."
    )

    images = []
    for i, img_rel in enumerate(entry["image"]):
        H, W = target_shapes[i][0], target_shapes[i][1]
        scene_id = _scene_id_from_path(img_rel)
        dataset  = detect_dataset(scene_id)
        img_path = os.path.join(spar_root, dataset, "images", img_rel)
        if not os.path.exists(img_path):
            images.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
        img = PIL.Image.open(img_path).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), PIL.Image.BILINEAR)
        images.append(np.array(img))

    return images


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    entry_id: str,
    frames: List[Dict],
    poses: np.ndarray,
    rel: np.ndarray,
    images: List[np.ndarray],
    out_dir: str,
    max_pts: int = 300_000,
    axis_len: float = 0.15,
    show_arrows: bool = True,
) -> str:
    """Build interactive Plotly HTML and save to *out_dir/visualization.html*.

    Parameters mirror coord_esti.py's visualize_results().
    """
    import plotly.graph_objects as go

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    traces = []

    # ---- coloured point cloud ----------------------------------------------
    all_pts, all_cols = [], []
    for fr, img in zip(frames, images):
        mask = fr["valid"]
        all_pts.append(fr["xyz"][mask])
        all_cols.append(img[mask])

    pts_np  = np.concatenate(all_pts,  axis=0)
    cols_np = np.concatenate(all_cols, axis=0)

    if len(pts_np) > max_pts:
        idx = np.random.choice(len(pts_np), max_pts, replace=False)
        pts_np  = pts_np[idx]
        cols_np = cols_np[idx]

    colors_str = [f"rgb({r},{g},{b})" for r, g, b in cols_np]

    traces.append(go.Scatter3d(
        x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors_str, opacity=0.8),
        name="point cloud",
        hoverinfo="skip",
    ))

    # ---- camera coordinate axes --------------------------------------------
    axis_colors = {"X": "red", "Y": "green", "Z": "blue"}
    _added = {k: False for k in axis_colors}

    for v_idx, (pose, fr) in enumerate(zip(poses, frames)):
        origin = pose[:3, 3]

        for ax_i, (lbl, col) in enumerate(axis_colors.items()):
            tip  = origin + pose[:3, ax_i] * axis_len
            show = not _added[lbl]
            traces.append(go.Scatter3d(
                x=[origin[0], tip[0]],
                y=[origin[1], tip[1]],
                z=[origin[2], tip[2]],
                mode="lines",
                line=dict(color=col, width=4),
                name=f"cam-{lbl}",
                legendgroup=f"cam-{lbl}",
                showlegend=show,
            ))
            _added[lbl] = True

        # camera label: index + frame_id
        traces.append(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            mode="text",
            text=[f"cam{v_idx}<br>({fr['frame_id']})"],
            textfont=dict(size=10, color="black"),
            showlegend=False,
        ))

    # ---- relative-transform arrows (cam-i origin → cam-j origin) ----------
    # Arrow represents where cam-j's origin lies when viewed from cam-i frame,
    # i.e. the translation part of T_{i→j}.
    if show_arrows:
        N = len(poses)
        arrow_added = False
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # T_{i→j}: transforms points from cam-i to cam-j frame
                # Tip = origin of cam-j expressed in first-frame world coords
                # (= poses[j][:3, 3])  — already available directly
                # Draw as thin dashed line from cam-i to cam-j origin
                oi = poses[i, :3, 3]
                oj = poses[j, :3, 3]
                traces.append(go.Scatter3d(
                    x=[oi[0], oj[0]],
                    y=[oi[1], oj[1]],
                    z=[oi[2], oj[2]],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    name="rel-transform",
                    legendgroup="rel-transform",
                    showlegend=not arrow_added,
                ))
                arrow_added = True

    # ---- world origin axes (first-frame camera = world origin) -------------
    world_len = axis_len * 3
    for ax_i, (lbl, col) in enumerate(
        {"World-X": "red", "World-Y": "green", "World-Z": "blue"}.items()
    ):
        tip = np.zeros(3)
        tip[ax_i] = world_len
        traces.append(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines",
            line=dict(color=col, width=8),
            name=lbl,
        ))

    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=6, color="black"),
        name="Origin (cam-0)",
    ))

    # ---- layout ------------------------------------------------------------
    dataset = "unknown"
    pos_path = os.path.join(os.path.dirname(out_dir), "3D_pos", f"{entry_id}.npz")  # fallback info
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"SPAR · {entry_id}  "
            f"({len(frames)} views, {len(pts_np):,} pts)"
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = out_path / "visualization.html"
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"[visualize_spar] Saved → {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Visualize SPAR 3D point clouds and camera poses."
    )
    p.add_argument("--id",       default=None,      help="Entry ID (default: random)")
    p.add_argument("--spar",     default=_SPAR_ROOT, help="spar/ root dir")
    p.add_argument("--json",     default=None,       help="JSON file path (auto: spar/train_10k.json)")
    p.add_argument("--out",      default=_VIS_DIR,   help="Output root dir")
    p.add_argument("--max-pts",  type=int, default=300_000, help="Max point-cloud points")
    p.add_argument("--axis-len", type=float, default=0.15,  help="Camera axis length (m)")
    p.add_argument("--no-arrows", action="store_true",
                   help="Hide relative-transform arrows between cameras")
    return p.parse_args()


def main():
    args = _parse_args()

    json_path = args.json or os.path.join(args.spar, "train_10k.json")
    with open(json_path) as f:
        data = json.load(f)

    id_to_entry = {e["id"]: e for e in data}

    if args.id is None:
        entry_id = random.choice(list(id_to_entry.keys()))
        print(f"[visualize_spar] Random entry: {entry_id}")
    else:
        entry_id = args.id

    if entry_id not in id_to_entry:
        print(f"ERROR: entry '{entry_id}' not found in JSON.")
        sys.exit(1)

    entry = id_to_entry[entry_id]

    print(f"[visualize_spar] Loading data for: {entry_id}")
    frames, poses, rel = load_entry_data(entry_id, args.spar)

    target_shapes = [(fr["xyz"].shape[0], fr["xyz"].shape[1]) for fr in frames]
    images = load_images(entry, args.spar, target_shapes)

    out_dir = os.path.join(args.out, entry_id)
    visualize(
        entry_id=entry_id,
        frames=frames,
        poses=poses,
        rel=rel,
        images=images,
        out_dir=out_dir,
        max_pts=args.max_pts,
        axis_len=args.axis_len,
        show_arrows=not args.no_arrows,
    )


if __name__ == "__main__":
    main()
