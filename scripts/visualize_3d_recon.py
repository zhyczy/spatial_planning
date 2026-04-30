"""
Visualize a multi-view 3D reconstruction (pts3d.npy + camera_pose.npy + image.png)
as a single self-contained interactive Plotly HTML file.

Usage:
  python visualize_3d_recon.py <scene_dir> [-o out.html] [--max_pts 30000]

scene_dir must contain:
  view_XXXX/
    image.png
    pts3d.npy       (H, W, 3) float32 — per-pixel 3D coords in world frame
    depth.npy       (H, W)
    mask.npy        (H, W) bool — valid pixels
    camera_pose.npy (4, 4) — view-to-world (or world-to-view; we display both options)
    intrinsics.npy  (3, 3)
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image


def load_view(view_dir: Path):
    pts = np.load(view_dir / "pts3d.npy")        # (H, W, 3)
    mask = np.load(view_dir / "mask.npy").astype(bool)
    pose = np.load(view_dir / "camera_pose.npy") # (4, 4)
    K = np.load(view_dir / "intrinsics.npy")
    img = np.array(Image.open(view_dir / "image.png").convert("RGB"))  # (H, W, 3) uint8
    return {"pts": pts, "mask": mask, "pose": pose, "K": K, "img": img}


def subsample(view, max_pts: int):
    H, W = view["mask"].shape
    valid = view["mask"]
    pts_flat = view["pts"].reshape(-1, 3)
    rgb_flat = view["img"].reshape(-1, 3)
    flat_mask = valid.reshape(-1)
    idx_valid = np.where(flat_mask)[0]
    if len(idx_valid) > max_pts:
        rng = np.random.default_rng(0)
        idx_valid = rng.choice(idx_valid, size=max_pts, replace=False)
    return pts_flat[idx_valid], rgb_flat[idx_valid]


def camera_frustum(pose: np.ndarray, K: np.ndarray,
                   img_size: tuple[int, int],
                   scale: float = 0.5) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return frustum vertices (in world frame) and edges as (i, j) pairs.
    pose is camera-to-world (R | t)."""
    H_img, W_img = img_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Image-plane corners in pixel coords
    corners_pix = np.array([[0, 0], [W_img, 0], [W_img, H_img], [0, H_img]])
    # Back-project to camera frame at depth = scale
    corners_cam = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_pix):
        x = (u - cx) / fx * scale
        y = (v - cy) / fy * scale
        z = scale
        corners_cam[i] = [x, y, z]
    origin_cam = np.zeros(3)
    verts_cam = np.vstack([origin_cam, corners_cam])  # (5, 3): [origin, c0, c1, c2, c3]
    # Transform to world
    R = pose[:3, :3]
    t = pose[:3, 3]
    verts_world = verts_cam @ R.T + t
    edges = [(0, 1), (0, 2), (0, 3), (0, 4),  # apex to corners
             (1, 2), (2, 3), (3, 4), (4, 1)]   # image-plane rectangle
    return verts_world, edges


def img_to_data_uri(img: np.ndarray, max_w: int = 320) -> str:
    """Encode an RGB array (H, W, 3) as a base64 data URI thumbnail."""
    pil = Image.fromarray(img)
    if pil.width > max_w:
        scale = max_w / pil.width
        pil = pil.resize((max_w, int(pil.height * scale)))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_figure(scene_dir: Path, max_pts: int) -> go.Figure:
    view_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("view_"))
    if not view_dirs:
        raise SystemExit(f"No view_* subdirectories under {scene_dir}")

    views = [load_view(vd) for vd in view_dirs]
    cmap = ["#3477eb", "#eb4934", "#34eb83", "#ebd534", "#a834eb", "#34ebe5"]

    fig = go.Figure()

    # ── Per-view point clouds (colored by RGB) ────────────────────────────────
    for i, v in enumerate(views):
        pts, rgb = subsample(v, max_pts)
        rgb_strs = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=rgb_strs, opacity=0.85),
            name=f"view_{i:04d} pts ({len(pts):,})",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra>view " + str(i) + "</extra>",
        ))

    # ── Camera frusta + origin markers ────────────────────────────────────────
    for i, v in enumerate(views):
        H_img, W_img = v["img"].shape[:2]
        verts, edges = camera_frustum(v["pose"], v["K"], (H_img, W_img), scale=0.6)
        col = cmap[i % len(cmap)]
        # frustum edges
        xe, ye, ze = [], [], []
        for a, b in edges:
            xe += [verts[a, 0], verts[b, 0], None]
            ye += [verts[a, 1], verts[b, 1], None]
            ze += [verts[a, 2], verts[b, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode="lines",
            line=dict(color=col, width=4),
            name=f"view_{i:04d} cam",
            hoverinfo="skip",
        ))
        # camera origin
        fig.add_trace(go.Scatter3d(
            x=[verts[0, 0]], y=[verts[0, 1]], z=[verts[0, 2]],
            mode="markers+text",
            marker=dict(size=6, color=col, symbol="diamond"),
            text=[f"v{i}"], textposition="top center", textfont=dict(color=col),
            name=f"view_{i:04d} origin",
            hovertemplate=f"camera {i}<br>pos=(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>",
        ))

    fig.update_layout(
        title=f"3D recon: {scene_dir.name}  ({len(views)} views, max {max_pts:,} pts/view)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0),
        height=850,
    )
    return fig, views


def write_html(fig: go.Figure, views: list, out_path: Path, scene_dir: Path):
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="plot3d")

    # Side panel: thumbnails of each view's image
    thumbs_html = ""
    for i, v in enumerate(views):
        uri = img_to_data_uri(v["img"], max_w=320)
        thumbs_html += (
            f'<div style="margin:6px;text-align:center;">'
            f'<div style="font:12px sans-serif;color:#444;">view_{i:04d}</div>'
            f'<img src="{uri}" style="max-width:320px;border:1px solid #ccc;" />'
            f"</div>"
        )

    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8" />
<title>3D recon — {scene_dir.name}</title>
<style>
  body {{ margin:0; font-family:sans-serif; background:#f5f5f5; }}
  .container {{ display:flex; flex-direction:row; }}
  .left  {{ flex:3; }}
  .right {{ flex:1; padding:12px; background:#fff; border-left:1px solid #ddd;
            max-width:360px; overflow-y:auto; }}
  h1 {{ font-size:16px; margin:8px 12px; }}
</style>
</head><body>
<h1>3D reconstruction · {scene_dir.parent.name} / {scene_dir.name}</h1>
<div class="container">
  <div class="left">{plot_html}</div>
  <div class="right">
    <h3 style="margin:6px 0;font-size:13px;">Source images</h3>
    {thumbs_html}
    <p style="font:11px sans-serif;color:#666;">
      Drag to rotate · scroll to zoom · double-click to reset.<br>
      Coordinates are in the first camera's frame.
    </p>
  </div>
</div>
</body></html>"""
    out_path.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path,
                    help="Path to scene directory containing view_XXXX/ subdirs")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output HTML file (default: <scene_dir>/recon_viz.html)")
    ap.add_argument("--max_pts", type=int, default=30000,
                    help="Max points per view (subsampled). Default 30k.")
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_path = args.out or scene_dir / "recon_viz.html"
    out_path = Path(out_path).resolve()

    fig, views = make_figure(scene_dir, args.max_pts)
    write_html(fig, views, out_path, scene_dir)
    print(f"Wrote {out_path}")
    print(f"  {len(views)} views, {args.max_pts:,} pts/view (cap), file size: "
          f"{out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
