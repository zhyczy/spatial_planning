"""
3D Coordinate and Camera Pose Estimation.

Routing:
  - Multi-view (≥2 images) → MapAnything  (multi-view stereo)
  - Single-view (1 image)  → Depth Pro    (monocular metric depth)

For each input image frame, estimates:
  - Per-pixel 3D world coordinates  (H x W x 3, float32)
  - Camera extrinsic pose            (4 x 4,     float32)
  - Camera intrinsic matrix          (3 x 3,     float32)
  - Valid pixel mask                 (H x W,     bool)

Weights are loaded entirely from local storage — no network access required.

Usage
-----
from coord_esti import CoordEstimator

estimator = CoordEstimator()          # loads model once
results   = estimator.estimate(image_paths)

# results is a list (one entry per view):
# [
#   {
#     "pts3d":      np.ndarray (H, W, 3),   # world-frame 3-D coords
#     "depth":      np.ndarray (H, W),       # metric depth (z-axis)
#     "camera_pose": np.ndarray (4, 4),      # cam-to-world matrix
#     "intrinsics": np.ndarray (3, 3),       # pinhole intrinsics
#     "mask":       np.ndarray (H, W, bool), # valid (non-edge) pixels
#     "image":      np.ndarray (H, W, 3),    # original image [0,255]
#   },
#   ...
# ]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_MAP_ANYTHING_DIR = _THIS_DIR / "external" / "map-anything"
_CHECKPOINT_DIR   = _THIS_DIR / "checkpoints" / "map-anything-weights"
_DEPTH_PRO_DIR    = _THIS_DIR / "external" / "ml-depth-pro" / "src"
_DEPTH_PRO_CKPT   = _THIS_DIR / "checkpoints" / "depth_pro" / "depth_pro.pt"

# Add map-anything to the Python path so its modules can be imported.
_MAP_ANYTHING_STR = str(_MAP_ANYTHING_DIR)
if _MAP_ANYTHING_STR not in sys.path:
    sys.path.insert(0, _MAP_ANYTHING_STR)

# Add depth-pro src to the Python path.
_DEPTH_PRO_STR = str(_DEPTH_PRO_DIR)
if _DEPTH_PRO_STR not in sys.path:
    sys.path.insert(0, _DEPTH_PRO_STR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESULTS_DIR   = _THIS_DIR / "coord_results"
# Torch hub dir: contains facebookresearch_dinov2_main/ and checkpoints/dinov2_vitg14_pretrain.pth
_DINOV2_DIR = _THIS_DIR / "checkpoints" / "dinov2"

# ---------------------------------------------------------------------------
# Local-weight configuration (mirrors demo_local_weight.py)
# ---------------------------------------------------------------------------

_LOCAL_CONFIG = {
    # Hydra train config — path is relative to the map-anything root
    "path": str(_MAP_ANYTHING_DIR / "configs" / "train.yaml"),
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    # Local checkpoint and its JSON config
    "checkpoint_path": str(_CHECKPOINT_DIR / "model.safetensors"),
    "config_json_path": str(_CHECKPOINT_DIR / "config.json"),
    "strict": False,
}


class CoordEstimator:
    """Load MapAnything once and run per-frame 3-D coordinate estimation.

    Routing:
      - Multi-view (≥2 images) → MapAnything
      - Single-view (1 image)  → Depth Pro (monocular metric depth)
    """

    def __init__(
        self,
        device: str | None = None,
        memory_efficient: bool = True,
        amp_dtype: str = "bf16",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_efficient = memory_efficient
        self.amp_dtype = amp_dtype
        self._model = self._load_model()
        self._depth_pro_model = None
        self._depth_pro_transform = None

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _load_model(self):
        from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local

        torch.hub.set_dir(str(_DINOV2_DIR))
        print(f"[CoordEstimator] Loading MapAnything from local weights on {self.device} …")
        model = initialize_mapanything_local(_LOCAL_CONFIG, self.device)
        print("[CoordEstimator] Model ready.")
        return model

    def _load_depth_pro(self):
        """Lazy-load Depth Pro model (only when a single-view input is encountered)."""
        if self._depth_pro_model is not None:
            return

        import depth_pro
        from depth_pro.depth_pro import DepthProConfig

        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=str(_DEPTH_PRO_CKPT),
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        print(f"[CoordEstimator] Loading Depth Pro from {_DEPTH_PRO_CKPT} on {self.device} …")
        model, transform = depth_pro.create_model_and_transforms(config=config)
        model = model.to(self.device).eval()
        self._depth_pro_model = model
        self._depth_pro_transform = transform
        print("[CoordEstimator] Depth Pro ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        images: Union[str, List[str], List[np.ndarray]],
        save_dir: Optional[Union[str, Path]] = None,
        run_name: Optional[str] = None,
        visualize: bool = False,
    ) -> List[Dict]:
        """Estimate per-pixel 3-D coordinates and camera parameters.

        Parameters
        ----------
        images:
            Either
            * a folder path (str) — all supported images inside are used, or
            * a list of image file paths (str), or
            * a list of uint8 RGB numpy arrays (H x W x 3).
        save_dir:
            Root directory under which results are written.  Defaults to
            ``spatial_planning/coord_results/``.  Pass ``None`` to skip saving.
        run_name:
            Sub-folder name inside *save_dir*.  Defaults to the basename of the
            image folder (for folder inputs) or ``"run"`` otherwise.

        Returns
        -------
        list of dict, one per input view:
            pts3d      – (H, W, 3) float32   world-frame 3-D point per pixel
            depth      – (H, W)    float32   metric depth along z-axis
            camera_pose– (4, 4)    float32   cam-to-world extrinsic matrix
            intrinsics – (3, 3)    float32   pinhole intrinsic matrix K
            mask       – (H, W)    bool      valid (non-masked) pixels
            image      – (H, W, 3) uint8     original image (0-255)
        """
        # ---- determine if single-view or multi-view --------------------
        import PIL.Image as PILImage

        if isinstance(images, list) and len(images) == 1:
            # single image path or array
            img = images[0]
            if isinstance(img, str):
                if run_name is None:
                    run_name = Path(img).parent.name or "run"
                img = np.array(PILImage.open(img).convert("RGB"), dtype=np.uint8)
            elif run_name is None:
                run_name = "run"
            results = self._estimate_single_view(img)
        else:
            # multi-view: original MapAnything path
            results = self._estimate_multi_view(images, run_name)
            if run_name is None:
                run_name = "run"

        if not results:
            raise ValueError("No valid images found.")

        # ---- save to disk ----------------------------------------------
        if save_dir is not None:
            out_dir = save_results(results, save_dir=save_dir, run_name=run_name)
            if visualize:
                visualize_results(results, out_dir=out_dir)

        return results

    # ------------------------------------------------------------------
    # Single-view path: Depth Pro
    # ------------------------------------------------------------------

    def _estimate_single_view(self, image_np: np.ndarray) -> List[Dict]:
        """Use Depth Pro to estimate per-pixel 3-D coords for one image.

        Parameters
        ----------
        image_np : (H, W, 3) uint8 RGB array

        Returns
        -------
        List with one result dict (same schema as multi-view output).
        """
        import PIL.Image
        import depth_pro

        self._load_depth_pro()

        pil_img = PIL.Image.fromarray(image_np).convert("RGB")
        W, H = pil_img.size

        # Preprocess and run inference
        img_tensor = self._depth_pro_transform(pil_img).to(self.device)  # (3, H', W')
        with torch.no_grad():
            prediction = self._depth_pro_model.infer(img_tensor, f_px=None)

        depth = prediction["depth"]          # (H', W') tensor, metric metres
        focallength_px = prediction["focallength_px"].item()

        # Resize depth back to original image size if needed
        dH, dW = depth.shape[-2], depth.shape[-1]
        if (dH, dW) != (H, W):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth_np = depth.cpu().float().numpy()  # (H, W)

        # Build intrinsic matrix K (pinhole, principal point at image centre)
        cx, cy = W / 2.0, H / 2.0
        fx = fy = focallength_px
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        # Back-project depth to 3-D camera-frame coordinates
        us = np.arange(W, dtype=np.float32)
        vs = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)          # (H, W)

        X = (uu - cx) * depth_np / fx
        Y = (vv - cy) * depth_np / fy
        Z = depth_np
        pts3d = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

        # Camera pose: identity (camera frame = world frame for single view)
        camera_pose = np.eye(4, dtype=np.float32)

        # Valid mask: positive finite depth
        mask = np.isfinite(depth_np) & (depth_np > 0)

        return [{
            "pts3d":       pts3d.astype(np.float32),
            "depth":       depth_np.astype(np.float32),
            "camera_pose": camera_pose,
            "intrinsics":  K,
            "mask":        mask,
            "image":       image_np,
        }]

    # ------------------------------------------------------------------
    # Multi-view path: MapAnything (original logic)
    # ------------------------------------------------------------------

    def _estimate_multi_view(
        self,
        images: Union[str, List[str], List[np.ndarray]],
        run_name: Optional[str],
    ) -> List[Dict]:
        from mapanything.utils.geometry import depthmap_to_world_frame
        from mapanything.utils.image import load_images

        if isinstance(images, str):
            views = load_images(images)
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            views = load_images(images)
        else:
            views = self._views_from_arrays(images)

        if not views:
            raise ValueError("No valid images found.")

        outputs = self._model.infer(
            views,
            memory_efficient_inference=self.memory_efficient,
            minibatch_size=1,
            use_amp=True,
            amp_dtype=self.amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )

        results: List[Dict] = []
        for pred in outputs:
            depthmap_t    = pred["depth_z"][0].squeeze(-1)         # (H, W)
            intrinsics_t  = pred["intrinsics"][0]                   # (3, 3)
            camera_pose_t = pred["camera_poses"][0]                 # (4, 4)

            pts3d_t, valid_mask_t = depthmap_to_world_frame(
                depthmap_t, intrinsics_t, camera_pose_t
            )

            model_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            valid_mask  = valid_mask_t.cpu().numpy()
            combined_mask = model_mask & valid_mask

            image_np = pred["img_no_norm"][0].cpu().numpy()
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

            results.append({
                "pts3d":       pts3d_t.cpu().numpy().astype(np.float32),
                "depth":       depthmap_t.cpu().numpy().astype(np.float32),
                "camera_pose": camera_pose_t.cpu().numpy().astype(np.float32),
                "intrinsics":  intrinsics_t.cpu().numpy().astype(np.float32),
                "mask":        combined_mask,
                "image":       image_np,
            })

        return results

    # ------------------------------------------------------------------
    # Helper: build views list from raw numpy arrays
    # ------------------------------------------------------------------

    @staticmethod
    def _views_from_arrays(arrays: List[np.ndarray]) -> List[Dict]:
        """Wrap numpy RGB arrays (H×W×3 uint8) into the dict format that
        load_images / model.infer() expects."""
        import torchvision.transforms as tvf
        from mapanything.utils.image import find_closest_aspect_ratio
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        import PIL.Image

        norm_type = "dinov2"
        resolution_set = 518
        img_norm_cfg = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=img_norm_cfg.mean, std=img_norm_cfg.std),
        ])

        # Determine target resolution from average aspect ratio
        aspect_ratios = [arr.shape[1] / arr.shape[0] for arr in arrays]
        avg_ar = sum(aspect_ratios) / len(aspect_ratios)
        target_w, target_h = find_closest_aspect_ratio(avg_ar, resolution_set)

        from mapanything.utils.cropping import crop_resize_if_necessary

        views = []
        for idx, arr in enumerate(arrays):
            pil_img = PIL.Image.fromarray(arr).convert("RGB")
            pil_resized = crop_resize_if_necessary(pil_img, resolution=(target_w, target_h))[0]
            W2, H2 = pil_resized.size
            views.append({
                "img":            ImgNorm(pil_resized)[None],
                "true_shape":     np.int32([H2, W2]),
                "idx":            idx,
                "instance":       str(idx),
                "data_norm_type": [norm_type],
            })

        return views


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_results(
    results: List[Dict],
    save_dir: Union[str, Path] = _RESULTS_DIR,
    run_name: Optional[str] = "run",
) -> Path:
    """Persist estimation results to *save_dir / run_name/*.

    Layout
    ------
    coord_results/
    └── <run_name>/
        ├── view_0000/
        │   ├── pts3d.npy          # (H, W, 3) float32 — world-frame 3-D coords
        │   ├── depth.npy          # (H, W)    float32 — metric depth
        │   ├── camera_pose.npy    # (4, 4)    float32 — cam-to-world matrix
        │   ├── intrinsics.npy     # (3, 3)    float32 — pinhole K matrix
        │   ├── mask.npy           # (H, W)    bool    — valid pixel mask
        │   └── image.png          # original image
        ├── view_0001/
        │   └── ...
        └── cameras.json           # all poses + intrinsics in one JSON file

    Returns the path to the run directory.
    """
    import PIL.Image

    out_dir = Path(save_dir) / (run_name or "run")
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_meta = []

    for idx, r in enumerate(results):
        view_dir = out_dir / f"view_{idx:04d}"
        view_dir.mkdir(exist_ok=True)

        np.save(view_dir / "pts3d.npy",       r["pts3d"])
        np.save(view_dir / "depth.npy",        r["depth"])
        np.save(view_dir / "camera_pose.npy",  r["camera_pose"])
        np.save(view_dir / "intrinsics.npy",   r["intrinsics"])
        np.save(view_dir / "mask.npy",         r["mask"])

        PIL.Image.fromarray(r["image"]).save(view_dir / "image.png")

        cameras_meta.append({
            "view":        idx,
            "camera_pose": r["camera_pose"].tolist(),
            "intrinsics":  r["intrinsics"].tolist(),
        })

    with open(out_dir / "cameras.json", "w") as f:
        json.dump(cameras_meta, f, indent=2)

    print(f"[CoordEstimator] Saved {len(results)} view(s) to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def visualize_results(
    results: List[Dict],
    out_dir: Union[str, Path],
    max_pts: int = 300_000,
    axis_len: float = 0.15,
) -> Path:
    """生成可交互的 3-D 可视化 HTML，保存到 *out_dir/visualization.html*。

    在浏览器中打开即可自由旋转、缩放、平移。内容：
    - 彩色点云（所有视角合并，随机下采样到 max_pts 个点）
    - 每个相机的位姿坐标轴（红 X / 绿 Y / 蓝 Z）+ 编号标注
    - 世界坐标系原点轴（更长、更粗）

    Parameters
    ----------
    results  : estimate() 的返回值
    out_dir  : 输出目录（visualization.html 存在此处）
    max_pts  : 点云最多显示的点数
    axis_len : 相机坐标轴箭头长度，单位与场景一致
    """
    import plotly.graph_objects as go

    out_dir = Path(out_dir)
    traces = []

    # ---- 彩色点云 ----------------------------------------------------------
    all_pts, all_cols = [], []
    for r in results:
        mask = r["mask"]
        all_pts.append(r["pts3d"][mask])
        all_cols.append(r["image"][mask])

    pts_np  = np.concatenate(all_pts,  axis=0).astype(np.float32)
    cols_np = np.concatenate(all_cols, axis=0)          # uint8 RGB

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

    # ---- 相机坐标轴 --------------------------------------------------------
    axis_colors = {"X": "red", "Y": "green", "Z": "blue"}
    _added_cam_legend = {k: False for k in axis_colors}

    for v_idx, r in enumerate(results):
        pose   = r["camera_pose"]       # (4, 4) cam-to-world
        origin = pose[:3, 3]

        for i, (lbl, col) in enumerate(axis_colors.items()):
            tip = origin + pose[:3, i] * axis_len
            show = not _added_cam_legend[lbl]
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
            _added_cam_legend[lbl] = True

        # 相机编号文字
        traces.append(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            mode="text",
            text=[f"cam{v_idx}"],
            textfont=dict(size=11, color="black"),
            showlegend=False,
        ))

    # ---- 世界坐标系原点轴 --------------------------------------------------
    world_len = axis_len * 3
    world_labels = {"World-X": "red", "World-Y": "green", "World-Z": "blue"}
    for i, (lbl, col) in enumerate(world_labels.items()):
        tip = np.zeros(3); tip[i] = world_len
        traces.append(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines",
            line=dict(color=col, width=8),
            name=lbl,
        ))

    # 原点标记
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=6, color="black"),
        name="Origin",
    ))

    # ---- 布局 --------------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3-D Reconstruction  ({len(results)} views, {len(pts_np):,} pts)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",          # 等比例坐标轴
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    save_path = out_dir / "visualization.html"
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    print(f"[CoordEstimator] Visualization saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def estimate_3d(
    images: Union[str, List[str], List[np.ndarray]],
    device: str | None = None,
    save_dir: Optional[Union[str, Path]] = _RESULTS_DIR,
    run_name: Optional[str] = None,
) -> List[Dict]:
    """One-shot helper: load model + run estimation in one call.

    For repeated calls, instantiate ``CoordEstimator`` directly to avoid
    reloading the model each time.
    """
    estimator = CoordEstimator(device=device)
    return estimator.estimate(images, save_dir=save_dir, run_name=run_name)
