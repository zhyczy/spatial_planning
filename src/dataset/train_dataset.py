import os
import sys
import logging
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch
from src.data_process.reconstruct_3d import detect_dataset, _scene_id_from_path

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DDP helpers ───────────────────────────────────────────────────────────────

local_rank: int = 0
world_size: int = 1


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ── paths ─────────────────────────────────────────────────────────────────────
SPAR_ROOT = os.path.join(
    _ROOT, "datasets/train/SPAR_7M/spar"
)
RECONSTRUCT_DIR = os.path.join(SPAR_ROOT, "reconstruct")
POS3D_DIR       = os.path.join(SPAR_ROOT, "3D_pos")
POSE_TOKEN  = "<pose>"
COORD_TOKEN = "<coord>"


# ── 3D coordinate helper ──────────────────────────────────────────────────────

def resize_xyz(
    xyz:      np.ndarray,
    target_h: int,
    target_w: int,
    valid:    np.ndarray | None = None,
) -> torch.Tensor:
    """
    Compute the mean 3D position of all valid pixels within each LLM patch.

    Args:
        xyz:      (H, W, 3) float  — per-pixel XYZ map (any float dtype)
        target_h: output patch rows (= image_grid_thw[1] // spatial_merge_size)
        target_w: output patch cols (= image_grid_thw[2] // spatial_merge_size)
        valid:    (H, W) bool mask — True where XYZ is reliable;
                  if None all pixels are treated as valid

    Returns:
        (target_h, target_w, 3) float32 tensor; patches with no valid pixels
        are set to zero.
    """
    H, W = xyz.shape[:2]
    xyz_f = xyz.astype(np.float32)                     # (H, W, 3)

    if valid is None:
        valid = np.ones((H, W), dtype=bool)

    # Stride per LLM patch (integer division; crop any remainder)
    stride_h = H // target_h
    stride_w = W // target_w
    H_crop   = target_h * stride_h
    W_crop   = target_w * stride_w
    xyz_f    = xyz_f[:H_crop, :W_crop]                 # (H_crop, W_crop, 3)
    valid    = valid[:H_crop, :W_crop].astype(np.float32)  # (H_crop, W_crop)

    # Reshape into patch blocks
    # (target_h, stride_h, target_w, stride_w, 3)
    xyz_blocks   = xyz_f.reshape(target_h, stride_h, target_w, stride_w, 3)
    valid_blocks = valid.reshape(target_h, stride_h, target_w, stride_w)

    # Masked sum → mean over the stride_h × stride_w pixel block per patch
    xyz_sum   = (xyz_blocks * valid_blocks[..., None]).sum(axis=(1, 3))  # (th, tw, 3)
    valid_cnt = valid_blocks.sum(axis=(1, 3))                            # (th, tw)

    denom    = np.maximum(valid_cnt, 1)[..., None]     # avoid ÷0
    xyz_mean = xyz_sum / denom                         # (target_h, target_w, 3)
    xyz_mean[valid_cnt == 0] = 0.0                     # patches with no valid px

    return torch.from_numpy(xyz_mean)                  # (target_h, target_w, 3)


def xyz_to_polar(xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian (x, y, z) → spherical (r, θ, α).

        r = sqrt(x²+y²+z²)          — radial distance  ∈ [0, ∞)
        θ = atan2(y, x)              — azimuth          ∈ [-π, π]
        α = atan2(sqrt(x²+y²), z)   — inclination      ∈ [0, π]

    Works on tensors of any shape (..., 3); returns same shape with the
    last dimension replaced by (r, θ, α).
    Patches with zero xyz (no valid pixels) stay at (0, 0, 0).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    zero_mask = (r == 0)
    r_safe = r.clamp(min=1e-8)
    theta = torch.atan2(y, x)
    alpha = torch.atan2(torch.sqrt(x**2 + y**2), z)
    # restore exact zero for invalid patches
    theta = torch.where(zero_mask, torch.zeros_like(theta), theta)
    alpha = torch.where(zero_mask, torch.zeros_like(alpha), alpha)
    return torch.stack([r, theta, alpha], dim=-1)


# ── MindCube training dataset ─────────────────────────────────────────────────

class MindCube_Train_Dataset(Dataset):
    """
    Training dataset for MindCube JSONL + 3d_results structure.

    Layout:
        <results_dir>/<id>/
            cameras.json          — list of {view, camera_pose (4×4), intrinsics}
            view_0000/
                image.png         — RGB image
                pts3d.npy         — (H, W, 3) per-pixel 3D coords
                mask.npy          — (H, W) bool valid mask
            view_0001/ ...

    Each JSONL line: { id, question, gt_answer, images, ... }

    Builds the same <pose>-token prompt as Train_Dataset (SPAR).
    relative_transform[i,j] = inv(pose_j) @ pose_i  (camera-to-world poses).
    """

    def __init__(
        self,
        jsonl_path:         str,
        results_dir:        str,
        processor,
        pose_token_id:      int,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        max_samples:        int | None = None,
        plus:               bool = False,
        no_pose:            bool = False,
    ):
        import json
        raw = []
        with open(jsonl_path) as fh:
            for line in fh:
                if line.strip():
                    raw.append(json.loads(line))

        self.samples = []
        for entry in raw:
            eid = entry.get("id", "")
            sample_dir = os.path.join(results_dir, eid)
            if not os.path.isdir(sample_dir):
                continue
            self.samples.append((entry, sample_dir))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.pose_token_id      = pose_token_id
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.plus               = plus
        self.no_pose            = no_pose
        self.log = log
        log.info(
            f"MindCube_Train_Dataset: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total) from {jsonl_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        # ── load images and per-pixel xyz ─────────────────────────────────────
        view_dirs = sorted(
            d for d in os.listdir(sample_dir) if d.startswith("view_")
        )
        images, xyz_raw_list, mask_raw_list = [], [], []
        for vd in view_dirs[: self.max_images]:
            img_path = os.path.join(sample_dir, vd, "image.png")
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except (FileNotFoundError, OSError):
                break
            pts3d_path = os.path.join(sample_dir, vd, "pts3d.npy")
            mask_path  = os.path.join(sample_dir, vd, "mask.npy")
            xyz_raw_list.append(
                np.load(pts3d_path).astype(np.float32)
                if os.path.exists(pts3d_path) else None
            )
            mask_raw_list.append(
                np.load(mask_path) if os.path.exists(mask_path) else None
            )

        N = len(images)
        if N < 2:
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has only {N} "
                f"valid images; need ≥ 2."
            )

        # ── load camera poses and compute relative transforms ─────────────────
        if not self.no_pose:
            poses = []
            for vd in view_dirs[:N]:
                cp_path = os.path.join(sample_dir, vd, "camera_pose.npy")
                poses.append(np.load(cp_path).astype(np.float64))  # (4, 4)

            pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
            rel_list = []
            for i, j in pairs:
                # T_{i→j}: transforms 3D point from camera-i to camera-j frame
                # = inv(pose_j) @ pose_i   (pose = camera-to-world)
                T = np.linalg.inv(poses[j]) @ poses[i]
                rel_list.append(T)
            gt_transforms = torch.tensor(
                np.stack(rel_list, axis=0), dtype=torch.float32
            )  # (N*(N-1), 4, 4)
        else:
            pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
            gt_transforms = None

        # ── build prompt ──────────────────────────────────────────────────────
        content: list = [{"type": "image", "image": img} for img in images]

        pose_sentences = []
        if not self.no_pose:
            pose_sentences = [
                f"The camera pose of image {j + 1} relative to image {i + 1} is "
                f"{POSE_TOKEN}."
                for (i, j) in pairs
            ]
            content.append({"type": "text", "text": " ".join(pose_sentences)})

        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        labels = None
        if (self.plus or self.no_pose) and _question and _answer:
            qa_content = list(content)
            if self.no_pose:
                qa_content.append({"type": "text", "text": _question})
            else:
                qa_content[-1] = {
                    "type": "text",
                    "text": " ".join(pose_sentences) + " " + _question,
                }
            text_full = self.processor.apply_chat_template(
                [{"role": "user",      "content": qa_content},
                 {"role": "assistant", "content": _answer}],
                tokenize=False, add_generation_prompt=False,
            )
            proc_out = self.processor(
                text=[text_full], images=images,
                return_tensors="pt", padding=False,
            )
            suffix_ids = self.processor.tokenizer(
                _answer + "<|im_end|>\n", add_special_tokens=False
            )["input_ids"]
            labels = proc_out["input_ids"].clone()
            labels[0, :-len(suffix_ids)] = -100
        else:
            messages = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            proc_out = self.processor(
                text=[prompt_text], images=images,
                return_tensors="pt", padding=False,
            )

        # ── 3D position maps (pts3d → patch-level xyz) ────────────────────────
        image_xyz = None
        try:
            thw_all = proc_out["image_grid_thw"]  # (N, 3)
            sms     = self.spatial_merge_size
            xyz_list = []
            for k in range(N):
                xyz_raw  = xyz_raw_list[k]
                mask_raw = mask_raw_list[k]
                thw_k    = thw_all[k]
                llm_h    = int(thw_k[1]) // sms
                llm_w    = int(thw_k[2]) // sms
                if xyz_raw is not None:
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=mask_raw))
                else:
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
            image_xyz = xyz_list
        except Exception as exc:
            self.log.debug(f"pts3d load failed for {sample_dir}: {exc}")
            image_xyz = None

        return {
            **proc_out,
            "gt_transforms": gt_transforms,
            "image_xyz":     image_xyz,
            "labels":        labels,
        }


class MindCube_Train_Dataset_Coord(Dataset):
    """
    MindCube training dataset with coordinate prediction support.

    Extends MindCube_Train_Dataset by:
      - Inserting <coord> tokens (one per LLM patch per image) into the prompt
      - Computing both patch-level image_xyz (for 4D M-RoPE) and
        sub-pixel image_xyz_hires (for PixelShuffle coord loss)
      - Always including QA supervision (like plus mode)
    """

    def __init__(
        self,
        jsonl_path:         str,
        results_dir:        str,
        processor,
        pose_token_id:      int,
        coord_token_id:     int,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        coord_upscale:      int = 4,
        max_samples:        int | None = None,
        no_cam:             bool = False,
    ):
        import json
        raw = []
        with open(jsonl_path) as fh:
            for line in fh:
                if line.strip():
                    raw.append(json.loads(line))

        self.samples = []
        for entry in raw:
            eid = entry.get("id", "")
            sample_dir = os.path.join(results_dir, eid)
            if not os.path.isdir(sample_dir):
                continue
            self.samples.append((entry, sample_dir))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.pose_token_id      = pose_token_id
        self.coord_token_id     = coord_token_id
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.coord_upscale      = coord_upscale
        self.no_cam             = no_cam
        self.log = log
        log.info(
            f"MindCube_Train_Dataset_Coord: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total) from {jsonl_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        # ── load images and per-pixel xyz ─────────────────────────────────────
        view_dirs = sorted(
            d for d in os.listdir(sample_dir) if d.startswith("view_")
        )
        images, xyz_raw_list, mask_raw_list = [], [], []
        for vd in view_dirs[: self.max_images]:
            img_path = os.path.join(sample_dir, vd, "image.png")
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except (FileNotFoundError, OSError):
                break
            pts3d_path = os.path.join(sample_dir, vd, "pts3d.npy")
            mask_path  = os.path.join(sample_dir, vd, "mask.npy")
            xyz_raw_list.append(
                np.load(pts3d_path).astype(np.float32)
                if os.path.exists(pts3d_path) else None
            )
            mask_raw_list.append(
                np.load(mask_path) if os.path.exists(mask_path) else None
            )

        N = len(images)
        if N < 2:
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has only {N} "
                f"valid images; need >= 2."
            )

        # ── load camera poses and compute relative transforms ─────────────────
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        if not self.no_cam:
            poses = []
            for vd in view_dirs[:N]:
                cp_path = os.path.join(sample_dir, vd, "camera_pose.npy")
                poses.append(np.load(cp_path).astype(np.float64))  # (4, 4)
            rel_list = []
            for i, j in pairs:
                T = np.linalg.inv(poses[j]) @ poses[i]
                rel_list.append(T)
            gt_transforms = torch.tensor(
                np.stack(rel_list, axis=0), dtype=torch.float32
            )  # (N*(N-1), 4, 4)
        else:
            gt_transforms = None

        # ── build prompt (pose + coord + QA) ──────────────────────────────────
        content: list = [{"type": "image", "image": img} for img in images]

        pose_sentences = [] if self.no_cam else [
            f"The camera pose of image {j + 1} relative to image {i + 1} is "
            f"{POSE_TOKEN}."
            for (i, j) in pairs
        ]

        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        if not (_question and _answer):
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        # ── Probe: process without coord tokens to get image_grid_thw ─────────
        probe_text_part = (
            (" ".join(pose_sentences) + " " if pose_sentences else "")
            + _question
        )
        content_probe = list(content) + [{"type": "text", "text": probe_text_part}]
        text_probe = self.processor.apply_chat_template(
            [{"role": "user", "content": content_probe}],
            tokenize=False, add_generation_prompt=False,
        )
        proc_probe = self.processor(
            text=[text_probe], images=images,
            return_tensors="pt", padding=False,
        )
        thw_all = proc_probe["image_grid_thw"]  # (N, 3)
        sms = self.spatial_merge_size

        # ── Build coord sentences (one <coord> token per LLM patch) ───────────
        coord_sentences = []
        for k in range(N):
            llm_h = int(thw_all[k][1]) // sms
            llm_w = int(thw_all[k][2]) // sms
            n_tok = llm_h * llm_w
            coord_tokens = "".join([COORD_TOKEN] * n_tok)
            coord_sentences.append(
                f"Image {k + 1} 3D spatial coordinates: {coord_tokens}."
            )

        # ── Build final prompt (pose if not no_cam) + coord + QA ─────────────
        parts = []
        if pose_sentences:
            parts.append(" ".join(pose_sentences))
        parts.append(" ".join(coord_sentences))
        parts.append(_question)
        content.append({"type": "text", "text": " ".join(parts)})
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": _answer}],
            tokenize=False, add_generation_prompt=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        # Supervise only the answer tokens
        suffix_ids = self.processor.tokenizer(
            _answer + "<|im_end|>\n", add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

        # ── 3D position maps (pts3d -> patch-level + sub-pixel) ───────────────
        image_xyz = None
        image_xyz_hires = None
        try:
            thw_all = proc_out["image_grid_thw"]  # (N, 3)
            up      = self.coord_upscale
            xyz_list = []
            xyz_hires_list = []
            for k in range(N):
                xyz_raw  = xyz_raw_list[k]
                mask_raw = mask_raw_list[k]
                thw_k    = thw_all[k]
                llm_h    = int(thw_k[1]) // sms
                llm_w    = int(thw_k[2]) // sms
                if xyz_raw is not None:
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=mask_raw))
                    xyz_hires_list.append(resize_xyz(xyz_raw, llm_h * up, llm_w * up, valid=mask_raw))
                else:
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                    xyz_hires_list.append(torch.zeros(llm_h * up, llm_w * up, 3))
            image_xyz = xyz_list
            image_xyz_hires = xyz_hires_list
        except Exception as exc:
            self.log.debug(f"pts3d load failed for {sample_dir}: {exc}")
            image_xyz = None
            image_xyz_hires = None

        return {
            **proc_out,
            "gt_transforms":  gt_transforms,
            "image_xyz":      image_xyz,
            "image_xyz_hires": image_xyz_hires,
            "labels":         labels,
        }


class MindCube_Train_Dataset_Coord_Polar(MindCube_Train_Dataset_Coord):
    """
    Variant of MindCube_Train_Dataset_Coord where image_xyz_hires is
    converted from Cartesian (x, y, z) to spherical (r, θ, α) before
    being returned.  image_xyz (patch-level, used for 4D M-RoPE) is
    kept in Cartesian so that RoPE position encoding is unchanged.

    Use with --polar flag in train_coordinate.py.
    """

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        if batch.get("image_xyz_hires") is not None:
            batch["image_xyz_hires"] = [
                xyz_to_polar(xyz) for xyz in batch["image_xyz_hires"]
            ]
        return batch
