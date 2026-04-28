import os
import re
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
    Convert Cartesian (x, y, z) → log-spherical (log r, θ, α).

        r  = sqrt(x²+y²+z²)           — radial distance  ∈ [0, ∞)
        log r                          — scale-invariant radial channel
        θ  = atan2(y, x)               — azimuth          ∈ [-π, π]
        α  = atan2(sqrt(x²+y²), z)    — inclination      ∈ [0, π]

    Why log r (not raw r):
      RoPE scores depend on (pos_i − pos_j). Using log r makes the radial
      difference log(r_i / r_j), which is invariant to a global scene scaling
      r → k·r. Raw r fails this (difference scales linearly with k), causing
      the same model to see very different attention patterns on MindCube
      (r ~ O(1)) vs SAT (r ~ O(100)). θ and α are angular and already
      scale-invariant.

    Works on tensors of any shape (..., 3); returns same shape with the
    last dimension replaced by (log r, θ, α).
    Patches with zero xyz (no valid pixels) stay at (0, 0, 0).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    zero_mask = (r == 0)
    r_safe = r.clamp(min=1e-8)
    log_r = torch.log(r_safe)
    theta = torch.atan2(y, x)
    alpha = torch.atan2(torch.sqrt(x**2 + y**2), z)
    # restore exact zero for invalid patches
    log_r = torch.where(zero_mask, torch.zeros_like(log_r), log_r)
    theta = torch.where(zero_mask, torch.zeros_like(theta), theta)
    alpha = torch.where(zero_mask, torch.zeros_like(alpha), alpha)
    return torch.stack([log_r, theta, alpha], dim=-1)


# ── MindCube training dataset ─────────────────────────────────────────────────

class MindCube_Train_Dataset(Dataset):
    """
    Training dataset for MindCube JSONL + 3d_results structure (LM-only, no pose prediction).

    Layout:
        <results_dir>/<id>/
            view_0000/
                image.png         — RGB image
                pts3d.npy         — (H, W, 3) per-pixel 3D coords
                mask.npy          — (H, W) bool valid mask
            view_0001/ ...

    Each JSONL line: { id, question, gt_answer, images, ... }

    Prompt: images only.
    Labels: always generated when QA is available (plus reserved for future use).
    """

    def __init__(
        self,
        jsonl_path:         str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        max_samples:        int | None = None,
        plus:               bool = False,
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
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.plus               = plus
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

        # ── build prompt (images only, no pose sentences) ─────────────────────
        content: list = [{"type": "image", "image": img} for img in images]

        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        labels = None
        if _question and _answer:
            qa_content = list(content)
            qa_content.append({"type": "text", "text": _question})
            # Wrap the bare letter in <answer>X</answer> so the supervised
            # assistant turn matches evaluation.py's regex parser. Pass
            # `enable_thinking=False` so the chat template auto-fills an
            # empty `<think></think>` block — without this, the deploy-time
            # prompt ends mid-`<think>` and the model produces reasoning
            # instead of the answer. See
            # md/bug_fix/train_eval_paradigm_mismatch.md §6.
            from .answer_format import format_answer, IM_END_NEWLINE
            formatted_answer = format_answer(_answer)
            text_full = self.processor.apply_chat_template(
                [{"role": "user",      "content": qa_content},
                 {"role": "assistant", "content": formatted_answer}],
                tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            proc_out = self.processor(
                text=[text_full], images=images,
                return_tensors="pt", padding=False,
            )
            # Loss mask covers the full `<answer>X</answer><|im_end|>\n`
            # suffix (8 tokens for Qwen3.5-VL), so the model gets gradient on
            # the tags too — not just the bare letter. This forces the
            # assistant turn to start with `<answer>` at deploy.
            suffix_ids = self.processor.tokenizer(
                formatted_answer + IM_END_NEWLINE, add_special_tokens=False
            )["input_ids"]
            labels = proc_out["input_ids"].clone()
            labels[0, :-len(suffix_ids)] = -100
        else:
            messages = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
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
            "image_xyz": image_xyz,
            "labels":    labels,
        }


class MindCube_Train_Dataset_Coord(Dataset):
    """
    MindCube training dataset with coordinate prediction support.

    Extends MindCube_Train_Dataset by:
      - Computing both patch-level image_xyz (for 4D M-RoPE) and
        sub-pixel image_xyz_hires (for PixelShuffle coord loss)
      - Always including QA supervision (like plus mode)

    The coord head reads LM hidden states at the <|image_pad|> vision-token
    positions directly; no dedicated per-patch text token is inserted.
    """

    def __init__(
        self,
        jsonl_path:         str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        coord_upscale:      int = 4,
        max_samples:        int | None = None,
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
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.coord_upscale      = coord_upscale
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

        # ── build prompt (images + QA) ────────────────────────────────────────
        content: list = [{"type": "image", "image": img} for img in images]

        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        if not (_question and _answer):
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        content.append({"type": "text", "text": _question})
        # Wrap the bare letter in <answer>X</answer> + disable thinking
        # mode so the deploy-time chat template ends with `</think>\n\n`
        # (= start of the supervised suffix), not mid-`<think>`. See
        # md/bug_fix/train_eval_paradigm_mismatch.md §6.
        from .answer_format import format_answer, IM_END_NEWLINE
        formatted_answer = format_answer(_answer)
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": formatted_answer}],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        # Supervise the full `<answer>X</answer><|im_end|>\n` suffix (tags
        # included), so the model learns to start the assistant turn with
        # the opening tag.
        suffix_ids = self.processor.tokenizer(
            formatted_answer + IM_END_NEWLINE, add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

        # ── 3D position maps (pts3d -> patch-level + sub-pixel) ───────────────
        image_xyz = None
        image_xyz_hires = None
        try:
            thw_all = proc_out["image_grid_thw"]  # (N, 3)
            sms     = self.spatial_merge_size
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
            "image_xyz":      image_xyz,
            "image_xyz_hires": image_xyz_hires,
            "labels":         labels,
        }


# ── Bucket classification + R_gt extraction ─────────────────────────────────
# Each extractor returns (bucket_tag, R_gt: np.ndarray (3,3) | None). R_gt is
# the raw pose-derived world→view rotation — it is NOT quantized to R_bins on
# the dataset side. Downstream, the training loop consumes R_gt directly via
# the SO(3) geodesic metric, so anchor quantization happens only implicitly
# through the softmax-over-θ soft prior.
#
# Buckets (see reward-shaping design):
#   'A' motion_query           — "in which direction did I move from the
#                                first view to the second view?"  Reasoning
#                                frame is view 0, so identity prior (R_gt = I).
#   'B' rotation_described     — title regex, 90°-multiple setup (pose-derived)
#   'C' other / unknown        — no rotation supervision (R_gt = None)
#                                Remaining C = E-pos-obj ("positioned where
#                                X is"): anchors to an object, no pose-
#                                derivable frame.
#   'D' viewpoint_anchor       — "from the viewpoint (presented) in image N"
#                                (pose-derived: anchor(view N))
#   'E' multi_view_scene       — "standing at the same spot … as shown in
#                                image N" without turn verb (pose-derived:
#                                anchor(view N)).
#   'H' hypothetical-action    — anchor(image N) ∘ ±90° yaw (pose + regex)
_BUCKET_B_TYPES = {
    "four_view", "three_view",
    "two_view_clockwise", "two_view_counterclockwise", "two_view_opposite",
}
_RE_IMG_REF = re.compile(
    r"same direction as shown in image\s+(\d+)", re.IGNORECASE
)
# Bucket H (hypothetical-action): "... as shown in image N, then I turn
# {left|right} and {go|move|walk} ...".  The explicit image-anchor + turn
# direction lets us derive R_gt = (C2W[0]ᵀ @ C2W[N]) @ R_bins[k_turn].
# The anchor half stays continuous (pose-derived); only the ±90° yaw uses
# R_bins[5/6], because the turn direction is genuinely a discrete right-
# angle in the dataset.
_RE_TURN = re.compile(
    r"(?:then\s+i|,\s*i)\s+turn(?:ed)?\s+(left|right)",
    re.IGNORECASE,
)
# Empirically derived (see docs): in MindCube 3d_results' camera convention,
# R_bins[6] ≈ Ry(+90°) = local yaw right; R_bins[5] ≈ Ry(-90°) = local yaw
# left. Verified on 50 two_view_clockwise / 50 two_view_counterclockwise
# samples — majority vote matches these indices.
# R_bins[16] = R_bins[5] @ R_bins[5] = R_bins[6] @ R_bins[6] = diag(-1, 1, -1)
# = Ry(180°) — the "turn around" action. Identity computed at module load
# (see R_bins verification in comments at _extract_kgt_bucket_b).
_K_TURN_RIGHT = 6
_K_TURN_LEFT  = 5
_K_TURN_180   = 16

# Bucket B turn extractor — captures the optional post-anchor rotation in
# the three B-style phrasings found in MindCube:
#   "turn 90 degrees to the left"   → k = _K_TURN_LEFT
#   "turn 90 degrees to the right"  → k = _K_TURN_RIGHT
#   "turn 180 degrees around"       → k = _K_TURN_180
# Only applied to the question *tail* after the image-anchor, so the
# scene-description intro (e.g. "Image 2 was taken after turning the
# camera 90 degrees to the right from ...") cannot produce a false match.
# Coverage verified on the MindCube training set: 864 / 864 B samples
# (756 with turn, 108 anchor-only) all classified exactly.
_RE_TURN_B = re.compile(
    r"turn\s+(90|180)\s+degrees?\s+(?:to\s+the\s+)?(left|right|around)",
    re.IGNORECASE,
)

# Bucket D (viewpoint_anchor): natural multi-view scenes where the question
# instructs the model to reason from a specific image's viewpoint — e.g.
# "From the viewpoint presented in image 2, what is to the left of X?".
# R_gt = C2W[0]ᵀ @ C2W[N]; same pose derivation as B but without
# the type-field filter (B's types are the synthetic 90°-turn layouts).
_RE_VIEWPOINT_IMG = re.compile(
    r"from\s+the\s+viewpoint(?:\s+presented)?\s+in\s+image\s+(\d+)",
    re.IGNORECASE,
)

# Bucket A (motion_query): "in which direction did I move from the first
# view to the second view?" — 4-option MCQ over translation directions.
# The question is framed from view 0's camera frame, so the rotation
# anchor prior is identity (R_gt = I). The GT direction answer itself
# is not used for rotation shaping.
_RE_DIR_MOVE = re.compile(
    r"in which direction did i move", re.IGNORECASE
)

# Cached R_bins (24, 3, 3) — matches rotation_rope_llm._build_chiral_cube_group
# (sorted descending by trace, R_bins[0] = I). Computed lazily on first use so
# the dataset module stays importable without pulling the model dependency.
_R_BINS_CACHE: torch.Tensor | None = None


def _get_r_bins() -> torch.Tensor:
    global _R_BINS_CACHE
    if _R_BINS_CACHE is None:
        from src.models.rotation_rope_llm import _build_chiral_cube_group
        _R_BINS_CACHE = _build_chiral_cube_group().float()     # (24, 3, 3)
    return _R_BINS_CACHE


def _load_relative_rotation(
    sample_dir: str,
    n:          int,
) -> np.ndarray | None:
    """Return C2W[0]ᵀ @ C2W[N] as a raw (3,3) float32 matrix, or None if any
    pose file is missing / unreadable. n=0 returns identity by construction.
    """
    if n == 0:
        return np.eye(3, dtype=np.float32)
    view_dirs = sorted(
        d for d in os.listdir(sample_dir) if d.startswith("view_")
    )
    if n < 0 or n >= len(view_dirs):
        return None
    p0 = os.path.join(sample_dir, view_dirs[0], "camera_pose.npy")
    pN = os.path.join(sample_dir, view_dirs[n], "camera_pose.npy")
    if not (os.path.exists(p0) and os.path.exists(pN)):
        return None
    try:
        C0 = np.load(p0).astype(np.float32)[:3, :3]
        CN = np.load(pN).astype(np.float32)[:3, :3]
        return (C0.T @ CN).astype(np.float32)
    except Exception:
        return None


def _extract_kgt_bucket_b(
    entry:      dict,
    sample_dir: str,
) -> tuple[str, np.ndarray | None]:
    """Return ('B', R_gt (3,3)) for Bucket B; ('C', None) otherwise.

    Bucket B uses entry['type'] to confirm the layout (synthetic 90°-multiple
    turns) and regex '(same direction as shown in) image N' to locate the
    anchor view. If the question *tail* (after the image-ref) also specifies
    a discrete action "turn 90 degrees {left|right}" / "turn 180 degrees
    around", that rotation is composed onto the anchor:

        R_gt = (C2W[0]ᵀ @ C2W[N]) @ R_turn,  R_turn ∈ {I, R_bins[5/6/16]}.

    The anchor half stays continuous (pose-derived); only the multiple-of-
    90° action uses R_bins, because those are the exact right angles of
    MindCube's synthetic layout. 87.5% of B samples carry such an action —
    ignoring it would leave R_gt systematically off by 90°–180°.
    """
    if entry.get("type") not in _BUCKET_B_TYPES:
        return "C", None
    q = entry.get("question", "")
    m = _RE_IMG_REF.search(q)
    if m is None:
        return "C", None
    n = int(m.group(1)) - 1                                   # 1-indexed → 0-idx
    R_anchor = _load_relative_rotation(sample_dir, n)
    if R_anchor is None:
        return "C", None

    # Only scan the tail — the intro sentence often says "Image 2 was
    # taken after turning the camera 90 degrees to the right ...", which
    # refers to a scene-layout fact, not the user's reasoning action.
    m_turn = _RE_TURN_B.search(q[m.end():])
    if m_turn is None:
        return "B", R_anchor.astype(np.float32)

    deg, direction = m_turn.group(1), m_turn.group(2).lower()
    if deg == "180" or direction == "around":
        k_turn = _K_TURN_180
    elif direction == "right":
        k_turn = _K_TURN_RIGHT
    else:                                                     # "left"
        k_turn = _K_TURN_LEFT
    R_turn = _get_r_bins().numpy()[k_turn]
    R_gt   = (R_anchor @ R_turn).astype(np.float32)
    return "B", R_gt


def _extract_kgt_bucket_h(
    entry:      dict,
    sample_dir: str,
) -> tuple[str, np.ndarray] | None:
    """Return ('H', R_gt (3,3)) for hypothetical-action samples that specify
    both an image anchor N and an explicit turn direction; None if the pattern
    does not match so the caller can fall through.

    Derivation: R_gt = R_anchor @ R_bins[k_turn], where
      R_anchor = C2W[0]ᵀ @ C2W[N]   (raw, unquantized)
      R_turn   = R_bins[6] (right) or R_bins[5] (left)
    The anchor half stays continuous; only the ±90° yaw is discrete because
    the turn direction in the dataset is a genuine right angle.
    """
    if entry.get("type") in _BUCKET_B_TYPES:
        return None
    q = entry.get("question", "")
    m_img  = _RE_IMG_REF.search(q)
    m_turn = _RE_TURN.search(q)
    if m_img is None or m_turn is None:
        return None
    n = int(m_img.group(1)) - 1
    direction = m_turn.group(1).lower()
    k_turn = _K_TURN_RIGHT if direction == "right" else _K_TURN_LEFT

    R_anchor = _load_relative_rotation(sample_dir, n)
    if R_anchor is None:
        return None
    R_turn = _get_r_bins().numpy()[k_turn]
    R_gt   = (R_anchor @ R_turn).astype(np.float32)
    return "H", R_gt


def _extract_kgt_bucket_d(
    entry:      dict,
    sample_dir: str,
) -> tuple[str, np.ndarray] | None:
    """Return ('D', R_gt (3,3)) for viewpoint-anchor samples ("From the
    viewpoint (presented) in image N, …"); None if the pattern does not
    match so the caller can fall through.

    R_gt = C2W[0]ᵀ @ C2W[N] — the raw rotation that takes the model from the
    current (view 0) frame into view N's camera frame.
    """
    q = entry.get("question", "")
    m = _RE_VIEWPOINT_IMG.search(q)
    if m is None:
        return None
    n = int(m.group(1)) - 1                                   # 1-indexed → 0-idx
    R = _load_relative_rotation(sample_dir, n)
    if R is None:
        return None
    return "D", R


def _extract_kgt_bucket_a(
    entry:      dict,
    sample_dir: str,
) -> tuple[str, np.ndarray] | None:
    """Return ('A', I) for motion_query samples ("in which direction did I
    move from the first view to the second view?"); None otherwise.

    R_gt = identity: the question is framed from view 0's frame, so the
    rotation-side prior is identity. The translation-direction MCQ answer
    is supervised via the ordinary answer-correctness reward, not by
    rotation shaping.

    sample_dir is accepted for signature consistency but unused here (no
    pose read needed for the identity anchor).
    """
    if _RE_DIR_MOVE.search(entry.get("question", "")) is None:
        return None
    return "A", np.eye(3, dtype=np.float32)


def _extract_kgt_bucket_e(
    entry:      dict,
    sample_dir: str,
) -> tuple[str, np.ndarray] | None:
    """Return ('E', R_gt (3,3)) for multi-view-scene samples that fix an
    image anchor without a turn verb — pattern "standing at the same spot
    and facing the same direction as shown in image N"; None otherwise.

    Must be called AFTER B and H in the chain:
      • B catches the same image-ref regex when type ∈ _BUCKET_B_TYPES.
      • H catches image-ref + turn-verb combos.
    So by the time this runs, any remaining image-ref match is E (scene is
    anchored to image N without hypothetical rotation). "positioned where X"
    samples (571 in training set) have no image anchor and fall through.

    R_gt = C2W[0]ᵀ @ C2W[N] — same pose derivation as D.
    """
    q = entry.get("question", "")
    m = _RE_IMG_REF.search(q)
    if m is None:
        return None
    n = int(m.group(1)) - 1                                   # 1-indexed → 0-idx
    R = _load_relative_rotation(sample_dir, n)
    if R is None:
        return None
    return "E", R


class MindCube_Train_Dataset_Rotation(MindCube_Train_Dataset_Coord):
    """
    Extends MindCube_Train_Dataset_Coord with camera-pose data.

    Additional items returned in each batch:
      cam_pos_frame0  (3,)    float32 — first-frame camera position in world
      gt_rotation     (3, 3)  float32 — world-to-camera rotation of frame 0
                              i.e. R_w2c = poses[0][:3, :3].T
                              (inverse = transpose for orthogonal matrices)

    If camera_pose.npy is missing for frame 0, cam_pos_frame0 is zeros and
    gt_rotation is the identity matrix.

    Note: RotationRoPEModel does not consume gt_rotation (no rot_loss);
    these fields are preserved for optional diagnostic use.
    """

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        entry, sample_dir = self.samples[idx]

        view_dirs = sorted(
            d for d in os.listdir(sample_dir) if d.startswith("view_")
        )

        # Load first-frame camera pose (camera-to-world, 4×4)
        cam_pos_frame0 = torch.zeros(3, dtype=torch.float32)
        gt_rotation    = torch.eye(3,  dtype=torch.float32)

        first_view = view_dirs[0] if view_dirs else None
        if first_view is not None:
            cp_path = os.path.join(sample_dir, first_view, "camera_pose.npy")
            if os.path.exists(cp_path):
                try:
                    pose = np.load(cp_path).astype(np.float32)  # (4, 4) C2W
                    cam_pos_frame0 = torch.tensor(
                        pose[:3, 3], dtype=torch.float32
                    )
                    # World-to-camera rotation:
                    #   R_w2c = inv(C2W)[:3, :3] = C2W[:3, :3].T
                    #   (rotation matrices: inv = transpose)
                    gt_rotation = torch.tensor(
                        pose[:3, :3].T, dtype=torch.float32
                    )
                except Exception as exc:
                    self.log.debug(
                        f"camera_pose load failed for {sample_dir}/{first_view}: {exc}"
                    )

        batch["cam_pos_frame0"] = cam_pos_frame0
        batch["gt_rotation"]    = gt_rotation

        # ── Reward-shaping supervision: bucket + raw R_gt rotation ───────
        # Priority: B (title regex) > H (anchor + turn) > D (viewpoint anchor)
        #         > E (same-spot anchor) > A (motion_query) > default ('C', None).
        # R_gt is the raw pose-derived (3,3) rotation — downstream consumes it
        # via SO(3) geodesic θ(R_k, R_gt), so NO quantization to R_bins here.
        bucket, R_gt = _extract_kgt_bucket_b(entry, sample_dir)
        if bucket != "B":
            for extractor in (
                _extract_kgt_bucket_h,
                _extract_kgt_bucket_d,
                _extract_kgt_bucket_e,
                _extract_kgt_bucket_a,
            ):
                res = extractor(entry, sample_dir)
                if res is not None:
                    bucket, R_gt = res
                    break
        batch["bucket"] = bucket                          # 'A'/'B'/'C'/'D'/'E'/'H'
        batch["has_gt"] = bool(R_gt is not None)
        batch["R_gt"]   = torch.tensor(
            R_gt if R_gt is not None else np.eye(3, dtype=np.float32),
            dtype=torch.float32,
        )
        return batch


_SAT_LETTERS = "ABCDEFGHIJ"


def _balance_sat_samples_by_qtype(
    samples: list,
    target_size: int,
    seed: int = 0,
    log=None,
) -> list:
    """Down-sample a SAT (entry, sample_dir) list so that the total count is
    ~``target_size`` and each ``question_type`` contributes uniformly.

    Per-category quota = ceil(target / n_categories) for the first
    (target % n_categories) categories, floor otherwise. If a category has
    fewer samples than its quota, all of them are kept (no upsampling).

    Sampling is deterministic given ``seed``.
    """
    import random
    from collections import defaultdict

    if target_size <= 0 or len(samples) <= target_size:
        return samples

    groups: dict[str, list] = defaultdict(list)
    for entry, sd in samples:
        qt = str(entry.get("question_type", "_unknown"))
        groups[qt].append((entry, sd))

    cats = sorted(groups.keys())
    n_cats = len(cats)
    base   = target_size // n_cats
    extra  = target_size % n_cats   # first `extra` categories get +1

    rng = random.Random(seed)
    out: list = []
    per_cat_taken: dict[str, int] = {}
    for i, qt in enumerate(cats):
        quota = base + (1 if i < extra else 0)
        pool  = groups[qt]
        if len(pool) <= quota:
            picked = pool
        else:
            picked = rng.sample(pool, quota)
        per_cat_taken[qt] = len(picked)
        out.extend(picked)

    rng.shuffle(out)   # avoid category-block ordering after concat
    if log is not None:
        log.info(
            f"_balance_sat_samples_by_qtype: target={target_size} "
            f"-> kept {len(out)} (per-category: {per_cat_taken})"
        )
    return out


def _format_sat_question(entry: dict) -> tuple[str, str]:
    """Format a SAT entry into (prompt_text, answer).

    If answer_choices is non-empty, the answer is the matching letter
    ("A", "B", ...); otherwise it's the raw correct_answer string.
    """
    question = entry.get("question", "")
    choices  = entry.get("answer_choices", []) or []
    correct  = entry.get("correct_answer", "")
    if choices:
        formatted = "\n".join(
            f"{_SAT_LETTERS[i]}. {c}" for i, c in enumerate(choices)
        )
        prompt_text = question + "\n" + formatted
        try:
            answer = _SAT_LETTERS[choices.index(correct)]
        except ValueError:
            answer = str(correct)
    else:
        prompt_text = question
        answer = str(correct)
    return prompt_text, answer


class SAT_Train_Dataset(Dataset):
    """
    SAT training dataset (LM-only, mirrors MindCube_Train_Dataset).

    Reads a SAT JSON list (NOT JSONL) where each entry has:
        database_idx, question, answer_choices, correct_answer, img_paths

    3D layout under ``results_dir`` (same as MindCube):
        <results_dir>/<database_idx>/view_XXXX/
            image.png, pts3d.npy, mask.npy, [camera_pose.npy]

    Returns the same per-item dict as MindCube_Train_Dataset:
        input_ids, attention_mask, pixel_values, image_grid_thw,
        image_xyz, labels.
    """

    def __init__(
        self,
        json_path:           str,
        results_dir:         str,
        processor,
        log,
        max_images:          int = 4,
        spatial_merge_size:  int = 2,
        max_samples:         int | None = None,
        balanced_categories: bool = False,
        target_size:         int | None = None,
        balance_seed:        int = 0,
    ):
        import json
        with open(json_path) as fh:
            raw = json.load(fh)

        self.samples = []
        for entry in raw:
            eid = str(entry.get("database_idx", ""))
            sample_dir = os.path.join(results_dir, eid)
            if not os.path.isdir(sample_dir):
                continue
            self.samples.append((entry, sample_dir))

        if balanced_categories and target_size is not None and target_size > 0:
            self.samples = _balance_sat_samples_by_qtype(
                self.samples, target_size=target_size,
                seed=balance_seed, log=log,
            )

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.log = log
        log.info(
            f"SAT_Train_Dataset: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total) from {json_path} "
            f"[balanced={balanced_categories}, target={target_size}]"
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
        if N < 1:
            raise RuntimeError(
                f"SAT sample {idx} (database_idx={entry.get('database_idx')}) "
                f"has no valid images under {sample_dir}."
            )

        # ── build prompt (QA + choices) ──────────────────────────────────────
        question_text, answer_text = _format_sat_question(entry)
        if not question_text or not answer_text:
            raise RuntimeError(
                f"SAT sample {idx} (database_idx={entry.get('database_idx')}) "
                f"has no QA pair."
            )

        content: list = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question_text})

        # New format (matches MindCube_Train_Dataset): wrap the SAT answer
        # in <answer>X</answer>, disable thinking. See
        # md/bug_fix/train_eval_paradigm_mismatch.md.
        from .answer_format import format_answer, IM_END_NEWLINE
        formatted_answer = format_answer(answer_text)
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": formatted_answer}],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        suffix_ids = self.processor.tokenizer(
            formatted_answer + IM_END_NEWLINE, add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

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
            "image_xyz": image_xyz,
            "labels":    labels,
        }


class SAT_Train_Dataset_Rotation(Dataset):
    """
    SAT training dataset for RotationRoPEModel.

    Reads a SAT train JSON (list, not JSONL) whose entries look like:
        {
            "database_idx": 127503,
            "question_type": "obj_movement",
            "question": "...",
            "answer_choices": ["no objects moved", "sofa was ..."],
            "correct_answer": "no objects moved",
            "img_paths": ["./data/train/image_127503_0.png", ...]
        }

    3D layout expected under ``results_dir``:
        <results_dir>/<database_idx>/view_XXXX/
            image.png          — RGB image
            pts3d.npy          — (H, W, 3) per-pixel 3D coords (world frame
                                 = view_0000 camera frame)
            mask.npy           — (H, W) bool valid mask
            camera_pose.npy    — (4, 4) camera-to-world

    Unlike MindCube_Train_Dataset_Coord, single-view samples (N=1) are
    allowed: SAT has both single- and multi-view entries. Camera-pair
    pose sentences are always suppressed (``no_cam=True`` in spirit).

    Returns the same per-item fields as MindCube_Train_Dataset_Rotation:
      input_ids, attention_mask, pixel_values, image_grid_thw,
      labels, image_xyz, image_xyz_hires, cam_pos_frame0, gt_rotation.
    """

    def __init__(
        self,
        json_path:          str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        coord_upscale:      int = 4,
        max_samples:        int | None = None,
    ):
        import json
        with open(json_path) as fh:
            raw = json.load(fh)

        self.samples = []
        for entry in raw:
            eid = str(entry.get("database_idx", ""))
            sample_dir = os.path.join(results_dir, eid)
            if not os.path.isdir(sample_dir):
                continue
            self.samples.append((entry, sample_dir))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.coord_upscale      = coord_upscale
        self.log = log
        log.info(
            f"SAT_Train_Dataset_Rotation: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total) from {json_path}"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _format_question(entry: dict) -> tuple[str, str]:
        """Return (prompt_text, answer_letter_or_text) for a SAT entry."""
        question = entry.get("question", "")
        choices  = entry.get("answer_choices", []) or []
        correct  = entry.get("correct_answer", "")
        if choices:
            formatted = "\n".join(
                f"{_SAT_LETTERS[i]}. {c}" for i, c in enumerate(choices)
            )
            prompt_text = question + "\n" + formatted
            try:
                answer = _SAT_LETTERS[choices.index(correct)]
            except ValueError:
                answer = str(correct)
        else:
            prompt_text = question
            answer = str(correct)
        return prompt_text, answer

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        # ── load images + per-pixel xyz ──────────────────────────────────────
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
        if N < 1:
            raise RuntimeError(
                f"SAT sample {idx} (database_idx={entry.get('database_idx')}) "
                f"has no valid images under {sample_dir}."
            )

        # ── build prompt (QA + choices) ──────────────────────────────────────
        question_text, answer_text = self._format_question(entry)
        if not question_text or not answer_text:
            raise RuntimeError(
                f"SAT sample {idx} (database_idx={entry.get('database_idx')}) "
                f"has no QA pair."
            )

        content: list = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": question_text})

        # New format (matches MindCube_Train_Dataset): wrap the SAT answer
        # in <answer>X</answer>, disable thinking.
        from .answer_format import format_answer, IM_END_NEWLINE
        formatted_answer = format_answer(answer_text)
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": formatted_answer}],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        suffix_ids = self.processor.tokenizer(
            formatted_answer + IM_END_NEWLINE, add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

        # ── 3D position maps (patch-level + sub-pixel) ───────────────────────
        image_xyz = None
        image_xyz_hires = None
        try:
            thw_all = proc_out["image_grid_thw"]  # (N, 3)
            sms     = self.spatial_merge_size
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
                    xyz_hires_list.append(
                        resize_xyz(xyz_raw, llm_h * up, llm_w * up, valid=mask_raw)
                    )
                else:
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                    xyz_hires_list.append(torch.zeros(llm_h * up, llm_w * up, 3))
            image_xyz = xyz_list
            image_xyz_hires = xyz_hires_list
        except Exception as exc:
            self.log.debug(f"pts3d load failed for {sample_dir}: {exc}")
            image_xyz = None
            image_xyz_hires = None

        # ── first-frame camera pose (for RotationRoPEModel) ──────────────────
        cam_pos_frame0 = torch.zeros(3, dtype=torch.float32)
        gt_rotation    = torch.eye(3,  dtype=torch.float32)
        first_view = view_dirs[0] if view_dirs else None
        if first_view is not None:
            cp_path = os.path.join(sample_dir, first_view, "camera_pose.npy")
            if os.path.exists(cp_path):
                try:
                    pose = np.load(cp_path).astype(np.float32)
                    cam_pos_frame0 = torch.tensor(pose[:3, 3], dtype=torch.float32)
                    gt_rotation    = torch.tensor(pose[:3, :3].T, dtype=torch.float32)
                except Exception as exc:
                    self.log.debug(
                        f"camera_pose load failed for {sample_dir}/{first_view}: {exc}"
                    )

        return {
            **proc_out,
            "image_xyz":       image_xyz,
            "image_xyz_hires": image_xyz_hires,
            "labels":          labels,
            "cam_pos_frame0":  cam_pos_frame0,
            "gt_rotation":     gt_rotation,
        }


class MindCube_Train_Dataset_Coord_Polar(MindCube_Train_Dataset_Coord):
    """
    Variant of MindCube_Train_Dataset_Coord where image_xyz_hires is
    converted from Cartesian (x, y, z) to log-spherical (log r, θ, α)
    via xyz_to_polar before being returned.  image_xyz (patch-level,
    used for 4D M-RoPE) is kept in Cartesian here — the RoPE layer
    converts it internally when polar=True is passed through
    CoordinateModel.forward → spa_model → get_vision_position_ids.

    Conventions (match xyz_to_polar and get_vision_position_ids):
        log r — scale-invariant radius
        θ     = atan2(y, x)          ∈ [-π, π]  — azimuth
        α     = atan2(√(x²+y²), z)   ∈ [0, π]   — inclination

    Use with --polar flag in train_coordinate.py.
    """

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        if batch.get("image_xyz_hires") is not None:
            batch["image_xyz_hires"] = [
                xyz_to_polar(xyz) for xyz in batch["image_xyz_hires"]
            ]
        return batch
