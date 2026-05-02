"""Training datasets for Qwen3.5-VL.

3d_results on disk live at the reconstruction model's MapAny native shape
(one of RESOLUTION_MAPPINGS[518], all /14 multiples but **not** /28).
Qwen3.5-VL's vision processor rounds image (H, W) to multiples of
patch_size × spatial_merge_size = 14 × 2 = 28 via smart_resize, and
resize_xyz block-averages pts3d to the LLM patch grid using *integer*
stride. Feeding Qwen the MapAny-shape image directly works (Qwen runs
smart_resize internally), but resize_xyz on the MapAny pts3d would
truncate edge pixels with non-/28 H or W.

We resolve this in __getitem__ by Qwen-aligning each loaded view (image +
pts3d + mask) to smart_resize's target shape *before* the Qwen processor
and resize_xyz run.
"""

import os
import math
from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch


# ── Qwen3.5-VL smart_resize alignment ─────────────────────────────────────────
# patch_size 14 × spatial_merge_size 2 = 28. Qwen's image_processor.smart_resize
# rounds H, W to multiples of this factor preserving aspect ratio, bounded by
# [min_pixels, max_pixels]. The defaults below match Qwen2.5/3.5-VL config
# (min = 56*56, max = 14*14*4*1280). If your processor uses different bounds,
# pass them via _qwen_align_view's kwargs.

_QWEN_FACTOR = 28
_QWEN_MIN_PIXELS = 56 * 56
_QWEN_MAX_PIXELS = 14 * 14 * 4 * 1280  # = 1_003_520


def _smart_resize_target(
    H: int,
    W: int,
    factor: int = _QWEN_FACTOR,
    min_pixels: int = _QWEN_MIN_PIXELS,
    max_pixels: int = _QWEN_MAX_PIXELS,
) -> tuple[int, int]:
    """Replicate Qwen2.5/3.5-VL's smart_resize target shape: round (H, W) to
    multiples of *factor*, bound area by [min_pixels, max_pixels] preserving
    aspect ratio. Returns (H_q, W_q)."""
    h_bar = round(H / factor) * factor
    w_bar = round(W / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((H * W) / max_pixels)
        h_bar = max(factor, math.floor(H / beta / factor) * factor)
        w_bar = max(factor, math.floor(W / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (H * W))
        h_bar = math.ceil(H * beta / factor) * factor
        w_bar = math.ceil(W * beta / factor) * factor
    return h_bar, w_bar


def _qwen_align_view(
    image: Image.Image,
    xyz:   np.ndarray | None,
    mask:  np.ndarray | None,
    factor:     int = _QWEN_FACTOR,
    min_pixels: int = _QWEN_MIN_PIXELS,
    max_pixels: int = _QWEN_MAX_PIXELS,
) -> tuple[Image.Image, np.ndarray | None, np.ndarray | None]:
    """Resample one view's image / pts3d / mask to Qwen smart_resize target.

    Bilinear for pts3d (continuous geometry), nearest for mask (preserves
    binary), LANCZOS for image (uint8 RGB).

    image is the canonical reference for shape: pts3d / mask must already be
    pixel-aligned with the image when this is called (we never resize them
    against image without resizing image too).
    """
    W_in, H_in = image.size                         # PIL .size = (W, H)
    if xyz is not None and xyz.shape[:2] != (H_in, W_in):
        raise ValueError(
            f"_qwen_align_view: pts3d {xyz.shape[:2]} != image (H,W) "
            f"{(H_in, W_in)} — saved data is not pixel-aligned"
        )
    if mask is not None and mask.shape != (H_in, W_in):
        raise ValueError(
            f"_qwen_align_view: mask {mask.shape} != image (H,W) "
            f"{(H_in, W_in)} — saved data is not pixel-aligned"
        )

    H_q, W_q = _smart_resize_target(H_in, W_in, factor, min_pixels, max_pixels)
    image_q = image.resize((W_q, H_q), resample=Image.LANCZOS)

    xyz_q: np.ndarray | None = None
    if xyz is not None:
        # (H, W, 3) → bilinear via torch.nn.functional.interpolate
        xyz_t = torch.from_numpy(xyz).permute(2, 0, 1).unsqueeze(0).float()
        xyz_q = (
            torch.nn.functional.interpolate(
                xyz_t, size=(H_q, W_q), mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
            .astype(np.float32)
        )

    mask_q: np.ndarray | None = None
    if mask is not None:
        mask_t = (
            torch.from_numpy(mask.astype(np.uint8))
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        mask_q = (
            torch.nn.functional.interpolate(
                mask_t, size=(H_q, W_q), mode="nearest"
            )
            .squeeze()
            .numpy()
            .astype(bool)
        )

    return image_q, xyz_q, mask_q


def _load_and_align_views(
    sample_dir: str,
    max_images: int,
) -> tuple[list[Image.Image], list[np.ndarray | None], list[np.ndarray | None], list[str]]:
    """Load up to *max_images* views from <sample_dir>/view_XXXX/ and apply
    Qwen alignment to each (image, pts3d, mask) tuple. Returns four parallel
    lists of equal length (empty if no view loads).

    Per-view file layout:
        view_XXXX/
            image.png        — RGB
            pts3d.npy        — (H, W, 3) float, optional
            mask.npy         — (H, W)    bool,  optional

    The first view that fails to decode the image breaks the loop (mirrors
    the original behavior where missing views truncate the sample early).
    """
    view_dirs_all = sorted(
        d for d in os.listdir(sample_dir) if d.startswith("view_")
    )
    images:     list[Image.Image]     = []
    xyz_list:   list[np.ndarray|None] = []
    mask_list:  list[np.ndarray|None] = []
    kept_views: list[str]             = []

    for vd in view_dirs_all[:max_images]:
        img_path = os.path.join(sample_dir, vd, "image.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            break

        pts3d_path = os.path.join(sample_dir, vd, "pts3d.npy")
        mask_path  = os.path.join(sample_dir, vd, "mask.npy")
        xyz  = np.load(pts3d_path).astype(np.float32) if os.path.exists(pts3d_path) else None
        mask = np.load(mask_path)                     if os.path.exists(mask_path)  else None

        # Hard assertion: pts3d / mask must be pixel-aligned with the image.
        # The reconstruction pipeline guarantees this; a mismatch means the
        # 3d_results entry is corrupted (mixed write, partial overwrite, etc.)
        # — fail loudly instead of silently masking the geometry.
        H_W = img.size[::-1]   # PIL .size = (W, H); reverse to (H, W)
        if xyz is not None:
            assert xyz.shape[:2] == H_W, (
                f"pts3d shape {xyz.shape[:2]} != image (H,W) {H_W} "
                f"at {sample_dir}/{vd}"
            )
        if mask is not None:
            assert mask.shape == H_W, (
                f"mask shape {mask.shape} != image (H,W) {H_W} "
                f"at {sample_dir}/{vd}"
            )

        img_q, xyz_q, mask_q = _qwen_align_view(img, xyz, mask)
        images.append(img_q)
        xyz_list.append(xyz_q)
        mask_list.append(mask_q)
        kept_views.append(vd)

    return images, xyz_list, mask_list, kept_views


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

    Note: expects (H, W) such that H % target_h == 0 and W % target_w == 0
    (true after _qwen_align_view, where stride = factor / spatial_merge_size = 14).
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
    valid    = valid[:H_crop, :W_crop].astype(np.float32)

    # Reshape into patch blocks
    xyz_blocks   = xyz_f.reshape(target_h, stride_h, target_w, stride_w, 3)
    valid_blocks = valid.reshape(target_h, stride_h, target_w, stride_w)

    # Masked sum → mean over the stride_h × stride_w pixel block per patch
    xyz_sum   = (xyz_blocks * valid_blocks[..., None]).sum(axis=(1, 3))
    valid_cnt = valid_blocks.sum(axis=(1, 3))

    denom    = np.maximum(valid_cnt, 1)[..., None]
    xyz_mean = xyz_sum / denom
    xyz_mean[valid_cnt == 0] = 0.0

    return torch.from_numpy(xyz_mean)


# ── MindCube training dataset ─────────────────────────────────────────────────

class MindCube_Train_Dataset(Dataset):
    """
    Training dataset for MindCube JSONL + 3d_results structure (LM-only,
    no pose prediction).

    Layout:
        <results_dir>/<id>/
            view_0000/
                image.png         — RGB image  (MapAny shape on disk)
                pts3d.npy         — (H, W, 3) per-pixel 3D coords
                mask.npy          — (H, W) bool valid mask
            view_0001/ ...

    All views are Qwen-aligned (smart_resize-rounded) at __getitem__ time.
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

        images, xyz_raw_list, mask_raw_list, _ = _load_and_align_views(
            sample_dir, self.max_images,
        )

        N = len(images)
        if N < 2:
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has only {N} "
                f"valid images; need ≥ 2."
            )

        # ── build prompt (images only, no pose sentences) ─────────────────────
        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        labels = None
        if _question and _answer:
            from .answer_format import (
                build_interleaved_content, format_answer_with_text, IM_END_NEWLINE,
            )
            qa_content = build_interleaved_content(_question, images)
            formatted_answer = format_answer_with_text(_answer, _question)
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
            suffix_ids = self.processor.tokenizer(
                formatted_answer + IM_END_NEWLINE, add_special_tokens=False
            )["input_ids"]
            labels = proc_out["input_ids"].clone()
            labels[0, :-len(suffix_ids)] = -100
        else:
            # No QA pair → image-only prompt (used by some MindCube preprocessing
            # entries that lack a question/answer field).
            content = [{"type": "image", "image": img} for img in images]
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

        images, xyz_raw_list, mask_raw_list, _ = _load_and_align_views(
            sample_dir, self.max_images,
        )

        N = len(images)
        if N < 2:
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has only {N} "
                f"valid images; need >= 2."
            )

        # ── build prompt (images + QA) ────────────────────────────────────────
        _question = entry.get("question", "")
        _answer   = entry.get("gt_answer", "")

        if not (_question and _answer):
            raise RuntimeError(
                f"MindCube sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        from .answer_format import (
            build_interleaved_content, format_answer_with_text, IM_END_NEWLINE,
        )
        content = build_interleaved_content(_question, images)
        formatted_answer = format_answer_with_text(_answer, _question)
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


# ── VST training dataset ──────────────────────────────────────────────────────

# VST entries declare a `type` (correspondence / qa / depth / ...). The
# 3d_results tree splits the same data across eight subdirectories named after
# the *task family*. The mapping is many-to-one (e.g. "depth" lives mostly in
# si_depth_comparison but a minority shares ids with si_distance), so each
# entry is resolved by trying its candidates in priority order.
_VST_TYPE_TO_SUBDIRS: dict[str, list[str]] = {
    "correspondence":     ["mi_correspondence"],
    "caption-multi-view": ["mi_scene_caption"],
    "qa":                 ["si_measurement"],
    "unknow":             ["si_scene_caption"],
    "depth":              ["si_depth_comparison", "si_distance"],
    "distance":           ["si_distance", "si_depth_comparison"],
    "multi_image":        ["mi_camera_motion", "mi_object_object_relation"],
}


def _build_vst_id_index(results_dir: str) -> dict[str, set[str]]:
    """One os.listdir per subdir under results_dir, cached as id-sets.

    Replaces O(N_entries × per-id isdir) with O(N_subdirs × listdir +
    N_entries × dict/set lookup), which is the difference between minutes
    and seconds for VST 500K on NFS.
    """
    index: dict[str, set[str]] = {}
    for sd in sorted(os.listdir(results_dir)):
        sd_path = os.path.join(results_dir, sd)
        if not os.path.isdir(sd_path):
            continue
        index[sd] = set(os.listdir(sd_path))
    return index


def _resolve_vst_sample_dir(
    results_dir: str, etype: str, eid: str,
    id_index: dict[str, set[str]],
) -> str | None:
    """Return <results_dir>/<subdir>/<eid> for the first matching subdir.

    Tries `_VST_TYPE_TO_SUBDIRS[etype]` first; if `etype` is unknown, falls
    back to every subdir registered in `id_index`. All checks are O(1) set
    lookups against the pre-built id index.
    """
    for sd in _VST_TYPE_TO_SUBDIRS.get(etype, []):
        ids = id_index.get(sd)
        if ids is not None and eid in ids:
            return os.path.join(results_dir, sd, eid)
    for sd, ids in id_index.items():
        if eid in ids:
            return os.path.join(results_dir, sd, eid)
    return None


def _vst_extract_qa_pairs(entry: dict) -> list[tuple[str, str]]:
    """Pair (human, gpt) turns into (question, answer) tuples.

    Most VST entries are single-turn, but `si_measurement` rows hold 3 Q&A
    pairs per sample (3 different objects in the same image). Returning the
    full list preserves all supervision instead of dropping turns 2-3.

    The human turn embeds one `<|image_pad|>` per attached image; we strip
    them because Qwen's chat template re-inserts vision placeholders from
    the `{"type": "image", ...}` content items.
    """
    pairs: list[tuple[str, str]] = []
    pending_q: str | None = None
    for turn in entry.get("conversations", []):
        who = turn.get("from", "")
        val = turn.get("value", "")
        if who == "human":
            pending_q = val.replace("<|image_pad|>", "").strip()
        elif who == "gpt" and pending_q is not None:
            answer = val.strip()
            if pending_q and answer:
                pairs.append((pending_q, answer))
            pending_q = None
    return pairs


def _vst_build_chat_with_labels(
    images: list,
    qa_pairs: list[tuple[str, str]],
    processor,
) -> tuple[dict, "torch.Tensor"]:
    """Tokenize a multi-turn VST chat and build the assistant-only label mask.

    Each assistant response is wrapped in ``<answer>{raw}</answer>`` so VST
    matches MindCube's parse target — eval can extract content with one
    regex regardless of the underlying VST subset (MCQ letter, "97 cm",
    long caption, etc.).

    Images are attached to the first user turn only; later turns are text.
    Labels start fully masked (-100); for each Q&A we locate the matching
    ``<answer>...</answer><|im_end|>\\n`` token subsequence and unmask it.
    A failure to locate a span (BPE drift across special-token boundaries)
    raises loudly rather than silently dropping supervision.
    """
    from .answer_format import IM_END_NEWLINE, build_interleaved_content

    messages: list = []
    for i, (q, a) in enumerate(qa_pairs):
        if i == 0:
            # First user turn carries all the images. Interleave any literal
            # `<image>` placeholders in the question with the actual image
            # objects (no-op for VST since its conversations don't contain
            # the placeholder string — verified — but kept consistent with
            # MindCube/SpinBench so all train+eval prompt builders share one
            # policy).
            user_content = build_interleaved_content(q, images)
        else:
            user_content = [{"type": "text", "text": q}]
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": f"<answer>{a}</answer>",
        })

    text_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    proc_out = processor(
        text=[text_full], images=images,
        return_tensors="pt", padding=False,
    )

    labels  = torch.full_like(proc_out["input_ids"], -100)
    ids_seq = proc_out["input_ids"][0].tolist()
    cursor  = 0
    for _, a in qa_pairs:
        suffix_ids = processor.tokenizer(
            f"<answer>{a}</answer>" + IM_END_NEWLINE,
            add_special_tokens=False,
        )["input_ids"]
        L = len(suffix_ids)
        found = -1
        for start in range(cursor, len(ids_seq) - L + 1):
            if ids_seq[start:start + L] == suffix_ids:
                found = start
                break
        if found < 0:
            preview = a[:60].replace("\n", " ")
            raise RuntimeError(
                f"VST: could not locate <answer>{preview}...</answer> token "
                f"span in tokenized chat (cursor={cursor}, L={L})"
            )
        labels[0, found:found + L] = proc_out["input_ids"][0, found:found + L]
        cursor = found + L

    return proc_out, labels


class VST_Train_Dataset(Dataset):
    """
    Training dataset for VST 500K (`vst_500k.json` + 3d_results tree).

    JSON layout (one list under `vst_500k.json`):
        {
          "id":            "<uuid or numeric>",
          "type":          "correspondence|qa|depth|distance|multi_image|...",
          "images":        ["images/<id>_0.jpg", ...],
          "conversations": [{"from": "human", "value": "<|image_pad|>...?"},
                            {"from": "gpt",   "value": "<answer text>"}]
        }

    Reconstruction layout:
        <results_dir>/<subdir>/<id>/
            cameras.json
            view_XXXX/{image.png, pts3d.npy, mask.npy, ...}
        where subdir is resolved from `type` via `_VST_TYPE_TO_SUBDIRS`.

    Differs from MindCube_Train_Dataset:
      - Loads JSON (not JSONL) and resolves the per-entry subdir.
      - Allows N >= 1 view (single-image VST tasks like qa/depth/distance).
      - Supports multi-turn QA (e.g. si_measurement) — all turns are kept
        and each assistant span is unmasked separately for CE loss.
      - Wraps every assistant response in ``<answer>...</answer>`` so the
        eval-time parsing target is the same as MindCube; the inner content
        stays free-form (MCQ letter, "97 cm", caption, etc.).
    """

    def __init__(
        self,
        json_path:          str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 8,
        spatial_merge_size: int = 2,
        max_samples:        int | None = None,
    ):
        import json
        with open(json_path) as fh:
            raw = json.load(fh)

        id_index = _build_vst_id_index(results_dir)

        self.samples = []
        n_no_dir = 0
        n_no_qa  = 0
        for entry in raw:
            eid   = str(entry.get("id", ""))
            etype = entry.get("type", "")
            sample_dir = _resolve_vst_sample_dir(
                results_dir, etype, eid, id_index,
            )
            if sample_dir is None:
                n_no_dir += 1
                continue
            if not _vst_extract_qa_pairs(entry):
                n_no_qa += 1
                continue
            self.samples.append((entry, sample_dir))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.log                = log
        log.info(
            f"VST_Train_Dataset: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total, skipped {n_no_dir} missing-dir, "
            f"{n_no_qa} empty-QA) from {json_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        images, xyz_raw_list, mask_raw_list, _ = _load_and_align_views(
            sample_dir, self.max_images,
        )

        N = len(images)
        if N < 1:
            raise RuntimeError(
                f"VST sample {idx} (id={entry.get('id')}) has no valid images."
            )

        qa_pairs = _vst_extract_qa_pairs(entry)
        if not qa_pairs:
            raise RuntimeError(
                f"VST sample {idx} (id={entry.get('id')}) has no Q&A pairs."
            )

        proc_out, labels = _vst_build_chat_with_labels(
            images, qa_pairs, self.processor,
        )

        # ── 3D position maps (pts3d → patch-level xyz) ───────────────────────
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


class VST_Train_Dataset_Coord(VST_Train_Dataset):
    """
    VST training dataset with sub-pixel coord supervision.

    Extends VST_Train_Dataset by also producing image_xyz_hires (upsampled by
    `coord_upscale`) for the PixelShuffle coord-head loss.
    """

    def __init__(
        self,
        json_path:          str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 8,
        spatial_merge_size: int = 2,
        coord_upscale:      int = 4,
        max_samples:        int | None = None,
    ):
        super().__init__(
            json_path=json_path, results_dir=results_dir, processor=processor,
            log=log, max_images=max_images,
            spatial_merge_size=spatial_merge_size, max_samples=max_samples,
        )
        self.coord_upscale = coord_upscale
        log.info(f"VST_Train_Dataset_Coord: coord_upscale={coord_upscale}")

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        images, xyz_raw_list, mask_raw_list, _ = _load_and_align_views(
            sample_dir, self.max_images,
        )

        N = len(images)
        if N < 1:
            raise RuntimeError(
                f"VST sample {idx} (id={entry.get('id')}) has no valid images."
            )

        qa_pairs = _vst_extract_qa_pairs(entry)
        if not qa_pairs:
            raise RuntimeError(
                f"VST sample {idx} (id={entry.get('id')}) has no Q&A pairs."
            )

        proc_out, labels = _vst_build_chat_with_labels(
            images, qa_pairs, self.processor,
        )

        # ── 3D position maps (pts3d → patch + sub-pixel) ─────────────────────
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
            "image_xyz":       image_xyz,
            "image_xyz_hires": image_xyz_hires,
            "labels":          labels,
        }


