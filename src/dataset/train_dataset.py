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
POSE_TOKEN = "<pose>"

# ── dataset ───────────────────────────────────────────────────────────────────

class Train_Dataset(Dataset):
    """
    One sample = one SPAR scene entry.

    For each entry with N images (capped at max_images):
      - Builds a multi-image chat prompt with A(N,2) = N*(N-1) <pose> tokens,
        one per ordered pair (i, j) with i≠j, enumerated as:
            (0,1), (0,2), ..., (0,N-1),
            (1,0), (1,2), ..., (1,N-1),
            ...
            (N-1,0), ..., (N-1,N-2)
      - Returns processor tensors + GT relative transforms T_{i→j}.

    GT transforms come from reconstruct/{entry_id}.npz:
        relative_transforms[i, j] = T_{i→j}
        = inv(poses_ff[j]) @ poses_ff[i]
        Transforms a 3-D point from camera-i frame to camera-j frame.
    """

    def __init__(
        self,
        json_path:           str,
        spar_root:           str,
        reconstruct_dir:     str,
        processor,
        pose_token_id:       int,
        log,
        max_images:          int = 4,
        pos3d_dir:           str | None = None,
        spatial_merge_size:  int = 2,
        max_samples:         int | None = None,
        plus:                bool = False,
        no_pose:             bool = False,
    ):
        import json
        with open(json_path) as fh:
            entries = json.load(fh)

        self.samples = []
        for e in entries:
            eid = e.get("id", "")
            npz = os.path.join(reconstruct_dir, f"{eid}.npz")
            if not (os.path.exists(npz) and e.get("image")):
                continue
            # also require 3D_pos file if a pos3d_dir is given
            if pos3d_dir is not None:
                p3d = os.path.join(pos3d_dir, f"{eid}.npz")
                if not os.path.exists(p3d):
                    continue
            else:
                p3d = None
            self.samples.append((e, npz, p3d))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.spar_root           = spar_root
        self.processor           = processor
        self.pose_token_id       = pose_token_id
        self.max_images          = max_images
        self.pos3d_dir           = pos3d_dir
        self.spatial_merge_size  = spatial_merge_size
        self.plus                = plus
        self.no_pose             = no_pose
        self.log = log
        log.info(f"SPARDataset: {len(self.samples)} valid entries "
                 f"(out of {len(entries)} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, npz_path, p3d_path = self.samples[idx]

        # ── load images ───────────────────────────────────────────────────────
        # Paths in the JSON are relative to {dataset}/images/; resolve using
        # detect_dataset() (scannet / scannetpp / structured3d).
        images = []
        for rel in entry["image"]:
            scene_id = _scene_id_from_path(rel)
            dataset  = detect_dataset(scene_id)
            full = os.path.join(self.spar_root, dataset, "images", rel)
            try:
                images.append(Image.open(full).convert("RGB"))
            except (FileNotFoundError, OSError):
                break
            if len(images) == self.max_images:
                break

        # ── validate against GT ───────────────────────────────────────────────
        data     = np.load(npz_path)
        n_stored = data["relative_transforms"].shape[0]
        N        = min(len(images), n_stored)

        if N < 2:
            raise RuntimeError(
                f"Sample {idx} (id={entry.get('id')}) has only {N} valid "
                f"images after loading; skipping. Check image paths."
            )

        images = images[:N]

        # ── GT transforms (skipped in no_pose ablation mode) ─────────────────
        if self.no_pose:
            gt_transforms = None
        else:
            rel_transforms = torch.tensor(
                data["relative_transforms"][:N, :N], dtype=torch.float32
            )  # (N, N, 4, 4)

            # All ordered pairs (i, j) with i≠j — A(N,2) = N*(N-1) pairs
            pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

            # GT: T_{i→j} for every ordered pair
            gt_transforms = torch.stack(
                [rel_transforms[i, j] for (i, j) in pairs], dim=0
            )  # (N*(N-1), 4, 4)

        # ── build multi-image prompt ──────────────────────────────────────────
        content: list = []
        for img in images:
            content.append({"type": "image", "image": img})

        pose_sentences = []
        if not self.no_pose:
            pose_sentences = [
                f"The camera pose of image {j + 1} relative to image {i + 1} is "
                f"{POSE_TOKEN}."
                for (i, j) in pairs
            ]
            content.append({"type": "text", "text": " ".join(pose_sentences)})

        # ── build prompt (plus / no_pose modes append QA for LM loss) ────────
        # Extract question/answer from conversations list (ShareGPT format)
        # or from flat "question"/"answer" fields.
        _convs = entry.get("conversations", [])
        _question = (
            entry.get("question")
            or next((c["value"] for c in _convs if c.get("from") == "human"), None)
        )
        _answer = (
            entry.get("answer")
            or next((c["value"] for c in _convs if c.get("from") == "gpt"), None)
        )

        labels = None
        if (self.plus or self.no_pose) and _question and _answer:
            question = _question
            answer   = _answer
            qa_content = list(content)
            if self.no_pose:
                # No pose sentences — just append the question
                qa_content.append({"type": "text", "text": question})
            else:
                # Replace the last text element to include the question
                qa_content[-1] = {
                    "type": "text",
                    "text": " ".join(pose_sentences) + " " + question,
                }
            # Full conversation with assistant answer
            text_full = self.processor.apply_chat_template(
                [{"role": "user", "content": qa_content},
                 {"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False,
            )
            proc_out = self.processor(
                text=[text_full], images=images,
                return_tensors="pt", padding=False,
            )
            # Supervise only the answer tokens at the tail of the sequence.
            # Mask everything before (including the <think> block) so the
            # model's reasoning behaviour is not suppressed.
            # <|im_end|> is a special token so suffix tokenisation is stable.
            suffix_ids = self.processor.tokenizer(
                answer + "<|im_end|>\n", add_special_tokens=False
            )["input_ids"]
            suffix_len = len(suffix_ids)
            labels = proc_out["input_ids"].clone()
            labels[0, :-suffix_len] = -100          # mask all except answer tokens
        else:
            messages = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            # ── tokenise + encode images ──────────────────────────────────────
            proc_out = self.processor(
                text   = [prompt_text],
                images = images,
                return_tensors = "pt",
                padding        = False,
            )

        # ── 3D position maps ──────────────────────────────────────────────────
        # Load per-pixel XYZ from 3D_pos file and downsample to the LLM patch
        # grid resolution consumed by get_rope_index() as image_xyz.
        # Each element: (llm_H_i, llm_W_i, 3) where
        #   llm_H_i = grid_thw[i,1] // spatial_merge_size
        #   llm_W_i = grid_thw[i,2] // spatial_merge_size
        image_xyz = None
        if p3d_path is not None:
            try:
                d3d   = np.load(p3d_path)
                thw_all = proc_out["image_grid_thw"]          # (N, 3)
                sms     = self.spatial_merge_size
                xyz_list = []
                n_3d = int(d3d["n_frames"])
                for k in range(min(N, n_3d)):
                    xyz_raw   = d3d[f"frame_{k}_xyz"]         # (H_img, W_img, 3)
                    valid_raw = d3d.get(f"frame_{k}_valid")   # (H_img, W_img) bool | None
                    thw_k     = thw_all[k]                    # (T, H, W) patch units
                    llm_h     = int(thw_k[1]) // sms
                    llm_w     = int(thw_k[2]) // sms
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=valid_raw))
                # Pad missing frames with zeros if needed
                for k in range(len(xyz_list), N):
                    thw_k = thw_all[k]
                    llm_h = int(thw_k[1]) // sms
                    llm_w = int(thw_k[2]) // sms
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                image_xyz = xyz_list                          # list of N tensors
            except Exception as exc:
                log.debug(f"3D_pos load failed for {p3d_path}: {exc}")
                image_xyz = None

        return {
            **proc_out,                      # input_ids, attention_mask,
                                             # pixel_values, image_grid_thw …
            "gt_transforms": gt_transforms,  # (N*(N-1), 4, 4)
            "image_xyz":     image_xyz,      # list of (llm_H, llm_W, 3) or None
            "labels":        labels,         # (1, seq_len) for plus mode, else None
        }


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

