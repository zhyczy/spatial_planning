"""
train_coordinate.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) with three
simultaneous supervision signals:

  1. Pose regression  — geodesic + L1 translation + cycle-consistency loss
                        (one <pose> token per ordered image pair)
  2. LM answer        — causal cross-entropy on answer tokens
                        (question appended to prompt; answer supervised)
  3. Coordinate       — L1 loss predicting per-patch mean 3D (x,y,z)
                        (down-sampled <coord> tokens → MLP → bilinear upsample)

Architecture:
  CoordinatePlusModel
    ├── SpaForConditionalGeneration  [backbone + LoRA adapters]
    │    ├── SpaVisionModel (ViT, frozen)
    │    └── SpaModel (LLM + 4D M-RoPE)
    ├── PoseRegressionHead      [MLP: hidden_dim*n_skip → 9D per <pose> token]
    └── CoordinateRegressionHead [MLP: hidden_dim → 3D + bilinear upsample]

Coordinate GT:
  For each image and each LLM patch token (after spatial merge), the GT is the
  mean (x,y,z) of all valid pixels that fall within that patch — exactly the
  values already computed by resize_xyz() and stored as image_xyz.

Total loss:
  loss = pose_loss + answer_weight * lm_loss + coord_weight * coord_loss

Usage:
  python train_coordinate.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_coordinate
"""

import argparse
import logging
import math
import os
import sys
from itertools import combinations

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None          # type: ignore[assignment]
    _WANDB_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from models.spa_emb import SpaForConditionalGeneration

sys.path.insert(0, os.path.join(_ROOT, "src", "data_process"))
from reconstruct_3d import detect_dataset, _scene_id_from_path  # noqa: E402

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
SPAR_ROOT       = os.path.join(_ROOT, "datasets/train/SPAR_7M/spar")
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
    xyz_f = xyz.astype(np.float32)

    if valid is None:
        valid = np.ones((H, W), dtype=bool)

    stride_h = H // target_h
    stride_w = W // target_w
    H_crop   = target_h * stride_h
    W_crop   = target_w * stride_w
    xyz_f    = xyz_f[:H_crop, :W_crop]
    valid    = valid[:H_crop, :W_crop].astype(np.float32)

    xyz_blocks   = xyz_f.reshape(target_h, stride_h, target_w, stride_w, 3)
    valid_blocks = valid.reshape(target_h, stride_h, target_w, stride_w)

    xyz_sum   = (xyz_blocks * valid_blocks[..., None]).sum(axis=(1, 3))  # (th, tw, 3)
    valid_cnt = valid_blocks.sum(axis=(1, 3))                            # (th, tw)

    denom    = np.maximum(valid_cnt, 1)[..., None]
    xyz_mean = xyz_sum / denom
    xyz_mean[valid_cnt == 0] = 0.0

    return torch.from_numpy(xyz_mean)   # (target_h, target_w, 3)


# ── rotation utilities ────────────────────────────────────────────────────────

def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    orig_dtype = r6d.dtype
    r6d = r6d.float()
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1).to(orig_dtype)   # (..., 3, 3)


def geodesic_loss(
    R_pred: torch.Tensor,
    R_gt:   torch.Tensor,
    eps:    float = 1e-6,
) -> torch.Tensor:
    R_pred = R_pred.float()
    R_gt   = R_gt.float()
    R_diff = torch.bmm(R_pred, R_gt.transpose(-1, -2))
    trace  = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos    = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()


def cycle_consistency_loss(R_preds: torch.Tensor, N: int) -> torch.Tensor:
    """
    Pair round-trips (N ≥ 2): R_{i→j} @ R_{j→i} = I
    Triangle cycles  (N ≥ 3): R_{a→b} @ R_{b→c} @ R_{c→a} = I
    Returns mean Frobenius error (scalar).
    """
    if N < 2:
        return R_preds.new_tensor(0.0)

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    p2k   = {p: k for k, p in enumerate(pairs)}
    I3    = torch.eye(3, device=R_preds.device, dtype=R_preds.dtype)
    errs  = []

    for i in range(N):
        for j in range(i + 1, N):
            cycle = R_preds[p2k[(i, j)]] @ R_preds[p2k[(j, i)]]
            errs.append(torch.norm(cycle - I3, p="fro"))

    for a, b, c in combinations(range(N), 3):
        cycle = R_preds[p2k[(a, b)]] @ R_preds[p2k[(b, c)]] @ R_preds[p2k[(c, a)]]
        errs.append(torch.norm(cycle - I3, p="fro"))

    return torch.stack(errs).mean()


# ── model components ──────────────────────────────────────────────────────────

class PoseRegressionHead(nn.Module):
    """MLP: input_dim → 9D (6D-rot + 3D-trans) per <pose> token."""

    def __init__(self, input_dim: int = 2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CoordinateRegressionHead(nn.Module):
    """
    Direct per-patch coordinate decoder: MLP applied to each <coord> token.

    One <coord> token per LLM patch.  Each token's hidden state is decoded to
    (x, y, z) by a small MLP; the output is reshaped to the patch grid.
    No downsampling or upsampling.
    """

    def __init__(self, hidden_dim: int = 2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (h*w, hidden_dim) — <coord> token hidden states
            h:      patch grid height
            w:      patch grid width
        Returns:
            (h, w, 3) — predicted xyz
        """
        return self.mlp(hidden).view(h, w, 3)


class CoordinatePlusModel(nn.Module):
    """
    SpaForConditionalGeneration (+ LoRA) with three supervision heads:

      1. PoseRegressionHead    → rot + trans + cycle loss   (at <pose> tokens)
      2. LM cross-entropy      → answer prediction loss     (at answer tokens)
      3. CoordinateRegressionHead → xyz coordinate loss
           - one <coord> token per LLM patch per image
           - each token's hidden state decoded directly to (x, y, z) via MLP

    Args:
        spa_model:          backbone (SpaForConditionalGeneration + LoRA)
        pose_head:          PoseRegressionHead
        coord_head:         CoordinateRegressionHead
        pose_token_id:      token id of <pose>
        coord_token_id:     token id of <coord>
        image_token_id:     token id of <|image_pad|>
        spatial_merge_size: spatial merge factor used by the vision encoder (2)
        skip_layers:        layer indices for skip-connection concat → pose head
        answer_weight:      scale factor for LM loss
        coord_weight:       scale factor for coordinate loss
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        pose_head:          PoseRegressionHead,
        coord_head:         CoordinateRegressionHead,
        pose_token_id:      int,
        coord_token_id:     int,
        image_token_id:     int,
        spatial_merge_size: int,
        skip_layers:        tuple[int, ...] = (-1,),
        answer_weight:      float = 1.0,
        coord_weight:       float = 1.0,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.pose_head          = pose_head
        self.coord_head         = coord_head
        self.pose_token_id      = pose_token_id
        self.coord_token_id     = coord_token_id
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.skip_layers        = list(skip_layers)
        self.answer_weight      = answer_weight
        self.coord_weight       = coord_weight

        # Hook on lm_head to capture last hidden state without
        # output_hidden_states=True (which defeats gradient checkpointing).
        self._lm_head_input: torch.Tensor | None = None
        for name, mod in self.spa_model.named_modules():
            if name.endswith("lm_head"):
                mod.register_forward_pre_hook(self._capture_lm_input)
                break

    def _capture_lm_input(self, module, args):
        """Pre-hook: capture the input to lm_head (= post-norm last hidden state)."""
        self._lm_head_input = args[0]

    def forward(
        self,
        input_ids:      torch.Tensor,           # (1, seq_len)
        attention_mask: torch.Tensor,           # (1, seq_len)
        pixel_values:   torch.Tensor | None,    # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor | None,    # (num_images, 3)  [T, H, W]
        gt_transforms:  torch.Tensor | None = None,  # (K, 4, 4)
        image_xyz:      list | None = None,     # list[i] = (llm_H_i, llm_W_i, 3)
        coord_scale:    float = 100.0,
        cycle_weight:   float = 0.0,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        # ── backbone (single forward pass) ────────────────────────────────────
        # When skip_layers==[-1], avoid output_hidden_states=True so that
        # gradient checkpointing can actually discard intermediate activations.
        # The lm_head pre-hook captures the last hidden state for us.
        only_last = (len(self.skip_layers) == 1 and self.skip_layers[0] == -1)
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = not only_last,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # ── hidden states for pose & coord heads ─────────────────────────────
        if only_last:
            # Captured by lm_head pre-hook (post-norm last hidden state)
            hidden_pose  = self._lm_head_input
            hidden_coord = self._lm_head_input
        else:
            if len(self.skip_layers) == 1:
                hidden_pose = outputs.hidden_states[self.skip_layers[0]]
            else:
                hidden_pose = torch.cat(
                    [outputs.hidden_states[i] for i in self.skip_layers], dim=-1
                )
            hidden_coord = outputs.hidden_states[-1]
        logits = outputs.logits
        del outputs

        # ── pose prediction ───────────────────────────────────────────────────
        pose_positions = (input_ids[0] == self.pose_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(pose_positions) == 0:
            return None, None, None

        latents  = hidden_pose[0, pose_positions]    # (K, input_dim)
        preds_9d = self.pose_head(latents)            # (K, 9)

        pose_loss = None
        _ldict: dict = {}
        if gt_transforms is not None:
            K  = preds_9d.shape[0]
            gt = gt_transforms[:K].to(preds_9d.device, dtype=preds_9d.dtype)
            R_gt   = gt[:, :3, :3]
            t_gt   = gt[:, :3,  3]
            R_pred = rot6d_to_rotmat(preds_9d[:, :6])
            t_pred = preds_9d[:, 6:]
            rot_loss   = geodesic_loss(R_pred, R_gt)
            trans_loss = F.l1_loss(t_pred, t_gt)
            pose_loss  = rot_loss + trans_loss
            _ldict = {"rot_loss": rot_loss.item(), "trans_loss": trans_loss.item()}
            if cycle_weight > 0.0:
                N = round((1 + math.sqrt(1 + 4 * K)) / 2)
                if N * (N - 1) == K:
                    c_loss    = cycle_weight * cycle_consistency_loss(R_pred, N)
                    pose_loss = pose_loss + c_loss
                    _ldict["cycle_loss"] = c_loss.item()
            _ldict["pose_loss"] = pose_loss.item()

        # ── LM answer-prediction loss ─────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()    # logits extracted above
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # ── per-patch 3D coordinate prediction ────────────────────────────────
        # One <coord> token per LLM patch; MLP decodes each directly to (x,y,z).
        coord_loss = None
        if image_xyz is not None and image_grid_thw is not None:
            coord_pos = (input_ids[0] == self.coord_token_id).nonzero(
                as_tuple=True
            )[0]

            sms   = self.spatial_merge_size
            start = 0
            per_img_losses: list[torch.Tensor] = []

            for k in range(min(len(image_xyz), len(image_grid_thw))):
                thw_k = image_grid_thw[k]
                llm_h = int(thw_k[1]) // sms
                llm_w = int(thw_k[2]) // sms
                n_tok = llm_h * llm_w

                if start + n_tok > len(coord_pos):
                    break

                coord_h_k = hidden_coord[0, coord_pos[start : start + n_tok]]
                pred_k = self.coord_head(coord_h_k, llm_h, llm_w)  # (llm_h, llm_w, 3)

                gt_k = image_xyz[k].to(pred_k.device, dtype=pred_k.dtype)
                per_img_losses.append(F.l1_loss(pred_k, gt_k))

                start += n_tok

            if per_img_losses:
                coord_loss = torch.stack(per_img_losses).mean()

        # ── combine losses ────────────────────────────────────────────────────
        loss = None
        if pose_loss is not None:
            loss = pose_loss
        if lm_loss is not None:
            _ldict["lm_loss"] = lm_loss.item()
            loss = (loss + self.answer_weight * lm_loss) if loss is not None \
                   else (self.answer_weight * lm_loss)
        if coord_loss is not None:
            _ldict["coord_loss"] = coord_loss.item()
            loss = (loss + self.coord_weight * coord_loss) if loss is not None \
                   else (self.coord_weight * coord_loss)

        return preds_9d, loss, (_ldict if _ldict else None)


# ── dataset ───────────────────────────────────────────────────────────────────

class SPARDataset(Dataset):
    """
    Dataset for coordinate training.  Every sample must have a QA pair and a
    3D position map (pos3d_dir).  Samples missing either are skipped at init.
    """

    def __init__(
        self,
        json_path:           str,
        spar_root:           str,
        reconstruct_dir:     str,
        processor,
        pose_token_id:       int,
        max_images:          int = 4,
        pos3d_dir:           str | None = None,
        spatial_merge_size:  int = 2,
        max_samples:         int | None = None,
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
            if pos3d_dir is not None:
                p3d = os.path.join(pos3d_dir, f"{eid}.npz")
                if not os.path.exists(p3d):
                    continue
            else:
                p3d = None
            self.samples.append((e, npz, p3d))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.spar_root          = spar_root
        self.processor          = processor
        self.pose_token_id      = pose_token_id
        self.max_images         = max_images
        self.pos3d_dir          = pos3d_dir
        self.spatial_merge_size = spatial_merge_size
        log.info(f"SPARDataset: {len(self.samples)} valid entries "
                 f"(out of {len(entries)} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, npz_path, p3d_path = self.samples[idx]

        # ── load images ───────────────────────────────────────────────────────
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
                f"images after loading; skipping."
            )

        images = images[:N]

        rel_transforms = torch.tensor(
            data["relative_transforms"][:N, :N], dtype=torch.float32
        )  # (N, N, 4, 4)

        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        gt_transforms = torch.stack(
            [rel_transforms[i, j] for (i, j) in pairs], dim=0
        )  # (N*(N-1), 4, 4)

        # ── build multi-image prompt ──────────────────────────────────────────
        content: list = []
        for img in images:
            content.append({"type": "image", "image": img})

        pose_sentences = [
            f"The camera pose of image {j + 1} relative to image {i + 1} is "
            f"{POSE_TOKEN}."
            for (i, j) in pairs
        ]

        # ── QA prompt ─────────────────────────────────────────────────────────
        _convs    = entry.get("conversations", [])
        _question = (
            entry.get("question")
            or next((c["value"] for c in _convs if c.get("from") == "human"), None)
        )
        _answer   = (
            entry.get("answer")
            or next((c["value"] for c in _convs if c.get("from") == "gpt"), None)
        )

        if not (_question and _answer):
            raise RuntimeError(
                f"Sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        # ── Probe image_grid_thw first (to know patch counts per image) ────────
        # Build a temporary prompt with pose but no coord, process to get thw
        content_probe = list(content)
        content_probe.append({
            "type": "text",
            "text": " ".join(pose_sentences) + " " + _question,
        })
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

        # ── Build coord sentences (one token per LLM patch) ────────────────────
        coord_sentences = []
        for k in range(N):
            llm_h = int(thw_all[k][1]) // sms
            llm_w = int(thw_all[k][2]) // sms
            n_tok = llm_h * llm_w
            coord_tokens = " ".join([COORD_TOKEN] * n_tok)
            coord_sentences.append(
                f"Image {k + 1} 3D spatial coordinates: {coord_tokens}."
            )

        # ── Build final prompt with dense coord tokens ──────────────────────────
        content.append({
            "type": "text",
            "text": (
                " ".join(pose_sentences)
                + " "
                + " ".join(coord_sentences)
                + " "
                + _question
            ),
        })
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": _answer}],
            tokenize=False, add_generation_prompt=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        # Supervise only the answer tokens (tail of sequence).
        # Mask everything before — including <think> block — so the model's
        # reasoning behaviour is not suppressed.
        suffix_ids = self.processor.tokenizer(
            _answer + "<|im_end|>\n", add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

        # ── 3D position maps ──────────────────────────────────────────────────
        image_xyz = None
        if p3d_path is not None:
            try:
                d3d     = np.load(p3d_path)
                thw_all = proc_out["image_grid_thw"]   # (N, 3)
                sms     = self.spatial_merge_size
                xyz_list = []
                n_3d = int(d3d["n_frames"])
                for k in range(min(N, n_3d)):
                    xyz_raw   = d3d[f"frame_{k}_xyz"]
                    valid_raw = d3d.get(f"frame_{k}_valid")
                    thw_k     = thw_all[k]
                    llm_h     = int(thw_k[1]) // sms
                    llm_w     = int(thw_k[2]) // sms
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=valid_raw))
                for k in range(len(xyz_list), N):
                    thw_k = thw_all[k]
                    llm_h = int(thw_k[1]) // sms
                    llm_w = int(thw_k[2]) // sms
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                image_xyz = xyz_list
            except Exception as exc:
                log.debug(f"3D_pos load failed for {p3d_path}: {exc}")
                image_xyz = None

        return {
            **proc_out,
            "gt_transforms": gt_transforms,
            "image_xyz":     image_xyz,
            "labels":        labels,
        }


def collate_fn(batch):
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# ── model building ────────────────────────────────────────────────────────────

def build_model(
    model_path:         str,
    pose_token_id:      int,
    coord_token_id:     int,
    image_token_id:     int,
    spatial_merge_size: int,
    lora_rank:          int = 16,
    freeze_vision:      bool = True,
    skip_layers:        tuple[int, ...] = (-1,),
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
) -> CoordinatePlusModel:
    """
    Load SpaForConditionalGeneration, patch 4D M-RoPE, apply LoRA,
    and attach PoseRegressionHead + CoordinateReadoutHead.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    section_size  = sum(orig_section) // 4
    config.text_config.rope_scaling["mrope_section"] = [section_size] * 4
    log.info(
        f"mrope_section: {orig_section} → {[section_size]*4}  "
        f"(4D M-RoPE for spatial tokens)"
    )

    spa = SpaForConditionalGeneration.from_pretrained(
        model_path,
        config              = config,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "sdpa",
    )

    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    spa.resize_token_embeddings(old_vocab + 2)
    log.info(f"Embedding table: {old_vocab} → {old_vocab + 2} (added <pose>, <coord>)")

    if freeze_vision:
        for p in spa.model.visual.parameters():
            p.requires_grad_(False)
        log.info("Vision encoder frozen.")

    lora_cfg = LoraConfig(
        r              = lora_rank,
        lora_alpha     = lora_rank * 2,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout = 0.05,
        bias         = "none",
        task_type    = TaskType.CAUSAL_LM,
    )
    spa = get_peft_model(spa, lora_cfg)
    spa.print_trainable_parameters()

    # Gradient checkpointing: trade ~20% speed for ~60% activation memory savings
    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Verify checkpointing propagated to the language model
    lm = spa.model.model.language_model if hasattr(spa.model, 'model') else spa.model.language_model
    gc_flag = getattr(lm, 'gradient_checkpointing', False)
    log.info(f"Gradient checkpointing enabled. language_model.gradient_checkpointing={gc_flag}")
    if not gc_flag:
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    hidden_dim = config.text_config.hidden_size

    # Pose head: uses skip-layer concat
    pose_input_dim = hidden_dim * len(skip_layers)
    pose_head = PoseRegressionHead(input_dim=pose_input_dim).to(torch.bfloat16)
    log.info(f"PoseRegressionHead input_dim={pose_input_dim} "
             f"(skip_layers={list(skip_layers)}, hidden={hidden_dim})")

    # Coordinate regression head: dense MLP decoder
    coord_head = CoordinateRegressionHead(hidden_dim=hidden_dim).to(torch.bfloat16)
    log.info(f"CoordinateRegressionHead hidden_dim={hidden_dim}")

    return CoordinatePlusModel(
        spa_model          = spa,
        pose_head          = pose_head,
        coord_head         = coord_head,
        pose_token_id      = pose_token_id,
        coord_token_id     = coord_token_id,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        skip_layers        = skip_layers,
        answer_weight      = answer_weight,
        coord_weight       = coord_weight,
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    _env_rank = os.environ.get("LOCAL_RANK")
    if _env_rank is not None:
        local_rank = int(_env_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        log.info(f"DDP: local_rank={local_rank}  world_size={world_size}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Single-GPU / CPU mode")

    # ── processor + tokeniser ─────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [POSE_TOKEN, COORD_TOKEN]}
    )
    pose_token_id  = tokenizer.convert_tokens_to_ids(POSE_TOKEN)
    coord_token_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
    rank0_print(f"<pose>  token id = {pose_token_id}")
    rank0_print(f"<coord> token id = {coord_token_id}")

    # image_token_id: <|image_pad|> in Qwen-VL tokeniser
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rank0_print(f"<|image_pad|> token id = {image_token_id}")

    # ── spatial_merge_size from vision config ─────────────────────────────────
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        args.model_path,
        pose_token_id      = pose_token_id,
        coord_token_id     = coord_token_id,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        lora_rank          = args.lora_rank,
        freeze_vision      = not args.train_vision,
        skip_layers        = tuple(args.skip_layers),
        answer_weight      = args.answer_weight,
        coord_weight       = args.coord_weight,
    )
    model = model.to(device)
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"[MEM] After model.to(device): {mem_gb:.2f} GiB allocated")

    if args.pos3d_dir == "":
        args.pos3d_dir = None
    rank0_print(f"pos3d_dir = {args.pos3d_dir}")

    # ── DDP ───────────────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)
        _model = model.module
    else:
        _model = model

    # ── dataset / loader ──────────────────────────────────────────────────────
    dataset = SPARDataset(
        json_path          = args.json_path,
        spar_root          = SPAR_ROOT,
        reconstruct_dir    = RECONSTRUCT_DIR,
        processor          = processor,
        pose_token_id      = pose_token_id,
        max_images         = args.max_images,
        pos3d_dir          = args.pos3d_dir,
        spatial_merge_size = spatial_merge_size,
        max_samples        = args.max_samples,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size,
                           rank=local_rank, shuffle=True)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = (sampler is None),
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        sampler     = sampler,
    )

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB (rank 0 only) ───────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and args.wandb_project and local_rank == 0
    if use_wandb:
        wandb.init(
            entity  = args.wandb_entity or None,
            project = args.wandb_project,
            name    = args.wandb_run_name or None,
            config  = vars(args),
            dir     = args.output_dir,
        )
        log.info(f"WandB run: {wandb.run.name}  project: {args.wandb_project}")
    elif args.wandb_project and not _WANDB_AVAILABLE and local_rank == 0:
        log.warning("wandb not installed — logging disabled. `pip install wandb`")

    model.train()

    global_step = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(loader):

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            gt_transforms  = batch["gt_transforms"].to(device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(device) for xyz in image_xyz]

            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            if step == 0 and local_rank == 0:
                n_img_tok = (input_ids[0] == image_token_id).sum().item()
                n_coord_tok = (input_ids[0] == coord_token_id).sum().item()
                pv_shape = tuple(pixel_values.shape) if pixel_values is not None else None
                mem_before = torch.cuda.memory_allocated(device) / 1e9
                log.info(
                    f"[MEM] Step 0: seq_len={input_ids.shape[1]}, "
                    f"img_tokens={n_img_tok}, coord_tokens={n_coord_tok}, "
                    f"pixel_values={pv_shape}, "
                    f"image_grid_thw={image_grid_thw}, "
                    f"mem_before_fwd={mem_before:.2f} GiB"
                )

            try:
                _, loss, loss_dict = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    pixel_values   = pixel_values,
                    image_grid_thw = image_grid_thw,
                    gt_transforms  = gt_transforms,
                    image_xyz      = image_xyz,
                    cycle_weight   = args.cycle_weight,
                    labels         = labels,
                )
            except Exception as exc:
                log.warning(f"[rank{local_rank}] Step {step} skipped: {exc}")
                continue

            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: no <pose> token, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            if loss_dict:
                for k, v in loss_dict.items():
                    running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if local_rank == 0:
                    avg_loss = running_loss / args.grad_accum
                    avg_loss_dict = {
                        k: v / args.grad_accum
                        for k, v in running_loss_dict.items()
                    }
                    running_loss = 0.0
                    running_loss_dict.clear()
                    current_lr = scheduler.get_last_lr()[0]
                    detail = "  ".join(
                        f"{k}={v:.4f}" for k, v in avg_loss_dict.items()
                    )
                    log.info(
                        f"epoch={epoch+1:02d}  step={global_step:05d}  "
                        f"loss={avg_loss:.4f}"
                        + (f"  ({detail})" if detail else "")
                        + f"  lr={current_lr:.2e}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr":   current_lr,
                                "epoch":      epoch + 1,
                                **{f"train/{k}": v for k, v in avg_loss_dict.items()},
                            },
                            step=global_step,
                        )

                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir,
                                         global_step)
                else:
                    running_loss = 0.0
                    running_loss_dict.clear()

    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step,
                         suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(
    model:      CoordinatePlusModel,
    tokenizer,
    output_dir: str,
    step:       int,
    suffix:     str = "",
) -> None:
    tag  = f"step_{step}" + (f"_{suffix}" if suffix else "")
    ckpt = os.path.join(output_dir, tag)
    os.makedirs(ckpt, exist_ok=True)

    model.spa_model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    torch.save(
        model.pose_head.state_dict(),
        os.path.join(ckpt, "pose_head.pt"),
    )
    torch.save(
        model.coord_head.state_dict(),
        os.path.join(ckpt, "coord_head.pt"),
    )
    log.info(f"Checkpoint saved → {ckpt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of SpaForConditionalGeneration "
                    "with pose + answer + coordinate supervision."
    )
    p.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"),
        help="Path to Qwen3.5-VL checkpoint",
    )
    p.add_argument(
        "--json_path",
        default=os.path.join(SPAR_ROOT, "train_10k.json"),
        help="Path to SPAR training JSON",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_coordinate"),
    )
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_rank",   type=int,   default=16)
    p.add_argument("--max_images",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--save_steps",  type=int,   default=200)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument(
        "--train_vision",
        action="store_true",
        help="Unfreeze the vision encoder (ViT) for fine-tuning",
    )
    p.add_argument(
        "--pos3d_dir",
        default=POS3D_DIR,
        help="Directory of 3D_pos *.npz files. Pass empty string to disable.",
    )
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-1],
        help="LLM layer indices concatenated before PoseRegressionHead. "
             "e.g. --skip_layers -4 -1",
    )
    p.add_argument(
        "--cycle_weight",
        type=float, default=0.1,
        help="Weight for rotation cycle-consistency loss (0 to disable).",
    )
    p.add_argument(
        "--answer_weight",
        type=float, default=1.0,
        help="Weight for the LM answer-prediction loss.",
    )
    p.add_argument(
        "--coord_weight",
        type=float, default=1.0,
        help="Weight for the per-patch coordinate prediction loss.",
    )
    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project",  default="", help="WandB project name.")
    p.add_argument("--wandb_entity",   default="", help="WandB entity.")
    p.add_argument("--wandb_run_name", default="", help="WandB run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
