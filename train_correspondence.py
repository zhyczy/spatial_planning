"""
train_correspondence.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) to predict
relative camera transformations between image pairs in a scene.

Architecture:
  SpaCorrespondenceModel
    ├── SpaForConditionalGeneration  [backbone + LoRA adapters]
    │    ├── SpaVisionModel (ViT, frozen)
    │    └── SpaModel (LLM + 4D M-RoPE)
    └── PoseRegressionHead           [MLP: hidden_dim → 9D]

Training strategy:
  - Each SPAR entry = one scene with N images
  - Prompt inserts one <pose> token per ordered pair (i→j), i≠j  → A(N,2) tokens
    e.g. N=2: 2 tokens (0→1, 1→0); N=3: 6 tokens (all permutations)
  - Hidden state at each <pose> token → PoseRegressionHead → (6D rot, 3D trans)
  - Loss = geodesic rotation loss + L1 translation loss

Coordinate convention:
  - Predicts T_{i→j} = relative_transforms[i, j] from the .npz GT files
  - T = [R | t; 0 | 1],  parameterised as 6D-rotation + 3D-translation

Usage:
  python train_correspondence.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_correspondence
"""

import argparse
import logging
import math
import os
import sys
from itertools import combinations

import wandb
_WANDB_AVAILABLE = True


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
SPAR_ROOT = os.path.join(
    _ROOT, "datasets/train/SPAR_7M/spar"
)
RECONSTRUCT_DIR = os.path.join(SPAR_ROOT, "reconstruct")
POS3D_DIR       = os.path.join(SPAR_ROOT, "3D_pos")
POSE_TOKEN = "<pose>"


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


# ── rotation utilities ────────────────────────────────────────────────────────

def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D continuous rotation → 3×3 rotation matrix via Gram-Schmidt.
    (..., 6)  →  (..., 3, 3)

    The first two columns of R are encoded in r6d[..., :3] and r6d[..., 3:6].
    Reference: Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.
    """
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    # eps=1e-6 guards against near-zero vectors at initialisation
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as columns: R = [b1 | b2 | b3]
    return torch.stack([b1, b2, b3], dim=-1)          # (..., 3, 3)


def geodesic_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean rotation-angle loss between predicted and GT rotation matrices.
    R_pred, R_gt: (N, 3, 3)
    """
    # R_diff = R_pred @ R_gt^T  (should be I when perfect)
    R_diff = torch.bmm(R_pred, R_gt.transpose(-1, -2))         # (N, 3, 3)
    trace  = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)         # (N,)
    cos    = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()


def cycle_consistency_loss(R_preds: torch.Tensor, N: int) -> torch.Tensor:
    """
    Rotation cycle-consistency loss over all C(N, 3) triangles.

    For each unordered triple {a, b, c}, penalises the deviation of
    R_{a→b} @ R_{b→c} @ R_{c→a} from the identity:
        loss = mean_triple  ||R_{a→b} R_{b→c} R_{c→a} - I||_F

    R_preds: (K, 3, 3)  where K = N*(N-1), ordered by
             [(i,j) for i in range(N) for j in range(N) if i != j]
    Returns scalar (0 when N < 3).
    """
    if N < 3:
        return R_preds.new_tensor(0.0)

    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    p2k   = {p: k for k, p in enumerate(pairs)}
    I3    = torch.eye(3, device=R_preds.device, dtype=R_preds.dtype)

    errs = []
    for a, b, c in combinations(range(N), 3):
        cycle = R_preds[p2k[(a, b)]] @ R_preds[p2k[(b, c)]] @ R_preds[p2k[(c, a)]]
        errs.append(torch.norm(cycle - I3, p="fro"))
    return torch.stack(errs).mean()


# ── model components ──────────────────────────────────────────────────────────

class PoseRegressionHead(nn.Module):
    """
    MLP regression head: input_dim → 9D pose vector.
    Output layout: first 6 dims = 6D rotation, last 3 dims = translation.

    input_dim = hidden_size * len(skip_layers), e.g.:
      skip_layers=[-1]     → input_dim = 2560  (last layer only)
      skip_layers=[-4,-1]  → input_dim = 5120  (concat of two layers)
    """

    def __init__(self, input_dim: int = 2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 9),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (input_dim,) or (N, input_dim) → (9,) or (N, 9)"""
        return self.mlp(latent)


class SpaCorrespondenceModel(nn.Module):
    """
    SpaForConditionalGeneration (+ LoRA) wrapped with a PoseRegressionHead.

    Forward:
      1. Runs the backbone with output_hidden_states=True.
      2. Finds all <pose> token positions in input_ids.
      3. Concatenates hidden states from skip_layers at each <pose> position
         (skip connection from intermediate layers preserves geometric detail).
      4. Passes each latent through PoseRegressionHead → 9D prediction.
      5. Optionally computes geodesic + translation + cycle-consistency loss.

    skip_layers: list of layer indices into outputs.hidden_states, e.g.
      [-1]      → last layer only  (input_dim = hidden_size)
      [-4, -1]  → concat 4th-from-last + last  (input_dim = 2 * hidden_size)

    Returns:
      preds_9d : (K, 9) tensor  —  K = number of <pose> tokens found
      loss     : scalar or None
    """

    def __init__(
        self,
        spa_model:      nn.Module,
        pose_head:      PoseRegressionHead,
        pose_token_id:  int,
        skip_layers:    tuple[int, ...] = (-1,),
    ):
        super().__init__()
        self.spa_model     = spa_model
        self.pose_head     = pose_head
        self.pose_token_id = pose_token_id
        self.skip_layers   = list(skip_layers)

    def forward(
        self,
        input_ids:      torch.Tensor,                    # (1, seq_len)
        attention_mask: torch.Tensor,                    # (1, seq_len)
        pixel_values:   torch.Tensor | None,             # (total_patches, C, H, W)
        image_grid_thw: torch.Tensor | None,             # (num_images, 3)
        gt_transforms:  torch.Tensor | None = None,      # (K, 4, 4)
        image_xyz:      list | None = None,              # list of (llm_H, llm_W, 3)
        coord_scale:    float = 100.0,
        cycle_weight:   float = 0.0,
        **kwargs,
    ):
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = True,
            return_dict          = True,
            image_xyz            = image_xyz,
            coord_scale          = coord_scale,
            **kwargs,
        )

        # Skip connection: concatenate hidden states from selected layers
        # outputs.hidden_states: tuple[num_layers+1] of (1, seq_len, hidden_dim)
        if len(self.skip_layers) == 1:
            hidden = outputs.hidden_states[self.skip_layers[0]]    # (1, seq, H)
        else:
            hidden = torch.cat(
                [outputs.hidden_states[i] for i in self.skip_layers], dim=-1
            )                                                        # (1, seq, H*n)

        # Locate <pose> tokens (batch item 0)
        pose_positions = (input_ids[0] == self.pose_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(pose_positions) == 0:
            return None, None

        # Predict pose at each <pose> token
        latents  = hidden[0, pose_positions]       # (K, input_dim)
        preds_9d = self.pose_head(latents)          # (K, 9)

        if gt_transforms is None:
            return preds_9d, None

        K  = preds_9d.shape[0]
        gt = gt_transforms[:K].to(preds_9d.device)  # (K, 4, 4)

        R_gt   = gt[:, :3, :3]                       # (K, 3, 3)
        t_gt   = gt[:, :3,  3]                       # (K, 3)
        R_pred = rot6d_to_rotmat(preds_9d[:, :6])    # (K, 3, 3)
        t_pred = preds_9d[:, 6:]                     # (K, 3)

        loss = geodesic_loss(R_pred, R_gt) + F.l1_loss(t_pred, t_gt)

        # Cycle consistency: R_{a→b} R_{b→c} R_{c→a} = I  for all triples
        if cycle_weight > 0.0:
            N = round((1 + math.sqrt(1 + 4 * K)) / 2)
            if N * (N - 1) == K:                     # valid A(N,2) count
                loss = loss + cycle_weight * cycle_consistency_loss(R_pred, N)

        return preds_9d, loss


# ── dataset ───────────────────────────────────────────────────────────────────

class SPARDataset(Dataset):
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
        max_images:          int = 4,
        pos3d_dir:           str | None = None,
        spatial_merge_size:  int = 2,
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

        self.spar_root           = spar_root
        self.processor           = processor
        self.pose_token_id       = pose_token_id
        self.max_images          = max_images
        self.pos3d_dir           = pos3d_dir
        self.spatial_merge_size  = spatial_merge_size
        log.info(f"SPARDataset: {len(self.samples)} valid entries "
                 f"(out of {len(entries)} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, npz_path, p3d_path = self.samples[idx]

        # ── load images ───────────────────────────────────────────────────────
        images = []
        for rel in entry["image"]:
            full = os.path.join(self.spar_root, rel)
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
            return self.__getitem__((idx + 1) % len(self))

        images = images[:N]

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

        pose_sentences = [
            f"The camera pose of image {j + 1} relative to image {i + 1} is "
            f"{POSE_TOKEN}."
            for (i, j) in pairs
        ]
        content.append({"type": "text", "text": " ".join(pose_sentences)})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # ── tokenise + encode images ──────────────────────────────────────────
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
        }


def collate_fn(batch):
    """
    Identity collation for batch_size=1.
    The processor already returns tensors with the correct shapes
    (input_ids: (1, seq_len), pixel_values: (total_patches, C, H, W), …).
    """
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# ── model building ────────────────────────────────────────────────────────────

def build_model(
    model_path:    str,
    pose_token_id: int,
    lora_rank:     int = 16,
    freeze_vision: bool = True,
    skip_layers:   tuple[int, ...] = (-1,),
) -> SpaCorrespondenceModel:
    """
    Load SpaForConditionalGeneration, patch 4D M-RoPE, apply LoRA,
    resize embeddings for <pose>, and attach PoseRegressionHead.
    """
    # ── config: switch mrope_section to 4 equal parts ─────────────────────────
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    section_size  = sum(orig_section) // 4          # e.g. 32 // 4 = 8
    config.text_config.rope_scaling["mrope_section"] = [section_size] * 4
    log.info(
        f"mrope_section: {orig_section} → {[section_size]*4}  "
        f"(4D M-RoPE for spatial tokens)"
    )

    # ── load model ─────────────────────────────────────────────────────────────
    spa = SpaForConditionalGeneration.from_pretrained(
        model_path,
        config             = config,
        torch_dtype        = torch.bfloat16,
        device_map         = "auto",
        attn_implementation= "flash_attention_2",
    )

    # ── resize embedding table for new <pose> token ───────────────────────────
    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    spa.resize_token_embeddings(old_vocab + 1)
    log.info(f"Embedding table: {old_vocab} → {old_vocab + 1} (added <pose>)")

    # ── optionally freeze vision encoder ─────────────────────────────────────
    if freeze_vision:
        for p in spa.model.visual.parameters():
            p.requires_grad_(False)
        log.info("Vision encoder frozen.")

    # ── apply LoRA to the language model ──────────────────────────────────────
    lora_cfg = LoraConfig(
        r              = lora_rank,
        lora_alpha     = lora_rank * 2,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    spa = get_peft_model(spa, lora_cfg)
    spa.print_trainable_parameters()

    # ── pose regression head ──────────────────────────────────────────────────
    hidden_dim = config.text_config.hidden_size
    input_dim  = hidden_dim * len(skip_layers)   # concat of skip_layers features
    pose_head  = PoseRegressionHead(input_dim=input_dim)
    log.info(f"PoseRegressionHead input_dim={input_dim} "
             f"(skip_layers={list(skip_layers)}, hidden={hidden_dim})")

    return SpaCorrespondenceModel(spa, pose_head, pose_token_id,
                                  skip_layers=skip_layers)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── processor + tokeniser ─────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [POSE_TOKEN]})
    pose_token_id = tokenizer.convert_tokens_to_ids(POSE_TOKEN)
    log.info(f"<pose> token id = {pose_token_id}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        args.model_path,
        pose_token_id  = pose_token_id,
        lora_rank      = args.lora_rank,
        freeze_vision  = not args.train_vision,
        skip_layers    = tuple(args.skip_layers),
    )
    # Move pose head to device (backbone is already placed by device_map)
    first_param_device = next(
        p.device for p in model.spa_model.parameters() if p.requires_grad
    )
    model.pose_head = model.pose_head.to(first_param_device)

    # Normalise --pos3d_dir (empty string → None to disable 3D pos loading)
    if args.pos3d_dir == "":
        args.pos3d_dir = None
    log.info(f"pos3d_dir = {args.pos3d_dir}")

    # ── resolve spatial_merge_size from vision config ─────────────────────────
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    log.info(f"spatial_merge_size = {spatial_merge_size}")

    # ── dataset / loader ──────────────────────────────────────────────────────
    dataset = SPARDataset(
        json_path           = args.json_path,
        spar_root           = SPAR_ROOT,
        reconstruct_dir     = RECONSTRUCT_DIR,
        processor           = processor,
        pose_token_id       = pose_token_id,
        max_images          = args.max_images,
        pos3d_dir           = args.pos3d_dir,
        spatial_merge_size  = spatial_merge_size,
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
    )

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and args.wandb_project
    if use_wandb:
        wandb.init(
            entity  = args.wandb_entity or None,
            project = args.wandb_project,
            name    = args.wandb_run_name or None,
            config  = vars(args),
            dir     = args.output_dir,
        )
        log.info(f"WandB run: {wandb.run.name}  project: {args.wandb_project}")
    elif args.wandb_project and not _WANDB_AVAILABLE:
        log.warning("wandb not installed — logging disabled. `pip install wandb`")

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):

            # ── move batch to device ──────────────────────────────────────────
            input_ids      = batch["input_ids"].to(first_param_device)
            attention_mask = batch["attention_mask"].to(first_param_device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            gt_transforms  = batch["gt_transforms"].to(first_param_device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(first_param_device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(first_param_device)

            # Move 3D position maps to device (list of tensors or None)
            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(first_param_device) for xyz in image_xyz]

            # ── forward + loss ────────────────────────────────────────────────
            try:
                _, loss = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    pixel_values   = pixel_values,
                    image_grid_thw = image_grid_thw,
                    gt_transforms  = gt_transforms,
                    image_xyz      = image_xyz,
                    cycle_weight   = args.cycle_weight,
                )
            except Exception as exc:
                log.warning(f"Step {step} skipped: {exc}")
                continue

            if loss is None:
                log.warning(f"Step {step}: no <pose> token found, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            # ── gradient accumulation ─────────────────────────────────────────
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / args.grad_accum
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]
                log.info(
                    f"epoch={epoch+1:02d}  step={global_step:05d}  "
                    f"loss={avg_loss:.4f}  "
                    f"lr={current_lr:.2e}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr":   current_lr,
                            "epoch":      epoch + 1,
                        },
                        step=global_step,
                    )

                # ── checkpoint ────────────────────────────────────────────────
                if global_step % args.save_steps == 0:
                    _save_checkpoint(model, tokenizer, args.output_dir,
                                     global_step)

    # Final checkpoint
    _save_checkpoint(model, tokenizer, args.output_dir, global_step,
                     suffix="final")
    log.info("Training complete.")
    if use_wandb:
        wandb.finish()


def _save_checkpoint(
    model: SpaCorrespondenceModel,
    tokenizer,
    output_dir: str,
    step: int,
    suffix: str = "",
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
    log.info(f"Checkpoint saved → {ckpt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of SpaForConditionalGeneration "
                    "for relative camera pose prediction."
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
        default=os.path.join(_ROOT, "checkpoints/spa_correspondence"),
    )
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lora_rank",    type=int,   default=16,
                   help="LoRA rank r")
    p.add_argument("--max_images",   type=int,   default=4,
                   help="Max images per scene (memory budget)")
    p.add_argument("--grad_accum",   type=int,   default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument(
        "--train_vision",
        action="store_true",
        help="Also unfreeze the vision encoder (ViT) for fine-tuning",
    )
    p.add_argument(
        "--pos3d_dir",
        default=POS3D_DIR,
        help="Directory of 3D_pos *.npz files (per-pixel XYZ for 4D M-RoPE). "
             "Pass empty string to disable.",
    )
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-8, -4, -1],
        help="LLM layer indices whose hidden states are concatenated before "
             "PoseRegressionHead. e.g. --skip_layers -4 -1 (default: -8 -4 -1)",
    )
    p.add_argument(
        "--cycle_weight",
        type=float, default=0.1,
        help="Weight for rotation cycle-consistency loss (0 to disable). "
             "Active only when N >= 3 views.",
    )
    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--wandb_project",
        default="",
        help="WandB project name. Leave empty to disable WandB logging.",
    )
    p.add_argument(
        "--wandb_entity",
        default="",
        help="WandB entity (username or team name). Leave empty to use default.",
    )
    p.add_argument(
        "--wandb_run_name",
        default="",
        help="WandB run name (optional; auto-generated when empty).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
