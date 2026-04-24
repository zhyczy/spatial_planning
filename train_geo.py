"""
train_geo.py

Geometry-focused fine-tuning, implementing the MVP recipe of
md/discussion/learning_real_3d_geometry.md (§1.A + §1.D, on top of the
decouple architecture from train_correspondence.py --decouple):

  lm_loss (answer CE)
    + contrast_weight * contrast_loss   ← §1.A multi-view triplet correspondence
    + match_weight    * match_loss      ← §1.D xyz-image match classifier

Position-embedding architecture — EXACTLY train_correspondence.py --decouple:
    - backbone:                SpaDecForConditionalGeneration
    - mrope_section:           [11, 11, 10] UNCHANGED  (Qwen original 3D M-RoPE
                               in rotary 64 dims, partial_rotary_factor=0.25)
    - XYZ RoPE:                new 66 dims in pass-through region (dims 64..129),
                               rope_theta=10000, Cartesian (x, y, z)
    - patch_attention_layers_dec applied after LoRA wrapping
    - polar=False (GeoModel never passes polar; we stay Cartesian)
    - coord_scale=100.0
train_geo.py does NOT modify the position-embedding path at all; it only adds
the §1.A contrast head (no params — triplet loss on hidden states) and the
§1.D match head (single nn.Linear(hidden_dim, 2)) on top.

1.A — multi-view contrastive correspondence:
    Within one sample's N views (MindCube has 4), for each ordered pair (i, j),
    build positive pairs (p, q) where patch p of image i and patch q of image j
    point to the same 3D location via an xyz L2 match below `contrast_eps`.
    A negative is a random other patch k ≠ p in image i. Minimize
        max(0, margin + ||h_i[p] - h_j[q]||² - ||h_i[p] - h_i[k]||²)
    on the last-layer vision-token hidden states.

1.D — xyz-image match classifier (anti-shortcut):
    With probability `match_prob`, permute the per-image xyz assignment inside
    the sample (image k gets image π(k)'s xyz). A small binary head reads the
    per-image pooled last-layer hidden state and predicts {matches, mismatches}.
    Cross-entropy. On permuted steps the contrast_loss is SKIPPED because
    positive-pair finding relies on correct xyz.
"""

import argparse
import logging
import os
import random
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import (
    SpaDecForConditionalGeneration,
    patch_attention_layers_dec,
)
from src.dataset import (
    MindCube_Train_Dataset,
    Eval_Dataset_Coord,
)

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


def collate_fn(batch):
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# ── model ─────────────────────────────────────────────────────────────────────

class GeoModel(nn.Module):
    """
    Decoupled-xyz backbone + two geometric auxiliary objectives on last-layer
    vision-token hidden states. See module docstring for losses.
    """

    def __init__(
        self,
        spa_model:          nn.Module,
        hidden_dim:         int,
        image_token_id:     int,
        spatial_merge_size: int,
        lm_weight:          float = 1.0,
        contrast_weight:    float = 1.0,
        match_weight:       float = 0.3,
        contrast_margin:    float = 0.5,
        contrast_eps:       float = 0.05,
        n_contrast_anchors: int   = 64,
        match_prob:         float = 0.5,
        coord_scale:        float = 100.0,
        use_contrast:       bool  = True,
        use_match:          bool  = True,
    ):
        super().__init__()
        self.spa_model          = spa_model
        self.image_token_id     = image_token_id
        self.spatial_merge_size = spatial_merge_size
        self.lm_weight          = lm_weight
        self.contrast_weight    = contrast_weight
        self.match_weight       = match_weight
        self.contrast_margin    = contrast_margin
        self.contrast_eps       = contrast_eps
        self.n_contrast_anchors = n_contrast_anchors
        self.match_prob         = match_prob
        self.coord_scale        = coord_scale
        self.use_contrast       = use_contrast
        self.use_match          = use_match

        # §1.D match head: per-image binary classifier on pooled vision hidden.
        # Only instantiated when match is enabled — saves params from being
        # registered with DDP and avoids unused-parameter noise.
        self.match_head: nn.Linear | None = (
            nn.Linear(hidden_dim, 2).to(torch.bfloat16) if use_match else None
        )

        # Capture post-norm last hidden state via lm_head pre-hook. Lets us keep
        # output_hidden_states=False (saves activation memory; also sidesteps
        # the NaN bug observed on SpaDecTextModel + gradient checkpointing when
        # output_hidden_states=True).
        self._last_hidden: torch.Tensor | None = None
        for name, mod in self.spa_model.named_modules():
            if name.endswith("lm_head"):
                mod.register_forward_pre_hook(self._capture_last_hidden)
                break

    def _capture_last_hidden(self, module, args):
        self._last_hidden = args[0]

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        image_xyz:      list | None,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        device = input_ids.device
        N = len(image_xyz) if image_xyz is not None else 0

        # ── §1.D: sample permutation (only when match is enabled) ────────────
        do_permute = False
        perm = list(range(N))
        match_labels = torch.ones(max(N, 1), dtype=torch.long, device=device)
        if (self.use_match and self.training and N >= 2
                and random.random() < self.match_prob):
            # Non-identity permutation
            for _ in range(8):
                cand = torch.randperm(N).tolist()
                if any(cand[i] != i for i in range(N)):
                    perm = cand
                    do_permute = True
                    break
            if do_permute:
                match_labels = torch.tensor(
                    [int(perm[i] == i) for i in range(N)],
                    dtype=torch.long, device=device,
                )

        image_xyz_fwd = (
            [image_xyz[p] for p in perm] if (do_permute and image_xyz is not None)
            else image_xyz
        )

        # ── Backbone forward ─────────────────────────────────────────────────
        outputs = self.spa_model(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            image_xyz            = image_xyz_fwd,
            coord_scale          = self.coord_scale,
            output_hidden_states = False,
            return_dict          = True,
            **kwargs,
        )
        logits = outputs.logits
        hidden = self._last_hidden   # (1, seq_len, D)
        del outputs

        loss_dict: dict = {}

        # ── LM answer loss ───────────────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].to(logits.device)
            mask = shift_labels[0] != -100
            _sl = shift_logits[0, mask]
            _sb = shift_labels[0, mask]
            if _sl.numel() > 0:
                lm_loss = F.cross_entropy(_sl, _sb)
                loss_dict["lm_loss"] = lm_loss.item()
                if not self.training:
                    loss_dict["acc"] = 1.0 if _sl[0].argmax(-1).item() == int(_sb[0].item()) else 0.0

        # ── Per-image vision-token hidden slices ──────────────────────────────
        # Only needed when contrast or match is enabled.
        per_image = []   # list[(llm_h, llm_w, Tensor[h*w, D])]
        need_slices = (self.use_contrast or self.use_match)
        if need_slices and image_grid_thw is not None and hidden is not None:
            vis_pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
            sms = self.spatial_merge_size
            start = 0
            for k in range(min(N, len(image_grid_thw))):
                thw = image_grid_thw[k]
                llm_h = int(thw[1]) // sms
                llm_w = int(thw[2]) // sms
                n_tok = llm_h * llm_w
                if start + n_tok > len(vis_pos):
                    break
                slice_h = hidden[0, vis_pos[start:start + n_tok]]  # (h*w, D)
                per_image.append((llm_h, llm_w, slice_h))
                start += n_tok

        # ── §1.D match head loss ─────────────────────────────────────────────
        match_loss = None
        if (self.use_match and self.match_head is not None
                and len(per_image) >= 2 and len(per_image) == N):
            pooled = torch.stack(
                [ph.mean(dim=0) for (_, _, ph) in per_image], dim=0
            )                                                     # (N, D)
            m_logits = self.match_head(pooled.to(self.match_head.weight.dtype))
            match_loss = F.cross_entropy(m_logits, match_labels[:N])
            loss_dict["match_loss"] = match_loss.item()
            if not self.training:
                loss_dict["match_acc"] = (
                    (m_logits.argmax(-1) == match_labels[:N]).float().mean().item()
                )

        # ── §1.A contrastive correspondence loss ─────────────────────────────
        # Skip on permuted samples: positive-pair finding depends on correct xyz.
        contrast_loss = None
        if (self.use_contrast and (not do_permute)
                and len(per_image) >= 2 and image_xyz is not None):
            contrast_loss = self._contrastive_loss(per_image, image_xyz)
            if contrast_loss is not None:
                loss_dict["contrast_loss"] = contrast_loss.item()

        # ── Combine ──────────────────────────────────────────────────────────
        total = None
        if lm_loss is not None:
            total = self.lm_weight * lm_loss
        if contrast_loss is not None:
            term = self.contrast_weight * contrast_loss
            total = term if total is None else total + term
        if match_loss is not None:
            term = self.match_weight * match_loss
            total = term if total is None else total + term

        return None, total, loss_dict

    def _contrastive_loss(self, per_image, image_xyz):
        """
        For each ordered view pair (i, j):
            - find patches p in image i whose nearest neighbor q in image j
              (by xyz L2) is within `contrast_eps`
            - sample up to `n_contrast_anchors` such anchors
            - negative k: random other patch in image i, k ≠ p
            - triplet margin loss with L2 distance

        xyz is used ONLY for correspondence indexing (no grad); hidden_states
        carry all the gradient.
        """
        device = per_image[0][2].device
        losses: list[torch.Tensor] = []

        for i in range(len(per_image)):
            hi = per_image[i][2]
            xyz_i = image_xyz[i].reshape(-1, 3).to(
                device=device, dtype=torch.float32
            )
            n_i = hi.shape[0]
            if xyz_i.shape[0] != n_i or n_i < 2:
                continue
            for j in range(len(per_image)):
                if i == j:
                    continue
                hj = per_image[j][2]
                xyz_j = image_xyz[j].reshape(-1, 3).to(
                    device=device, dtype=torch.float32
                )
                n_j = hj.shape[0]
                if xyz_j.shape[0] != n_j or n_j < 1:
                    continue

                with torch.no_grad():
                    dmat = torch.cdist(xyz_i, xyz_j)              # (n_i, n_j)
                    dmin, qmin = dmat.min(dim=1)                  # (n_i,)
                    valid_idx = (dmin < self.contrast_eps).nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    if valid_idx.numel() > self.n_contrast_anchors:
                        sel = torch.randperm(valid_idx.numel(), device=device)[
                            : self.n_contrast_anchors
                        ]
                        valid_idx = valid_idx[sel]
                    p = valid_idx
                    q = qmin[p]
                    # Negative: random other patch in image i, avoid collision with p
                    k = torch.randint(0, n_i, p.shape, device=device)
                    same = (k == p)
                    k = torch.where(same, (k + 1) % n_i, k)

                ha = hi[p].float()
                hp = hj[q].float()
                hn = hi[k].float()
                dap = (ha - hp).pow(2).sum(dim=-1)
                dan = (ha - hn).pow(2).sum(dim=-1)
                losses.append(F.relu(self.contrast_margin + dap - dan).mean())

        if not losses:
            return None
        return torch.stack(losses).mean()


# ── model building ────────────────────────────────────────────────────────────

def build_spa_model(
    model_path:    str,
    lora_rank:     int  = 16,
    freeze_vision: bool = True,
) -> nn.Module:
    """
    SpaDecForConditionalGeneration + LoRA + decouple attention patch.
    Mirrors train_correspondence.py --decouple (Cartesian XYZ RoPE, θ=10000).
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    log.info(
        f"mrope_section: {orig_section} (UNCHANGED) + XYZ RoPE (66 dims, θ=10000) "
        f"in pass-through region [Cartesian (x, y, z)]"
    )
    spa = SpaDecForConditionalGeneration.from_pretrained(
        model_path,
        config              = config,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "sdpa",
    )
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

    n = patch_attention_layers_dec(spa)
    log.info(f"Wrapped {n} attention layers with SpaDecAttentionWrapper.")

    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = spa.model.model.language_model if hasattr(spa.model, "model") else spa.model.language_model
    gc_flag = getattr(lm, "gradient_checkpointing", False)
    log.info(f"Gradient checkpointing enabled. language_model.gradient_checkpointing={gc_flag}")
    if not gc_flag:
        lm.gradient_checkpointing = True
    return spa


# ── checkpoint ────────────────────────────────────────────────────────────────

def _save_checkpoint(model, tokenizer, output_dir, step, suffix=""):
    tag = f"step_{step}" + (f"_{suffix}" if suffix else "")
    ckpt = os.path.join(output_dir, tag)
    os.makedirs(ckpt, exist_ok=True)
    model.spa_model.save_pretrained(ckpt)
    if getattr(model, "match_head", None) is not None:
        torch.save(model.match_head.state_dict(), os.path.join(ckpt, "match_head.pt"))
    tokenizer.save_pretrained(ckpt)
    log.info(f"Checkpoint saved -> {ckpt}")


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

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rank0_print(f"<|image_pad|> token id = {image_token_id}")

    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    spa = build_spa_model(
        args.model_path,
        lora_rank     = args.lora_rank,
        freeze_vision = not args.train_vision,
    )
    hidden_dim = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True
    ).text_config.hidden_size
    use_contrast = not args.no_contrast
    use_match    = not args.no_match
    if not (use_contrast or use_match):
        log.warning("Both --no_contrast and --no_match set: only lm_loss will be used.")
    model = GeoModel(
        spa_model          = spa,
        hidden_dim         = hidden_dim,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        lm_weight          = args.lm_weight,
        contrast_weight    = args.contrast_weight,
        match_weight       = args.match_weight,
        contrast_margin    = args.contrast_margin,
        contrast_eps       = args.contrast_eps,
        n_contrast_anchors = args.n_contrast_anchors,
        match_prob         = args.match_prob,
        coord_scale        = args.coord_scale,
        use_contrast       = use_contrast,
        use_match          = use_match,
    ).to(device)
    log.info(f"GeoModel: use_contrast={use_contrast}  use_match={use_match}")

    if world_size > 1:
        # find_unused_parameters=False: contrast has no learnable parameters
        # (triplet L2 on hidden states only), and match_head either participates
        # every step (use_match=True and N ≥ 2) or is None (use_match=False).
        # LoRA params in q/k/v/o/gate/up/down_proj always receive lm_loss
        # gradient. So no parameter is unused across iterations under default
        # max_images ≥ 2; keeping this False avoids the per-step autograd-graph
        # traversal overhead and matches DDP's preference.
        #
        # Caveat: if a sample has only N<2 images (rare, e.g. max_images=1
        # smoke tests), match_head won't receive gradient on that step and
        # DDP will raise. Keep max_images ≥ 2 when use_match=True.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        _model = model.module
    else:
        _model = model

    # ── train dataset ─────────────────────────────────────────────────────────
    train_dataset = MindCube_Train_Dataset(
        jsonl_path         = args.json_path,
        results_dir        = args.mindcube_results_dir,
        processor          = processor,
        log                = log,
        max_images         = args.max_images,
        spatial_merge_size = spatial_merge_size,
        max_samples        = args.max_samples,
    )
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size,
                           rank=local_rank, shuffle=True)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=collate_fn, sampler=train_sampler,
    )

    # ── eval loaders (use Eval_Dataset_Coord so image_xyz is available at eval,
    #    letting GeoModel run contrast_loss / match_loss on test data too) ─────
    _eval_dir = os.path.join(_ROOT, "datasets/evaluation")
    test_loaders = {}
    test_samplers = {}
    for _ds_name, _ds_jsonl, _ds_results, _q_key, _a_key in [
        ("mindcube",
         os.path.join(_eval_dir, "MindCube", "MindCube_tinybench.jsonl"),
         os.path.join(_eval_dir, "MindCube", "3d_results"),
         "question", "gt_answer"),
        ("spinbench",
         os.path.join(_eval_dir, "spinbench_data", "test.jsonl"),
         os.path.join(_eval_dir, "spinbench_data", "3d_results"),
         "problem", "answer"),
    ]:
        try:
            ds = Eval_Dataset_Coord(
                _ds_jsonl,
                _ds_results,
                processor,
                log,
                max_images         = args.max_images,
                spatial_merge_size = spatial_merge_size,
                question_key       = _q_key,
                answer_key         = _a_key,
            )
            _eval_sampler = (
                DistributedSampler(ds, num_replicas=world_size, rank=local_rank, shuffle=False)
                if world_size > 1 else None
            )
            test_loaders[_ds_name] = DataLoader(
                ds, batch_size=1, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_fn,
                sampler=_eval_sampler, pin_memory=True,
            )
            test_samplers[_ds_name] = _eval_sampler
            log.info(f"Eval dataset '{_ds_name}': {len(ds)} samples")
        except Exception as exc:
            log.warning(f"Failed to load eval dataset '{_ds_name}': {exc}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    rank_log_file = os.path.join(
        args.output_dir,
        f"train_rank{local_rank}.log" if world_size > 1 else "train.log",
    )
    rank_handler = logging.FileHandler(rank_log_file, mode="w", encoding="utf-8")
    rank_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(rank_handler)
    if world_size > 1 and local_rank == 0:
        summary_log = os.path.join(args.output_dir, "train.log")
        h = logging.FileHandler(summary_log, mode="w", encoding="utf-8")
        h.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(h)

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
    running_ldict: dict = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [x.to(device) for x in image_xyz]
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            _, loss, ldict = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                pixel_values   = pixel_values,
                image_grid_thw = image_grid_thw,
                image_xyz      = image_xyz,
                labels         = labels,
            )
            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: no supervision, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            if ldict:
                for k, v in ldict.items():
                    running_ldict[k] = running_ldict.get(k, 0.0) + v

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss  = running_loss / args.grad_accum
                avg_ldict = {k: v / args.grad_accum for k, v in running_ldict.items()}
                running_loss = 0.0
                running_ldict.clear()

                if world_size > 1:
                    keys = sorted(avg_ldict.keys())
                    vals = [avg_loss] + [avg_ldict[k] for k in keys]
                    t = torch.tensor(vals, dtype=torch.float64, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= world_size
                    avg_loss  = t[0].item()
                    avg_ldict = {k: t[i + 1].item() for i, k in enumerate(keys)}

                if local_rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    detail = "  ".join(f"{k}={v:.4f}" for k, v in avg_ldict.items())
                    log.info(
                        f"[train] epoch={epoch+1:02d}  step={global_step:05d}  "
                        f"loss={avg_loss:.4f}" + (f"  ({detail})" if detail else "")
                        + f"  lr={lr:.2e}  (agg across {world_size} GPU)"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr":   lr,
                                "epoch":      epoch + 1,
                                **{f"train/{k}": v for k, v in avg_ldict.items()},
                            },
                            step=global_step,
                        )
                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir, global_step)

                # ── periodic eval (lm + contrast + match losses via GeoModel) ─
                # Calls `model(...)` rather than spa_model directly, so
                # Eval_Dataset_Coord's image_xyz flows through decouple's XYZ
                # RoPE and GeoModel's loss_dict reports all three heads on test
                # data. GeoModel.forward gates permutation on `self.training`,
                # so eval is always un-permuted → contrast fires every eval
                # step; match_loss is computed on the trivial all-matches case
                # (CE against label=1 for every image).
                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    _spa = _model.spa_model
                    _spa_gc = getattr(_spa, "gradient_checkpointing", False)
                    _lm_inner = _spa.language_model if hasattr(_spa, "language_model") else None
                    _lm_gc = getattr(_lm_inner, "gradient_checkpointing", False) if _lm_inner else False
                    if _spa_gc:
                        _spa.gradient_checkpointing = False
                    if _lm_inner and _lm_gc:
                        _lm_inner.gradient_checkpointing = False

                    for ds_name, loader in test_loaders.items():
                        if test_samplers.get(ds_name) is not None:
                            test_samplers[ds_name].set_epoch(global_step)
                        local_count = 0
                        local_loss_sums: dict[str, float] = {}

                        for tb in loader:
                            t_ids  = tb["input_ids"].to(device)
                            t_mask = tb["attention_mask"].to(device)
                            t_pv   = tb.get("pixel_values")
                            t_thw  = tb.get("image_grid_thw")
                            t_lbl  = tb.get("labels")
                            t_xyz  = tb.get("image_xyz")
                            if t_pv is not None:
                                t_pv = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None:
                                t_thw = t_thw.to(device)
                            if t_lbl is not None:
                                t_lbl = t_lbl.to(device)
                            if t_xyz is not None:
                                t_xyz = [x.to(device) for x in t_xyz]

                            try:
                                with torch.inference_mode():
                                    _, _, eval_ldict = model(
                                        input_ids      = t_ids,
                                        attention_mask = t_mask,
                                        pixel_values   = t_pv,
                                        image_grid_thw = t_thw,
                                        image_xyz      = t_xyz,
                                        labels         = t_lbl,
                                    )
                            except Exception as exc:
                                log.debug(f"Eval skip ({ds_name}): {exc}")
                                continue
                            if not eval_ldict:
                                continue
                            local_count += 1
                            for k, v in eval_ldict.items():
                                local_loss_sums[k] = local_loss_sums.get(k, 0.0) + v

                        # Aggregate across all ranks
                        _loss_keys = sorted(local_loss_sums.keys())
                        if world_size > 1:
                            _vals = [float(local_count)] + [
                                local_loss_sums.get(k, 0.0) for k in _loss_keys
                            ]
                            stats = torch.tensor(_vals, dtype=torch.float64, device=device)
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            total_count = int(stats[0].item())
                            agg_sums = {k: stats[i + 1].item() for i, k in enumerate(_loss_keys)}
                        else:
                            total_count = local_count
                            agg_sums = dict(local_loss_sums)

                        # match_loss / match_acc in eval are degenerate: the
                        # §1.D permutation is gated by self.training, so eval
                        # always sees correct xyz → label=[1,…,1] (trivial CE).
                        # Drop them from both the INFO line and wandb so the
                        # plots don't carry misleading flat curves.
                        _eval_keys = [
                            k for k in _loss_keys
                            if k not in ("match_loss", "match_acc")
                        ]

                        if total_count > 0 and local_rank == 0:
                            detail = "  ".join(
                                f"{k}={agg_sums[k] / total_count:.4f}"
                                for k in _eval_keys
                            )
                            log.info(
                                f"[eval] step={global_step:05d}  {ds_name}  "
                                + detail + f"  (n={total_count})"
                            )
                            if use_wandb:
                                wandb.log(
                                    {
                                        f"eval/{ds_name}_{k}": agg_sums[k] / total_count
                                        for k in _eval_keys
                                    },
                                    step=global_step,
                                )
                    if _spa_gc:
                        _spa.gradient_checkpointing = True
                    if _lm_inner and _lm_gc:
                        _lm_inner.gradient_checkpointing = True
                    model.train()

    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step, suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LM + §1.A multi-view contrastive + §1.D xyz-image match "
                    "fine-tuning of SpaDecForConditionalGeneration (decouple)."
    )
    p.add_argument("--model_path",
                   default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"))
    p.add_argument("--json_path",
                   default=os.path.join(_ROOT, "datasets/train/MindCube/MindCube_train.jsonl"))
    p.add_argument("--mindcube_results_dir",
                   default=os.path.join(_ROOT, "datasets/train/MindCube/3d_results"))
    p.add_argument("--output_dir",
                   default=os.path.join(_ROOT, "checkpoints/spa_geo"))
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lora_rank",    type=int,   default=16)
    p.add_argument("--max_images",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--eval_steps",   type=int,   default=100)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--train_vision", action="store_true",
                   help="Unfreeze ViT.")
    p.add_argument("--coord_scale",  type=float, default=100.0)
    # ── Geo loss weights ──
    p.add_argument("--lm_weight",        type=float, default=1.0)
    p.add_argument("--contrast_weight",  type=float, default=1.0,
                   help="§1.A triplet contrastive correspondence weight.")
    p.add_argument("--match_weight",     type=float, default=0.3,
                   help="§1.D xyz-image match head CE weight.")
    p.add_argument("--contrast_margin",  type=float, default=0.5,
                   help="Triplet margin.")
    p.add_argument("--contrast_eps",     type=float, default=0.05,
                   help="xyz L2 threshold for positive-pair selection (scene units).")
    p.add_argument("--n_contrast_anchors", type=int, default=64,
                   help="Max anchor patches per (i, j) image pair.")
    p.add_argument("--match_prob",       type=float, default=0.5,
                   help="Probability of permuting xyz for the match head "
                        "(contrast_loss is skipped on permuted steps).")
    # ── Ablation switches ──
    p.add_argument("--no_contrast", action="store_true",
                   help="Disable §1.A contrast loss entirely (skip its computation). "
                        "Use alongside default match settings to measure §1.D's gain "
                        "in isolation.")
    p.add_argument("--no_match", action="store_true",
                   help="Disable §1.D match loss entirely (skip permutation + match "
                        "head forward). Use alongside default contrast settings to "
                        "measure §1.A's gain in isolation.")
    # ── WandB ──
    p.add_argument("--wandb_project",  default="")
    p.add_argument("--wandb_entity",   default="")
    p.add_argument("--wandb_run_name", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
