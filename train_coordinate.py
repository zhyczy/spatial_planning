"""
train_coordinate.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) with three
simultaneous supervision signals:

  1. Pose regression  — geodesic + L1 translation + cycle-consistency loss
                        (one <pose> token per ordered image pair)
  2. LM answer        — causal cross-entropy on answer tokens
                        (question appended to prompt; answer supervised)
  3. Coordinate       — L1 loss predicting sub-pixel 3D (x,y,z)
                        (one <coord> token per LLM patch -> Linear + PixelShuffle)

Architecture:
  CoordinatePlusModel
    +-- SpaForConditionalGeneration  [backbone + LoRA adapters]
    |    +-- SpaVisionModel (ViT, frozen)
    |    +-- SpaModel (LLM + 4D M-RoPE)
    +-- PoseRegressionHead      [MLP: hidden_dim*n_skip -> 9D per <pose> token]
    +-- CoordinateRegressionHead [Linear + PixelShuffle -> sub-pixel 3D]

Coordinate GT:
  For each image and each LLM patch token (after spatial merge), the GT is the
  mean (x,y,z) of all valid pixels that fall within that patch -- exactly the
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None          # type: ignore[assignment]
    _WANDB_AVAILABLE = False

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import CoordinateRegressionHead, CoordinatePlusModel, CoordinateModel, PoseRegressionHead, SpaForConditionalGeneration
from src.dataset import MindCube_Train_Dataset_Coord, Eval_Dataset, load_testing_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- DDP helpers ---------------------------------------------------------------

local_rank: int = 0
world_size: int = 1


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

POSE_TOKEN  = "<pose>"
COORD_TOKEN = "<coord>"


def collate_fn(batch):
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# -- model building ------------------------------------------------------------

def build_model(
    model_path:         str,
    pose_token_id:      int,
    coord_token_id:     int,
    image_token_id:     int,
    spatial_merge_size: int,
    coord_upscale:      int = 4,
    lora_rank:          int = 16,
    freeze_vision:      bool = True,
    skip_layers:        tuple[int, ...] = (-1,),
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
) -> CoordinatePlusModel:
    """
    Load SpaForConditionalGeneration, patch 4D M-RoPE, apply LoRA,
    and attach PoseRegressionHead + CoordinateRegressionHead.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    total = sum(orig_section)
    xyz_size = (total - 2) // 3
    new_section = [2, xyz_size, xyz_size, xyz_size]
    config.text_config.rope_scaling["mrope_section"] = new_section
    log.info(
        f"mrope_section: {orig_section} -> {new_section}  "
        f"(4D M-RoPE: 2 for t, {xyz_size} each for x/y/z)"
    )

    spa = SpaForConditionalGeneration.from_pretrained(
        model_path,
        config              = config,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "sdpa",
    )

    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    spa.resize_token_embeddings(old_vocab + 2)
    log.info(f"Embedding table: {old_vocab} -> {old_vocab + 2} (added <pose>, <coord>)")

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

    # Coordinate regression head: Linear + PixelShuffle (ultra-shallow)
    coord_head = CoordinateRegressionHead(
        hidden_dim=hidden_dim, upscale_factor=coord_upscale,
    ).to(torch.bfloat16)
    log.info(f"CoordinateRegressionHead hidden_dim={hidden_dim} upscale={coord_upscale}")

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


def build_coord_only_model(
    model_path:         str,
    coord_token_id:     int,
    image_token_id:     int,
    spatial_merge_size: int,
    coord_upscale:      int = 4,
    lora_rank:          int = 16,
    freeze_vision:      bool = True,
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
) -> CoordinateModel:
    """
    Ablation build: CoordinateModel (no pose head, no <pose> tokens needed).
    Supervision: LM answer loss + per-patch coordinate loss only.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    total = sum(orig_section)
    xyz_size = (total - 2) // 3
    new_section = [2, xyz_size, xyz_size, xyz_size]
    config.text_config.rope_scaling["mrope_section"] = new_section
    log.info(
        f"mrope_section: {orig_section} -> {new_section}  "
        f"(4D M-RoPE: 2 for t, {xyz_size} each for x/y/z)"
    )

    spa = SpaForConditionalGeneration.from_pretrained(
        model_path,
        config              = config,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "sdpa",
    )

    old_vocab = spa.model.language_model.embed_tokens.weight.shape[0]
    spa.resize_token_embeddings(old_vocab + 1)   # only <coord>
    log.info(f"Embedding table: {old_vocab} -> {old_vocab + 1} (added <coord>)")

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

    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = spa.model.model.language_model if hasattr(spa.model, 'model') else spa.model.language_model
    gc_flag = getattr(lm, 'gradient_checkpointing', False)
    log.info(f"Gradient checkpointing enabled. language_model.gradient_checkpointing={gc_flag}")
    if not gc_flag:
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    hidden_dim = config.text_config.hidden_size
    coord_head = CoordinateRegressionHead(
        hidden_dim=hidden_dim, upscale_factor=coord_upscale,
    ).to(torch.bfloat16)
    log.info(f"CoordinateRegressionHead hidden_dim={hidden_dim} upscale={coord_upscale}")

    return CoordinateModel(
        spa_model          = spa,
        coord_head         = coord_head,
        coord_token_id     = coord_token_id,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        answer_weight      = answer_weight,
        coord_weight       = coord_weight,
    )


# -- training loop -------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    # -- DDP initialisation ----------------------------------------------------
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

    # -- processor + tokeniser -------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if args.no_cam:
        tokenizer.add_special_tokens({"additional_special_tokens": [COORD_TOKEN]})
        pose_token_id  = None
        coord_token_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
        rank0_print(f"[no_cam] <coord> token id = {coord_token_id}")
    else:
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

    # -- spatial_merge_size from vision config ---------------------------------
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # -- model -----------------------------------------------------------------
    if args.no_cam:
        model = build_coord_only_model(
            args.model_path,
            coord_token_id     = coord_token_id,
            image_token_id     = image_token_id,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
            lora_rank          = args.lora_rank,
            freeze_vision      = not args.train_vision,
            answer_weight      = args.answer_weight,
            coord_weight       = args.coord_weight,
        )
        log.info("Using CoordinateModel (ablation: no pose head)")
    else:
        model = build_model(
            args.model_path,
            pose_token_id      = pose_token_id,
            coord_token_id     = coord_token_id,
            image_token_id     = image_token_id,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
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

    # -- DDP -------------------------------------------------------------------
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)
        _model = model.module
    else:
        _model = model

    # -- dataset / loader ------------------------------------------------------
    train_dataset = MindCube_Train_Dataset_Coord(
        jsonl_path         = args.json_path,
        results_dir        = args.mindcube_results_dir,
        processor          = processor,
        pose_token_id      = pose_token_id,
        coord_token_id     = coord_token_id,
        log                = log,
        max_images         = args.max_images,
        spatial_merge_size = spatial_merge_size,
        coord_upscale      = args.coord_upscale,
        max_samples        = args.max_samples,
    )
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size,
                           rank=local_rank, shuffle=True)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = 1,
        shuffle     = (train_sampler is None),
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        sampler     = train_sampler,
    )

    # -- test datasets (for periodic LM loss evaluation) -----------------------
    _eval_dir = os.path.join(_ROOT, "datasets/evaluation")
    test_loaders = {}
    test_samplers = {}
    for _ds_name, _ds_dir in [
        ("mindcube",  os.path.join(_eval_dir, "MindCube")),
        ("spinbench", os.path.join(_eval_dir, "spinbench_data")),
    ]:
        try:
            raw_samples = load_testing_dataset(data_dir=_ds_dir, dataset=_ds_name)
            ds = Eval_Dataset(raw_samples, processor)
            _eval_sampler = (
                DistributedSampler(ds, num_replicas=world_size,
                                   rank=local_rank, shuffle=False)
                if world_size > 1 else None
            )
            test_loaders[_ds_name] = DataLoader(
                ds, batch_size=1, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_fn,
                sampler=_eval_sampler,
            )
            test_samplers[_ds_name] = _eval_sampler
            log.info(f"Eval dataset '{_ds_name}': {len(ds)} samples")
        except Exception as exc:
            log.warning(f"Failed to load eval dataset '{_ds_name}': {exc}")

    # -- optimiser -------------------------------------------------------------
    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # -- logging to file -------------------------------------------------------
    rank_log_file = os.path.join(
        args.output_dir,
        f"train_rank{local_rank}.log" if world_size > 1 else "train.log"
    )
    rank_handler = logging.FileHandler(rank_log_file, mode="w", encoding="utf-8")
    rank_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S"
    ))
    log.addHandler(rank_handler)

    if world_size > 1 and local_rank == 0:
        summary_log_file = os.path.join(args.output_dir, "train.log")
        summary_handler = logging.FileHandler(summary_log_file, mode="w", encoding="utf-8")
        summary_handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S"
        ))
        log.addHandler(summary_handler)
        rank0_print(f"Per-rank logs: train_rank*.log  |  Summary log: {summary_log_file}")
    else:
        rank0_print(f"Logging to {rank_log_file}")

    # -- WandB (rank 0 only) ---------------------------------------------------
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
        log.warning("wandb not installed -- logging disabled. `pip install wandb`")

    model.train()

    global_step = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):

            # -- move batch to device ------------------------------------------
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            gt_transforms  = (
                None if args.no_cam
                else batch["gt_transforms"].to(device)
            )

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(device) for xyz in image_xyz]

            image_xyz_hires = batch.get("image_xyz_hires")
            if image_xyz_hires is not None:
                image_xyz_hires = [xyz.to(device) for xyz in image_xyz_hires]

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

            # -- forward + loss ------------------------------------------------
            try:
                _, loss, loss_dict = model(
                    input_ids       = input_ids,
                    attention_mask  = attention_mask,
                    pixel_values    = pixel_values,
                    image_grid_thw  = image_grid_thw,
                    gt_transforms   = gt_transforms,
                    image_xyz       = image_xyz,
                    image_xyz_hires = image_xyz_hires,
                    cycle_weight    = args.cycle_weight,
                    labels          = labels,
                )
            except Exception as exc:
                log.warning(f"[rank{local_rank}] Step {step} skipped: {exc}")
                continue

            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: no supervision signal, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            if loss_dict:
                for k, v in loss_dict.items():
                    running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

            # -- gradient accumulation -----------------------------------------
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / args.grad_accum
                avg_loss_dict = {
                    k: v / args.grad_accum
                    for k, v in running_loss_dict.items()
                }
                running_loss = 0.0
                running_loss_dict.clear()

                # All-reduce training losses across ranks
                if world_size > 1:
                    _loss_keys = sorted(avg_loss_dict.keys())
                    _loss_vals = [avg_loss] + [avg_loss_dict[k] for k in _loss_keys]
                    _loss_t = torch.tensor(_loss_vals, dtype=torch.float64, device=device)
                    dist.all_reduce(_loss_t, op=dist.ReduceOp.SUM)
                    _loss_t /= world_size
                    avg_loss = _loss_t[0].item()
                    avg_loss_dict = {k: _loss_t[i + 1].item() for i, k in enumerate(_loss_keys)}

                if local_rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    detail = "  ".join(
                        f"{k}={v:.4f}" for k, v in avg_loss_dict.items()
                    )
                    log.info(
                        f"[train] epoch={epoch+1:02d}  global_step={global_step:05d}  "
                        f"loss={avg_loss:.4f}"
                        + (f"  ({detail})" if detail else "")
                        + f"  lr={current_lr:.2e}  "
                        f"(aggregated across {world_size} GPU{'s' if world_size > 1 else ''})"
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

                    # -- checkpoint ------------------------------------------------
                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir,
                                         global_step)

                # -- periodic evaluation on test sets --------------------------
                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    _spa = _model.spa_model if hasattr(_model, 'spa_model') else _model

                    # Save and disable gradient checkpointing during eval to enable kv_cache
                    _spa_gc_flag = getattr(_spa, 'gradient_checkpointing', False)
                    _lm = _spa.language_model if hasattr(_spa, 'language_model') else None
                    _lm_gc_flag = getattr(_lm, 'gradient_checkpointing', False) if _lm else False
                    if _spa_gc_flag:
                        _spa.gradient_checkpointing = False
                    if _lm and _lm_gc_flag:
                        _lm.gradient_checkpointing = False

                    for ds_name, loader in test_loaders.items():
                        if ds_name in test_samplers and test_samplers[ds_name] is not None:
                            test_samplers[ds_name].set_epoch(global_step)
                        local_loss_sum = 0.0
                        local_count = 0
                        for test_batch in loader:
                            t_ids   = test_batch["input_ids"].to(device)
                            t_mask  = test_batch["attention_mask"].to(device)
                            t_pv    = test_batch.get("pixel_values")
                            t_thw   = test_batch.get("image_grid_thw")
                            t_labels = test_batch.get("labels")
                            if t_pv is not None:
                                t_pv = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None:
                                t_thw = t_thw.to(device)
                            if t_labels is not None:
                                t_labels = t_labels.to(device)
                            try:
                                with torch.no_grad():
                                    out = _spa(
                                        input_ids=t_ids, attention_mask=t_mask,
                                        pixel_values=t_pv, image_grid_thw=t_thw,
                                        return_dict=True,
                                        kv_cache=(ds_name == "spinbench"),
                                    )
                                    logits = out.logits
                                    shift_logits = logits[..., :-1, :].contiguous()
                                    shift_labels = t_labels[..., 1:].contiguous()
                                    lm_loss = F.cross_entropy(
                                        shift_logits.view(-1, shift_logits.size(-1)),
                                        shift_labels.view(-1),
                                        ignore_index=-100,
                                    )
                                    local_loss_sum += lm_loss.item()
                                    local_count += 1
                            except Exception as exc:
                                log.debug(f"Eval skip ({ds_name}): {exc}")
                                continue

                        # Aggregate across all ranks
                        if world_size > 1:
                            stats = torch.tensor(
                                [local_loss_sum, local_count],
                                dtype=torch.float64, device=device,
                            )
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            total_loss = stats[0].item()
                            total_count = int(stats[1].item())
                        else:
                            total_loss = local_loss_sum
                            total_count = local_count

                        if total_count > 0 and local_rank == 0:
                            avg = total_loss / total_count
                            log.info(
                                f"[eval] global_step={global_step:05d}  "
                                f"{ds_name}_lm_loss={avg:.4f}  "
                                f"(n={total_count} samples, aggregated across "
                                f"{world_size} GPU{'s' if world_size > 1 else ''})"
                            )
                            if use_wandb:
                                wandb.log(
                                    {f"eval/{ds_name}_lm_loss": avg},
                                    step=global_step,
                                )

                    # Restore gradient checkpointing
                    if _spa_gc_flag:
                        _spa.gradient_checkpointing = True
                    if _lm and _lm_gc_flag:
                        _lm.gradient_checkpointing = True

                    model.train()

    # Final checkpoint (rank 0 only)
    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step,
                         suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(
    model:      CoordinatePlusModel | CoordinateModel,
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
    if hasattr(model, "pose_head"):
        torch.save(
            model.pose_head.state_dict(),
            os.path.join(ckpt, "pose_head.pt"),
        )
    torch.save(
        model.coord_head.state_dict(),
        os.path.join(ckpt, "coord_head.pt"),
    )
    log.info(f"Checkpoint saved -> {ckpt}")


# -- CLI -----------------------------------------------------------------------

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
        default=os.path.join(_ROOT, "datasets/train/MindCube/MindCube_train.jsonl"),
        help="Path to MindCube training JSONL",
    )
    p.add_argument(
        "--mindcube_results_dir",
        default=os.path.join(_ROOT, "datasets/train/MindCube/3d_results"),
        help="Directory containing per-sample 3d_results folders for MindCube training",
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
    p.add_argument("--eval_steps",  type=int,   default=100)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument(
        "--train_vision",
        action="store_true",
        help="Unfreeze the vision encoder (ViT) for fine-tuning",
    )
    p.add_argument(
        "--no_cam",
        action="store_true",
        help="Ablation: remove camera pose regression head. Uses CoordinateModel "
             "(coord loss + LM loss only, no <pose> tokens needed).",
    )
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-8, -4, -1],
        help="LLM layer indices concatenated before PoseRegressionHead. "
             "e.g. --skip_layers -4 -1 (default: -8 -4 -1)",
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
    p.add_argument(
        "--coord_upscale",
        type=int, default=4,
        help="PixelShuffle upscale factor for coord head. "
             "Each <coord> token predicts upscale^2 sub-pixel (x,y,z) values.",
    )
    # -- WandB -----------------------------------------------------------------
    p.add_argument("--wandb_project",  default="", help="WandB project name.")
    p.add_argument("--wandb_entity",   default="", help="WandB entity.")
    p.add_argument("--wandb_run_name", default="", help="WandB run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
