"""
train_coordinate.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) with two
simultaneous supervision signals:

    1. LM answer   — causal cross-entropy on answer tokens
                                     (question appended to prompt; answer supervised)
    2. Coordinate  — L1 loss predicting sub-pixel 3D (x,y,z)
                                     (decoded from vision-token hidden states)

Architecture:
    CoordinateModel
    +-- SpaForConditionalGeneration  [backbone + LoRA adapters]
        |    +-- SpaVisionModel (ViT, optional frozen)
    |    +-- SpaModel (LLM + 4D M-RoPE)
        +-- CoordinateRegressionHead [DepthPredictionTransformer -> sub-pixel 3D]

Coordinate GT:
  For each image and each LLM patch token (after spatial merge), the GT is the
  mean (x,y,z) of all valid pixels that fall within that patch -- exactly the
  values already computed by resize_xyz() and stored as image_xyz.

Total loss:
    loss = answer_weight * lm_loss + coord_weight * coord_loss

Camera transform prediction is removed in this script.

Usage:
  python train_coordinate.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_coordinate
"""

import argparse
import logging
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
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import DepthPredictionTransformer, CoordinateModel, SpaForConditionalGeneration
from src.dataset import MindCube_Train_Dataset_Coord, MindCube_Train_Dataset_Coord_Polar, Eval_Dataset_Coord, xyz_to_polar

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


def collate_fn(batch):
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# -- model building ------------------------------------------------------------

def build_model(
    model_path:         str,
    image_token_id:     int,
    spatial_merge_size: int,
    coord_upscale:      int = 4,
    lora_rank:          int = 16,
    freeze_vision:      bool = True,
    skip_layers:        tuple[int, ...] = (-1,),
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
    polar:              bool  = False,
    full_rotary:        bool  = False,
) -> CoordinateModel:
    """
    Build CoordinateModel with LM + coordinate supervision.
    Camera transform prediction is removed.

    If ``full_rotary`` is True, partial_rotary_factor is forced to 1.0 so every
    head_dim dimension gets RoPE (vs. default 0.25 where 75% of dims bypass).
    This quadruples the number of RoPE freq bands (32 → 128 for head_dim=256)
    so the mrope_section is rebuilt to sum = head_dim // 2.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if full_rotary:
        # Enable full rotary: every head_dim dim gets RoPE. Breaks pretraining
        # convention of 75% content-only dims; relies on LoRA to adapt.
        config.text_config.rope_scaling["partial_rotary_factor"] = 1.0
        if hasattr(config.text_config, "rope_parameters") and config.text_config.rope_parameters is not None:
            config.text_config.rope_parameters["partial_rotary_factor"] = 1.0
        total = int(config.text_config.head_dim) // 2  # 128
        # Match Qwen's original t=11 allocation; split remainder evenly across x/y/z.
        t_size = 11
    else:
        orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
        total = sum(orig_section)  # 32
        t_size = 2
    xyz_size = (total - t_size) // 3
    new_section = [t_size, xyz_size, xyz_size, xyz_size]
    config.text_config.rope_scaling["mrope_section"] = new_section
    log.info(
        f"mrope_section -> {new_section}  sum={sum(new_section)}  "
        f"(4D M-RoPE: {t_size} for t, {xyz_size} each for x/y/z; "
        f"partial_rotary={'1.0 (full)' if full_rotary else '0.25 (default)'})"
    )

    spa = SpaForConditionalGeneration.from_pretrained(
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

    # Gradient checkpointing: trade ~20% speed for ~60% activation memory savings
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
    coord_head = DepthPredictionTransformer(
        hidden_dim=hidden_dim, upscale_factor=coord_upscale,
    ).to(torch.bfloat16)
    log.info(f"DepthPredictionTransformer hidden_dim={hidden_dim} upscale={coord_upscale}")

    return CoordinateModel(
        spa_model          = spa,
        coord_head         = coord_head,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        skip_layers        = skip_layers,
        answer_weight      = answer_weight,
        coord_weight       = coord_weight,
        polar              = polar,
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

    # image_token_id: <|image_pad|> in Qwen-VL tokeniser
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rank0_print(f"<|image_pad|> token id = {image_token_id}")

    # -- spatial_merge_size from vision config ---------------------------------
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # -- coord_scale (scalar or per-axis) --------------------------------------
    if args.coord_scale_xyz is not None:
        coord_scale_final = tuple(float(s) for s in args.coord_scale_xyz)
        rank0_print(f"coord_scale per-axis (x, y, z) = {coord_scale_final}")
    else:
        coord_scale_final = float(args.coord_scale)
        rank0_print(f"coord_scale (scalar, all axes) = {coord_scale_final}")

    # -- model -----------------------------------------------------------------
    model = build_model(
        args.model_path,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        coord_upscale      = args.coord_upscale,
        lora_rank          = args.lora_rank,
        freeze_vision      = not args.train_vision,
        skip_layers        = tuple(args.skip_layers),
        answer_weight      = args.answer_weight,
        coord_weight       = args.coord_weight,
        polar              = args.polar,
        full_rotary        = args.full,
    )
    log.info("Using CoordinateModel (camera transform prediction removed)")

    # Toggle visual-interleave RoPE layout (t at high-freq end, x/y/z round-robin)
    if args.interleave_vision:
        # PEFT-wrapped path: model.spa_model.model.model.language_model.rotary_emb
        # Fall back to shallower paths if structure differs.
        _rotary = None
        for _name, _mod in model.spa_model.named_modules():
            if _name.endswith("language_model.rotary_emb"):
                _rotary = _mod
                break
        if _rotary is None:
            raise RuntimeError("Could not find language_model.rotary_emb on spa_model")
        _rotary.visual_interleave = True
        rank0_print(
            "[RoPE] visual_interleave=True: t at high-freq end (bands 0..s0-1), "
            "x/y/z round-robin through remaining bands."
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
    _CoordDataset = MindCube_Train_Dataset_Coord_Polar if args.polar else MindCube_Train_Dataset_Coord
    train_dataset = _CoordDataset(
        args.json_path,
        args.mindcube_results_dir,
        processor,
        log,
        max_images         = args.max_images,
        spatial_merge_size = spatial_merge_size,
        coord_upscale      = args.coord_upscale,
        max_samples        = args.max_samples,
    )
    if args.polar:
        log.info(
            "Polar mode: image_xyz_hires GT converted to (log r, θ, α) "
            "[θ=azimuth ∈ [-π,π],  α=inclination ∈ [0,π]]; "
            "visual-token RoPE also uses log-spherical positions."
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

    # -- test datasets (full format: same prompt as training) ------------------
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
                coord_upscale      = args.coord_upscale,
                question_key       = _q_key,
                answer_key         = _a_key,
            )
            _eval_sampler = (
                DistributedSampler(ds, num_replicas=world_size,
                                   rank=local_rank, shuffle=False)
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
                pv_shape = tuple(pixel_values.shape) if pixel_values is not None else None
                mem_before = torch.cuda.memory_allocated(device) / 1e9
                log.info(
                    f"[MEM] Step 0: seq_len={input_ids.shape[1]}, "
                    f"img_tokens={n_img_tok}, "
                    f"pixel_values={pv_shape}, "
                    f"image_grid_thw={image_grid_thw}, "
                    f"mem_before_fwd={mem_before:.2f} GiB"
                )

            # -- forward + loss ------------------------------------------------
  
            _, loss, loss_dict = model(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                pixel_values    = pixel_values,
                image_grid_thw  = image_grid_thw,
                image_xyz       = image_xyz,
                image_xyz_hires = image_xyz_hires,
                coord_scale     = coord_scale_final,
                labels          = labels,
            )

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

                    # Disable gradient checkpointing during eval
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

                        local_count = 0
                        local_loss_sums: dict[str, float] = {}

                        for test_batch in loader:
                            t_ids   = test_batch["input_ids"].to(device)
                            t_mask  = test_batch["attention_mask"].to(device)
                            t_pv    = test_batch.get("pixel_values")
                            t_thw   = test_batch.get("image_grid_thw")
                            t_labels = test_batch.get("labels")
                            t_xyz   = test_batch.get("image_xyz")
                            t_xyz_h = test_batch.get("image_xyz_hires")

                            if t_pv is not None:
                                t_pv = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None:
                                t_thw = t_thw.to(device)
                            if t_labels is not None:
                                t_labels = t_labels.to(device)
                            if t_xyz is not None:
                                t_xyz = [x.to(device) for x in t_xyz]
                            if t_xyz_h is not None:
                                t_xyz_h = [x.to(device) for x in t_xyz_h]
                                if args.polar:
                                    t_xyz_h = [xyz_to_polar(x) for x in t_xyz_h]

                        
                            with torch.inference_mode():
                                _, loss, loss_dict = model(
                                    input_ids       = t_ids,
                                    attention_mask  = t_mask,
                                    pixel_values    = t_pv,
                                    image_grid_thw  = t_thw,
                                    image_xyz       = t_xyz,
                                    image_xyz_hires = t_xyz_h,
                                    coord_scale     = coord_scale_final,
                                    labels          = t_labels,
                                )
                            if loss is None:
                                continue
                            local_count += 1
                            if loss_dict:
                                for k, v in loss_dict.items():
                                    local_loss_sums[k] = local_loss_sums.get(k, 0.0) + v
                            

                        # Aggregate across all ranks
                        _loss_keys = sorted(local_loss_sums.keys())
                        if world_size > 1:
                            _vals = [float(local_count)] + [local_loss_sums.get(k, 0.0) for k in _loss_keys]
                            stats = torch.tensor(_vals, dtype=torch.float64, device=device)
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            total_count = int(stats[0].item())
                            agg_sums = {k: stats[i + 1].item() for i, k in enumerate(_loss_keys)}
                        else:
                            total_count = local_count
                            agg_sums = dict(local_loss_sums)

                        if total_count > 0 and local_rank == 0:
                            detail = "  ".join(
                                f"{k}={agg_sums[k] / total_count:.4f}"
                                for k in _loss_keys
                            )
                            log.info(
                                f"[eval] global_step={global_step:05d}  {ds_name}  "
                                + detail
                                + f"  (n={total_count}, {world_size} GPU{'s' if world_size > 1 else ''})"
                            )
                            if use_wandb:
                                _main_keys = {"coord_loss", "lm_loss", "acc"}
                                wandb.log(
                                    {
                                        **(
                                            {f"eval/{ds_name}_{k}": agg_sums[k] / total_count
                                             for k in _loss_keys if k in _main_keys}
                                        ),
                                        **(
                                            {f"eval_sub_loss/{ds_name}_{k}": agg_sums[k] / total_count
                                             for k in _loss_keys if k not in _main_keys}
                                        ),
                                    },
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
    model:      CoordinateModel,
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
        model.coord_head.state_dict(),
        os.path.join(ckpt, "coord_head.pt"),
    )
    log.info(f"Checkpoint saved -> {ckpt}")


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of SpaForConditionalGeneration "
                    "with answer + coordinate supervision."
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
        "--polar",
        action="store_true",
        help="Use log-spherical (log r, θ=azimuth, α=inclination) instead of "
             "Cartesian for both the coord-loss target (via "
             "MindCube_Train_Dataset_Coord_Polar) and the visual-token 4D M-RoPE "
             "positions (get_vision_position_ids polar branch).",
    )
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-8, -4, -1],
        help="LLM layer indices used by CoordinateModel. "
             "e.g. --skip_layers -4 -1 (default: -8 -4 -1)",
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
             "Each vision patch predicts upscale^2 sub-pixel (x,y,z) values.",
    )
    p.add_argument(
        "--coord_scale",
        type=float, default=100.0,
        help="Scalar multiplier applied to xyz before RoPE discretization. "
             "Used as the default for all three axes when --coord_scale_xyz is unset.",
    )
    p.add_argument(
        "--coord_scale_xyz",
        type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
        help="Per-axis scales (scale_x scale_y scale_z) applied to xyz before RoPE "
             "discretization. Overrides --coord_scale. Useful because x/y/z are "
             "assigned to freq bands with very different inv_freq ranges; picking "
             "different scales lets each axis land in its own useful freq region.",
    )
    p.add_argument(
        "--interleave_vision",
        action="store_true",
        help="Use interleaved M-RoPE layout for visual tokens: t keeps its "
             "mrope_section[0] bands at the high-freq end, then x/y/z round-robin "
             "through the remaining bands so each spans the full freq range. "
             "Removes the need for per-axis scales because x/y/z become symmetric.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Force partial_rotary_factor=1.0 so every head_dim dimension gets "
             "RoPE (vs. default 0.25 where 75%% of dims are content-only "
             "pass-through). Rebuilds mrope_section to sum=head_dim//2 "
             "(e.g. 32 -> 128 for head_dim=256, giving [2, 42, 42, 42]). "
             "Breaks Qwen's pretrained content/position split; only LoRA can "
             "adapt. Use with caution — expect degraded LM loss initially.",
    )
    # -- WandB -----------------------------------------------------------------
    p.add_argument("--wandb_project",  default="", help="WandB project name.")
    p.add_argument("--wandb_entity",   default="", help="WandB entity.")
    p.add_argument("--wandb_run_name", default="", help="WandB run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
