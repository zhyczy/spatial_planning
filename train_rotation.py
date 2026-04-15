"""
train_rotation.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) with a
single-pass rotation-aware coordinate prediction pipeline using
differentiable M-RoPE.

Architecture
~~~~~~~~~~~~
RotationRoPEModel
+-- SpaForConditionalGeneration [backbone + LoRA]
|    +-- SpaVisionModel (ViT, frozen)
|    +-- SpaModel (LLM + 4D M-RoPE, manual decoder loop)
+-- CameraTokenRotationEncoder   [shallow transformer, predicts R]
+-- DepthPredictionTransformer   [coordinate head]

Pipeline
~~~~~~~~
Single pass  Image token XYZ positions pass through the rotation
             encoder to produce R, which rotates them in-place to a
             canonical frame.  The rotated float position_ids are fed
             into a DifferentiableMRoPE and the MLLM decoder runs
             manually layer-by-layer so gradients from lm_loss and
             coord_loss flow back through (cos, sin) into R and
             rotation_enc end-to-end (no GT rotation supervision).

Coordinate head
        Decodes hidden states → sub-pixel (x, y, z) predictions in the
        rotated frame.  GT = R.detach() @ xyz_hires (prevents the
        trivial R→0 collapse cheat).

Total loss
    loss = answer_weight * lm_loss + coord_weight * coord_loss

Usage
~~~~~
  python train_rotation.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_rotation
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

from src.models import (
    DepthPredictionTransformer,
    CameraTokenRotationEncoder,
    RotationRoPEModel,
    SpaForConditionalGeneration,
)
from src.models.spa_emb import SpaTextRotaryEmbedding
from src.dataset import (
    MindCube_Train_Dataset_Rotation,
    SAT_Train_Dataset_Rotation,
    Eval_Dataset_Coord,
)

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
    coord_upscale:      int   = 4,
    lora_rank:          int   = 16,
    freeze_vision:      bool  = True,
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
    rot_nhead:          int   = 4,
    rot_dim_feedforward: int  = 2048,
    rot_num_layers:     int   = 2,
) -> RotationRoPEModel:
    """Build RotationRoPEModel with LM + coordinate supervision
    (rotation learned end-to-end via differentiable M-RoPE)."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])
    total    = sum(orig_section)
    xyz_size = (total - 2) // 3
    new_section = [2, xyz_size, xyz_size, xyz_size]
    config.text_config.rope_scaling["mrope_section"] = new_section
    log.info(
        f"mrope_section: {orig_section} -> {new_section}  "
        f"(4D M-RoPE: 2 for t, {xyz_size} each for x/y/z)"
    )

    # MLLM explicit head_dim (e.g. 256 for Qwen3.5-4B).
    # This is config.head_dim, NOT hidden_size // num_attention_heads.
    # The rotation encoder must use the same head_dim so RoPE freq mapping
    # is byte-for-byte identical.
    mllm_head_dim = getattr(config.text_config, "head_dim", None) or (
        config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    log.info(f"mllm_head_dim={mllm_head_dim}  rot_nhead={rot_nhead}  "
             f"→ rotation_enc d_model={rot_nhead * mllm_head_dim}")

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

    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = (spa.model.model.language_model
          if hasattr(spa.model, "model") else spa.model.language_model)
    gc_flag = getattr(lm, "gradient_checkpointing", False)
    log.info(f"Gradient checkpointing enabled. language_model.gradient_checkpointing={gc_flag}")
    if not gc_flag:
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    hidden_dim = config.text_config.hidden_size

    # Build a SpaTextRotaryEmbedding from the SAME (updated) config so that
    # inv_freq, rope_theta, and mrope_section are byte-for-byte identical to
    # what the MLLM backbone uses.
    rot_rope_emb = SpaTextRotaryEmbedding(config=config.text_config).to(torch.bfloat16)

    rotation_enc = CameraTokenRotationEncoder(
        hidden_dim      = hidden_dim,
        mllm_head_dim   = mllm_head_dim,
        rope_emb        = rot_rope_emb,
        nhead           = rot_nhead,
        dim_feedforward = rot_dim_feedforward,
        num_layers      = rot_num_layers,
    ).to(torch.bfloat16)
    log.info(
        f"CameraTokenRotationEncoder  hidden_dim={hidden_dim}  "
        f"mllm_head_dim={mllm_head_dim}  nhead={rot_nhead}  "
        f"d_model={rotation_enc.d_model}  "
        f"dim_feedforward={rot_dim_feedforward}  num_layers={rot_num_layers}"
    )

    coord_head = DepthPredictionTransformer(
        hidden_dim=hidden_dim, upscale_factor=coord_upscale,
    ).to(torch.bfloat16)
    log.info(f"DepthPredictionTransformer hidden_dim={hidden_dim} upscale={coord_upscale}")

    return RotationRoPEModel(
        spa_model          = spa,
        rotation_enc       = rotation_enc,
        coord_head         = coord_head,
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

    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rank0_print(f"<|image_pad|> token id = {image_token_id}")

    # -- spatial_merge_size from vision config ---------------------------------
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # -- model -----------------------------------------------------------------
    model = build_model(
        args.model_path,
        image_token_id      = image_token_id,
        spatial_merge_size  = spatial_merge_size,
        coord_upscale       = args.coord_upscale,
        lora_rank           = args.lora_rank,
        freeze_vision       = not args.train_vision,
        answer_weight       = args.answer_weight,
        coord_weight        = args.coord_weight,
        rot_nhead           = args.rot_nhead,
        rot_dim_feedforward = args.rot_dim_feedforward,
        rot_num_layers      = args.rot_num_layers,
    )
    model = model.to(device)
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"[MEM] After model.to(device): {mem_gb:.2f} GiB allocated")

    # -- DDP -------------------------------------------------------------------
    if world_size > 1:
        # find_unused_parameters=True so the rotation_enc warm-up phase
        # (epoch < begin_round) is allowed: rotation_enc is skipped in
        # forward, so its params receive no grad, and DDP must tolerate that.
        model  = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        _model = model.module
    else:
        _model = model

    # -- dataset / loader ------------------------------------------------------
    if args.training_dataset == "mindcube":
        train_dataset = MindCube_Train_Dataset_Rotation(
            args.json_path,
            args.results_dir,
            processor,
            None,
            log,
            max_images         = args.max_images,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
            max_samples        = args.max_samples,
            no_cam             = True,
        )
    elif args.training_dataset == "sat":
        train_dataset = SAT_Train_Dataset_Rotation(
            args.json_path,
            args.results_dir,
            processor,
            log,
            max_images         = args.max_images,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
            max_samples        = args.max_samples,
        )
    else:
        raise ValueError(f"Unknown --training_dataset: {args.training_dataset}")
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

    # -- eval datasets ---------------------------------------------------------
    _eval_dir = os.path.join(_ROOT, "datasets/evaluation")
    test_loaders  = {}
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
                None,
                log,
                max_images         = args.max_images,
                spatial_merge_size = spatial_merge_size,
                coord_upscale      = args.coord_upscale,
                no_cam             = True,
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
    # Split into two mutually-exclusive groups:
    #   (a) rotation_enc — train-from-scratch, strict RoPE-aware clip
    #   (b) rest (LoRA + coord_head) — standard fine-tune regime
    rotation_enc_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc" in n
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc" not in n
    ]
    trainable = rotation_enc_params + other_params
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,        "lr": args.lr,              "name": "lora"},
            {"params": rotation_enc_params, "lr": args.rotation_enc_lr, "name": "rotation_enc"},
        ],
        weight_decay=0.01,
    )
    log.info(
        f"Optimizer groups: lora/coord_head={len(other_params)} params "
        f"@ lr={args.lr}, rotation_enc={len(rotation_enc_params)} params "
        f"@ lr={args.rotation_enc_lr}"
    )
    log.info("=" * 72)
    log.info(
        f">>> begin_round = {args.begin_round}  "
        f"(rotation_enc activates at epoch {args.begin_round}; "
        f"epochs 0..{args.begin_round - 1} use R=I identity)"
        if args.begin_round > 0
        else f">>> begin_round = 0  (rotation_enc active from epoch 0)"
    )
    log.info(
        f">>> no_coord    = {args.no_coord}  "
        + ("(coord_loss + coord_head DISABLED for entire run)"
           if args.no_coord
           else "(coord_loss active)")
    )
    log.info("=" * 72)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -- logging to file -------------------------------------------------------
    rank_log_file = os.path.join(
        args.output_dir,
        f"train_rank{local_rank}.log" if world_size > 1 else "train.log"
    )
    rank_handler = logging.FileHandler(rank_log_file, mode="w", encoding="utf-8")
    rank_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(rank_handler)

    if world_size > 1 and local_rank == 0:
        summary_log_file = os.path.join(args.output_dir, "train.log")
        summary_handler = logging.FileHandler(summary_log_file, mode="w", encoding="utf-8")
        summary_handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(summary_handler)
        rank0_print(f"Per-rank logs: train_rank*.log  |  Summary: {summary_log_file}")
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

    global_step  = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Warm-up phase: skip rotation_enc for epoch < begin_round so the
        # LoRA + coord_head learn against an identity rotation first.
        use_rot_enc = epoch >= args.begin_round
        if local_rank == 0:
            log.info(
                f"[epoch {epoch+1:02d}] use_rotation_enc={use_rot_enc} "
                f"(begin_round={args.begin_round})"
            )

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
                n_img_tok  = (input_ids[0] == image_token_id).sum().item()
                pv_shape   = tuple(pixel_values.shape) if pixel_values is not None else None
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
                input_ids        = input_ids,
                attention_mask   = attention_mask,
                pixel_values     = pixel_values,
                image_grid_thw   = image_grid_thw,
                image_xyz        = image_xyz,
                image_xyz_hires  = image_xyz_hires,
                labels           = labels,
                coord_scale      = args.coord_scale,
                use_rotation_enc = use_rot_enc,
                use_coord_loss   = not args.no_coord,
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
                # Independent grad-norm clipping: rotation_enc is strict
                # (RoPE high-frequency gradient amplification), LoRA/coord_head
                # uses standard fine-tune clip.
                if rotation_enc_params:
                    torch.nn.utils.clip_grad_norm_(
                        rotation_enc_params, max_norm=args.rotation_enc_clip
                    )
                torch.nn.utils.clip_grad_norm_(
                    other_params, max_norm=args.lora_clip
                )
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

                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir, global_step)

                # -- periodic evaluation ---------------------------------------
                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    _spa = _model.spa_model if hasattr(_model, "spa_model") else _model

                    _spa_gc = getattr(_spa, "gradient_checkpointing", False)
                    _lm     = getattr(_spa, "language_model", None)
                    _lm_gc  = getattr(_lm,  "gradient_checkpointing", False) if _lm else False
                    if _spa_gc:
                        _spa.gradient_checkpointing = False
                    if _lm and _lm_gc:
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

                            with torch.inference_mode():
                                _, loss, loss_dict = model(
                                    input_ids        = t_ids,
                                    attention_mask   = t_mask,
                                    pixel_values     = t_pv,
                                    image_grid_thw   = t_thw,
                                    image_xyz        = t_xyz,
                                    image_xyz_hires  = t_xyz_h,
                                    labels           = t_labels,
                                    coord_scale      = args.coord_scale,
                                    use_rotation_enc = use_rot_enc,
                                    use_coord_loss   = not args.no_coord,
                                )
                            if loss is None:
                                continue
                            local_count += 1
                            if loss_dict:
                                for k, v in loss_dict.items():
                                    local_loss_sums[k] = local_loss_sums.get(k, 0.0) + v

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
                            agg_sums    = dict(local_loss_sums)

                        if total_count > 0 and local_rank == 0:
                            detail = "  ".join(
                                f"{k}={agg_sums[k] / total_count:.4f}" for k in _loss_keys
                            )
                            log.info(
                                f"[eval] global_step={global_step:05d}  {ds_name}  "
                                + detail
                                + f"  (n={total_count}, "
                                f"{world_size} GPU{'s' if world_size > 1 else ''})"
                            )
                            if use_wandb:
                                _main_keys = {"coord_loss", "lm_loss"}
                                wandb.log(
                                    {
                                        **{f"eval/{ds_name}_{k}": agg_sums[k] / total_count
                                           for k in _loss_keys if k in _main_keys},
                                        **{f"eval_sub/{ds_name}_{k}": agg_sums[k] / total_count
                                           for k in _loss_keys if k not in _main_keys},
                                    },
                                    step=global_step,
                                )

                    if _spa_gc:
                        _spa.gradient_checkpointing = True
                    if _lm and _lm_gc:
                        _lm.gradient_checkpointing = True

                    model.train()

    # Final checkpoint
    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step, suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(
    model:      RotationRoPEModel,
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
        model.rotation_enc.state_dict(),
        os.path.join(ckpt, "rotation_enc.pt"),
    )
    torch.save(
        model.coord_head.state_dict(),
        os.path.join(ckpt, "coord_head.pt"),
    )
    log.info(f"Checkpoint saved -> {ckpt}")


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-pass differentiable-RoPE rotation-aware "
                    "coordinate prediction training "
                    "(LoRA fine-tuning of SpaForConditionalGeneration)."
    )
    p.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"),
        help="Path to Qwen3.5-VL checkpoint",
    )
    p.add_argument(
        "--training_dataset",
        choices=["mindcube", "sat"],
        default="sat",
        help="Which training dataset class to use.",
    )
    p.add_argument(
        "--json_path",
        default=os.path.join(_ROOT, "datasets/train/SAT/train_36k.json"),
        help="Path to train JSON/JSONL. mindcube → JSONL; sat → JSON list.",
    )
    p.add_argument(
        "--results_dir",
        default=os.path.join(_ROOT, "datasets/train/SAT/3d_results"),
        help="Directory containing per-entry 3d_results/<id>/view_XXXX/.",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_rotation"),
    )
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument(
        "--begin_round", type=int, default=1,
        help="Epoch index (0-based) at which to start training the "
             "CameraTokenRotationEncoder.  Earlier epochs run with R=I "
             "(identity rotation).  Default 1 = skip epoch 0, enable from "
             "epoch 1 onwards.",
    )
    p.add_argument("--lr",              type=float, default=2e-4,
                   help="learning rate for LoRA + coord_head group")
    p.add_argument("--rotation_enc_lr", type=float, default=2e-4,
                   help="learning rate for rotation_enc (train-from-scratch)")
    p.add_argument("--lora_clip",           type=float, default=1.0,
                   help="grad-norm clip for LoRA + coord_head group")
    p.add_argument("--rotation_enc_clip",   type=float, default=0.3,
                   help="grad-norm clip for rotation_enc "
                        "(strict, due to RoPE high-freq gradient amplification)")
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
        help="Unfreeze the vision encoder (ViT) for fine-tuning.",
    )
    p.add_argument("--answer_weight", type=float, default=1.0)
    p.add_argument("--coord_weight",  type=float, default=1.0)
    p.add_argument(
        "--no_coord", action="store_true",
        help="Disable the coordinate L1 loss and coord_head forward entirely. "
             "Only the LM loss supervises training; coord_head receives no "
             "gradient and is effectively frozen.",
    )
    p.add_argument(
        "--coord_upscale", type=int, default=4,
        help="PixelShuffle upscale factor for the coordinate head.",
    )
    # Coordinate scale (must match MLLM's coord_scale in SpaModel)
    p.add_argument(
        "--coord_scale", type=float, default=100.0,
        help="Multiplier applied to float XYZ before rounding to integer "
             "RoPE indices.  Must be the same value used everywhere "
             "(SpaModel.get_vision_position_ids, rotation encoder PE, "
             "coord head GT).  Default 100 maps ±10 m → ±1000.",
    )
    # Rotation encoder hyper-parameters
    # d_model = rot_nhead × mllm_head_dim  (mllm_head_dim read from config,
    # e.g. 256 for Qwen3.5-4B  →  rot_nhead=4 gives d_model=1024)
    p.add_argument(
        "--rot_nhead", type=int, default=4,
        help="Number of attention heads in CameraTokenRotationEncoder. "
             "d_model = rot_nhead × config.head_dim (e.g. 4×256=1024).",
    )
    p.add_argument("--rot_dim_feedforward", type=int, default=2048)
    p.add_argument("--rot_num_layers",      type=int, default=2)
    # WandB
    p.add_argument("--wandb_project",  default="", help="WandB project name.")
    p.add_argument("--wandb_entity",   default="", help="WandB entity.")
    p.add_argument("--wandb_run_name", default="", help="WandB run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
