"""
train_enhance.py

Baseline that does NOT fine-tune Qwen3.5-VL.  Instead, each merged-patch
vision token receives a sinusoidal 3D positional encoding built from the
patch-level (x, y, z) coordinate map.  The LLM weights stay frozen — the
only optimised module is `coord_head`.

Architecture:
    EnhanceModel
    ├── Qwen3_5ForConditionalGeneration   (frozen — vision + language)
    │       └── 3D PE injected element-wise into merged image embeds
    │           (formula: per-axis sinusoidal, see src/models/enhance_llm.py)
    └── DepthPredictionTransformer        (trainable — sub-pixel xyz)

Loss:
    loss = answer_weight * lm_loss (logging only, frozen)
         + coord_weight  * coord_loss   ← the real training signal

Compare against train_coordinate.py:
  • train_coordinate.py: LoRA on Qwen + 4D M-RoPE (xyz)  + coord_head
  • train_enhance.py:    Frozen Qwen + 3D PE on patches  + coord_head

Usage:
  python train_enhance.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_enhance
"""

import argparse
import datetime
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
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import (
    DepthPredictionTransformer,
    EnhanceModel,
)
from src.dataset import VST_Train_Dataset_Coord, Eval_Dataset_Coord

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
    skip_layers:        tuple[int, ...] = (-1,),
    answer_weight:      float = 1.0,
    coord_weight:       float = 1.0,
    use_coord_head:     bool  = True,
) -> EnhanceModel:
    """
    Build EnhanceModel: frozen Qwen3.5-VL + sinusoidal 3D PE on patches +
    optional coord_head.  No LoRA, no M-RoPE changes — Qwen is loaded with
    its original config.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        config              = config,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "sdpa",
    )

    # Freeze everything in the base model — vision + language.
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()  # disable dropout for the frozen backbone
    log.info("Qwen3.5-VL fully frozen (vision + language). No LoRA.")

    coord_head: torch.nn.Module | None = None
    if use_coord_head:
        hidden_dim = config.text_config.hidden_size
        coord_head = DepthPredictionTransformer(
            hidden_dim=hidden_dim, upscale_factor=coord_upscale,
        ).to(torch.bfloat16)
        log.info(
            f"DepthPredictionTransformer hidden_dim={hidden_dim} "
            f"upscale={coord_upscale} (trainable)"
        )

    return EnhanceModel(
        base_model         = base,
        coord_head         = coord_head,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        skip_layers        = skip_layers,
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
        # 1h NCCL timeout. Default is 10min, which is too short when a slow
        # rank lags behind the periodic-eval all_reduce on long-sequence
        # samples (mirrors train_correspondence.py / train_atten.py).
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=1),
        )
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
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        coord_upscale      = args.coord_upscale,
        skip_layers        = tuple(args.skip_layers),
        answer_weight      = args.answer_weight,
        coord_weight       = args.coord_weight,
        use_coord_head     = not args.no_coord_head,
    )

    from src.dataset import compute_letter_offset
    model.letter_offset = compute_letter_offset(processor.tokenizer)
    log.info(f"letter_offset = {model.letter_offset}")
    log.warning(
        "[eval] using LETTER_OFFSET teacher-forced probe — acc is INFLATED "
        "on datasets like SpinBench. Use evaluation.py deploy eval for true acc."
    )
    log.info("Using EnhanceModel (frozen Qwen3.5 + sinusoidal 3D PE).")

    model = model.to(device)
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"[MEM] After model.to(device): {mem_gb:.2f} GiB allocated")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    rank0_print(
        f"Trainable params: {n_train:,} / {n_total:,} "
        f"({100.0 * n_train / max(n_total, 1):.4f}%)"
    )

    # -- DDP -------------------------------------------------------------------
    if world_size > 1:
        # find_unused_parameters=True: the frozen base model has no grads,
        # so DDP must tolerate parameters that are not used in backward.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        _model = model.module
    else:
        _model = model

    # -- dataset / loader ------------------------------------------------------
    train_dataset = VST_Train_Dataset_Coord(
        args.json_path,
        args.vst_results_dir,
        processor,
        log,
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

    # -- test datasets ---------------------------------------------------------
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

    # -- optimiser -------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found. Pass --coord_head (default) so "
            "DepthPredictionTransformer is constructed."
        )
    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
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

    # -- WandB -----------------------------------------------------------------
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

    # Keep frozen Qwen in eval() mode even after model.train() — this is
    # important for layers with stochastic behaviour (dropout). Only the
    # coord_head should be trained.
    model.train()
    if hasattr(_model, "base_model"):
        _model.base_model.eval()

    global_step  = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            mm_token_type_ids = batch.get("mm_token_type_ids")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(device)

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

            _, loss, loss_dict = model(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                pixel_values    = pixel_values,
                image_grid_thw  = image_grid_thw,
                mm_token_type_ids = mm_token_type_ids,
                image_xyz       = image_xyz,
                image_xyz_hires = image_xyz_hires,
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
                        _save_checkpoint(_model, tokenizer, args.output_dir,
                                         global_step)

                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
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
                            t_mm    = test_batch.get("mm_token_type_ids")
                            t_labels = test_batch.get("labels")
                            t_xyz   = test_batch.get("image_xyz")
                            t_xyz_h = test_batch.get("image_xyz_hires")

                            if t_pv is not None:
                                t_pv = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None:
                                t_thw = t_thw.to(device)
                            if t_mm is not None:
                                t_mm = t_mm.to(device)
                            if t_labels is not None:
                                t_labels = t_labels.to(device)
                            if t_xyz is not None:
                                t_xyz = [x.to(device) for x in t_xyz]
                            if t_xyz_h is not None:
                                t_xyz_h = [x.to(device) for x in t_xyz_h]

                            with torch.inference_mode():
                                _, loss, loss_dict = model(
                                    input_ids         = t_ids,
                                    attention_mask    = t_mask,
                                    pixel_values      = t_pv,
                                    image_grid_thw    = t_thw,
                                    mm_token_type_ids = t_mm,
                                    image_xyz         = t_xyz,
                                    image_xyz_hires   = t_xyz_h,
                                    labels            = t_labels,
                                )
                            if loss_dict is None:
                                continue
                            local_count += 1
                            for k, v in loss_dict.items():
                                local_loss_sums[k] = local_loss_sums.get(k, 0.0) + v

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
                            _front  = f"acc={agg_sums['acc'] / total_count:.4f}  " if "acc" in agg_sums else ""
                            _rest   = "  ".join(
                                f"{k}={agg_sums[k] / total_count:.4f}"
                                for k in _loss_keys if k != "acc"
                            )
                            log.info(
                                f"[eval] global_step={global_step:05d}  {ds_name}  "
                                + _front + _rest
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

                    model.train()
                    if hasattr(_model, "base_model"):
                        _model.base_model.eval()

    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step,
                         suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(
    model:      EnhanceModel,
    tokenizer,
    output_dir: str,
    step:       int,
    suffix:     str = "",
) -> None:
    """Save only the trainable coord_head + tokenizer.

    The frozen Qwen weights are unchanged from the pretrained checkpoint,
    so we don't re-write them.  Loading is the inverse:
        coord_head.load_state_dict(torch.load("coord_head.pt"))
    """
    tag  = f"step_{step}" + (f"_{suffix}" if suffix else "")
    ckpt = os.path.join(output_dir, tag)
    os.makedirs(ckpt, exist_ok=True)

    tokenizer.save_pretrained(ckpt)
    if model.coord_head is not None:
        torch.save(
            model.coord_head.state_dict(),
            os.path.join(ckpt, "coord_head.pt"),
        )
    log.info(f"Checkpoint saved -> {ckpt}")


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen Qwen3.5-VL + sinusoidal 3D PE on patches "
                    "(baseline; only coord_head is trained)."
    )
    p.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"),
        help="Path to Qwen3.5-VL checkpoint",
    )
    p.add_argument(
        "--json_path",
        default=os.path.join(_ROOT, "datasets/train/VST_parsed/vst_500k.json"),
    )
    p.add_argument(
        "--vst_results_dir",
        default=os.path.join(_ROOT, "datasets/train/VST/3d_results"),
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_enhance"),
    )
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_images",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--save_steps",  type=int,   default=1000)
    p.add_argument("--eval_steps",  type=int,   default=200)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-1],
        help="LLM layer index from which coord_head reads hidden states.",
    )
    p.add_argument("--answer_weight", type=float, default=1.0)
    p.add_argument("--coord_weight",  type=float, default=1.0)
    p.add_argument(
        "--coord_upscale", type=int, default=4,
        help="PixelShuffle factor for coord_head (each patch predicts "
             "upscale^2 sub-pixel xyz).",
    )
    p.add_argument(
        "--no_coord_head", action="store_true",
        help="Skip the coord_head entirely (pure inference baseline — "
             "nothing trainable; the script will refuse to start).",
    )
    p.add_argument("--wandb_project",  default="")
    p.add_argument("--wandb_entity",   default="")
    p.add_argument("--wandb_run_name", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
