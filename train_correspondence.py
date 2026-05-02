"""
train_correspondence.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) to predict
answers from multi-image prompts with optional 4D M-RoPE spatial conditioning.

Architecture:
    AnswerOnlyModel
    ├── SpaForConditionalGeneration  [backbone + LoRA adapters]
    │    ├── SpaVisionModel (ViT, frozen)
    │    └── SpaModel (LLM + 4D M-RoPE)

Training strategy:
  - Each SPAR entry = one scene with N images
    - Prompt uses images + question (no pairwise camera-language template)
    - Supervision is LM answer loss (with optional xyz conditioning in RoPE)

Coordinate convention:
    - For non-vanilla mode, per-patch xyz is used in vision-token M-RoPE

Usage:
  python train_correspondence.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_correspondence
"""

import argparse
import datetime
import logging
import os
import sys

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None          # type: ignore[assignment]
    _WANDB_AVAILABLE = False


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import (
    AnswerOnlyModel,
    SpaForConditionalGeneration,
    SpaDecForConditionalGeneration,
    patch_attention_layers_dec,
)
from src.dataset import (
    VST_Train_Dataset,
    Eval_Dataset_Coord,
    extract_answer_letter,
)

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Eval generation cap. Sized to fit the longest VST supervised answer
# (unknow type ≈ 3987 tokens incl. wrappers), so the model can finish
# `</answer>` even when it has drifted to long-form outputs from training.
# Greedy decoding stops at EOS naturally; this is the safety ceiling.
_EVAL_MAX_NEW_TOKENS = 4096


class _StopOnAnswerClose(StoppingCriteria):
    """Halt generation as soon as the model emits the `</answer>` close tag.

    extract_answer_letter only looks at `<answer>…</answer>`, so anything
    after the close tag is dead weight. Stopping early bounds per-sample
    eval time by the actual answer length rather than `_EVAL_MAX_NEW_TOKENS`,
    which prevents NCCL all_reduce timeouts on the slowest rank when a few
    samples would otherwise generate to the cap.
    """

    def __init__(self, tokenizer):
        ids = tokenizer.encode("</answer>", add_special_tokens=False)
        self._stop_ids = torch.tensor(ids, dtype=torch.long)
        self._L = len(ids)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self._L:
            return False
        last = input_ids[0, -self._L:]
        return bool(torch.equal(last, self._stop_ids.to(last.device)))

# ── DDP helpers ───────────────────────────────────────────────────────────────

local_rank: int = 0
world_size: int = 1


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def collate_fn(batch):
    """
    Identity collation for batch_size=1.
    The processor already returns tensors with the correct shapes
    (input_ids: (1, seq_len), pixel_values: (total_patches, C, H, W), …).
    """
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


def _resolve_language_model(root: nn.Module) -> nn.Module | None:
    """BFS through PEFT/DDP/AnswerOnly wrapper layers to find the inner
    Qwen3_5/Spa/SpaDec text-decoder module. Mirrors the helper in
    train_atten.py so both scripts find LM the same way regardless of mode.
    """
    queue = [root]
    seen: set[int] = set()
    while queue:
        cur = queue.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        lm = getattr(cur, "language_model", None)
        if isinstance(lm, nn.Module) and hasattr(lm, "layers"):
            return lm
        for attr in ("spa_model", "module", "base_model", "model"):
            nxt = getattr(cur, attr, None)
            if isinstance(nxt, nn.Module) and id(nxt) not in seen:
                queue.append(nxt)
    return None


# ── model building ────────────────────────────────────────────────────────────
def build_model(
    model_path:    str,
    lora_rank:     int = 16,
    freeze_vision: bool = True,
    vanilla:       bool = False,
    decouple:      bool = False,
    xyz_rope_dim:  int  = 66,
) -> nn.Module:
    """
    Load backbone, patch M-RoPE, apply LoRA, return an answer model.

    vanilla=False  → 4D M-RoPE (t, x, y, z) with image_xyz spatial embedding
    vanilla=True   → original 3D M-RoPE, no image_xyz
    decouple=True  → keep Qwen original 3D M-RoPE [11,11,10] in the rotary 64
                     dims (UNCHANGED) and add a NEW XYZ RoPE in dims 64..129
                     (66 dims, sequential x|y|z, rope_theta=10000) with
                     **Cartesian** xyz. Text tokens get xyz=(0,0,0). Mutually
                     exclusive with --vanilla.
    """
    if decouple and vanilla:
        raise ValueError("--decouple is mutually exclusive with --vanilla.")
    if xyz_rope_dim % 6 != 0 or xyz_rope_dim <= 0 or xyz_rope_dim > 192:
        raise ValueError(
            f"--xyz_rope_dim must be a positive multiple of 6 ≤ 192 "
            f"(pass-through region); got {xyz_rope_dim}."
        )

    use_decouple = decouple

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])

    if vanilla:
        # ── vanilla: keep original 3D M-RoPE, use stock Qwen model ───────────
        log.info(f"mrope_section: {orig_section} (original 3D M-RoPE, vanilla ablation)")
        spa = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config             = config,
            torch_dtype        = torch.bfloat16,
            attn_implementation= "sdpa",
        )
    elif use_decouple:
        # ── decouple: keep Qwen original 3D M-RoPE in the rotary 64 dims,
        #             add new XYZ RoPE in pass-through dims 64..129 (Cartesian).
        # θ = 10000 — wider spectrum for raw meters (wavelength 6.3..29K ≈
        # covers 0.06m..300m at scale=100). ─────────────────────────────────
        _xyz_theta = 10000.0
        log.info(
            f"mrope_section: {orig_section} (UNCHANGED — Qwen original 3D M-RoPE) "
            f"+ new XYZ RoPE ({xyz_rope_dim} dims, rope_theta={_xyz_theta:g}) in pass-through region "
            f"[input: Cartesian (x, y, z)]"
        )
        spa = SpaDecForConditionalGeneration.from_pretrained(
            model_path,
            config             = config,
            torch_dtype        = torch.bfloat16,
            attn_implementation= "sdpa",
        )
        # Swap in the requested xyz_dim. xyz_rotary_emb has no trainable params
        # (only an inv_freq buffer), so replacing it post-from_pretrained is
        # safe and happens before LoRA wrapping. SpaDecAttentionWrapper reads
        # xyz_dim from cos.shape[-1] at runtime, so no other change needed.
        if xyz_rope_dim != 66:
            from src.models.spa_emb_dec import SpaXYZRotaryEmbedding
            _lm = spa.model.language_model
            _old = _lm.xyz_rotary_emb
            _new = SpaXYZRotaryEmbedding(
                xyz_dim             = xyz_rope_dim,
                rope_theta          = _xyz_theta,
                default_coord_scale = _old.default_coord_scale,
            )
            _lm.xyz_rotary_emb = _new.to(next(_lm.parameters()).device)
            log.info(
                f"[XYZ RoPE] xyz_dim={xyz_rope_dim} (n_per_axis={xyz_rope_dim // 6}), "
                f"theta={_xyz_theta:g}"
            )
    else:
        # ── 4D M-RoPE ────────────────────────────────────────────────────────
        total = sum(orig_section)                       # e.g. 32
        xyz_size = (total - 2) // 3                     # e.g. (32-2)//3 = 10
        new_section = [2, xyz_size, xyz_size, xyz_size] # [2, 10, 10, 10]
        config.text_config.rope_scaling["mrope_section"] = new_section
        log.info(
            f"mrope_section: {orig_section} → {new_section}  "
            f"(4D M-RoPE: 2 for t, {xyz_size} each for x/y/z)"
        )
        spa = SpaForConditionalGeneration.from_pretrained(
            model_path,
            config             = config,
            torch_dtype        = torch.bfloat16,
            attn_implementation= "sdpa",
        )

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

    # ── patch attention layers for decouple mode (after LoRA) ────────────────
    if use_decouple:
        n = patch_attention_layers_dec(spa)
        log.info(f"Wrapped {n} attention layers with SpaDecAttentionWrapper.")

    # Gradient checkpointing — same pattern as train_atten.py
    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = _resolve_language_model(spa)
    if lm is not None and not getattr(lm, "gradient_checkpointing", False):
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    # All modes use the same coord_scale = 100 convention
    # (4D: cm-equivalent for M-RoPE position; decouple: cm-equivalent for
    # the pass-through XYZ RoPE → wavelength range 0.063m .. 272m at θ=10000)
    _coord_scale = 100.0

    use_xyz = not vanilla
    log.info(
        f"AnswerOnlyModel (use_xyz={use_xyz}, "
        f"decouple={use_decouple}, coord_scale={_coord_scale})"
    )
    return AnswerOnlyModel(
        spa,
        use_xyz     = use_xyz,
        coord_scale = _coord_scale,
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    # ── DDP initialisation ────────────────────────────────────────────────────
    _env_rank = os.environ.get("LOCAL_RANK")
    if _env_rank is not None:
        local_rank = int(_env_rank)
        # 1h NCCL timeout. Default is 10min, which is too short when periodic
        # eval has to generate up to _EVAL_MAX_NEW_TOKENS per sample on a few
        # outlier samples — the slowest rank can fall behind the eval-loop
        # all_reduce and trip the watchdog.
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

    # ── logging to file (set up BEFORE build_model so its logs land in train.log) ─
    os.makedirs(args.output_dir, exist_ok=True)
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

    # Stamp the full CLI args once so train.log captures every config knob
    # (xyz_rope_dim, decouple, lora_rank, etc.) regardless of mode.
    log.info(f"[CONFIG] {vars(args)}")

    # ── processor + tokeniser ─────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        args.model_path,
        lora_rank      = args.lora_rank,
        freeze_vision  = not args.train_vision,
        vanilla        = args.vanilla,
        decouple       = args.decouple,
        xyz_rope_dim   = args.xyz_rope_dim,
    )

    model = model.to(device)

    # ── resolve spatial_merge_size from vision config ─────────────────────────
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # ── DDP wrapping ──────────────────────────────────────────────────────────
    # find_unused_parameters=False: DDP only tracks requires_grad=True params;
    # frozen backbone weights are invisible to it. All trainable LoRA params
    # participate in every forward pass, so the extra graph traversal is unnecessary.
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)
        _model = model.module   # unwrapped reference for checkpointing
    else:
        _model = model

    # ── dataset / loader ──────────────────────────────────────────────────────
    rank0_print(
        f"Loading VST_Train_Dataset from {args.json_path} "
        f"(results: {args.vst_results_dir})"
    )
    train_dataset = VST_Train_Dataset(
        json_path          = args.json_path,
        results_dir        = args.vst_results_dir,
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
        train_dataset,
        batch_size  = 1,
        shuffle     = (train_sampler is None),
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        sampler     = train_sampler,
    )

    # ── test datasets (for periodic LM loss + first-token acc evaluation) ─────
    # Uses Eval_Dataset_Coord so that image_xyz is loaded from pts3d and passed
    # to the model at eval time — matching the training input distribution for
    # all xyz-using modes (default 4D / decouple).
    # coord_upscale=1 to skip the unused image_xyz_hires (save memory).
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
            coord_upscale      = 1,
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
            sampler=_eval_sampler,
        )
        test_samplers[_ds_name] = _eval_sampler
        log.info(f"Eval dataset '{_ds_name}': {len(ds)} samples")
        

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

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

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):

            # ── move batch to device (input template aligned with train_atten.py) ─
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            # mm_token_type_ids: emitted by AutoProcessor. For atten this is
            # required (vision_mask derivation); for 4D / decouple it lets
            # get_rope_index identify vision-token positions cleanly rather
            # than rederiving from input_ids token ids.
            mm_token_type_ids = batch.get("mm_token_type_ids")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(device)

            # Move 3D position maps to device (list of tensors or None)
            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(device) for xyz in image_xyz]

            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            # ── forward + loss ────────────────────────────────────────────────
            _, loss, _ldict = model(
                input_ids         = input_ids,
                attention_mask    = attention_mask,
                pixel_values      = pixel_values,
                image_grid_thw    = image_grid_thw,
                image_xyz         = image_xyz,
                mm_token_type_ids = mm_token_type_ids,
                labels            = labels,
            )
            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: loss is None, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / args.grad_accum
                running_loss = 0.0

                if world_size > 1:
                    _t = torch.tensor([avg_loss], dtype=torch.float64, device=device)
                    dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                    avg_loss = (_t / world_size)[0].item()

                if local_rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log.info(
                        f"[train] epoch={epoch+1:02d}  global_step={global_step:05d}  "
                        f"loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                        f"(aggregated across {world_size} GPU{'s' if world_size > 1 else ''})"
                    )
                    if use_wandb:
                        wandb.log(
                            {"train/loss": avg_loss,
                             "train/lr":   current_lr,
                             "epoch":      epoch + 1},
                            step=global_step,
                        )
                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir, global_step)

                # ── periodic eval (deploy-aligned generative, mirrors train_atten.py) ─
                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    _spa = _model.spa_model if hasattr(_model, "spa_model") else _model
                    _stop_criteria = StoppingCriteriaList(
                        [_StopOnAnswerClose(tokenizer)]
                    )

                    # Disable GC for KV cache during generate()
                    _spa_gc_flag = getattr(_spa, "gradient_checkpointing", False)
                    _lm = _resolve_language_model(_spa)
                    _lm_gc_flag = getattr(_lm, "gradient_checkpointing", False) if _lm else False
                    if _spa_gc_flag:    _spa.gradient_checkpointing = False
                    if _lm and _lm_gc_flag: _lm.gradient_checkpointing = False

                    for ds_name, loader in test_loaders.items():
                        if ds_name in test_samplers and test_samplers[ds_name] is not None:
                            test_samplers[ds_name].set_epoch(global_step)
                        loss_sum, acc_sum, count = 0.0, 0.0, 0
                        for tb in loader:
                            t_ids   = tb["input_ids"].to(device)
                            t_mask  = tb["attention_mask"].to(device)
                            t_pv    = tb.get("pixel_values")
                            t_thw   = tb.get("image_grid_thw")
                            t_mm    = tb.get("mm_token_type_ids")
                            t_lbl   = tb.get("labels")
                            t_xyz   = tb.get("image_xyz")
                            if t_pv  is not None: t_pv  = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None: t_thw = t_thw.to(device)
                            if t_mm  is not None: t_mm  = t_mm.to(device)
                            if t_lbl is not None: t_lbl = t_lbl.to(device)
                            if t_xyz is not None: t_xyz = [x.to(device) for x in t_xyz]

                            with torch.no_grad():
                                # ── lm_loss: cheap teacher-forced forward over
                                # full sequence. Input template mirrors the
                                # training step (incl. mm_token_type_ids) +
                                # mode-conditional xyz kwargs.
                                _fwd_kwargs = dict(
                                    input_ids            = t_ids,
                                    attention_mask       = t_mask,
                                    pixel_values         = t_pv,
                                    image_grid_thw       = t_thw,
                                    mm_token_type_ids    = t_mm,
                                    output_hidden_states = False,
                                    return_dict          = True,
                                )
                                if not args.vanilla and t_xyz is not None:
                                    _fwd_kwargs["image_xyz"] = t_xyz
                                out = _spa(**_fwd_kwargs)
                                logits = out.logits
                                sl = logits[..., :-1, :].contiguous()
                                sb = t_lbl[..., 1:].contiguous()
                                lm_loss = F.cross_entropy(
                                    sl.view(-1, sl.size(-1)),
                                    sb.view(-1),
                                    ignore_index=-100,
                                )
                                loss_sum += lm_loss.item()
                                count += 1

                                # ── Generative acc (deploy-aligned with
                                # evaluation.py): slice prompt to right before
                                # the supervised answer span, autoregressively
                                # generate, then run extract_answer_letter
                                # on the decoded text. No fixed-offset probe →
                                # no letter/non-letter bias. Matches
                                # train_atten.py periodic eval.
                                ans_idx = (t_lbl[0] != -100).nonzero(as_tuple=False).flatten()
                                if ans_idx.numel() == 0:
                                    continue
                                ans_start = ans_idx[0].item()

                                p_ids  = t_ids[:, :ans_start]
                                p_mask = t_mask[:, :ans_start]
                                p_mm   = t_mm[:, :ans_start] if t_mm is not None else None
                                _coord_scale = 100.0
                                # Inner backbone for mode-specific position prep:
                                #   vanilla  → Qwen3_5ForConditionalGeneration  (no prep)
                                #   decouple → SpaDecForConditionalGeneration → .model = SpaDecModel  (._compute_xyz_pos)
                                #   default  → SpaForConditionalGeneration → .model = SpaModel       (.get_rope_index)
                                _backbone = _spa.base_model.model if hasattr(_spa, "base_model") else _spa

                                # Per-mode generate prep + call. Each branch builds its own
                                # gen_kwargs and calls _spa.generate(...) directly — no
                                # indirection through evaluation.run_inference_spa.
                                if args.vanilla:
                                    # Stock Qwen 3D M-RoPE: HF computes position_ids itself.
                                    generated = _spa.generate(
                                        input_ids         = p_ids,
                                        attention_mask    = p_mask,
                                        pixel_values      = t_pv,
                                        image_grid_thw    = t_thw,
                                        mm_token_type_ids = p_mm,
                                        max_new_tokens    = _EVAL_MAX_NEW_TOKENS,
                                        do_sample         = False,
                                        pad_token_id      = tokenizer.eos_token_id,
                                        stopping_criteria = _stop_criteria,
                                    )
                                elif args.decouple:
                                    # Decouple: keep Qwen 3D M-RoPE in rotary dims +
                                    # new XYZ RoPE in pass-through. HF's generate() strips
                                    # non-standard kwargs (mm_token_type_ids, image_xyz), so
                                    # pre-compute xyz_pos on the prompt and stash on the
                                    # language_model; SpaDecModel.forward picks it up when
                                    # mm_token_type_ids is None on decode steps.
                                    xyz_pos = _backbone.model._compute_xyz_pos(
                                        input_ids         = p_ids,
                                        mm_token_type_ids = p_mm,
                                        image_grid_thw    = t_thw,
                                        attention_mask    = p_mask,
                                        image_xyz         = t_xyz,
                                    )
                                    _backbone.model.language_model._xyz_pos     = xyz_pos
                                    _backbone.model.language_model._coord_scale = _coord_scale
                                    generated = _spa.generate(
                                        input_ids         = p_ids,
                                        attention_mask    = p_mask,
                                        pixel_values      = t_pv,
                                        image_grid_thw    = t_thw,
                                        max_new_tokens    = _EVAL_MAX_NEW_TOKENS,
                                        do_sample         = False,
                                        pad_token_id      = tokenizer.eos_token_id,
                                        coord_scale       = _coord_scale,
                                        stopping_criteria = _stop_criteria,
                                    )
                                else:
                                    # Default 4D M-RoPE: pre-compute 5D position_ids so
                                    # generate()'s _prepare_position_ids_for_generation is
                                    # bypassed (otherwise SpaForConditionalGeneration's
                                    # *args/**kwargs forward signature defeats inspect-based
                                    # detection and the model falls back to 3D position_ids).
                                    position_ids, _ = _backbone.model.get_rope_index(
                                        input_ids         = p_ids,
                                        mm_token_type_ids = p_mm,
                                        image_grid_thw    = t_thw,
                                        video_grid_thw    = None,
                                        attention_mask    = p_mask,
                                        image_xyz         = t_xyz,
                                        coord_scale       = _coord_scale,
                                    )
                                    _gen_kwargs = dict(
                                        input_ids         = p_ids,
                                        attention_mask    = p_mask,
                                        pixel_values      = t_pv,
                                        image_grid_thw    = t_thw,
                                        position_ids      = position_ids,
                                        max_new_tokens    = _EVAL_MAX_NEW_TOKENS,
                                        do_sample         = False,
                                        pad_token_id      = tokenizer.eos_token_id,
                                        coord_scale       = _coord_scale,
                                        stopping_criteria = _stop_criteria,
                                    )
                                    if t_xyz is not None:
                                        _gen_kwargs["image_xyz"] = t_xyz
                                    generated = _spa.generate(**_gen_kwargs)

                            # Decode model output and supervised GT, extract letter.
                            trimmed = generated[0][p_ids.shape[1]:]
                            pred_text = tokenizer.decode(
                                trimmed.tolist(), skip_special_tokens=True,
                            )
                            pred_letter = extract_answer_letter(pred_text)

                            gt_ids = t_lbl[0, ans_start:]
                            gt_ids = gt_ids[gt_ids != -100]
                            gt_text = tokenizer.decode(
                                gt_ids.tolist(), skip_special_tokens=True,
                            )
                            gt_letter = extract_answer_letter(gt_text)

                            if (pred_letter and gt_letter
                                    and pred_letter.lower() == gt_letter.lower()):
                                acc_sum += 1.0

                        if world_size > 1:
                            stats = torch.tensor([loss_sum, acc_sum, count],
                                                 dtype=torch.float64, device=device)
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            loss_sum, acc_sum, count = (
                                stats[0].item(), stats[1].item(), int(stats[2].item()),
                            )

                        if count > 0 and local_rank == 0:
                            avg_l = loss_sum / count
                            avg_a = acc_sum / count
                            log.info(
                                f"[eval] global_step={global_step:05d}  "
                                f"{ds_name}_lm_loss={avg_l:.4f}  "
                                f"{ds_name}_acc={avg_a:.4f}  (n={count})"
                            )
                            if use_wandb:
                                wandb.log(
                                    {f"eval/{ds_name}_lm_loss": avg_l,
                                     f"eval/{ds_name}_acc":     avg_a},
                                    step=global_step,
                                )

                    if _spa_gc_flag:    _spa.gradient_checkpointing = True
                    if _lm and _lm_gc_flag: _lm.gradient_checkpointing = True
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
    model: nn.Module,
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
    log.info(f"Checkpoint saved → {ckpt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of SpaForConditionalGeneration "
                    "for answer prediction with optional 4D M-RoPE spatial conditioning."
    )
    p.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"),
        help="Path to Qwen3.5-VL checkpoint",
    )
    p.add_argument(
        "--json_path",
        default=os.path.join(_ROOT, "datasets/train/VST_parsed/vst_500k.json"),
        help="Path to VST training JSON (vst_500k.json).",
    )
    p.add_argument(
        "--vst_results_dir",
        default=os.path.join(_ROOT, "datasets/train/VST/3d_results"),
        help="Root of VST 3d_results tree (subdirs per task family).",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_correspondence"),
    )
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int,   default=100,
                   help="Linear warmup steps before cosine decay.")
    p.add_argument("--lora_rank",    type=int,   default=16,
                   help="LoRA rank r")
    p.add_argument("--max_images",   type=int,   default=4,
                   help="Max images per scene (memory budget)")
    p.add_argument("--grad_accum",   type=int,   default=16,
                   help="Gradient accumulation steps")
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--eval_steps",   type=int,   default=100)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Truncate dataset to this many samples (None = use all)")
    p.add_argument(
        "--train_vision",
        action="store_true",
        help="Also unfreeze the vision encoder (ViT) for fine-tuning",
    )
    p.add_argument(
        "--vanilla",
        action="store_true",
        help="Use original Qwen 3D M-RoPE instead of 4D M-RoPE (no image_xyz). "
             "only LM answer loss.",
    )
    p.add_argument(
        "--decouple",
        action="store_true",
        help="Decoupled position embedding: keep Qwen original 3D M-RoPE [11,11,10] "
             "in the rotary 64 dims (UNCHANGED) and add a NEW XYZ RoPE (66 dims, "
             "sequential x|y|z each 11 bands, rope_theta=10000) in pass-through "
             "dims 64..129. Text tokens default to xyz=(0,0,0) → identity rotation. "
             "Mutually exclusive with --vanilla.",
    )
    p.add_argument(
        "--xyz_rope_dim",
        type=int, default=66,
        help="Total head_dim units allocated to the XYZ RoPE in the pass-through "
             "region under --decouple (each axis x/y/z gets xyz_rope_dim/6 "
             "frequency bands). Must be a positive multiple of 6 ≤ 192 "
             "(pass-through region size). Default 66 (= 11 bands per axis). "
             "No effect without --decouple.",
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
