"""
train_atten.py

LoRA fine-tuning of Qwen3.5-VL with per-layer geometric attention bias
(spatial_attention_block.py + spatial_attention_llm.py).

Full design notes — see md/model_design/spatial_attention.md.

Architecture (single mode, no flags to switch it):
    AnswerOnlyModel                                ← LM-loss-only training shell
    └── Qwen3_5ForConditionalGeneration  [LoRA + per-layer bias]
         ├── Qwen3_5VisionModel (ViT, frozen)
         └── SpatialAttnVanillaModel               ← swapped inline (build_model);
              │                                      pads multi-sample image_xyz
              │                                      via pad_sequence → (B, N_max, 3)
              └── SpatialAttnVanillaTextModel      ← punches V↔V hole in
                   │                                  causal_mask (prefill only)
                   │                                  so geometry can flow both ways
                   └── decoder layers — every STANDARD-attn self_attn is wrapped
                       by SpatialAttnWrapper (linear-attn layers are skipped:
                       their kernel ignores the additive attention_mask, so
                       the bias would be dropped silently and DDP would flag
                       the unused params)
                       (per-pair 2-layer MLP: 4 → hidden → GELU → num_heads,
                        added as per-head bias on vision-vision attention,
                        with torch.where defense to preserve causal/padding
                        barriers from the additive sum)

Position embedding: Qwen3.5 ORIGINAL 3D M-RoPE [11, 11, 10] (UNCHANGED).
The spatial signal enters only through the per-layer attention-bias module;
position_ids stay exactly as Qwen ships them.

Loss: LM answer cross-entropy only (no contrast / match heads).

Three-mask safety system (all parameter-free):
  1. vision_mask          (in SpatialAttentionBias) — bias ≡ 0 outside V×V cells.
  2. V↔V prefix-mask hole (in SpatialAttnVanillaTextModel) — opens the upper
     triangle of the V×V sub-block so geometric bias works in both directions.
  3. torch.where defense  (in SpatialAttnWrapper) — re-pins any cell whose
     original mask was a hard barrier (< -1e4), so bias can never punch through.

What gets trained:
  • LoRA adapters on q/k/v/o/gate/up/down_proj
  • Per-layer SpatialAttentionBias.mlp — a 2-layer MLP
        Linear(4 → hidden_dim) → GELU → Linear(hidden_dim → num_heads)
    Input is the per-pair edge feature (n_x, n_y, n_z, d) where (n_x,n_y,n_z)
    is the unit direction of (p_j − p_i) and d is its magnitude. Output W₂
    layer is zero-initialized so B ≡ 0 at step 0; W₁ keeps Kaiming default.
    Modules are added AFTER LoRA wrapping by patch_attention_layers_spatial.

Batch handling:
  Model side accepts `image_xyz` as either list[Tensor] (B=1 shorthand) or
  list[list[Tensor]] (true batched form), and pads to (B, N_max, 3) via
  torch.nn.utils.rnn.pad_sequence inside SpatialAttnVanillaModel.forward.
  NOTE: `collate_fn` below still asserts len(batch)==1 — the model is ready
  for B>1 but the collate function needs a multi-sample upgrade (text-side
  pad + multi-sample image_xyz collection) before it can actually be enabled.

Checkpoint format:
  step_<N>/
    adapter_*           ← PEFT LoRA adapter (saved via save_pretrained)
    spatial_bias.pt     ← {fully-qualified module name → state_dict} for every
                          SpatialAttentionBias instance. Load via
                          load_spatial_bias_modules() at eval / inference time.
                          Keys per module: mlp.0.{weight,bias}, mlp.2.{weight,bias}
                          — pre-MLP checkpoints (proj.{weight,bias}) are NOT
                          loadable; re-train from the LoRA adapter.

Usage:
    torchrun --nproc_per_node 4 train_atten.py \
        --model_path checkpoints/Qwen3.5-4B \
        --output_dir train_records/spatial_attn
"""

import argparse
import datetime
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
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
    AnswerOnlyModel,
    SpatialAttnVanillaModel,
    SpatialAttentionBias,
    patch_attention_layers_spatial,
)
from src.dataset import (
    VST_Train_Dataset,
    MindCube_Train_Dataset,
    Eval_Dataset_Coord,
    compute_letter_offset,
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
    # NOTE: model side (SpatialAttnVanillaModel + SpatialAttentionBias) is
    # already batch-aware — image_xyz gets pad_sequence'd to (B, N_max, 3) and
    # bias scatters per-sample. This collate is the remaining bottleneck:
    # before lifting the assert, also pad the text-side tensors (input_ids,
    # attention_mask, labels) and gather image_xyz into list[list[Tensor]]
    # form (one inner list per sample). See md/model_design/spatial_attention.md
    # §5 (Batch Handling).
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# ── model building ────────────────────────────────────────────────────────────

def build_model(
    model_path:            str,
    lora_rank:             int   = 16,
    freeze_vision:         bool  = True,
    bias_init_scale:       float = 0.0,
    freeze_linear_attn:    bool  = False,
    couple:                bool  = False,
) -> nn.Module:
    """
    Load stock Qwen3.5-VL → swap inner backbone for SpatialAttnVanillaModel
    (so `image_xyz` → _spatial_cache routing works) → freeze vision → apply
    LoRA → patch attention with SpatialAttnWrapper → return AnswerOnlyModel.

    RoPE stays exactly as Qwen ships it (3D M-RoPE).  The only spatial
    conditioning added on top is the per-layer SpatialAttentionBias.
    """
    spa = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype        = torch.bfloat16,
        attn_implementation= "sdpa",
    )

    # ── Swap the backbone for the cache-aware variant ─────────────────────────
    # SpatialAttnVanillaModel inherits from Qwen3_5Model and adds NO new
    # parameters — its forward override only populates _spatial_cache before
    # delegating. So state_dict shapes/keys match exactly and a strict load
    # carries pretrained weights over with no remap.
    new_inner = SpatialAttnVanillaModel(spa.config)
    new_inner.load_state_dict(spa.model.state_dict(), strict=True)
    spa.model = new_inner.to(dtype=torch.bfloat16)
    # Re-tie lm_head.weight ↔ embed_tokens.weight after the swap (the original
    # tying broke when we replaced spa.model). _tied_weights_keys is configured
    # on Qwen3_5ForConditionalGeneration; tie_weights() honors it.
    spa.tie_weights()

    if freeze_vision:
        for p in spa.model.visual.parameters():
            p.requires_grad_(False)
        log.info("Vision encoder frozen.")

    # ── LoRA on the language model ────────────────────────────────────────────
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    target_modules: list[str] | str = proj_names
    if freeze_linear_attn:
        # Inspect layer_type BEFORE PEFT wrapping. Decoder layers live under
        # spa.model.language_model.layers (Qwen3.5-VL keeps visual and LM
        # separate). Reuse _resolve_language_model so we don't hard-code the
        # path. Build a regex that only matches full-attn layer indices, so
        # PEFT skips linear-attn layers entirely — no LoRA modules are created.
        _lm_pre = _resolve_language_model(spa)
        if _lm_pre is None:
            raise RuntimeError(
                "freeze_linear_attn=True but failed to locate language_model "
                "with .layers under spa — cannot determine layer types."
            )
        full_idx, lin_idx = [], []
        for i, layer in enumerate(_lm_pre.layers):
            (lin_idx if getattr(layer, "layer_type", None) == "linear_attention"
                     else full_idx).append(i)
        if not lin_idx:
            log.info("freeze_linear_attn=True but no linear-attn layers found; "
                     "applying LoRA to all layers.")
        else:
            idx_alt = "|".join(str(i) for i in full_idx)
            projs   = "|".join(proj_names)
            # PEFT runs re.fullmatch against the full module name; restrict
            # match to layers.{full_idx}.(self_attn|mlp).{proj}.
            target_modules = (
                rf"^.*\.layers\.({idx_alt})\.(self_attn|mlp)\.({projs})$"
            )
            log.info(
                f"freeze_linear_attn=True: LoRA on {len(full_idx)} full-attn "
                f"layers, skipping {len(lin_idx)} linear-attn layers "
                f"(idx={lin_idx[:8]}{'...' if len(lin_idx) > 8 else ''})."
            )
    lora_cfg = LoraConfig(
        r              = lora_rank,
        lora_alpha     = lora_rank * 2,
        target_modules = target_modules,
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    spa = get_peft_model(spa, lora_cfg)
    spa.print_trainable_parameters()

    # ── Wrap every self_attn with SpatialAttnWrapper (AFTER LoRA) ─────────────
    n_wrapped = patch_attention_layers_spatial(spa, bias_init_scale=bias_init_scale)
    log.info(f"Wrapped {n_wrapped} self_attn layers with SpatialAttnWrapper.")

    # Cast newly-created bias_module params to bf16 so they match the rest
    # of the (bf16) backbone. patch_attention_layers_spatial creates fresh
    # nn.Linears AFTER spa.model.to(bf16), so without this they default to
    # fp32 → SDPA bias-dtype mismatch (must equal query.dtype).
    for module in spa.modules():
        if isinstance(module, SpatialAttentionBias):
            module.to(dtype=torch.bfloat16)

    # SpatialAttentionBias.mlp params start with requires_grad=True (fresh
    # nn.Linears inside an nn.Sequential) — they are NOT inside the LoRA
    # freeze. Count them for sanity.
    n_bias_params = sum(
        p.numel() for n, p in spa.named_parameters()
        if "bias_module" in n and p.requires_grad
    )
    log.info(
        f"Trainable SpatialAttentionBias params: {n_bias_params} "
        f"(={n_wrapped} layers × 2-layer MLP "
        f"[Linear(4→hidden) + GELU + Linear(hidden→num_heads)])"
    )

    # ── decoupled (bias-only) mode ────────────────────────────────────────────
    # When --couple is OFF (default), freeze every trainable parameter that is
    # NOT inside a SpatialAttentionBias module — including LoRA adapters. This
    # gives 100% of the gradient signal to the geometric MLP so W₁ can actually
    # learn instead of staying near random init while LoRA absorbs the task.
    if not couple:
        n_frozen = 0
        for name, p in spa.named_parameters():
            if "bias_module" not in name and p.requires_grad:
                p.requires_grad_(False)
                n_frozen += 1
        log.info(
            f"[couple=False] froze {n_frozen} non-bias_module parameter tensors "
            f"(LoRA + everything else); only SpatialAttentionBias trains."
        )
    else:
        log.info("[couple=True] joint training: LoRA + SpatialAttentionBias both train.")

    # Gradient checkpointing — same pattern as train_correspondence.py
    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = _resolve_language_model(spa)
    if lm is not None and not getattr(lm, "gradient_checkpointing", False):
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    # use_xyz=True: AnswerOnlyModel passes image_xyz down so the
    # SpatialAttnVanillaModel.forward can populate _spatial_cache for the
    # per-layer attention bias. coord_scale is not used in vanilla mode
    # (the bias module learns its own scale via Linear).
    return AnswerOnlyModel(spa, use_xyz=True)


# ── helpers: walk through PEFT/DDP layers to find language_model ──────────────

def _resolve_language_model(root: nn.Module) -> nn.Module | None:
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


# ── checkpointing: PEFT adapter + spatial_bias.pt ────────────────────────────

def _save_checkpoint(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    step: int,
    suffix: str = "",
) -> None:
    """Save PEFT adapter + the per-layer SpatialAttentionBias state_dicts."""
    tag  = f"step_{step}" + (f"_{suffix}" if suffix else "")
    ckpt = os.path.join(output_dir, tag)
    os.makedirs(ckpt, exist_ok=True)

    # PEFT adapter (LoRA)
    model.spa_model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    # SpatialAttentionBias modules — keyed by their fully-qualified module name
    bias_state: dict = {}
    for name, mod in model.spa_model.named_modules():
        if isinstance(mod, SpatialAttentionBias):
            bias_state[name] = {k: v.detach().cpu()
                                for k, v in mod.state_dict().items()}
    if bias_state:
        torch.save(bias_state, os.path.join(ckpt, "spatial_bias.pt"))
        log.info(
            f"Checkpoint saved → {ckpt} "
            f"(LoRA + spatial_bias.pt with {len(bias_state)} layers)"
        )
    else:
        log.warning(
            f"Checkpoint saved → {ckpt} but no SpatialAttentionBias modules "
            f"found — was patch_attention_layers_spatial called?"
        )


def load_spatial_bias_modules(model: nn.Module, ckpt_dir: str) -> int:
    """
    Reload spatial_bias.pt into a freshly-built model. Returns the number of
    modules loaded. Use at eval / inference time after instantiating the model
    and calling patch_attention_layers_spatial.
    """
    path = os.path.join(ckpt_dir, "spatial_bias.pt")
    if not os.path.isfile(path):
        log.warning(f"No spatial_bias.pt at {path} — skipping.")
        return 0
    bias_state = torch.load(path, map_location="cpu")
    n_loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, SpatialAttentionBias) and name in bias_state:
            mod.load_state_dict(bias_state[name])
            n_loaded += 1
    log.info(f"Loaded spatial_bias.pt: {n_loaded}/{len(bias_state)} modules.")
    return n_loaded


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    # ── DDP init ──────────────────────────────────────────────────────────────
    _env_rank = os.environ.get("LOCAL_RANK")
    if _env_rank is not None:
        local_rank = int(_env_rank)
        # 1h NCCL timeout (default 10min) — kept generous so any periodic-eval
        # rank skew or slow ckpt save can't trip the watchdog.
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

    # ── per-rank log file ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
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

    log.info(f"[CONFIG] {vars(args)}")

    # ── processor / model ─────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    # MCQ letter-probe: position of the letter token inside `<answer>X</answer>`.
    # Qwen3.5 BPE merges `>X` into one token, so the letter sits at offset 2,
    # not at ans_start (which is the `<` of `<answer>`).
    letter_offset = compute_letter_offset(tokenizer)
    model = build_model(
        args.model_path,
        lora_rank          = args.lora_rank,
        freeze_vision      = not args.train_vision,
        bias_init_scale    = args.bias_w2_init_scale,
        freeze_linear_attn = args.freeze,
        couple             = args.couple,
    ).to(device)

    # ── spatial_merge_size from config.json ───────────────────────────────────
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        _model = model.module
    else:
        _model = model

    # ── train dataset (VST or MindCube, dispatched via --dataset) ─────────────
    if args.dataset == "mindcube":
        rank0_print(
            f"Loading MindCube_Train_Dataset from {args.json_path} "
            f"(results: {args.vst_results_dir})"
        )
        train_dataset = MindCube_Train_Dataset(
            jsonl_path         = args.json_path,
            results_dir        = args.vst_results_dir,
            processor          = processor,
            log                = log,
            spatial_merge_size = spatial_merge_size,
            max_samples        = args.max_samples,
        )
    else:
        rank0_print(
            f"Loading VST_Train_Dataset from {args.json_path} "
            f"(results: {args.vst_results_dir})"
        )
        train_dataset = VST_Train_Dataset(
            json_path          = args.json_path,
            results_dir        = args.vst_results_dir,
            processor          = processor,
            log                = log,
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

    # ── eval datasets (MindCube tinybench + SpinBench + MMSIBench,
    #    all three with image_xyz) ─────────────────────────────────────────────
    _eval_dir = os.path.join(_ROOT, "datasets/evaluation")
    test_loaders, test_samplers = {}, {}
    for _ds_name, _ds_jsonl, _ds_results, _q_key, _a_key in [
        ("mindcube",
         os.path.join(_eval_dir, "MindCube", "MindCube_tinybench.jsonl"),
         os.path.join(_eval_dir, "MindCube", "3d_results"),
         "question", "gt_answer"),
        ("spinbench",
         os.path.join(_eval_dir, "spinbench_data", "test.jsonl"),
         os.path.join(_eval_dir, "spinbench_data", "3d_results"),
         "problem", "answer"),
        ("mmsibench",
         os.path.join(_eval_dir, "MMSIBench", "data", "test_data_final.json"),
         os.path.join(_eval_dir, "MMSIBench", "3d_results"),
         "question", "answer"),
    ]:
        ds = Eval_Dataset_Coord(
            _ds_jsonl, _ds_results, processor, log,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = 1,
            max_samples        = args.max_eval_samples,
            question_key       = _q_key,
            answer_key         = _a_key,
        )
        _es = (DistributedSampler(ds, num_replicas=world_size,
                                    rank=local_rank, shuffle=False)
                if world_size > 1 else None)
        test_loaders[_ds_name] = DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            sampler=_es,
        )
        test_samplers[_ds_name] = _es
        log.info(f"Eval dataset '{_ds_name}': {len(ds)} samples")


    # ── optimiser ─────────────────────────────────────────────────────────────
    # bias_module params get a boosted LR so W₂ (zero/small-init output layer)
    # builds up meaningful scale faster, allowing gradients to flow to W₁.
    bias_params  = [(n, p) for n, p in model.named_parameters()
                    if p.requires_grad and "bias_module" in n]
    other_params = [(n, p) for n, p in model.named_parameters()
                    if p.requires_grad and "bias_module" not in n]
    trainable = [p for _, p in other_params] + [p for _, p in bias_params]
    bias_lr = args.lr * args.bias_lr_scale
    log.info(
        f"Optimizer param groups: "
        f"LoRA/other {len(other_params)} params @ lr={args.lr:.2e}  |  "
        f"bias_module {len(bias_params)} params @ lr={bias_lr:.2e} "
        f"(scale={args.bias_lr_scale}x)"
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for _, p in other_params], "lr": args.lr},
            {"params": [p for _, p in bias_params],  "lr": bias_lr},
        ],
        weight_decay=0.01,
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    # ── WandB (rank 0) ────────────────────────────────────────────────────────
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
        log.warning("wandb not installed — `pip install wandb` to enable logging.")

    global_step  = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            # mm_token_type_ids: emitted by AutoProcessor, REQUIRED for our
            # SpatialAttnVanillaModel to derive vision_mask. If we forget to
            # forward it, _spatial_cache stays None → wrappers bypass →
            # bias_module params receive no gradient → DDP failure.
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
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

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

                # ── periodic eval (MCQ letter-probe via teacher-forced argmax) ─
                # Free-form generative eval lives in evaluation.py. Here we run
                # a single teacher-forced forward and check whether the logit
                # that *predicts* the letter token argmaxes onto the GT letter
                # token id. Cheap (one forward, no autoregressive decode) and
                # monotone with generative MCQ acc, but not bit-equivalent —
                # see evaluation.py for the deploy-aligned numbers.
                if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    _spa = _model.spa_model

                    for ds_name, loader in test_loaders.items():
                        if test_samplers.get(ds_name) is not None:
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
                                out = _spa(
                                    input_ids            = t_ids,
                                    attention_mask       = t_mask,
                                    pixel_values         = t_pv,
                                    image_grid_thw       = t_thw,
                                    image_xyz            = t_xyz,
                                    mm_token_type_ids    = t_mm,
                                    output_hidden_states = False,
                                    return_dict          = True,
                                )
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

                                ans_idx = (t_lbl[0] != -100).nonzero(as_tuple=False).flatten()
                                if ans_idx.numel() == 0:
                                    continue
                                # ans_start is the `<` of `<answer>`; the letter
                                # token sits at ans_start + letter_offset (Qwen3.5
                                # BPE merges `>X` into one token so it's not at
                                # ans_start itself). The logit that predicts
                                # position p is logits[p-1].
                                letter_pos = ans_idx[0].item() + letter_offset
                                if letter_pos == 0 or letter_pos >= t_lbl.shape[1]:
                                    continue
                                pred_id = logits[0, letter_pos - 1].argmax().item()
                                gt_id   = t_lbl[0, letter_pos].item()
                                if pred_id == gt_id:
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
        description="LoRA + per-layer SpatialAttentionBias fine-tuning, LM loss only."
    )
    p.add_argument("--model_path",
                   default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"))
    p.add_argument("--json_path",
                   default=os.path.join(_ROOT, "datasets/train/VST_parsed/vst_500k.json"),
                   help="VST entries JSON (e.g. vst_500k.json).")
    p.add_argument("--vst_results_dir",
                   default=os.path.join(_ROOT, "datasets/train/VST/3d_results"),
                   help="Root of VST 3d_results tree (subdirs per task family). "
                        "For --dataset mindcube, point this at the flat MindCube "
                        "3d_results dir instead.")
    p.add_argument("--dataset", choices=["vst", "mindcube"], default="vst",
                   help="Train dataset family: 'vst' uses VST_Train_Dataset "
                        "(JSON list + subdir results tree); 'mindcube' uses "
                        "MindCube_Train_Dataset (JSONL + flat results dir).")
    p.add_argument("--output_dir",
                   default=os.path.join(_ROOT, "train_records/spatial_attn"))
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--warmup_steps",type=int,   default=100,
                   help="Linear warmup steps before cosine decay.")
    p.add_argument("--lora_rank",   type=int,   default=16)
    p.add_argument("--grad_accum",  type=int,   default=16)
    p.add_argument("--save_steps",  type=int,   default=200)
    p.add_argument("--eval_steps",  type=int,   default=100)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Truncate dataset to this many samples (per source).")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Truncate each eval dataset (mindcube/spinbench/mmsibench) to "
                        "this many samples. Useful for smoke tests.")
    p.add_argument("--train_vision", action="store_true",
                   help="Also unfreeze the ViT.")
    p.add_argument("--freeze", action="store_true",
                   help="Do not apply LoRA to linear-attention layers — only "
                       "full-attention layers (where SpatialAttentionBias also "
                       "lives) receive LoRA on q/k/v/o + gate/up/down. "
                       "Linear-attn layer params stay fully frozen.")
    p.add_argument("--bias_lr_scale", type=float, default=10.0,
                   help="LR multiplier for SpatialAttentionBias params vs LoRA. "
                        "bias_module gets lr * bias_lr_scale. Default 10.")
    p.add_argument("--bias_w2_init_scale", type=float, default=0.01,
                   help="W₂ (output layer) init std for SpatialAttentionBias. "
                        "0 = exact zero-init (pretrained behavior preserved but "
                        "W₁ gradients dead until W₂ moves); >0 = N(0,scale) init "
                        "allowing W₁ to train from step 0. Default 0.01.")
    p.add_argument("--couple", action="store_true",
                   help="Joint-train SpatialAttentionBias + LoRA together "
                        "(historical behavior). Default OFF: only the bias "
                        "module is trainable, LoRA adapters stay frozen at "
                        "init so all gradient signal reaches W₁/W₂.")
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity",  default="")
    p.add_argument("--wandb_run_name", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
