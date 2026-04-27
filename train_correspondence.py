"""
train_correspondence.py

LoRA fine-tuning of SpaForConditionalGeneration (Qwen3.5-VL) to predict
answers from multi-image prompts with optional 4D M-RoPE spatial conditioning.

Architecture:
    AnswerOnlyModel / AnswerRelativeModel
    ├── SpaForConditionalGeneration  [backbone + LoRA adapters]
    │    ├── SpaVisionModel (ViT, frozen)
    │    └── SpaModel (LLM + 4D M-RoPE)

Training strategy:
  - Each SPAR entry = one scene with N images
    - Prompt uses images + question (no pairwise camera-language template)
    - Supervision is LM answer loss (with optional xyz conditioning in RoPE)

Coordinate convention:
    - For non-vanilla mode, per-patch xyz is used in vision-token M-RoPE
    - For --relative mode, coordinates are transformed per query frame

Usage:
  python train_correspondence.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_correspondence
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import (
    AnswerOnlyModel,
    AnswerRelativeModel,
    SpaForConditionalGeneration,
    SpaRelativeForConditionalGeneration,
    SpaDecForConditionalGeneration,
    patch_attention_layers,
    patch_attention_layers_dec,
)
from src.dataset import (
    MindCube_Train_Dataset,
    MindCube_Train_Dataset_Relative,
    SAT_Train_Dataset,
    SAT_Train_Dataset_Relative,
    Eval_Dataset_Coord,
)
from torch.utils.data import ConcatDataset

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

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
    lora_rank:     int = 16,
    freeze_vision: bool = True,
    vanilla:       bool = False,
    polar:         bool = False,
    relative:      bool = False,
    decouple:      bool = False,
    xyz_rope_dim:  int  = 66,
) -> nn.Module:
    """
    Load backbone, patch M-RoPE, apply LoRA, return an answer model.

    vanilla=False  → 4D M-RoPE (t, x, y, z) with image_xyz spatial embedding
    vanilla=True   → original 3D M-RoPE, no image_xyz
    polar=True     → use the decouple architecture (Qwen 3D M-RoPE in the
                     rotary 64 dims + new XYZ RoPE in pass-through dims
                     64..129) BUT feed log-spherical (log r, θ, α) into the
                     XYZ RoPE. Matches xyz_to_polar convention:
                         log r = log||xyz||,
                         θ     = atan2(y, x) ∈ [-π, π],
                         α     = atan2(√(x²+y²), z) ∈ [0, π].
                     Text tokens stay xyz=(0,0,0) → identity rotation.
                     Mutually exclusive with --vanilla and --decouple.
    relative=True  → per-query-frame coordinate transform (SpaRelativeForConditionalGeneration);
                     dataset must return image_xyz_relative instead of image_xyz;
                     incompatible with vanilla; defaults to polar coordinates
    decouple=True  → keep Qwen original 3D M-RoPE [11,11,10] in the rotary 64
                     dims (UNCHANGED) and add a NEW XYZ RoPE in dims 64..129
                     (66 dims, sequential x|y|z, rope_theta=10000) with
                     **Cartesian** xyz. Text tokens get xyz=(0,0,0). For
                     log-spherical input, use --polar (which is mutually
                     exclusive with --decouple). Mutually exclusive with
                     --vanilla / --relative / --polar.
    """
    if relative and vanilla:
        raise ValueError("--relative and --vanilla are mutually exclusive.")
    if polar and vanilla:
        raise ValueError("--polar and --vanilla are mutually exclusive.")
    if decouple and (vanilla or relative):
        raise ValueError(
            "--decouple is mutually exclusive with --vanilla / --relative."
        )
    if polar and decouple:
        raise ValueError(
            "--polar already implies the decouple architecture (with log-spherical "
            "XYZ RoPE); don't combine it with --decouple. Use --polar alone for "
            "log-spherical or --decouple alone for Cartesian."
        )
    if xyz_rope_dim % 6 != 0 or xyz_rope_dim <= 0 or xyz_rope_dim > 192:
        raise ValueError(
            f"--xyz_rope_dim must be a positive multiple of 6 ≤ 192 "
            f"(pass-through region); got {xyz_rope_dim}."
        )

    # --polar implies the decouple architecture with log-spherical XYZ RoPE in
    # the pass-through region. --decouple alone uses Cartesian xyz. The two
    # are mutually exclusive.
    use_decouple = decouple or polar
    polar_xyz    = polar
    effective_polar = polar or relative

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
        #             add new XYZ RoPE in pass-through dims 64..129.
        # If polar_xyz=True (triggered by --polar), the XYZ RoPE consumes
        # log-spherical (log r, θ, α) instead of raw Cartesian xyz. ──────────
        # Theta choice:
        #   decouple + Cartesian (--decouple): θ = 10000 — wider spectrum for
        #       raw meters (wavelength 6.3..29K ≈ covers 0.06m..300m at scale=100).
        #   decouple + log-spherical (--polar): θ = 1000 — narrower spectrum;
        #       polar's effective dynamic range (log r ≈ O(1-3), angles ≤ 2π) is
        #       smaller, so a tighter ladder keeps more of the 11 bands in the
        #       useful region.
        _xyz_theta = 1000.0 if polar_xyz else 10000.0
        _mode = "log-spherical (log r, θ, α)" if polar_xyz else "Cartesian (x, y, z)"
        log.info(
            f"mrope_section: {orig_section} (UNCHANGED — Qwen original 3D M-RoPE) "
            f"+ new XYZ RoPE ({xyz_rope_dim} dims, rope_theta={_xyz_theta:g}) in pass-through region "
            f"[input: {_mode}]"
        )
        spa = SpaDecForConditionalGeneration.from_pretrained(
            model_path,
            config             = config,
            torch_dtype        = torch.bfloat16,
            attn_implementation= "sdpa",
        )
        # Swap in the requested xyz_dim / theta. xyz_rotary_emb has no trainable
        # params (only an inv_freq buffer), so replacing it post-from_pretrained
        # is safe and happens before LoRA wrapping. SpaDecAttentionWrapper reads
        # xyz_dim from cos.shape[-1] at runtime, so no other change needed.
        if _xyz_theta != 10000.0 or xyz_rope_dim != 66:
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
        spa_cls = SpaRelativeForConditionalGeneration if relative else SpaForConditionalGeneration
        spa = spa_cls.from_pretrained(
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

    # ── patch attention layers for relative mode (after LoRA) ────────────────
    if relative:
        n = patch_attention_layers(spa)
        log.info(f"Wrapped {n} attention layers with SpaRelativeAttentionWrapper.")

    # ── patch attention layers for decouple mode (after LoRA) ────────────────
    if use_decouple:
        n = patch_attention_layers_dec(spa)
        log.info(f"Wrapped {n} attention layers with SpaDecAttentionWrapper.")

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

    # All modes use the same coord_scale = 100 convention
    # (4D / polar: cm-equivalent for M-RoPE position; decouple: cm-equivalent for
    # the pass-through XYZ RoPE → wavelength range 0.063m .. 272m at θ=10000)
    _coord_scale = 100.0

    if relative:
        log.info(
            f"AnswerRelativeModel (per-query-frame relative coords, polar={effective_polar}, "
            f"coord_scale={_coord_scale})"
        )
        return AnswerRelativeModel(spa, polar=effective_polar, coord_scale=_coord_scale)

    use_xyz = not vanilla
    log.info(
        f"AnswerOnlyModel (use_xyz={use_xyz}, polar={effective_polar and use_xyz}, "
        f"decouple={use_decouple}, coord_scale={_coord_scale})"
    )
    return AnswerOnlyModel(
        spa,
        use_xyz     = use_xyz,
        polar       = effective_polar and use_xyz,
        coord_scale = _coord_scale,
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    # ── DDP initialisation ────────────────────────────────────────────────────
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
    # (xyz_rope_dim, decouple, polar, lora_rank, etc.) regardless of mode.
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
        polar          = args.polar,
        relative       = args.relative,
        decouple       = args.decouple,
        xyz_rope_dim   = args.xyz_rope_dim,
    )

    # Toggle visual-interleave RoPE layout (t at high-freq end, x/y/z round-robin).
    # Only meaningful for the 4D M-RoPE path. --polar and --decouple both use
    # the Qwen original 3D M-RoPE in the rotary 64 dims (plus a separate XYZ
    # RoPE in pass-through), so interleave has no effect there.
    _uses_4d_mrope = not (args.vanilla or args.decouple or args.polar)
    if args.interleave_vision and _uses_4d_mrope:
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
    elif args.interleave_vision and args.vanilla:
        log.warning("--interleave_vision has no effect with --vanilla (3D M-RoPE).")
    elif args.interleave_vision and (args.decouple or args.polar):
        log.warning(
            "--interleave_vision has no effect with --decouple / --polar "
            "(Qwen original 3D M-RoPE in rotary region; XYZ RoPE in pass-through "
            "has its own symmetric spectrum)."
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
    # --datasets controls which sources are concatenated. Default: mindcube only.
    # Each source uses its own (json_path, results_dir) pair from CLI args, and
    # the per-mode dataset class (MindCube_*_Train_Dataset[_Relative] vs
    # SAT_Train_Dataset[_Relative]) handles format-specific parsing.
    _ds_classes = {
        ("mindcube", False): MindCube_Train_Dataset,
        ("mindcube", True):  MindCube_Train_Dataset_Relative,
        ("sat",      False): SAT_Train_Dataset,
        ("sat",      True):  SAT_Train_Dataset_Relative,
    }
    _ds_paths = {
        "mindcube": (args.json_path,     args.mindcube_results_dir),
        "sat":      (args.sat_json_path, args.sat_results_dir),
    }

    # Two-phase build: construct MindCube first (so we can use its size as the
    # SAT down-sample target when both sources are mixed), then construct SAT
    # with category-balanced sub-sampling. Single-source runs skip balancing.
    _is_mixed = (len(args.datasets) > 1)
    _built: dict[str, "torch.utils.data.Dataset"] = {}
    _build_order = sorted(
        args.datasets, key=lambda n: 0 if n == "mindcube" else 1
    )
    for _name in _build_order:
        if _name not in _ds_paths:
            raise ValueError(f"Unknown dataset '{_name}'. Choices: mindcube, sat.")
        _cls     = _ds_classes[(_name, args.relative)]
        _jp, _rd = _ds_paths[_name]
        rank0_print(f"Loading {_cls.__name__} from {_jp} (results: {_rd})")
        # SAT classes use json_path (JSON list); MindCube classes use jsonl_path.
        _path_kw = "json_path" if _name == "sat" else "jsonl_path"
        _kwargs: dict = dict(
            results_dir        = _rd,
            processor          = processor,
            log                = log,
            max_images         = args.max_images,
            spatial_merge_size = spatial_merge_size,
            max_samples        = args.max_samples,
        )
        _kwargs[_path_kw] = _jp

        # Mixed (MindCube + SAT): per-category balanced down-sample of SAT to
        # MindCube's size (~10k → ~1.67k per question_type × 6 cats).
        if _name == "sat" and _is_mixed and "mindcube" in _built:
            _kwargs["balanced_categories"] = True
            _kwargs["target_size"]         = len(_built["mindcube"])
            _kwargs["balance_seed"]        = 0

        _built[_name] = _cls(**_kwargs)

    # Preserve the user-facing dataset order (args.datasets) for the concat.
    _per_dataset_loaders = [_built[n] for n in args.datasets]

    if len(_per_dataset_loaders) == 1:
        train_dataset = _per_dataset_loaders[0]
    else:
        train_dataset = ConcatDataset(_per_dataset_loaders)
        _per_ds_sizes = {n: len(_built[n]) for n in args.datasets}
        rank0_print(
            f"ConcatDataset: {len(train_dataset)} total samples across "
            f"{len(_per_dataset_loaders)} datasets {args.datasets} "
            f"(per-dataset sizes: {_per_ds_sizes})"
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
    # all xyz-using modes (default 4D / polar / relative / decouple).
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
        try:
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
                relative           = args.relative,
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
        except Exception as exc:
            log.warning(f"Failed to load eval dataset '{_ds_name}': {exc}")


    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    # Create output directory (all ranks can do this safely with exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── logging to file ───────────────────────────────────────────────────────
    # Per-rank log file (all ranks)
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

    # Aggregate log file (rank 0 only, for multi-GPU)
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
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):

            # ── move batch to device ──────────────────────────────────────────
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            # Move 3D position maps to device (list of tensors or None)
            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(device) for xyz in image_xyz]

            # Relative mode: per-frame xyz (list of (N_frames, H, W, 3) tensors)
            image_xyz_relative = batch.get("image_xyz_relative")
            if image_xyz_relative is not None:
                image_xyz_relative = [xyz.to(device) for xyz in image_xyz_relative]

            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            # ── forward + loss ────────────────────────────────────────────────
            if args.relative:
                _, loss, loss_dict = model(
                    input_ids          = input_ids,
                    attention_mask     = attention_mask,
                    pixel_values       = pixel_values,
                    image_grid_thw     = image_grid_thw,
                    image_xyz_relative = image_xyz_relative,
                    labels             = labels,
                )
            else:
                _, loss, loss_dict = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    pixel_values   = pixel_values,
                    image_grid_thw = image_grid_thw,
                    image_xyz      = image_xyz,
                    labels         = labels,
                )
            
            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: loss is None, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            if loss_dict:
                for k, v in loss_dict.items():
                    running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

            # ── gradient accumulation ─────────────────────────────────────────
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

                    # ── checkpoint ────────────────────────────────────────────
                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir,
                                         global_step)

                # ── periodic evaluation on test sets ──────────────────────────
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
                        local_acc_sum  = 0.0
                        local_count    = 0
                        for test_batch in loader:
                            t_ids   = test_batch["input_ids"].to(device)
                            t_mask  = test_batch["attention_mask"].to(device)
                            t_pv    = test_batch.get("pixel_values")
                            t_thw   = test_batch.get("image_grid_thw")
                            t_labels = test_batch.get("labels")
                            t_xyz     = test_batch.get("image_xyz")
                            t_xyz_rel = test_batch.get("image_xyz_relative")
                            if t_pv is not None:
                                t_pv = t_pv.to(device, dtype=torch.bfloat16)
                            if t_thw is not None:
                                t_thw = t_thw.to(device)
                            if t_labels is not None:
                                t_labels = t_labels.to(device)
                            if t_xyz is not None:
                                t_xyz = [x.to(device) for x in t_xyz]
                            if t_xyz_rel is not None:
                                t_xyz_rel = [x.to(device) for x in t_xyz_rel]
                            try:
                                # Match training input distribution per mode:
                                #   vanilla       — no xyz kwargs (stock Qwen)
                                #   --relative    — image_xyz_relative (per-frame)
                                #   --polar       — image_xyz + polar=True (xyz→log-spherical in M-RoPE)
                                #   default/decouple — image_xyz
                                _eval_fwd_kwargs = dict(
                                    input_ids=t_ids, attention_mask=t_mask,
                                    pixel_values=t_pv, image_grid_thw=t_thw,
                                    return_dict=True,
                                    kv_cache=(ds_name == "spinbench"),
                                )
                                if not args.vanilla:
                                    if args.relative and t_xyz_rel is not None:
                                        _eval_fwd_kwargs["image_xyz_relative"] = t_xyz_rel
                                    elif t_xyz is not None:
                                        _eval_fwd_kwargs["image_xyz"] = t_xyz
                                    if args.polar:
                                        _eval_fwd_kwargs["polar"] = True
                                with torch.no_grad():
                                    out = _spa(**_eval_fwd_kwargs)
                                    logits = out.logits
                                    shift_logits = logits[..., :-1, :].contiguous()
                                    shift_labels = t_labels[..., 1:].contiguous()
                                    lm_loss = F.cross_entropy(
                                        shift_logits.view(-1, shift_logits.size(-1)),
                                        shift_labels.view(-1),
                                        ignore_index=-100,
                                    )
                                    local_loss_sum += lm_loss.item()

                                    # Top-1 accuracy on first answer token
                                    # (matches coordinate_llm.py convention).
                                    _mask  = shift_labels[0] != -100
                                    _sl_m  = shift_logits[0, _mask]
                                    _sb_m  = shift_labels[0, _mask]
                                    if _sl_m.numel() > 0:
                                        _pred = _sl_m[0].argmax(-1).item()
                                        _tgt  = int(_sb_m[0].item())
                                        local_acc_sum += 1.0 if _pred == _tgt else 0.0
                                    local_count += 1
                            except Exception as exc:
                                log.debug(f"Eval skip ({ds_name}): {exc}")
                                continue

                        # Aggregate across all ranks
                        if world_size > 1:
                            stats = torch.tensor(
                                [local_loss_sum, local_acc_sum, local_count],
                                dtype=torch.float64, device=device,
                            )
                            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                            total_loss  = stats[0].item()
                            total_acc   = stats[1].item()
                            total_count = int(stats[2].item())
                        else:
                            total_loss  = local_loss_sum
                            total_acc   = local_acc_sum
                            total_count = local_count

                        if total_count > 0 and local_rank == 0:
                            avg_loss = total_loss / total_count
                            avg_acc  = total_acc  / total_count
                            log.info(
                                f"[eval] global_step={global_step:05d}  "
                                f"{ds_name}_lm_loss={avg_loss:.4f}  "
                                f"{ds_name}_acc={avg_acc:.4f}  "
                                f"(n={total_count} samples, aggregated across {world_size} GPU{'s' if world_size > 1 else ''})"
                            )
                            if use_wandb:
                                wandb.log(
                                    {
                                        f"eval/{ds_name}_lm_loss": avg_loss,
                                        f"eval/{ds_name}_acc":     avg_acc,
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
        default=os.path.join(_ROOT, "datasets/train/MindCube/MindCube_train.jsonl"),
        help="Path to MindCube training JSONL",
    )
    p.add_argument(
        "--mindcube_results_dir",
        default=os.path.join(_ROOT, "datasets/train/MindCube/3d_results"),
        help="Directory containing per-sample 3d_results folders for MindCube training",
    )
    p.add_argument(
        "--sat_json_path",
        default=os.path.join(_ROOT, "datasets/train/SAT/train_36k.json"),
        help="Path to SAT training JSON list (only used when 'sat' in --datasets)",
    )
    p.add_argument(
        "--sat_results_dir",
        default=os.path.join(_ROOT, "datasets/train/SAT/3d_results"),
        help="Directory containing per-sample 3d_results folders for SAT training",
    )
    p.add_argument(
        "--datasets",
        nargs="+", default=["mindcube"], choices=["mindcube", "sat"],
        help="Which training datasets to concatenate (one or more). "
             "Default: mindcube only. Example: --datasets mindcube sat "
             "→ ConcatDataset of MindCube_train.jsonl + SAT/train_36k.json. "
             "Each dataset's loader uses its own --{name}_json_path/--{name}_results_dir "
             "(or --json_path/--mindcube_results_dir for mindcube).",
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
        "--relative",
        action="store_true",
        help="Enable relative mode: per-query-frame coordinate transformation in "
             "4D M-RoPE. Q from frame f sees ALL K tokens' xyz in frame-f camera "
             "coordinates. Uses MindCube_Train_Dataset_Relative and "
             "SpaRelativeForConditionalGeneration. Incompatible with --vanilla. "
             "Polar coordinates are enabled by default in this mode.",
    )
    p.add_argument(
        "--polar",
        action="store_true",
        help="Convert per-patch Cartesian (x, y, z) → log-spherical (log ρ, θ, α) "
             "for the 4D M-RoPE vision-token position embedding. "
             "log ρ = log||xyz|| (scale-invariant; RoPE pos-diff = log(ρ_i/ρ_j)), "
             "θ = atan2(y,x) ∈ [-π,π] (azimuth), "
             "α = atan2(√(x²+y²), z) ∈ [0,π] (inclination). "
             "Matches train_coordinate.py polar convention. "
             "No effect when --vanilla is set. In --relative mode, polar is on by default.",
    )
    p.add_argument(
        "--interleave_vision",
        action="store_true",
        help="Use interleaved M-RoPE layout for visual tokens: t keeps its "
             "mrope_section[0] bands at the high-freq end, then x/y/z round-robin "
             "through the remaining bands so each spans the full freq range. "
             "Independent of --polar; combinable with --polar / --relative. "
             "No effect when --vanilla or --decouple is set.",
    )
    p.add_argument(
        "--decouple",
        action="store_true",
        help="Decoupled position embedding: keep Qwen original 3D M-RoPE [11,11,10] "
             "in the rotary 64 dims (UNCHANGED) and add a NEW XYZ RoPE (66 dims, "
             "sequential x|y|z each 11 bands, rope_theta=1000) in pass-through "
             "dims 64..129. Text tokens default to xyz=(0,0,0) → identity rotation. "
             "Mutually exclusive with --vanilla / --polar / --relative.",
    )
    p.add_argument(
        "--xyz_rope_dim",
        type=int, default=66,
        help="Total head_dim units allocated to the XYZ RoPE in the pass-through "
             "region under --decouple / --polar (each axis x/y/z gets xyz_rope_dim/6 "
             "frequency bands). Must be a positive multiple of 6 ≤ 192 "
             "(pass-through region size). Default 66 (= 11 bands per axis). "
             "No effect without --decouple / --polar.",
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
