"""
train_alternate.py

Alternating two-phase LoRA fine-tuning of SpaForConditionalGeneration
(Qwen3.5-VL) with a single-pass rotation-aware coordinate prediction
pipeline using differentiable M-RoPE.

Schedule (per epoch — two full passes over the dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each epoch iterates the train_loader TWICE.  Pass 0 is Phase A, pass 1 is
Phase B, so with --epochs N both the LoRA branch and the rotation_enc
branch see all N full passes through the data.

Phase A (first full pass of an epoch):
    * rotation_enc  : frozen
    * LoRA          : trainable (lm_loss)
    * coord_head    : trainable (coord_loss)
    * forward R     : epoch 0      → R = I  (rotation_enc untrained)
                      epoch ≥ 1    → R from current (frozen) rotation_enc

Phase B (second full pass of an epoch):
    * rotation_enc  : trainable   (gradient from lm_loss only)
    * LoRA          : frozen
    * coord_head    : trainable   (coord_loss on detached hidden so the
                                   rotation_enc is NOT updated by the
                                   coord_loss path)

Learning rate
~~~~~~~~~~~~~
Cosine annealing with warm restarts (CosineAnnealingWarmRestarts,
T_0 = 1 phase-pass, T_mult = 1).  Each phase-pass is one cycle: the
LR starts at its base value, cosine-decays across that single pass,
and is "kicked" back up the moment its phase begins again.

Three independent schedules (one per trainable group):
  * LoRA         (active in Phase A) → restart at start of every
                                       Phase A pass.
  * rotation_enc (active in Phase B) → restart at start of every
                                       Phase B pass.
  * coord_head   (active in BOTH)    → restart at start of every
                                       pass (A and B).
Cycle length for each group = steps_per_epoch // grad_accum optim
steps.  Warm restarts matter here because Phase B keeps updating R
between epochs, so by the time Phase A runs again LoRA is facing a
shifted coordinate frame — a monotonic decay would leave it with no
LR "budget" left to adapt.  Monotonic cosine decay is still used by
the other (non-alternating) training scripts.
Set via --lr_phase_a / --lr_phase_b / --lr_coord_head (fall back to
--lr / --rotation_enc_lr / --lr respectively).

Architecture
~~~~~~~~~~~~
RotationRoPEModel
+-- SpaForConditionalGeneration [backbone + LoRA]
|    +-- SpaVisionModel (ViT, frozen)
|    +-- SpaModel (LLM + 4D M-RoPE, manual decoder loop)
+-- CameraTokenRotationEncoder   [shallow transformer, predicts R]
+-- DepthPredictionTransformer   [coordinate head]

Usage
~~~~~
  python train_alternate.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_rotation_alternate \\
      --lr_phase_a 2e-4 --lr_phase_b 5e-5
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
    SpaDecForConditionalGeneration,
    patch_attention_layers_dec,
)
from src.models.rotation_rope_llm import (
    _build_chiral_cube_group,
)
from src.models.spa_emb import SpaTextRotaryEmbedding
from src.models.spa_emb_dec import SpaXYZRotaryEmbedding
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
    decouple:           bool  = False,
    xyz_rope_dim:       int   = 66,
) -> RotationRoPEModel:
    """Build RotationRoPEModel with LM + coordinate supervision
    (rotation learned end-to-end via differentiable M-RoPE).

    decouple=True  → keep Qwen original 3D M-RoPE [11,11,10] in the rotary 64
                     dims (UNCHANGED) and add a NEW XYZ RoPE in dims 64..129
                     (xyz_rope_dim dims, sequential x|y|z, rope_theta=10000)
                     fed with **R-rotated Cartesian xyz**. R is still predicted
                     by rotation_enc (gradient flows through the XYZ RoPE
                     since SpaXYZRotaryEmbedding has no @torch.no_grad).
    xyz_rope_dim   → total dims for the pass-through XYZ RoPE (must be a
                     positive multiple of 6 ≤ 192). Only used with --decouple.
    """
    if xyz_rope_dim % 6 != 0 or xyz_rope_dim <= 0 or xyz_rope_dim > 192:
        raise ValueError(
            f"--xyz_rope_dim must be a positive multiple of 6 ≤ 192 "
            f"(pass-through region); got {xyz_rope_dim}."
        )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])

    if decouple:
        # Decouple: keep Qwen original 3D M-RoPE in rotary 64 dims; new XYZ
        # RoPE lives in the pass-through region (dims 64..129 by default).
        log.info(
            f"mrope_section: {orig_section} (UNCHANGED — Qwen original 3D M-RoPE) "
            f"+ new XYZ RoPE ({xyz_rope_dim} dims, rope_theta=10000) in pass-through region "
            f"[input: R-rotated Cartesian (x, y, z)]"
        )
    else:
        total       = sum(orig_section)
        xyz_size    = (total - 2) // 3
        new_section = [2, xyz_size, xyz_size, xyz_size]
        config.text_config.rope_scaling["mrope_section"] = new_section
        log.info(
            f"mrope_section: {orig_section} -> {new_section}  "
            f"(4D M-RoPE: 2 for t, {xyz_size} each for x/y/z)"
        )

    mllm_head_dim = getattr(config.text_config, "head_dim", None) or (
        config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    log.info(f"mllm_head_dim={mllm_head_dim}  rot_nhead={rot_nhead}  "
             f"→ rotation_enc d_model={rot_nhead * mllm_head_dim}")

    if decouple:
        spa = SpaDecForConditionalGeneration.from_pretrained(
            model_path,
            config              = config,
            torch_dtype         = torch.bfloat16,
            attn_implementation = "sdpa",
        )
        # Optional XYZ RoPE swap (matches train_correspondence.py: only swap
        # when the requested xyz_rope_dim deviates from the SpaDec default 66).
        if xyz_rope_dim != 66:
            _lm  = spa.model.language_model
            _old = _lm.xyz_rotary_emb
            _new = SpaXYZRotaryEmbedding(
                xyz_dim             = xyz_rope_dim,
                rope_theta          = 10000.0,
                default_coord_scale = _old.default_coord_scale,
            )
            _lm.xyz_rotary_emb = _new.to(next(_lm.parameters()).device)
            log.info(
                f"[XYZ RoPE] xyz_dim={xyz_rope_dim} "
                f"(n_per_axis={xyz_rope_dim // 6}), theta=10000"
            )
    else:
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

    # Patch attention layers AFTER LoRA so adapters wrap the base attention,
    # then SpaDecAttentionWrapper wraps the LoRA-adapted attention.
    if decouple:
        n_patched = patch_attention_layers_dec(spa)
        log.info(
            f"Wrapped {n_patched} attention layers with SpaDecAttentionWrapper."
        )

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

    # rotation_enc's positional encoding mirrors the LLM main-path RoPE:
    #   • non-decouple → 4D M-RoPE [2, xyz_size, xyz_size, xyz_size] over
    #                    `_build_token_txyz_int` integer positions.
    #   • decouple     → stock 3D M-RoPE [11, 11, 10] over `_build_token_thw_int`
    #                    integer positions PLUS a separate XYZ RoPE on
    #                    pass-through dims via `SpaXYZRotaryEmbedding`.
    # Build a private text_config copy so the rotation_enc's mrope_section
    # is set independently of the backbone's config.text_config.
    import copy as _copy
    _rot_text_config = _copy.deepcopy(config.text_config)
    _rot_text_config.rope_scaling = dict(_rot_text_config.rope_scaling)
    if decouple:
        _rot_text_config.rope_scaling["mrope_section"] = list(orig_section)
    else:
        _orig_total   = sum(orig_section)
        _rot_xyz_size = (_orig_total - 2) // 3
        _rot_text_config.rope_scaling["mrope_section"] = [
            2, _rot_xyz_size, _rot_xyz_size, _rot_xyz_size,
        ]
    rot_rope_emb = SpaTextRotaryEmbedding(config=_rot_text_config).to(torch.bfloat16)

    rot_xyz_rope_emb = None
    if decouple:
        rot_xyz_rope_emb = SpaXYZRotaryEmbedding(
            xyz_dim             = xyz_rope_dim,
            rope_theta          = 10000.0,
            default_coord_scale = 100.0,
        ).to(torch.bfloat16)
        log.info(
            f"rotation_enc XYZ RoPE: xyz_dim={xyz_rope_dim} "
            f"(n_per_axis={xyz_rope_dim // 6}), theta=10000 "
            f"[independent instance from main backbone]"
        )

    rotation_enc = CameraTokenRotationEncoder(
        hidden_dim      = hidden_dim,
        mllm_head_dim   = mllm_head_dim,
        rope_emb        = rot_rope_emb,
        nhead           = rot_nhead,
        dim_feedforward = rot_dim_feedforward,
        num_layers      = rot_num_layers,
        decouple        = decouple,
        xyz_rotary_emb  = rot_xyz_rope_emb,
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
        decouple           = decouple,
    )


# -- training loop -------------------------------------------------------------

def _set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)


def _phase_a_reg_step(
    _model:         RotationRoPEModel,
    batch_tensors:  dict,
    args:           argparse.Namespace,
    R_bins:         torch.Tensor,
    grad_accum:     int,
    device:         torch.device,
    use_rot_enc:    bool = True,
):
    """Phase A composite regularized loss (replaces plain lm_loss).

        L_total = L_CE(R_opt)
                + (λ1/N) · Σ_{k∈S}  L_CE(R_k)
                + (λ2/N) · Σ_{k∈S} max(0, L_opt − L_k + α · dist(R_k, R_opt))

    where L_x = L_CE(R_x), S is a stratified sample of N=6 anchors drawn
    from the 24 chiral-cube rotations (R_bins from train_rl's group), and
    dist is the SO(3) geodesic angle θ = acos((tr(R_k^T · R_opt) − 1) / 2).

    Stratified sampling
    ~~~~~~~~~~~~~~~~~~~
    Per step, all 24 distances d_k = dist(R_bins[k], R_opt) are computed
    (24 cheap 3×3 matmuls — no LLM forward), argsort'd, and partitioned
    into three equal tiers of 8 (near / middle / far by rank).  Two
    anchors are drawn uniformly at random from each tier, giving N=6
    per step.  Every call resamples — no forced-include anchor, no
    cross-step memoization.  Proportional allocation (tier sizes 8,8,8
    × equal picks 2,2,2) makes (1/N)Σ_{k∈S} an unbiased estimator of
    (1/24)Σ_{k=0..23}, so the 1/N normalization preserves the λ1/λ2
    scales you'd get from summing over all 24 anchors.

    When ``use_rot_enc=False`` (epoch 0, before rotation_enc has been
    trained), R_opt is pinned to R_bins[0] = I.  The rotation_enc call
    and token_txyz_int construction are both skipped — the regularizer
    still runs, but its "optimal" anchor is identity, matching what R
    would be in the plain lm_loss path at this stage.

    Memory: 1 + N grad LLM forwards per step would still be heavy if
    all graphs were kept alive for a single combined backward.  Instead:
        • Pass 1 (no_grad): 1 + N cheap forwards → scalar L values to
          decide the hinge-active set and per-term weights.
        • Pass 2 (grad, in-loop backward): re-forward each R with its
          closed-form per-term weight and ``backward()`` immediately so
          that iteration's activations are released before the next.

    All backwards are pre-scaled by ``1/grad_accum`` — the caller must
    NOT re-apply that factor.
    """
    input_ids       = batch_tensors["input_ids"]
    attention_mask  = batch_tensors["attention_mask"]
    pixel_values    = batch_tensors["pixel_values"]
    image_grid_thw  = batch_tensors["image_grid_thw"]
    labels          = batch_tensors["labels"]
    image_xyz       = batch_tensors["image_xyz"]
    image_xyz_hires = batch_tensors["image_xyz_hires"]

    lam1  = float(args.reg_lambda1)
    lam2  = float(args.reg_lambda2)
    alpha = float(args.reg_alpha)

    ldict: dict = {}

    # ── Pass 1: no_grad — shared encode, R_opt, full-24 distances, then
    #           rank-stratified sample S of N=6 anchors, then L_k on S only.
    with torch.no_grad():
        inputs_embeds = _model.encode_inputs(
            input_ids, pixel_values, image_grid_thw,
        )
        if use_rot_enc:
            R_opt, _ = _model._call_rotation_enc(
                inputs_embeds  = inputs_embeds,
                input_ids      = input_ids,
                image_xyz      = image_xyz,
                image_grid_thw = image_grid_thw,
                coord_scale    = args.coord_scale,
            )
            R_opt = R_opt.float().detach()
        else:
            # Epoch-0 bootstrap: rotation_enc is untrained, pin R_opt = I.
            R_opt = R_bins[0].clone()

        # SO(3) geodesic distance for ALL 24 anchors (matmul only; no LLM
        # forward).  Needed before sampling so we can stratify by rank.
        R_rel        = R_bins.transpose(-1, -2) @ R_opt              # (24,3,3)
        trace        = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)      # (24,)
        cos_ang      = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        dists_tensor = torch.arccos(cos_ang)                          # (24,) rad
        dists        = dists_tensor.tolist()

        # Rank-based stratified sampling: argsort(d_k) into [0:8]/[8:16]/
        # [16:24] (near / middle / far tiers), then 2 uniformly-random
        # picks per tier → N=6 anchors this step.  Resampled every call.
        order = torch.argsort(dists_tensor).tolist()
        S: list[int] = []
        for tier_start in (0, 8, 16):
            tier  = order[tier_start:tier_start + 8]
            picks = torch.randperm(8)[:2].tolist()
            S.extend(tier[p] for p in picks)

        # Pass-1 L_k only for the sampled anchors.
        L_k_vals_S: list[float] = []
        dists_S:    list[float] = []
        for k in S:
            lm_k, _, _ = _model.compute_losses_from_R(
                R                   = R_bins[k],
                inputs_embeds       = inputs_embeds,
                input_ids           = input_ids,
                attention_mask      = attention_mask,
                image_xyz           = image_xyz,
                image_xyz_hires     = None,
                image_grid_thw      = image_grid_thw,
                labels              = labels,
                coord_scale         = args.coord_scale,
                use_coord_loss      = False,
                detach_coord_hidden = True,
                compute_reward      = False,
            )
            L_k_vals_S.append(
                float(lm_k.item()) if lm_k is not None else float("inf")
            )
            dists_S.append(dists[k])

        lm_opt, _, _ = _model.compute_losses_from_R(
            R                   = R_opt,
            inputs_embeds       = inputs_embeds,
            input_ids           = input_ids,
            attention_mask      = attention_mask,
            image_xyz           = image_xyz,
            image_xyz_hires     = None,
            image_grid_thw      = image_grid_thw,
            labels              = labels,
            coord_scale         = args.coord_scale,
            use_coord_loss      = False,
            detach_coord_hidden = True,
            compute_reward      = False,
        )
        L_opt_val = (
            float(lm_opt.item()) if lm_opt is not None else float("inf")
        )

    # Early bail: without a finite L_opt there is no main-term gradient.
    if not math.isfinite(L_opt_val):
        return None, None

    # Hinge active over sampled S only; infinite L_k falls through inactive.
    N        = len(S)
    active_S: list[bool] = []
    for i in range(N):
        if not math.isfinite(L_k_vals_S[i]):
            active_S.append(False)
            continue
        hinge_val = L_opt_val - L_k_vals_S[i] + alpha * dists_S[i]
        active_S.append(hinge_val > 0.0)
    n_active = sum(active_S)

    # Per-term closed-form weights derived from
    #   ∂L_total/∂θ = (1 + (λ2/N)·n_active)·∂L_opt
    #               + Σ_{i∈S} [(λ1/N) − (λ2/N)·I[active_i]]·∂L_i
    w_opt = 1.0 + (lam2 / float(N)) * n_active
    w_k_S = [
        (lam1 / float(N)) - ((lam2 / float(N)) if active_S[i] else 0.0)
        for i in range(N)
    ]

    scale           = 1.0 / float(grad_accum)
    lm_opt_val_g    = 0.0
    coord_loss_val  = 0.0

    # ── Pass 2a: R_opt with grad — carries the answer_weight·L_opt term
    #            and the (unchanged) coord_loss at the predicted pose.
    lm_opt_g, coord_opt_g, _ld_opt = _model.compute_losses_from_R(
        R                   = R_opt,
        inputs_embeds       = inputs_embeds,
        input_ids           = input_ids,
        attention_mask      = attention_mask,
        image_xyz           = image_xyz,
        image_xyz_hires     = image_xyz_hires,
        image_grid_thw      = image_grid_thw,
        labels              = labels,
        coord_scale         = args.coord_scale,
        use_coord_loss      = (not args.no_coord),
        detach_coord_hidden = False,
        compute_reward      = False,
    )
    step_loss_opt = torch.zeros((), device=device)
    if lm_opt_g is not None:
        step_loss_opt = step_loss_opt + (args.answer_weight * w_opt) * lm_opt_g
        lm_opt_val_g  = float(lm_opt_g.item())
    if coord_opt_g is not None:
        step_loss_opt  = step_loss_opt + args.coord_weight * coord_opt_g
        coord_loss_val = float(coord_opt_g.item())
    if step_loss_opt.requires_grad:
        (scale * step_loss_opt).backward()

    # ── Pass 2b: sampled anchors (k ∈ S) — one grad forward + in-loop
    #           backward each.  w_k=0 anchors (e.g. active with λ1=λ2)
    #           are skipped entirely; so are any with non-finite L_k.
    for i, k in enumerate(S):
        wk = w_k_S[i]
        if abs(wk) < 1e-9 or not math.isfinite(L_k_vals_S[i]):
            continue
        lm_k_g, _, _ = _model.compute_losses_from_R(
            R                   = R_bins[k],
            inputs_embeds       = inputs_embeds,
            input_ids           = input_ids,
            attention_mask      = attention_mask,
            image_xyz           = image_xyz,
            image_xyz_hires     = None,
            image_grid_thw      = image_grid_thw,
            labels              = labels,
            coord_scale         = args.coord_scale,
            use_coord_loss      = False,
            detach_coord_hidden = True,
            compute_reward      = False,
        )
        if lm_k_g is None:
            continue
        step_loss_k = (args.answer_weight * wk) * lm_k_g
        (scale * step_loss_k).backward()

    # True L_total at sampled S (matches the mathematical formula; uses
    # Pass-1 values so every term — including skipped-w_k anchors and the
    # α·d_k margin constants — is captured).  Reported for logging only;
    # gradients already flowed via the per-term backward() above.
    finite_L  = [v for v in L_k_vals_S if math.isfinite(v)]
    lm_k_mean = (sum(finite_L) / len(finite_L)) if finite_L else 0.0
    lm_k_min  = min(finite_L) if finite_L else 0.0
    hinge_sum = 0.0
    for i in range(N):
        if active_S[i]:
            hinge_sum += (L_opt_val - L_k_vals_S[i] + alpha * dists_S[i])
    reg_lm_term    = (lam1 / float(N)) * sum(finite_L)
    reg_hinge_term = (lam2 / float(N)) * hinge_sum
    total_loss_val = (
        args.answer_weight * (lm_opt_val_g + reg_lm_term + reg_hinge_term)
        + args.coord_weight * coord_loss_val
    )

    ldict["lm_loss"]         = lm_opt_val_g
    if coord_loss_val:
        ldict["coord_loss"]  = coord_loss_val
    ldict["reg_lm_kmean"]    = lm_k_mean
    ldict["reg_lm_kmin"]     = lm_k_min
    ldict["reg_hinge_sum"]   = hinge_sum
    ldict["reg_n_active"]    = float(n_active)
    ldict["reg_n_sampled"]   = float(N)
    ldict["reg_w_opt"]       = float(w_opt)

    return total_loss_val, ldict


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
        decouple            = args.decouple,
        xyz_rope_dim        = args.xyz_rope_dim,
    )
    # Letter-position offset inside the masked answer suffix. Probed
    # dynamically — Qwen BPE merges `>X` into one token, so the letter
    # sits at index 2 not 3. Plumbed into RotationRoPEModel.forward so its
    # eval-time `_ldict["acc"]` reports letter-prediction accuracy. See
    # md/bug_fix/train_eval_paradigm_mismatch.md.
    from src.dataset import compute_letter_offset
    model.letter_offset = compute_letter_offset(processor.tokenizer)
    log.info(f"letter_offset = {model.letter_offset}")
    model = model.to(device)
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"[MEM] After model.to(device): {mem_gb:.2f} GiB allocated")

    # 24 chiral cube anchors — only materialized when --reg is active.
    R_bins_dev: torch.Tensor | None = None
    if args.reg:
        R_bins_dev = _build_chiral_cube_group().to(
            device=device, dtype=torch.float32,
        )
        log.info(
            f"[--reg] Phase A composite loss enabled: "
            f"lambda1={args.reg_lambda1}  lambda2={args.reg_lambda2}  "
            f"alpha={args.reg_alpha}  (R_bins on {device})"
        )

    # -- DDP -------------------------------------------------------------------
    if world_size > 1:
        # find_unused_parameters=True because different phases freeze
        # different parameter sets; DDP must tolerate params that receive
        # no grad in a given step.
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
            log,
            max_images         = args.max_images,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
            max_samples        = args.max_samples,
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
    # Three mutually-exclusive groups so the phase scheduler can toggle
    # requires_grad at phase boundaries:
    #   (a) rotation_enc  — train-from-scratch, strict RoPE-aware clip
    #   (b) coord_head    — standard fine-tune regime
    #   (c) lora (rest)   — standard fine-tune regime
    rotation_enc_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc" in n
    ]
    coord_head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "coord_head" in n and "rotation_enc" not in n
    ]
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc" not in n and "coord_head" not in n
    ]
    other_params = lora_params + coord_head_params   # kept for grad-clip / legacy logs
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params,         "lr": args.lr,              "name": "lora"},
            {"params": coord_head_params,   "lr": args.lr,              "name": "coord_head"},
            {"params": rotation_enc_params, "lr": args.rotation_enc_lr, "name": "rotation_enc"},
        ],
        weight_decay=0.01,
    )
    log.info(
        f"Optimizer groups: lora={len(lora_params)} @ lr={args.lr}, "
        f"coord_head={len(coord_head_params)} @ lr={args.lr}, "
        f"rotation_enc={len(rotation_enc_params)} @ lr={args.rotation_enc_lr}"
    )

    # Resolve per-group base LRs. Phase A/B each run an independent
    # cosine schedule over their own optimizer steps; coord_head runs
    # a separate cosine advanced in both phases.
    lr_phase_a_base = args.lr_phase_a if args.lr_phase_a is not None else args.lr
    lr_phase_b_base = (
        args.lr_phase_b if args.lr_phase_b is not None else args.rotation_enc_lr
    )
    lr_coord_head_base = (
        args.lr_coord_head if args.lr_coord_head is not None else args.lr
    )

    log.info("=" * 72)
    log.info(
        f">>> no_coord    = {args.no_coord}  "
        + ("(coord_loss + coord_head DISABLED for entire run)"
           if args.no_coord
           else "(coord_loss active)")
    )
    log.info(
        f">>> decouple    = {args.decouple}  "
        + (f"(SpaDec backbone, 3D M-RoPE + XYZ RoPE in pass-through "
           f"dims; xyz_rope_dim={args.xyz_rope_dim})"
           if args.decouple
           else "(4D M-RoPE with rotated xyz embedded in position_ids, default)")
    )
    log.info(
        ">>> alternating two-phase schedule (two full passes per epoch):\n"
        "    Phase A (first full pass): rotation_enc frozen; train LoRA "
        "+ coord_head. In epoch 0 rotation_enc is still untrained so "
        "R=I; from epoch 1 onwards Phase A uses the current (frozen) "
        "rotation_enc output in forward.\n"
        "    Phase B (second full pass): LoRA frozen; train "
        "rotation_enc via lm_loss only (coord_loss path detaches hidden "
        "so rotation_enc is not updated by coord_loss); coord_head still "
        "trained by coord_loss."
    )
    log.info(
        f">>> lr_phase_a  = {lr_phase_a_base:.2e}  "
        f"(LoRA — warm-restart at every Phase A pass)"
    )
    log.info(
        f">>> lr_phase_b  = {lr_phase_b_base:.2e}  "
        f"(rotation_enc — warm-restart at every Phase B pass)"
    )
    log.info(
        f">>> lr_coord_head = {lr_coord_head_base:.2e}  "
        f"(coord_head — warm-restart at the start of EVERY pass)"
    )
    log.info("=" * 72)

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

    steps_per_epoch = len(train_loader)
    current_phase = None                 # "A" | "B"

    # Cosine annealing with warm restarts.  T_0 = one phase-pass so
    # every phase transition is a restart; all three groups use the
    # same cycle length but each counter is reset independently
    # (LoRA on entering Phase A, rotation_enc on entering Phase B,
    # coord_head on entering EITHER phase).
    cycle_optim_steps = max(
        1, steps_per_epoch // args.grad_accum
    )
    phase_a_cycle_step     = 0
    phase_b_cycle_step     = 0
    coord_head_cycle_step  = 0

    for epoch in range(args.epochs):
        # Two full passes per epoch: pass_idx 0 → Phase A, 1 → Phase B.
        for pass_idx in range(2):
            new_phase = "A" if pass_idx == 0 else "B"
            if new_phase != current_phase:
                if new_phase == "A":
                    # Phase A: freeze rotation_enc; train LoRA + coord_head.
                    _set_requires_grad(rotation_enc_params, False)
                    _set_requires_grad(lora_params,         True)
                    _set_requires_grad(coord_head_params,   True)
                    # Warm restart: LoRA kicks back to its base LR on
                    # entering Phase A (rotation_enc just updated R,
                    # LoRA is facing a slightly shifted frame).
                    phase_a_cycle_step    = 0
                    coord_head_cycle_step = 0
                else:
                    # Phase B: freeze LoRA; train rotation_enc (via lm_loss)
                    # and coord_head (via coord_loss on detached hidden).
                    _set_requires_grad(rotation_enc_params, True)
                    _set_requires_grad(lora_params,         False)
                    _set_requires_grad(coord_head_params,   True)
                    # Warm restart: rotation_enc + coord_head kick back
                    # to their base LRs on entering Phase B.
                    phase_b_cycle_step    = 0
                    coord_head_cycle_step = 0
                current_phase = new_phase
                if local_rank == 0:
                    if new_phase == "A":
                        msg = ("(train LoRA+coord_head; R=I — "
                               "rotation_enc untrained)"
                               if epoch == 0
                               else "(train LoRA+coord_head; R from "
                                    "trained rotation_enc)")
                    else:
                        msg = ("(freeze LoRA, train rotation_enc via "
                               "lm_loss, coord_head via coord_loss on "
                               "detached hidden)")
                    log.info(
                        f"[epoch {epoch+1:02d} pass {pass_idx}] "
                        f"→ Phase {new_phase} full pass  {msg}"
                    )

            # Phase A in epoch 0: rotation_enc not trained yet → R=I.
            # After first Phase B completes, rotation_enc has weights,
            # so Phase A in subsequent epochs uses the trained R.
            use_rot_enc         = (new_phase == "B") or (epoch > 0)
            detach_coord_hidden = (new_phase == "B")

            # Distinct DDP shuffle per pass: epoch*2 + pass_idx so Phase A
            # and Phase B see different mini-batch orderings.
            if train_sampler is not None:
                train_sampler.set_epoch(epoch * 2 + pass_idx)

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
                # --reg Phase A branch: composite loss over R_opt + 24 anchors
                # with in-loop backward; rotation_enc stays frozen, so R_opt
                # is detached and the LLM path is the sole grad source.
                if args.reg and current_phase == "A" and R_bins_dev is not None:
                    loss_val, loss_dict = _phase_a_reg_step(
                        _model        = _model,
                        batch_tensors = {
                            "input_ids":       input_ids,
                            "attention_mask":  attention_mask,
                            "pixel_values":    pixel_values,
                            "image_grid_thw":  image_grid_thw,
                            "image_xyz":       image_xyz,
                            "image_xyz_hires": image_xyz_hires,
                            "labels":          labels,
                        },
                        args        = args,
                        R_bins      = R_bins_dev,
                        grad_accum  = args.grad_accum,
                        device      = device,
                        use_rot_enc = use_rot_enc,
                    )
                    if loss_val is None:
                        log.warning(f"[rank{local_rank}] Step {step}: "
                                    "no reg signal, skipping.")
                        continue
                    running_loss += loss_val
                    if loss_dict:
                        for k, v in loss_dict.items():
                            running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v
                else:
                    _, loss, loss_dict = model(
                        input_ids           = input_ids,
                        attention_mask      = attention_mask,
                        pixel_values        = pixel_values,
                        image_grid_thw      = image_grid_thw,
                        image_xyz           = image_xyz,
                        image_xyz_hires     = image_xyz_hires,
                        labels              = labels,
                        coord_scale         = args.coord_scale,
                        use_rotation_enc    = use_rot_enc,
                        use_coord_loss      = not args.no_coord,
                        detach_coord_hidden = detach_coord_hidden,
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
                    if rotation_enc_params:
                        torch.nn.utils.clip_grad_norm_(
                            rotation_enc_params, max_norm=args.rotation_enc_clip
                        )
                    torch.nn.utils.clip_grad_norm_(
                        other_params, max_norm=args.lora_clip
                    )
                    optimizer.step()
                    # Cosine annealing with warm restarts.  All three
                    # groups share the same cycle length (= one phase-
                    # pass); each counter is reset at the start of its
                    # own phase so the LR "kicks" back up at every
                    # phase boundary.  coord_head is trained in BOTH
                    # phases so its counter is reset at EVERY pass.
                    coord_head_cycle_step += 1
                    _t_ch  = min(coord_head_cycle_step, cycle_optim_steps)
                    _cos_ch = 0.5 * (1 + math.cos(
                        math.pi * _t_ch / cycle_optim_steps
                    ))
                    _lr_ch = lr_coord_head_base * _cos_ch
                    for _g in optimizer.param_groups:
                        if _g.get("name") == "coord_head":
                            _g["lr"] = _lr_ch

                    if current_phase == "A":
                        phase_a_cycle_step += 1
                        _t = min(phase_a_cycle_step, cycle_optim_steps)
                        _cos = 0.5 * (1 + math.cos(
                            math.pi * _t / cycle_optim_steps
                        ))
                        _lr = lr_phase_a_base * _cos
                        for _g in optimizer.param_groups:
                            if _g.get("name") == "lora":
                                _g["lr"] = _lr
                    else:
                        phase_b_cycle_step += 1
                        _t = min(phase_b_cycle_step, cycle_optim_steps)
                        _cos = 0.5 * (1 + math.cos(
                            math.pi * _t / cycle_optim_steps
                        ))
                        _lr = lr_phase_b_base * _cos
                        for _g in optimizer.param_groups:
                            if _g.get("name") == "rotation_enc":
                                _g["lr"] = _lr
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
                        _active_group = (
                            "lora" if current_phase == "A" else "rotation_enc"
                        )
                        current_lr = next(
                            g["lr"] for g in optimizer.param_groups
                            if g.get("name") == _active_group
                        )
                        coord_head_lr = next(
                            g["lr"] for g in optimizer.param_groups
                            if g.get("name") == "coord_head"
                        )
                        detail = "  ".join(
                            f"{k}={v:.4f}" for k, v in avg_loss_dict.items()
                        )
                        log.info(
                            f"[train][Phase {current_phase}] "
                            f"epoch={epoch+1:02d}  global_step={global_step:05d}  "
                            f"loss={avg_loss:.4f}"
                            + (f"  ({detail})" if detail else "")
                            + f"  lr={current_lr:.2e}  coord_lr={coord_head_lr:.2e}  "
                            f"(aggregated across {world_size} GPU{'s' if world_size > 1 else ''})"
                        )
                        if use_wandb:
                            # Items shown when --reg is off stay in `train/`
                            # (lm_loss, coord_loss, R_trace, …); the --reg
                            # composite breakdown (`reg_*`) routes to
                            # `train_sub/`.
                            wandb.log(
                                {
                                    "train/loss":          avg_loss,
                                    "train/lr":            current_lr,
                                    "train/coord_head_lr": coord_head_lr,
                                    "train/phase":         0 if current_phase == "A" else 1,
                                    "epoch":               epoch + 1,
                                    **{f"train/{k}": v
                                       for k, v in avg_loss_dict.items()
                                       if not k.startswith("reg_")},
                                    **{f"train_sub/{k}": v
                                       for k, v in avg_loss_dict.items()
                                       if k.startswith("reg_")},
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
                            # Per-sample rotation angle (deg) for visualization.
                            local_R_angles: list[float] = []

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
                                    inputs_embeds = _model.encode_inputs(
                                        t_ids, t_pv, t_thw,
                                    )
                                    R_pred = None
                                    if use_rot_enc and t_xyz is not None and t_thw is not None:
                                        R_pred, _ = _model._call_rotation_enc(
                                            inputs_embeds  = inputs_embeds,
                                            input_ids      = t_ids,
                                            image_xyz      = t_xyz,
                                            image_grid_thw = t_thw,
                                            coord_scale    = args.coord_scale,
                                        )

                                    lm_loss, coord_loss, loss_dict = _model.compute_losses_from_R(
                                        R                   = R_pred,
                                        inputs_embeds       = inputs_embeds,
                                        input_ids           = t_ids,
                                        attention_mask      = t_mask,
                                        image_xyz           = t_xyz,
                                        image_xyz_hires     = t_xyz_h,
                                        image_grid_thw      = t_thw,
                                        labels              = t_labels,
                                        coord_scale         = args.coord_scale,
                                        use_coord_loss      = not args.no_coord,
                                        detach_coord_hidden = detach_coord_hidden,
                                        compute_reward      = True,
                                    )
                                if lm_loss is None and coord_loss is None:
                                    continue
                                local_count += 1
                                if loss_dict is None:
                                    loss_dict = {}
                                if R_pred is not None:
                                    loss_dict["R_trace"] = float(R_pred.trace().item())
                                for k, v in loss_dict.items():
                                    local_loss_sums[k] = local_loss_sums.get(k, 0.0) + float(v)
                                # Rotation angle (deg) via θ = acos((tr(R)-1)/2)
                                # — same convention as evaluation.py.
                                if R_pred is not None:
                                    _R = R_pred.detach().float()
                                    _tr = _R[0, 0] + _R[1, 1] + _R[2, 2]
                                    _cos = ((_tr - 1.0) / 2.0).clamp(-1.0, 1.0)
                                    local_R_angles.append(
                                        float(torch.acos(_cos) * (180.0 / math.pi))
                                    )

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
                                    _main_keys = {"coord_loss", "lm_loss", "acc"}
                                    wandb.log(
                                        {
                                            **{f"eval/{ds_name}_{k}": agg_sums[k] / total_count
                                               for k in _loss_keys if k in _main_keys},
                                            **{f"eval_sub/{ds_name}_{k}": agg_sums[k] / total_count
                                               for k in _loss_keys if k not in _main_keys},
                                        },
                                        step=global_step,
                                    )

                            # -- R visualization ------------------------------
                            # Gather per-sample rotation angles across ranks
                            # and log summary stats + histogram to wandb.
                            if world_size > 1:
                                _gathered: list = [None] * world_size
                                dist.all_gather_object(_gathered, local_R_angles)
                                all_angles = [
                                    a for lst in _gathered for a in (lst or [])
                                ]
                            else:
                                all_angles = list(local_R_angles)

                            if all_angles and local_rank == 0:
                                _n   = len(all_angles)
                                _mean = sum(all_angles) / _n
                                _var = sum(
                                    (a - _mean) ** 2 for a in all_angles
                                ) / _n
                                _std = _var ** 0.5
                                _min = min(all_angles)
                                _max = max(all_angles)
                                log.info(
                                    f"[eval-R] global_step={global_step:05d}  "
                                    f"{ds_name}  angle_deg: mean={_mean:.2f}  "
                                    f"std={_std:.2f}  min={_min:.2f}  "
                                    f"max={_max:.2f}  (n={_n})"
                                )
                                if use_wandb:
                                    wandb.log(
                                        {
                                            f"eval_R/{ds_name}_angle_deg_mean": _mean,
                                            f"eval_R/{ds_name}_angle_deg_std":  _std,
                                            f"eval_R/{ds_name}_angle_deg_min":  _min,
                                            f"eval_R/{ds_name}_angle_deg_max":  _max,
                                            f"eval_R/{ds_name}_angle_deg_hist":
                                                wandb.Histogram(all_angles),
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
        description="Alternating two-phase single-pass differentiable-RoPE "
                    "rotation-aware coordinate prediction training "
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
        default=os.path.join(_ROOT, "checkpoints/spa_rotation_alternate"),
    )
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",              type=float, default=2e-4,
                   help="Default Phase A base LR (LoRA + coord_head). "
                        "Used when --lr_phase_a is unset.")
    p.add_argument("--rotation_enc_lr", type=float, default=2e-4,
                   help="Default Phase B base LR (rotation_enc + coord_head). "
                        "Used when --lr_phase_b is unset.")
    p.add_argument(
        "--lr_phase_a", type=float, default=None,
        help="Base LR for Phase A (LoRA + coord_head) with an independent "
             "cosine decay that counts only Phase A optimizer steps. "
             "Falls back to --lr when unset.",
    )
    p.add_argument(
        "--lr_phase_b", type=float, default=None,
        help="Base LR for Phase B (rotation_enc) with an independent "
             "cosine decay that counts only Phase B optimizer steps. "
             "Falls back to --rotation_enc_lr when unset.",
    )
    p.add_argument(
        "--lr_coord_head", type=float, default=None,
        help="Base LR for coord_head with an independent cosine decay "
             "that is advanced in BOTH phases (horizon = 2 * epochs * "
             "steps_per_epoch // grad_accum). Falls back to --lr when "
             "unset.",
    )
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
        "--decouple", action="store_true",
        help="Decoupled XYZ RoPE architecture: keep Qwen original 3D M-RoPE "
             "[11,11,10] in the rotary 64 dims (UNCHANGED) and add a separate "
             "XYZ RoPE in pass-through dims 64..129 fed with R-rotated "
             "Cartesian xyz. Mirrors --decouple in train_correspondence.py.",
    )
    p.add_argument(
        "--xyz_rope_dim", type=int, default=66,
        help="Total dims for the pass-through XYZ RoPE under --decouple "
             "(must be a positive multiple of 6 ≤ 192). "
             "Each axis x/y/z gets xyz_rope_dim/6 frequencies. "
             "No effect without --decouple.",
    )
    p.add_argument(
        "--coord_upscale", type=int, default=4,
        help="PixelShuffle upscale factor for the coordinate head.",
    )
    p.add_argument(
        "--coord_scale", type=float, default=100.0,
        help="Multiplier applied to float XYZ before rounding to integer "
             "RoPE indices.  Must be the same value used everywhere "
             "(SpaModel.get_vision_position_ids, rotation encoder PE, "
             "coord head GT).  Default 100 maps ±10 m → ±1000.",
    )
    p.add_argument(
        "--rot_nhead", type=int, default=4,
        help="Number of attention heads in CameraTokenRotationEncoder. "
             "d_model = rot_nhead × config.head_dim (e.g. 4×256=1024).",
    )
    p.add_argument("--rot_dim_feedforward", type=int, default=2048)
    p.add_argument("--rot_num_layers",      type=int, default=2)

    # ── Phase A composite regularization (--reg) ──────────────────────────
    # L_total = L_CE(R_opt) + (λ1/N) Σ_{k∈S} L_CE(R_k)
    #         + (λ2/N) Σ_{k∈S} max(0, L_opt − L_k + α · dist(R_k, R_opt))
    # Replaces the plain lm_loss in Phase A; S is a rank-stratified sample
    # of N=6 anchors from the 24 chiral-cube rotations (2 per tier of
    # near/middle/far d_k), resampled every step.  Rotation_enc stays
    # frozen in Phase A — R_opt is its detached prediction (R_opt = I at
    # epoch 0) and grads flow only through LoRA/lm_head/coord_head.  Cost:
    # 1 + N no_grad + up to 1 + N grad LLM forwards per step (in-loop
    # backward keeps activation memory bounded; w_k=0 anchors skipped).
    p.add_argument("--reg", action="store_true",
                   help="Enable Phase A composite regularized loss "
                        "(core + full-view + distance-aware margin, "
                        "rank-stratified sampling N=6 of 24 anchors).")
    p.add_argument("--reg_lambda1", type=float, default=0.3,
                   help="Weight on the sampled-anchor mean L_CE "
                        "(full-view generalization term).")
    p.add_argument("--reg_lambda2", type=float, default=0.3,
                   help="Weight on the distance-aware margin term "
                        "suppressing non-optimal anchors.")
    p.add_argument("--reg_alpha",   type=float, default=0.8,
                   help="Distance coefficient in the margin term (radians): "
                        "L_opt must be below L_k by at least α · θ(R_k, R_opt).")

    p.add_argument("--wandb_project",  default="", help="WandB project name.")
    p.add_argument("--wandb_entity",   default="", help="WandB entity.")
    p.add_argument("--wandb_run_name", default="", help="WandB run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
