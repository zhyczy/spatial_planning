"""
train_rl.py

Alternating two-phase LoRA fine-tuning of SpaForConditionalGeneration
(Qwen3.5-VL) where Phase B is replaced with GRPO-based RL over a
discretised SO(3) action space.

Architecture
~~~~~~~~~~~~
RotationRLModel = RotationRoPEModel + CameraTokenRotationEncoderRL
    +-- SpaForConditionalGeneration [backbone + LoRA]
    +-- CameraTokenRotationEncoderRL
    |     +-- shallow 4D M-RoPE encoder (2 layers)
    |     +-- head_cls  : Linear(d, 24)   — anchor-classifier logits
    |     +-- head_res  : Linear(d, 72)   — per-anchor axis-angle (hybrid only)
    |     +-- R_bins    : 24 chiral cube rotations (buffer)
    +-- DepthPredictionTransformer   [coordinate head]

Final rotation composition (hybrid):
    R_final(k) = R_bins[k] @ exp(hat(residual_clamp * tanh(residual_all[k])))
Discrete (--action_space discrete): R_final(k) = R_bins[k] only.

Schedule (per epoch — two full passes over the dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase A (first full pass): standard SFT
    * rotation_enc frozen     (epoch 0: logits=0, residual=0 → R = R_bins[0] = I)
    * LoRA       trainable    (lm_loss)
    * coord_head trainable    (coord_loss)

Phase B (second full pass): GRPO RL
    * LoRA            frozen
    * head_cls        trainable via REINFORCE     (reward = 0/1 MCQ accuracy)
    * head_res        trainable via RWR-weighted  (loss_lm only)
    * rot_enc backbone trainable via lm_loss      (shared signal)
    * coord_head      trainable via loss_coord    (R detached, hidden detached)

Gradient flow in Phase B
~~~~~~~~~~~~~~~~~~~~~~~~
    head_cls  ← policy gradient     ← REINFORCE(log π · advantage)
    head_res  ← differentiable SFT  ← loss_lm on correct anchors (RWR)
    coord_head ← differentiable SFT ← loss_coord, R detached, hidden detached
    (head_res NEVER receives loss_coord; coord_head NEVER receives loss_lm.)

GRPO objective (per batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Enumerate all G=24 cube anchors in a no_grad reward pass
     → rewards[k] ∈ {0, 1} (binary accuracy of MCQ answer)
  2. Group-normalised advantage:  A[k] = (r[k] - mean) / (std + ε)
     (If std < ε → group is degenerate; skip RL term but still do SFT/KL.)
  3. L_rl    = -(log_probs · A.detach()).mean()                 # 1-step REINFORCE
  4. L_ent   = -entropy_beta · H(π)                              # exploration
  5. L_kl    = kl_lambda · KL(π || Uniform)                      # anti-collapse
  6. L_sft   (hybrid) = Σ_k w_k · (w_lm · lm_loss_k + w_coord · coord_loss_k)
                        for k ∈ top-K correct anchors,
                        w_k = softmax(-lm_loss / τ)              # RWR
            (discrete) = w_coord · coord_loss(argmax k)
  7. L_total = rl_weight · (L_rl + L_ent + L_kl) + sft_weight · L_sft

Learning rate
~~~~~~~~~~~~~
Cosine annealing with warm restarts; one cycle per phase-pass.
Five independent schedules: LoRA, coord_head, rot_bb, head_cls, head_res.

Usage
~~~~~
  python train_rl.py \\
      --model_path checkpoints/Qwen3.5-4B \\
      --output_dir checkpoints/spa_rotation_rl \\
      --action_space hybrid \\
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.models import (
    DepthPredictionTransformer,
    CameraTokenRotationEncoderRL,
    RotationRoPEModel,
    SpaForConditionalGeneration,
)
from src.models.rotation_rope_llm import _build_token_txyz_int
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
    model_path:          str,
    image_token_id:      int,
    spatial_merge_size:  int,
    coord_upscale:       int   = 4,
    lora_rank:           int   = 16,
    freeze_vision:       bool  = True,
    answer_weight:       float = 1.0,
    coord_weight:        float = 1.0,
    rot_nhead:           int   = 4,
    rot_dim_feedforward: int   = 2048,
    rot_num_layers:      int   = 2,
    relative:            bool  = False,
    action_space:        str   = "hybrid",
    residual_clamp:      float = math.pi / 6,
) -> RotationRoPEModel:
    """Build RotationRoPEModel with the RL encoder (24-class + residual)."""
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

    mllm_head_dim = getattr(config.text_config, "head_dim", None) or (
        config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    log.info(
        f"mllm_head_dim={mllm_head_dim}  rot_nhead={rot_nhead}  "
        f"→ rotation_enc d_model={rot_nhead * mllm_head_dim}"
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

    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lm = (spa.model.model.language_model
          if hasattr(spa.model, "model") else spa.model.language_model)
    if not getattr(lm, "gradient_checkpointing", False):
        lm.gradient_checkpointing = True

    hidden_dim = config.text_config.hidden_size

    rot_rope_emb = SpaTextRotaryEmbedding(config=config.text_config).to(torch.bfloat16)

    rotation_enc = CameraTokenRotationEncoderRL(
        hidden_dim      = hidden_dim,
        mllm_head_dim   = mllm_head_dim,
        rope_emb        = rot_rope_emb,
        nhead           = rot_nhead,
        dim_feedforward = rot_dim_feedforward,
        num_layers      = rot_num_layers,
        dropout         = 0.0,                  # MUST be 0 for RL (det. no_grad == grad)
        action_space    = action_space,
        residual_clamp  = residual_clamp,
    ).to(torch.bfloat16)
    log.info(
        f"CameraTokenRotationEncoderRL  hidden_dim={hidden_dim}  "
        f"mllm_head_dim={mllm_head_dim}  nhead={rot_nhead}  "
        f"d_model={rotation_enc.d_model}  action_space={action_space}  "
        f"residual_clamp={residual_clamp:.4f} rad "
        f"({math.degrees(residual_clamp):.1f}°)"
    )

    cam_dim = rotation_enc.d_model if relative else 0
    coord_head = DepthPredictionTransformer(
        hidden_dim=hidden_dim, upscale_factor=coord_upscale,
        cam_dim=cam_dim,
    ).to(torch.bfloat16)
    log.info(
        f"DepthPredictionTransformer hidden_dim={hidden_dim} upscale={coord_upscale}"
        f"{f'  cam_dim={cam_dim}' if relative else ''}"
    )

    return RotationRoPEModel(
        spa_model          = spa,
        rotation_enc       = rotation_enc,
        coord_head         = coord_head,
        image_token_id     = image_token_id,
        spatial_merge_size = spatial_merge_size,
        answer_weight      = answer_weight,
        coord_weight       = coord_weight,
    )


# -- gradient utilities --------------------------------------------------------

def _set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)


# -- Phase A step (SFT with deterministic argmax anchor) -----------------------

def _phase_a_step(
    _model:        RotationRoPEModel,
    batch_tensors: dict,
    args:          argparse.Namespace,
    use_rot_enc:   bool,
):
    """Standard SFT using argmax(head_cls) + (optionally) argmax residual.

    Returns:
        (loss, loss_dict)  with loss already scaled by answer/coord weights.
    """
    input_ids       = batch_tensors["input_ids"]
    attention_mask  = batch_tensors["attention_mask"]
    pixel_values    = batch_tensors["pixel_values"]
    image_grid_thw  = batch_tensors["image_grid_thw"]
    image_xyz       = batch_tensors["image_xyz"]
    image_xyz_hires = batch_tensors["image_xyz_hires"]
    labels          = batch_tensors["labels"]
    device          = input_ids.device

    inputs_embeds = _model.encode_inputs(input_ids, pixel_values, image_grid_thw)

    R        = None
    cam_feat = None
    if use_rot_enc and image_xyz is not None and image_grid_thw is not None:
        token_txyz_int = _build_token_txyz_int(
            input_ids, _model.image_token_id,
            image_xyz, image_grid_thw, _model.spatial_merge_size,
            args.coord_scale,
        )
        logits, residual_all, cam_feat = _model.rotation_enc(
            inputs_embeds.detach(), token_txyz_int,
        )
        k_star = int(logits.argmax(-1).item())
        R = _model.rotation_enc.compose_R(k_star, residual_all)

    lm_loss, coord_loss, _ldict = _model.compute_losses_from_R(
        R                   = R,
        inputs_embeds       = inputs_embeds,
        input_ids           = input_ids,
        attention_mask      = attention_mask,
        image_xyz           = image_xyz,
        image_xyz_hires     = image_xyz_hires,
        image_grid_thw      = image_grid_thw,
        labels              = labels,
        coord_scale         = args.coord_scale,
        use_coord_loss      = not args.no_coord,
        use_relative        = args.relative,
        detach_coord_hidden = False,          # Phase A: coord flows normally
        cam_feat            = cam_feat,
        compute_reward      = False,
    )

    loss = None
    if lm_loss is not None:
        loss = _model.answer_weight * lm_loss
    if coord_loss is not None:
        loss = (loss + _model.coord_weight * coord_loss) if loss is not None \
                else (_model.coord_weight * coord_loss)
    return loss, _ldict, R


# -- Phase B step (GRPO) -------------------------------------------------------

def _phase_b_grpo_step(
    _model:          RotationRoPEModel,
    batch_tensors:   dict,
    args:            argparse.Namespace,
    grad_accum:      int                    = 1,
    policy_opt:      torch.optim.Optimizer | None = None,
    head_cls_params: list | None            = None,
    rot_bb_params:   list | None            = None,
):
    """One GRPO update over all 24 cube anchors.

    Discrete mode: true PPO inner loop.
        - ``log_probs_old`` and group-normalised advantages are snapshotted
          before the loop.
        - K = ``args.ppo_inner_epochs`` inner iterations each re-forward
          ``rotation_enc``, compute the clipped surrogate
          ``L_rl + L_ent + L_kl``, ``policy_opt.zero_grad / backward / step``.
          No feature caching — rot_bb weights change between iterations, so
          cam_feat must be regenerated every time.
        - head_cls / rot_bb grads are clipped inside the inner loop; the
          outer grad_accum pipeline does NOT touch these params in discrete.
        - Zero-variance batch: inner loop is skipped entirely (no L_rl /
          L_ent / L_kl).

    Hybrid mode: unchanged — single-step surrogate in Stage 2, reg_loss
    backward in Stage 4, all scaled by ``1/grad_accum``.

    Stage 3 (coord-head SFT) still runs its own in-loop ``backward(retain_graph=True)``
    scaled by ``1/grad_accum``; only the coord_head / LoRA path is affected.

    Returns:
        (loss_total_scalar: float, loss_dict, R_used) — scalar total loss
        value (already ``backward``-ed internally), dict of diagnostics, and
        the R used in the RWR/argmax path (for R-visualisation).
    """
    input_ids       = batch_tensors["input_ids"]
    attention_mask  = batch_tensors["attention_mask"]
    pixel_values    = batch_tensors["pixel_values"]
    image_grid_thw  = batch_tensors["image_grid_thw"]
    image_xyz       = batch_tensors["image_xyz"]
    image_xyz_hires = batch_tensors["image_xyz_hires"]
    labels          = batch_tensors["labels"]
    device          = input_ids.device

    _ldict: dict = {}

    # ── Encoding shared across all 24 no_grad + K grad forwards ─────────
    inputs_embeds = _model.encode_inputs(input_ids, pixel_values, image_grid_thw)

    token_txyz_int = _build_token_txyz_int(
        input_ids, _model.image_token_id,
        image_xyz, image_grid_thw, _model.spatial_merge_size,
        args.coord_scale,
    )
    # Encoder forward WITH grad — logits & residual_all carry head_cls /
    # head_res gradient back to their heads (and to encoder backbone).
    logits, residual_all, cam_feat = _model.rotation_enc(
        inputs_embeds.detach(), token_txyz_int,
    )   # logits (24,), residual_all (24,3) or None, cam_feat (d_model,)

    # ── Stage 1: 24 × no_grad reward collection ──────────────────────────
    rewards   = torch.zeros(24, dtype=torch.float32, device=device)
    lm_losses = torch.full((24,), float("inf"), dtype=torch.float32, device=device)

    # Consistency Protocol: evaluate with R_bins[k] @ exp(delta_k.detach()) so the
    # reward reflects the *current combined* pose (anchor + residual), matching
    # the Stage-3 SFT forward.  Inside no_grad, compose_R auto-detaches; in
    # discrete mode residual_all is None and compose_R falls back to R_bins[k].
    with torch.no_grad():
        for k in range(24):
            R_k = _model.rotation_enc.compose_R(k, residual_all)
            lm_loss_k, _, _ldict_k = _model.compute_losses_from_R(
                R                   = R_k,
                inputs_embeds       = inputs_embeds,
                input_ids           = input_ids,
                attention_mask      = attention_mask,
                image_xyz           = image_xyz,
                image_xyz_hires     = None,             # coord not needed for reward
                image_grid_thw      = image_grid_thw,
                labels              = labels,
                coord_scale         = args.coord_scale,
                use_coord_loss      = False,
                use_relative        = args.relative,
                detach_coord_hidden = True,
                cam_feat            = None,
                compute_reward      = True,
            )
            rewards[k]   = float(_ldict_k.get("acc", 0.0))
            if lm_loss_k is not None:
                lm_losses[k] = float(_ldict_k.get("lm_loss", float("inf")))

    n_correct = int((rewards > 0.5).sum().item())
    _ldict["n_correct"]   = float(n_correct)
    _ldict["reward_mean"] = float(rewards.mean().item())
    _ldict["reward_std"]  = float(rewards.std(unbiased=False).item())
    _ldict["lm_loss_best_anchor"] = float(lm_losses.min().item())

    # Discrete reward shaping: within each outcome group (correct / wrong)
    # add a zero-mean lm_loss-based perturbation. Group means are preserved
    # (correct stays at 1, wrong stays at 0), so the correct-vs-wrong contrast
    # is untouched while lm_loss breaks intra-group ties and provides a dense
    # signal in the all-correct / all-wrong degenerate batches.
    if _model.rotation_enc.action_space == "discrete":
        finite_mask    = torch.isfinite(lm_losses)
        rewards_shaped = rewards.clone()
        n_shaped_groups = 0
        for group_mask in (rewards > 0.5, rewards < 0.5):
            g   = group_mask & finite_mask
            n_g = int(g.sum().item())
            if n_g >= 2:
                lm_g = lm_losses[g]
                w    = F.softmax(-lm_g / args.rwr_tau, dim=0)
                rewards_shaped[g] = rewards_shaped[g] + args.w_lm * (w - 1.0 / n_g)
                n_shaped_groups += 1
        rewards = rewards_shaped
        _ldict["reward_shape_groups"] = float(n_shaped_groups)
        _ldict["reward_shape_spread"] = float(rewards.max() - rewards.min())

    # ── Stage 2: policy-gradient update(s) ──────────────────────────────
    # Discrete:  K inner PPO epochs, each one re-forwards rotation_enc and
    #            calls policy_opt.step() inline (no grad-accum on policy
    #            params).  NO feature caching — rot_bb weights change between
    #            iterations, so cam_feat must be regenerated every time.
    # Hybrid:    unchanged — single-step surrogate; Stage 4 handles backward.
    r_mean = rewards.mean()
    r_std  = rewards.std(unbiased=False)

    if _model.rotation_enc.action_space == "discrete":
        # Snapshot old policy (constant across inner epochs).
        with torch.no_grad():
            log_probs_old = F.log_softmax(logits, dim=-1)

        # Defaults for zero-variance skip.
        L_rl  = torch.zeros((), device=device)
        L_ent = torch.zeros((), device=device)
        L_kl  = torch.zeros((), device=device)
        _ldict["entropy"]   = 0.0
        _ldict["L_rl"]      = 0.0
        _ldict["rl_active"] = 0.0

        if r_std.item() > 1e-6 and policy_opt is not None:
            advantages = ((rewards - r_mean) / (r_std + 1e-8)).detach()
            K = int(args.ppo_inner_epochs)
            for inner_step in range(K):
                # Re-forward rotation_enc (rot_bb + head_cls). inputs_embeds
                # is already detached upstream, so only the shallow M-RoPE
                # encoder graph is rebuilt — cheap.
                logits_i, _res_i, _cam_i = _model.rotation_enc(
                    inputs_embeds.detach(), token_txyz_int,
                )
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                probs_i     = log_probs_i.exp()
                H_pi_i      = -(probs_i * log_probs_i).sum()

                ratio_i = torch.exp(log_probs_i - log_probs_old)      # (24,)
                surr1   = ratio_i * advantages
                surr2   = torch.clamp(
                    ratio_i, 1.0 - args.ppo_clip_eps,
                             1.0 + args.ppo_clip_eps,
                ) * advantages
                L_rl_i  = -torch.min(surr1, surr2).mean()
                L_ent_i = -args.entropy_beta * H_pi_i
                L_kl_i  = args.kl_lambda * (math.log(24) - H_pi_i)
                # No /grad_accum scaling: this is a standalone inner step.
                policy_loss_i = args.rl_weight * (L_rl_i + L_ent_i + L_kl_i)

                policy_opt.zero_grad(set_to_none=True)
                policy_loss_i.backward()
                if head_cls_params:
                    torch.nn.utils.clip_grad_norm_(
                        head_cls_params, max_norm=args.head_cls_clip
                    )
                if rot_bb_params:
                    torch.nn.utils.clip_grad_norm_(
                        rot_bb_params, max_norm=args.rot_bb_clip
                    )
                policy_opt.step()

                # Log stats from the final iteration (closest to the
                # post-update policy).
                if inner_step == K - 1:
                    L_rl  = L_rl_i.detach()
                    L_ent = L_ent_i.detach()
                    L_kl  = L_kl_i.detach()
                    _ldict["entropy"]    = float(H_pi_i.item())
                    _ldict["L_rl"]       = float(L_rl_i.item())
                    _ldict["ratio_mean"] = float(ratio_i.mean().item())
                    _ldict["ratio_max"]  = float(ratio_i.max().item())
                    _ldict["clip_frac"]  = float(
                        ((ratio_i < 1.0 - args.ppo_clip_eps) |
                         (ratio_i > 1.0 + args.ppo_clip_eps)).float().mean().item())
            _ldict["rl_active"]        = 1.0
            _ldict["ppo_inner_epochs"] = float(K)

    else:
        # Hybrid mode (unchanged single-step surrogate + Stage-4 backward).
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        H_pi      = -(probs * log_probs).sum()
        if r_std.item() > 1e-6:
            advantages = (rewards - r_mean) / (r_std + 1e-8)
            log_probs_old = log_probs.detach()
            ratio = torch.exp(log_probs - log_probs_old)
            adv   = advantages.detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.ppo_clip_eps,
                                        1.0 + args.ppo_clip_eps) * adv
            L_rl  = -torch.min(surr1, surr2).mean()
            L_ent = -args.entropy_beta * H_pi
            L_kl  = args.kl_lambda * (math.log(24) - H_pi)
            _ldict["rl_active"]   = 1.0
            _ldict["ratio_mean"]  = float(ratio.mean().item())
            _ldict["ratio_max"]   = float(ratio.max().item())
            _ldict["clip_frac"]   = float(
                ((ratio < 1.0 - args.ppo_clip_eps) |
                 (ratio > 1.0 + args.ppo_clip_eps)).float().mean().item())
        else:
            L_rl  = torch.zeros((), device=device)
            L_ent = torch.zeros((), device=device)
            L_kl  = torch.zeros((), device=device)
            _ldict["rl_active"] = 0.0
        _ldict["entropy"]  = float(H_pi.item())
        _ldict["L_rl"]     = float(L_rl.item())

    # ── Stage 3: differentiable SFT (RWR weighted / argmax) ──────────────
    #   hybrid  : grad forward on up to K_MAX_GRAD correct anchors, weighted
    #             by softmax(-lm_loss / τ).  head_res receives loss_lm;
    #             coord_head receives loss_coord (hidden detached).
    #   discrete: single grad forward on argmax(logits) — coord_loss only
    #             (head_cls is updated only via policy gradient).
    #
    # **In-loop backward**: each SFT iteration calls backward(retain_graph=True)
    # immediately so the LLM+Vision activation graph of *this* iteration is
    # released before the next forward.  retain_graph=True keeps the small
    # rotation_enc graph alive for: (i) subsequent SFT iterations that re-use
    # residual_all/cam_feat, and (ii) the final reg-loss backward on logits.
    sft_coef     = args.sft_weight / grad_accum
    L_sft_scalar = 0.0              # Σ weights[i] · step_loss (unscaled) for logs
    R_used       = None             # for R-visualisation

    if _model.rotation_enc.action_space == "hybrid":
        correct_idx = torch.where(rewards > 0.5)[0]
        if correct_idx.numel() > 0:
            correct_lm = lm_losses[correct_idx]
            # Top-K truncation by lowest lm_loss (best anchors first).
            K = int(min(args.k_max_grad, correct_idx.numel()))
            if correct_idx.numel() > K:
                topk = torch.topk(-correct_lm, k=K)
                keep = topk.indices
                correct_idx = correct_idx[keep]
                correct_lm  = correct_lm[keep]
            # RWR weights: favour anchors with smaller lm_loss.
            weights = F.softmax(-correct_lm / args.rwr_tau, dim=0)
            _ldict["sft_n_anchors"] = float(correct_idx.numel())

            for i, k_idx in enumerate(correct_idx.tolist()):
                R_k_grad = _model.rotation_enc.compose_R(k_idx, residual_all)
                lm_loss_g, coord_loss_g, _ldict_g = _model.compute_losses_from_R(
                    R                   = R_k_grad,
                    inputs_embeds       = inputs_embeds,
                    input_ids           = input_ids,
                    attention_mask      = attention_mask,
                    image_xyz           = image_xyz,
                    image_xyz_hires     = image_xyz_hires,
                    image_grid_thw      = image_grid_thw,
                    labels              = labels,
                    coord_scale         = args.coord_scale,
                    use_coord_loss      = not args.no_coord,
                    use_relative        = args.relative,
                    detach_coord_hidden = True,
                    cam_feat            = cam_feat,
                    compute_reward      = False,
                )
                step_loss = torch.zeros((), device=device)
                if lm_loss_g is not None:
                    step_loss = step_loss + args.w_lm    * lm_loss_g
                if coord_loss_g is not None:
                    step_loss = step_loss + args.w_coord * coord_loss_g
                weighted = weights[i] * step_loss

                # Backward this iteration NOW → free the LLM/Vision graph.
                (sft_coef * weighted).backward(retain_graph=True)
                L_sft_scalar += float(weighted.item())

                if i == 0:
                    R_used = R_k_grad.detach()          # record best anchor's R
        else:
            _ldict["sft_n_anchors"] = 0.0
    else:
        # Discrete: no residual head → head_res is None.  Use argmax anchor
        # for the coord_loss grad forward (coord_head updates only).
        k_star = int(logits.argmax(-1).item())
        R_star = _model.rotation_enc.compose_R(k_star, None)
        _, coord_loss_g, _ldict_g = _model.compute_losses_from_R(
            R                   = R_star,
            inputs_embeds       = inputs_embeds,
            input_ids           = input_ids,
            attention_mask      = attention_mask,
            image_xyz           = image_xyz,
            image_xyz_hires     = image_xyz_hires,
            image_grid_thw      = image_grid_thw,
            labels              = labels,
            coord_scale         = args.coord_scale,
            use_coord_loss      = not args.no_coord,
            use_relative        = args.relative,
            detach_coord_hidden = True,
            cam_feat            = cam_feat,
            compute_reward      = False,
        )
        if coord_loss_g is not None:
            step_loss = args.w_coord * coord_loss_g
            (sft_coef * step_loss).backward(retain_graph=True)
            L_sft_scalar = float(step_loss.item())
        R_used = R_star.detach()

    _ldict["L_sft"] = L_sft_scalar

    # ── Stage 4: final backward — reg loss on logits only ────────────────
    # Discrete: inner PPO loop already ran policy_opt.zero_grad/backward/step
    # K times, so no outer backward is needed for the policy term. Hybrid:
    # original single-step pipeline — backward reg_loss into the shared
    # optimizer's rot_bb/head_cls/head_res groups.
    if _model.rotation_enc.action_space == "hybrid":
        reg_loss = (args.rl_weight / grad_accum) * (L_rl + L_ent + L_kl)
        reg_loss.backward()

    loss_total_scalar = (
        args.rl_weight * (float(L_rl.item()) + float(L_ent.item()) + float(L_kl.item()))
        + args.sft_weight * L_sft_scalar
    )
    _ldict["L_total"] = loss_total_scalar

    return loss_total_scalar, _ldict, R_used


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
        relative            = args.relative,
        action_space        = args.action_space,
        residual_clamp      = args.residual_clamp,
    )
    model = model.to(device)
    if local_rank == 0:
        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"[MEM] After model.to(device): {mem_gb:.2f} GiB allocated")

    # -- DDP -------------------------------------------------------------------
    if world_size > 1:
        # find_unused_parameters=True because:
        #   (a) different phases freeze different parameter sets
        #   (b) Phase B RWR: number of correct anchors (K) varies per rank →
        #       head_res receives grad on some ranks but not others.
        model  = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        _model = model.module
    else:
        _model = model

    # -- dataset / loader ------------------------------------------------------
    if args.training_dataset == "mindcube":
        train_dataset = MindCube_Train_Dataset_Rotation(
            args.json_path, args.results_dir, processor, None, log,
            max_images         = args.max_images,
            spatial_merge_size = spatial_merge_size,
            coord_upscale      = args.coord_upscale,
            max_samples        = args.max_samples,
            no_cam             = True,
        )
    elif args.training_dataset == "sat":
        train_dataset = SAT_Train_Dataset_Rotation(
            args.json_path, args.results_dir, processor, log,
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
        train_dataset, batch_size=1, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=collate_fn,
        sampler=train_sampler,
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
                _ds_jsonl, _ds_results, processor, None, log,
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

    # -- parameter groups ------------------------------------------------------
    # Five mutually-exclusive groups for the phase-aware toggling:
    #   head_cls     — policy head    (RL-only; strict clip)
    #   head_res     — residual head  (RWR-only; standard clip)
    #   rot_bb       — encoder layers + input_proj + cam_token (shared signal)
    #   coord_head   — depth predictor (active in BOTH phases)
    #   lora         — LoRA on SpaModel (Phase A only)
    head_cls_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc.head_cls" in n
    ]
    head_res_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "rotation_enc.head_res" in n
    ]
    rot_bb_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and "rotation_enc" in n
        and "head_cls" not in n
        and "head_res" not in n
    ]
    coord_head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "coord_head" in n and "rotation_enc" not in n
    ]
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and "rotation_enc" not in n
        and "coord_head"   not in n
    ]

    lr_phase_a_base    = args.lr_phase_a    if args.lr_phase_a    is not None else args.lr
    lr_phase_b_base    = args.lr_phase_b    if args.lr_phase_b    is not None else args.rotation_enc_lr
    lr_coord_head_base = args.lr_coord_head if args.lr_coord_head is not None else args.lr

    # Split optimizer (discrete only): head_cls + rot_bb live on policy_opt
    # and are stepped K times per rollout inside _phase_b_grpo_step. Their
    # base LR is divided by K so effective per-rollout update magnitude
    # stays comparable to the previous single-step setup.
    # In hybrid mode policy_opt is None and all five groups share optimizer
    # with the original grad-accum pipeline.
    K_ppo = max(1, int(args.ppo_inner_epochs))
    if args.action_space == "discrete":
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params,       "lr": args.lr,            "name": "lora"},
                {"params": coord_head_params, "lr": args.lr,            "name": "coord_head"},
                {"params": head_res_params,   "lr": lr_phase_b_base,    "name": "head_res"},
            ],
            weight_decay=0.01,
        )
        policy_opt = torch.optim.AdamW(
            [
                {"params": rot_bb_params,
                 "lr": lr_phase_b_base / K_ppo,                         "name": "rot_bb"},
                {"params": head_cls_params,
                 "lr": lr_phase_b_base * args.head_cls_lr_scale / K_ppo,"name": "head_cls"},
            ],
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params,       "lr": args.lr,                                 "name": "lora"},
                {"params": coord_head_params, "lr": args.lr,                                 "name": "coord_head"},
                {"params": rot_bb_params,     "lr": lr_phase_b_base,                         "name": "rot_bb"},
                {"params": head_cls_params,   "lr": lr_phase_b_base * args.head_cls_lr_scale,"name": "head_cls"},
                {"params": head_res_params,   "lr": lr_phase_b_base,                         "name": "head_res"},
            ],
            weight_decay=0.01,
        )
        policy_opt = None
    log.info(
        f"Optim groups: lora={len(lora_params)} "
        f"coord_head={len(coord_head_params)} "
        f"rot_bb={len(rot_bb_params)} "
        f"head_cls={len(head_cls_params)} "
        f"head_res={len(head_res_params)}"
        f"  (policy_opt={'on' if policy_opt is not None else 'off'}, K={K_ppo})"
    )

    log.info("=" * 72)
    log.info(f">>> action_space   = {args.action_space}")
    log.info(f">>> residual_clamp = {args.residual_clamp:.4f} rad "
             f"({math.degrees(args.residual_clamp):.1f}°)")
    log.info(f">>> group_size     = 24 (exhaustive cube-group enumeration)")
    log.info(f">>> k_max_grad     = {args.k_max_grad}")
    log.info(f">>> rwr_tau        = {args.rwr_tau}")
    log.info(f">>> entropy_beta   = {args.entropy_beta}")
    log.info(f">>> kl_lambda      = {args.kl_lambda}")
    log.info(f">>> rl_weight      = {args.rl_weight}")
    log.info(f">>> sft_weight     = {args.sft_weight}")
    log.info(f">>> w_lm           = {args.w_lm}")
    log.info(f">>> w_coord        = {args.w_coord}")
    log.info(f">>> no_coord       = {args.no_coord}")
    log.info(f">>> relative       = {args.relative}")
    log.info(f">>> lr_phase_a     = {lr_phase_a_base:.2e}  (LoRA, warm-restart per Phase A)")
    log.info(f">>> lr_phase_b     = {lr_phase_b_base:.2e}  (rot_bb/head_res, warm-restart per Phase B)")
    log.info(f">>> head_cls_lr    = {lr_phase_b_base * args.head_cls_lr_scale:.2e}"
             f"  (scale={args.head_cls_lr_scale}, strict clip {args.head_cls_clip})")
    if args.action_space == "discrete":
        log.info(f">>> ppo_inner_K    = {K_ppo}  (policy_opt LR divided by K:"
                 f" rot_bb={lr_phase_b_base / K_ppo:.2e}, "
                 f"head_cls={lr_phase_b_base * args.head_cls_lr_scale / K_ppo:.2e})")
        log.info(f">>> ppo_clip_eps   = {args.ppo_clip_eps}")
    log.info(f">>> lr_coord_head  = {lr_coord_head_base:.2e}  (warm-restart every pass)")
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

    model.train()

    global_step  = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    steps_per_epoch = len(train_loader)
    current_phase   = None
    cycle_optim_steps = max(1, steps_per_epoch // args.grad_accum)
    phase_a_cycle_step    = 0
    phase_b_cycle_step    = 0
    coord_head_cycle_step = 0

    for epoch in range(args.epochs):
        for pass_idx in range(2):
            new_phase = "A" if pass_idx == 0 else "B"
            if new_phase != current_phase:
                if new_phase == "A":
                    _set_requires_grad(head_cls_params,   False)
                    _set_requires_grad(head_res_params,   False)
                    _set_requires_grad(rot_bb_params,     False)
                    _set_requires_grad(lora_params,       True)
                    _set_requires_grad(coord_head_params, True)
                    phase_a_cycle_step    = 0
                    coord_head_cycle_step = 0
                else:
                    _set_requires_grad(head_cls_params,   True)
                    _set_requires_grad(head_res_params,   True)
                    _set_requires_grad(rot_bb_params,     True)
                    _set_requires_grad(lora_params,       False)
                    _set_requires_grad(coord_head_params, True)
                    phase_b_cycle_step    = 0
                    coord_head_cycle_step = 0
                current_phase = new_phase
                if local_rank == 0:
                    if new_phase == "A":
                        msg = ("(train LoRA+coord_head; R≈I — encoder zero-init)"
                               if epoch == 0
                               else "(train LoRA+coord_head; R from argmax anchor)")
                    else:
                        msg = (f"(GRPO — action_space={args.action_space}, "
                               f"G=24, K_max={args.k_max_grad})")
                    log.info(f"[epoch {epoch+1:02d} pass {pass_idx}] "
                             f"→ Phase {new_phase} full pass  {msg}")

            use_rot_enc = (new_phase == "B") or (epoch > 0)

            if train_sampler is not None:
                train_sampler.set_epoch(epoch * 2 + pass_idx)

            for step, batch in enumerate(train_loader):
                # -- move batch to device --------------------------------
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

                batch_tensors = {
                    "input_ids":       input_ids,
                    "attention_mask":  attention_mask,
                    "pixel_values":    pixel_values,
                    "image_grid_thw":  image_grid_thw,
                    "image_xyz":       image_xyz,
                    "image_xyz_hires": image_xyz_hires,
                    "labels":          labels,
                }

                if step == 0 and local_rank == 0:
                    n_img_tok  = (input_ids[0] == image_token_id).sum().item()
                    mem_before = torch.cuda.memory_allocated(device) / 1e9
                    log.info(
                        f"[MEM] Phase {new_phase} step 0: "
                        f"seq_len={input_ids.shape[1]}, img_tokens={n_img_tok}, "
                        f"image_grid_thw={image_grid_thw}, "
                        f"mem_before_fwd={mem_before:.2f} GiB"
                    )

                # -- forward + loss --------------------------------------
                if new_phase == "A":
                    loss, loss_dict, _ = _phase_a_step(
                        _model, batch_tensors, args, use_rot_enc,
                    )
                    if loss is None:
                        log.warning(f"[rank{local_rank}] step {step}: no signal, skip.")
                        continue
                    (loss / args.grad_accum).backward()
                    running_loss += loss.item()
                else:
                    # Phase B does per-iteration backward internally (see
                    # _phase_b_grpo_step's in-loop backward) and already
                    # applies the 1/grad_accum factor, so no .backward() here.
                    # In discrete mode the policy sub-optimizer runs K inner
                    # steps of its own inside _phase_b_grpo_step.
                    loss_scalar, loss_dict, _ = _phase_b_grpo_step(
                        _model, batch_tensors, args,
                        grad_accum=args.grad_accum,
                        policy_opt=policy_opt,
                        head_cls_params=head_cls_params,
                        rot_bb_params=rot_bb_params,
                    )
                    running_loss += loss_scalar
                if loss_dict:
                    for k, v in loss_dict.items():
                        running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

                # -- gradient accumulation / step ------------------------
                if (step + 1) % args.grad_accum == 0:
                    # Per-group grad clipping. In discrete mode head_cls /
                    # rot_bb are clipped inside the PPO inner loop and live
                    # on policy_opt (already stepped) — skip them here.
                    _skip_policy_clip = (
                        args.action_space == "discrete" and policy_opt is not None
                    )
                    if head_cls_params and not _skip_policy_clip:
                        torch.nn.utils.clip_grad_norm_(
                            head_cls_params, max_norm=args.head_cls_clip
                        )
                    if head_res_params:
                        torch.nn.utils.clip_grad_norm_(
                            head_res_params, max_norm=args.head_res_clip
                        )
                    if rot_bb_params and not _skip_policy_clip:
                        torch.nn.utils.clip_grad_norm_(
                            rot_bb_params, max_norm=args.rot_bb_clip
                        )
                    if lora_params or coord_head_params:
                        torch.nn.utils.clip_grad_norm_(
                            lora_params + coord_head_params, max_norm=args.lora_clip
                        )
                    optimizer.step()

                    # Cosine warm-restart schedule (coord_head every pass;
                    # LoRA on Phase A; rot_bb/head_res/head_cls on Phase B).
                    coord_head_cycle_step += 1
                    _t_ch  = min(coord_head_cycle_step, cycle_optim_steps)
                    _cos_ch = 0.5 * (1 + math.cos(math.pi * _t_ch / cycle_optim_steps))
                    _lr_ch  = lr_coord_head_base * _cos_ch
                    for _g in optimizer.param_groups:
                        if _g.get("name") == "coord_head":
                            _g["lr"] = _lr_ch

                    if current_phase == "A":
                        phase_a_cycle_step += 1
                        _t   = min(phase_a_cycle_step, cycle_optim_steps)
                        _cos = 0.5 * (1 + math.cos(math.pi * _t / cycle_optim_steps))
                        _lr  = lr_phase_a_base * _cos
                        for _g in optimizer.param_groups:
                            if _g.get("name") == "lora":
                                _g["lr"] = _lr
                    else:
                        phase_b_cycle_step += 1
                        _t   = min(phase_b_cycle_step, cycle_optim_steps)
                        _cos = 0.5 * (1 + math.cos(math.pi * _t / cycle_optim_steps))
                        _lr_b   = lr_phase_b_base                   * _cos
                        _lr_cls = lr_phase_b_base * args.head_cls_lr_scale * _cos
                        # In discrete mode head_cls / rot_bb live on policy_opt
                        # with base LR already divided by K — reapply the same
                        # division to keep "K inner steps ≈ one big step".
                        _policy_scale = (1.0 / K_ppo) if args.action_space == "discrete" else 1.0
                        for _g in optimizer.param_groups:
                            if _g.get("name") == "rot_bb":
                                _g["lr"] = _lr_b
                            elif _g.get("name") == "head_res":
                                _g["lr"] = _lr_b
                            elif _g.get("name") == "head_cls":
                                _g["lr"] = _lr_cls
                        if policy_opt is not None:
                            for _g in policy_opt.param_groups:
                                if _g.get("name") == "rot_bb":
                                    _g["lr"] = _lr_b   * _policy_scale
                                elif _g.get("name") == "head_cls":
                                    _g["lr"] = _lr_cls * _policy_scale

                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss      = running_loss / args.grad_accum
                    avg_loss_dict = {k: v / args.grad_accum for k, v in running_loss_dict.items()}
                    running_loss  = 0.0
                    running_loss_dict.clear()

                    if world_size > 1:
                        _keys = sorted(avg_loss_dict.keys())
                        _vals = [avg_loss] + [avg_loss_dict[k] for k in _keys]
                        _t    = torch.tensor(_vals, dtype=torch.float64, device=device)
                        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                        _t /= world_size
                        avg_loss      = _t[0].item()
                        avg_loss_dict = {k: _t[i + 1].item() for i, k in enumerate(_keys)}

                    if local_rank == 0:
                        if current_phase == "A":
                            _active_group = "lora"
                        else:
                            _active_group = "head_res" \
                                if args.action_space == "hybrid" else "head_cls"
                        # In discrete mode head_cls lives on policy_opt.
                        _lr_sources = [optimizer]
                        if policy_opt is not None:
                            _lr_sources.append(policy_opt)
                        current_lr = next(
                            g["lr"] for opt in _lr_sources
                            for g in opt.param_groups
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
                            f"(aggregated across {world_size} GPU"
                            f"{'s' if world_size > 1 else ''})"
                        )
                        if use_wandb:
                            wandb.log(
                                {
                                    "train/loss":          avg_loss,
                                    "train/lr":            current_lr,
                                    "train/coord_head_lr": coord_head_lr,
                                    "train/phase":         0 if current_phase == "A" else 1,
                                    "epoch":               epoch + 1,
                                    **{f"train/{k}": v for k, v in avg_loss_dict.items()},
                                },
                                step=global_step,
                            )
                        if global_step % args.save_steps == 0:
                            _save_checkpoint(_model, tokenizer, args.output_dir, global_step)

                    # -- periodic evaluation -------------------------------
                    if test_loaders and global_step > 0 and global_step % args.eval_steps == 0:
                        _run_eval(
                            model, _model, test_loaders, test_samplers,
                            args, device, global_step, use_wandb,
                        )

    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step, suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


# -- eval loop (argmax anchor, deterministic) ----------------------------------

def _run_eval(
    model, _model, test_loaders, test_samplers,
    args, device, global_step, use_wandb,
):
    """Deterministic eval using argmax anchor + (hybrid) argmax residual."""
    model.eval()
    _spa    = _model.spa_model
    _spa_gc = getattr(_spa, "gradient_checkpointing", False)
    _lm     = getattr(_spa, "language_model", None)
    _lm_gc  = getattr(_lm,  "gradient_checkpointing", False) if _lm else False
    if _spa_gc:
        _spa.gradient_checkpointing = False
    if _lm and _lm_gc:
        _lm.gradient_checkpointing = False

    _is_hybrid = _model.rotation_enc.action_space == "hybrid"
    _topk      = max(1, min(int(args.eval_topk), 24))

    for ds_name, loader in test_loaders.items():
        if ds_name in test_samplers and test_samplers[ds_name] is not None:
            test_samplers[ds_name].set_epoch(global_step)

        local_count     = 0
        local_loss_sums: dict[str, float] = {}
        local_R_angles:   list[float] = []
        local_entropies:  list[float] = []
        local_k_stars:    list[int]   = []
        local_topk_hits:  list[float] = []
        local_delta_xyz:  list[list[float]] = []   # hybrid only: (3,) per sample

        for batch in loader:
            t_ids   = batch["input_ids"].to(device)
            t_mask  = batch["attention_mask"].to(device)
            t_pv    = batch.get("pixel_values")
            t_thw   = batch.get("image_grid_thw")
            t_labels = batch.get("labels")
            t_xyz   = batch.get("image_xyz")
            t_xyz_h = batch.get("image_xyz_hires")
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
                inputs_embeds = _model.encode_inputs(t_ids, t_pv, t_thw)
                token_txyz_int = _build_token_txyz_int(
                    t_ids, _model.image_token_id,
                    t_xyz, t_thw, _model.spatial_merge_size,
                    args.coord_scale,
                )
                logits, residual_all, cam_feat = _model.rotation_enc(
                    inputs_embeds, token_txyz_int,
                )
                probs    = torch.softmax(logits.float(), dim=-1)            # (24,)
                entropy  = float(-(probs * (probs.clamp_min(1e-12)).log()).sum().item())
                topk_idx = logits.topk(_topk).indices.tolist()              # list[int]
                k_star   = int(topk_idx[0])
                R = _model.rotation_enc.compose_R(k_star, residual_all)

                lm_loss, coord_loss, _ldict = _model.compute_losses_from_R(
                    R                   = R,
                    inputs_embeds       = inputs_embeds,
                    input_ids           = t_ids,
                    attention_mask      = t_mask,
                    image_xyz           = t_xyz,
                    image_xyz_hires     = t_xyz_h,
                    image_grid_thw      = t_thw,
                    labels              = t_labels,
                    coord_scale         = args.coord_scale,
                    use_coord_loss      = not args.no_coord,
                    use_relative        = args.relative,
                    detach_coord_hidden = False,
                    cam_feat            = cam_feat,
                    compute_reward      = True,
                )

                # Pass@K: short-circuit once any top-K anchor is correct.
                topk_hit = float(_ldict.get("acc", 0.0)) if _ldict else 0.0
                if topk_hit < 0.5 and _topk > 1:
                    for k_other in topk_idx[1:]:
                        R_o = _model.rotation_enc.compose_R(int(k_other), residual_all)
                        _, _, _ld_o = _model.compute_losses_from_R(
                            R                   = R_o,
                            inputs_embeds       = inputs_embeds,
                            input_ids           = t_ids,
                            attention_mask      = t_mask,
                            image_xyz           = t_xyz,
                            image_xyz_hires     = t_xyz_h,
                            image_grid_thw      = t_thw,
                            labels              = t_labels,
                            coord_scale         = args.coord_scale,
                            use_coord_loss      = False,
                            use_relative        = args.relative,
                            detach_coord_hidden = False,
                            cam_feat            = cam_feat,
                            compute_reward      = True,
                        )
                        if _ld_o and float(_ld_o.get("acc", 0.0)) > 0.5:
                            topk_hit = 1.0
                            break

            if lm_loss is None and coord_loss is None:
                continue
            local_count += 1
            if _ldict:
                for k, v in _ldict.items():
                    local_loss_sums[k] = local_loss_sums.get(k, 0.0) + float(v)

            _R = R.detach().float()
            _tr = _R[0, 0] + _R[1, 1] + _R[2, 2]
            _cos = ((_tr - 1.0) / 2.0).clamp(-1.0, 1.0)
            local_R_angles.append(float(torch.acos(_cos) * (180.0 / math.pi)))

            local_entropies.append(entropy)
            local_k_stars.append(k_star)
            local_topk_hits.append(topk_hit)

            if _is_hybrid and residual_all is not None:
                # Effective axis-angle actually applied to rotation for k_star.
                _delta = (args.residual_clamp
                          * torch.tanh(residual_all[k_star].float())).tolist()
                local_delta_xyz.append([float(_delta[0]),
                                        float(_delta[1]),
                                        float(_delta[2])])

        _keys = sorted(local_loss_sums.keys())
        if world_size > 1:
            _vals = [float(local_count)] + [local_loss_sums.get(k, 0.0) for k in _keys]
            stats = torch.tensor(_vals, dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_count = int(stats[0].item())
            agg_sums    = {k: stats[i + 1].item() for i, k in enumerate(_keys)}
        else:
            total_count = local_count
            agg_sums    = dict(local_loss_sums)

        if total_count > 0 and local_rank == 0:
            detail = "  ".join(
                f"{k}={agg_sums[k] / total_count:.4f}" for k in _keys
            )
            log.info(f"[eval] global_step={global_step:05d}  {ds_name}  "
                     + detail
                     + f"  (n={total_count}, {world_size} GPU"
                     f"{'s' if world_size > 1 else ''})")
            if use_wandb:
                _main = {"coord_loss", "lm_loss", "acc"}
                wandb.log(
                    {
                        **{f"eval/{ds_name}_{k}": agg_sums[k] / total_count
                           for k in _keys if k in _main},
                        **{f"eval_sub/{ds_name}_{k}": agg_sums[k] / total_count
                           for k in _keys if k not in _main},
                    },
                    step=global_step,
                )

        # Rotation angle visualisation (θ = acos((tr R - 1) / 2)).
        if world_size > 1:
            _gathered = [None] * world_size
            dist.all_gather_object(_gathered, local_R_angles)
            all_angles = [a for lst in _gathered for a in (lst or [])]
        else:
            all_angles = list(local_R_angles)
        if all_angles and local_rank == 0:
            _n    = len(all_angles)
            _mean = sum(all_angles) / _n
            _std  = (sum((a - _mean) ** 2 for a in all_angles) / _n) ** 0.5
            _mn   = min(all_angles)
            _mx   = max(all_angles)
            log.info(
                f"[eval-R] global_step={global_step:05d}  {ds_name}  "
                f"angle_deg: mean={_mean:.2f}  std={_std:.2f}  "
                f"min={_mn:.2f}  max={_mx:.2f}  (n={_n})"
            )
            if use_wandb:
                wandb.log(
                    {
                        f"eval_R/{ds_name}_angle_deg_mean": _mean,
                        f"eval_R/{ds_name}_angle_deg_std":  _std,
                        f"eval_R/{ds_name}_angle_deg_min":  _mn,
                        f"eval_R/{ds_name}_angle_deg_max":  _mx,
                        f"eval_R/{ds_name}_angle_deg_hist": wandb.Histogram(all_angles),
                    },
                    step=global_step,
                )

        # -- Policy analytics: entropy, argmax-k histogram, Pass@K -------------
        if world_size > 1:
            _g_ent:  list = [None] * world_size
            _g_ks:   list = [None] * world_size
            _g_hit:  list = [None] * world_size
            dist.all_gather_object(_g_ent, local_entropies)
            dist.all_gather_object(_g_ks,  local_k_stars)
            dist.all_gather_object(_g_hit, local_topk_hits)
            all_entropies = [x for lst in _g_ent for x in (lst or [])]
            all_k_stars   = [x for lst in _g_ks  for x in (lst or [])]
            all_topk_hits = [x for lst in _g_hit for x in (lst or [])]
        else:
            all_entropies = list(local_entropies)
            all_k_stars   = list(local_k_stars)
            all_topk_hits = list(local_topk_hits)

        if all_entropies and local_rank == 0:
            _n      = len(all_entropies)
            _H_mean = sum(all_entropies) / _n
            _H_std  = (sum((h - _H_mean) ** 2 for h in all_entropies) / _n) ** 0.5
            _topk_acc = (sum(all_topk_hits) / _n) if all_topk_hits else 0.0
            # Uniform coverage fraction = |unique k_stars| / 24.
            _unique_k = len(set(all_k_stars))
            log.info(
                f"[eval-pi] global_step={global_step:05d}  {ds_name}  "
                f"H(pi): mean={_H_mean:.3f}  std={_H_std:.3f}  "
                f"top{_topk}_acc={_topk_acc:.4f}  "
                f"unique_k={_unique_k}/24  (n={_n})"
            )
            if use_wandb:
                _k_tensor = torch.tensor(all_k_stars, dtype=torch.int64)
                _k_counts = torch.bincount(_k_tensor, minlength=24).tolist()
                _log = {
                    f"eval_pi/{ds_name}_entropy_mean":  _H_mean,
                    f"eval_pi/{ds_name}_entropy_std":   _H_std,
                    f"eval_pi/{ds_name}_entropy_hist":  wandb.Histogram(all_entropies),
                    f"eval_pi/{ds_name}_top{_topk}_acc": _topk_acc,
                    f"eval_pi/{ds_name}_unique_k":     _unique_k,
                    f"eval_pi/{ds_name}_kstar_hist":   wandb.Histogram(
                        np_histogram=(_k_counts, list(range(25)))
                    ),
                }
                wandb.log(_log, step=global_step)

        # -- Residual analytics (hybrid only): per-DoF + magnitude -------------
        if _is_hybrid:
            if world_size > 1:
                _g_d: list = [None] * world_size
                dist.all_gather_object(_g_d, local_delta_xyz)
                all_deltas = [v for lst in _g_d for v in (lst or [])]
            else:
                all_deltas = list(local_delta_xyz)

            if all_deltas and local_rank == 0:
                _dt  = torch.tensor(all_deltas, dtype=torch.float32)    # (N,3)
                _mag = _dt.norm(dim=-1)                                 # (N,)
                _per_dof_mean = _dt.mean(dim=0).tolist()
                _per_dof_std  = _dt.std(dim=0, unbiased=False).tolist()
                _mag_mean = float(_mag.mean().item())
                _mag_std  = float(_mag.std(unbiased=False).item())
                log.info(
                    f"[eval-res] global_step={global_step:05d}  {ds_name}  "
                    f"|delta|: mean={_mag_mean:.4f}  std={_mag_std:.4f}  "
                    f"delta_xyz_mean=({_per_dof_mean[0]:+.4f},"
                    f"{_per_dof_mean[1]:+.4f},{_per_dof_mean[2]:+.4f})  "
                    f"(n={_dt.shape[0]})"
                )
                if use_wandb:
                    wandb.log(
                        {
                            f"eval_res/{ds_name}_delta_x_mean": _per_dof_mean[0],
                            f"eval_res/{ds_name}_delta_y_mean": _per_dof_mean[1],
                            f"eval_res/{ds_name}_delta_z_mean": _per_dof_mean[2],
                            f"eval_res/{ds_name}_delta_x_std":  _per_dof_std[0],
                            f"eval_res/{ds_name}_delta_y_std":  _per_dof_std[1],
                            f"eval_res/{ds_name}_delta_z_std":  _per_dof_std[2],
                            f"eval_res/{ds_name}_delta_mag_mean": _mag_mean,
                            f"eval_res/{ds_name}_delta_mag_std":  _mag_std,
                            f"eval_res/{ds_name}_delta_mag_hist":
                                wandb.Histogram(_mag.tolist()),
                        },
                        step=global_step,
                    )

    if _spa_gc:
        _spa.gradient_checkpointing = True
    if _lm and _lm_gc:
        _lm.gradient_checkpointing = True
    model.train()


# -- checkpoint ----------------------------------------------------------------

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
    torch.save(model.rotation_enc.state_dict(),
               os.path.join(ckpt, "rotation_enc.pt"))
    torch.save(model.coord_head.state_dict(),
               os.path.join(ckpt, "coord_head.pt"))
    log.info(f"Checkpoint saved -> {ckpt}")


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alternating two-phase training: SFT (Phase A) + GRPO RL "
                    "(Phase B) over 24 chiral cube anchors with optional "
                    "3-DoF residual."
    )
    p.add_argument("--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"))
    p.add_argument("--training_dataset", choices=["mindcube", "sat"], default="sat")
    p.add_argument("--json_path",
        default=os.path.join(_ROOT, "datasets/train/SAT/train_36k.json"))
    p.add_argument("--results_dir",
        default=os.path.join(_ROOT, "datasets/train/SAT/3d_results"))
    p.add_argument("--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_rotation_rl"))
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--rotation_enc_lr",type=float, default=5e-5)
    p.add_argument("--lr_phase_a",     type=float, default=None)
    p.add_argument("--lr_phase_b",     type=float, default=None)
    p.add_argument("--lr_coord_head",  type=float, default=None)

    p.add_argument("--lora_clip",      type=float, default=1.0)
    p.add_argument("--rot_bb_clip",    type=float, default=0.3)
    p.add_argument("--head_cls_clip",  type=float, default=0.3)
    p.add_argument("--head_res_clip",  type=float, default=1.0)
    p.add_argument("--head_cls_lr_scale", type=float, default=0.5,
                   help="Multiplier on lr_phase_b for head_cls (policy head).")

    p.add_argument("--lora_rank",      type=int, default=16)
    p.add_argument("--max_images",     type=int, default=4)
    p.add_argument("--grad_accum",     type=int, default=8)
    p.add_argument("--save_steps",     type=int, default=200)
    p.add_argument("--eval_steps",     type=int, default=100)
    p.add_argument("--eval_topk",      type=int, default=5,
                   help="K for top-K accuracy at eval (Pass@K over top-K "
                        "policy anchors). Each sample runs up to K-1 extra "
                        "inference_mode forwards. Set 0/1 to disable.")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--train_vision",   action="store_true")

    p.add_argument("--answer_weight", type=float, default=1.0)
    p.add_argument("--coord_weight",  type=float, default=1.0)
    p.add_argument("--no_coord",      action="store_true")
    p.add_argument("--relative",      action="store_true")
    p.add_argument("--coord_upscale", type=int,   default=4)
    p.add_argument("--coord_scale",   type=float, default=100.0)

    p.add_argument("--rot_nhead",           type=int, default=4)
    p.add_argument("--rot_dim_feedforward", type=int, default=2048)
    p.add_argument("--rot_num_layers",      type=int, default=2)

    # RL-specific args
    p.add_argument("--action_space",   choices=["discrete", "hybrid"], default="hybrid",
                   help="'discrete' = head_cls only; 'hybrid' = head_cls + head_res.")
    p.add_argument("--residual_clamp", type=float, default=math.pi / 6,
                   help="Max axis-angle magnitude (rad) per DoF in head_res. "
                        "Default π/6 ≈ 0.5236 rad ≈ 30°.")
    p.add_argument("--k_max_grad",     type=int,   default=4,
                   help="Max number of correct anchors to run grad forward in "
                        "RWR (top-K by lowest lm_loss). Bounds backward cost.")
    p.add_argument("--rwr_tau",        type=float, default=1.0,
                   help="Temperature for RWR softmax weights over -lm_loss.")
    p.add_argument("--entropy_beta",   type=float, default=0.01,
                   help="Coefficient for entropy bonus (-β·H).")
    p.add_argument("--kl_lambda",      type=float, default=0.001,
                   help="Coefficient for KL(π || Uniform).")
    p.add_argument("--rl_weight",      type=float, default=1.0,
                   help="Scalar on (L_rl + L_ent + L_kl).")
    p.add_argument("--ppo_clip_eps",   type=float, default=0.2,
                   help="PPO clip range ε for the GRPO surrogate "
                        "min(ρ·A, clip(ρ,1-ε,1+ε)·A).")
    p.add_argument("--ppo_inner_epochs", type=int, default=3,
                   help="K = number of PPO inner epochs per rollout (discrete). "
                        "Each epoch re-forwards rotation_enc, computes clipped "
                        "surrogate, and steps policy_opt once. Base LR for "
                        "head_cls/rot_bb is divided by K so effective per-"
                        "rollout update magnitude stays comparable.")
    p.add_argument("--sft_weight",     type=float, default=1.0,
                   help="Scalar on the differentiable-SFT (RWR / argmax coord) term.")
    p.add_argument("--w_lm",           type=float, default=1.0,
                   help="Weight on lm_loss inside each per-anchor SFT step.")
    p.add_argument("--w_coord",        type=float, default=1.0,
                   help="Weight on coord_loss inside each per-anchor SFT step.")

    p.add_argument("--wandb_project",  default="")
    p.add_argument("--wandb_entity",   default="")
    p.add_argument("--wandb_run_name", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
