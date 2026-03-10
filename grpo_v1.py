"""
Iterative GRPO: Group Relative Policy Optimization for the spatial Planning Model.

Architecture:
  - Planning Model : Qwen3-VL-4B-Instruct  (trainable, LoRA)
  - Executor       : Flux2Klein-4B          (frozen image generator)
  - Critic / Judge : Qwen3-VL-8B            (frozen scorer)

Each iteration executes a closed loop:
  Step A  – On-policy rollout: sample G instruction sets per question (T=0.9)
  Step B  – Execution: generate images from instructions via Flux2Klein (multi-GPU)
  Step C  – Labeling: score generated images with Critic (multi-GPU)
  Step D  – GRPO Dataset: normalize rewards → advantages, flatten to training examples
  Step E  – GRPO Training: advantage-weighted policy gradient via train_planner_grpo.py
  Step F  – LoRA Merge: merge adapter into base model for next iteration's rollout

GRPO vs DPO:
  DPO  — picks best/worst pair per question → preference comparison loss
  GRPO — uses ALL G rollouts with reward-derived advantages → policy gradient loss
         (no information discarded; lower-variance gradient estimate)

Data accumulation (same as iterative DPO):
  Each iteration accumulates training examples from all prior iterations to
  prevent catastrophic forgetting.

Usage:
  # Pilot: SAT 2000 samples, 5 iterations, LoRA
  python grpo.py \\
    --dataset sat \\
    --data_path datasets/evaluation/SAT/train_action_consequence.json \\
    --image_root datasets/evaluation/SAT \\
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \\
    --critic_model_path  checkpoints/Qwen3-VL-8B-Instruct \\
    --flux_ckpt checkpoints/flux2-klein-4B \\
    --max_samples 2000 \\
    --num_iterations 5

  # Full scale (~172K samples)
  python grpo.py \\
    --dataset sat \\
    --data_path datasets/evaluation/SAT/train_action_consequence.json \\
    --image_root datasets/evaluation/SAT \\
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \\
    --critic_model_path  checkpoints/Qwen3-VL-8B-Instruct \\
    --max_samples -1 \\
    --num_iterations 5

  # Resume from iteration 3
  python grpo.py --resume_from results/grpo/sat_xxx --resume_iter 3
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

# ── Add spatial_planning to path ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from generate_dpo_data import (
    run_execution,
    run_labeling,
    run_rollouts,
)
from generate_image_instructions import DATASET_LOADERS, chunk_dataset

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(samples: list, n_shards: int, seed: int = 42) -> List[list]:
    """Shuffle and split *samples* into *n_shards* balanced shards."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    shards = [[] for _ in range(n_shards)]
    for i, idx in enumerate(indices):
        shards[i % n_shards].append(samples[idx])
    return shards


def load_grpo_examples(path: Path) -> list:
    """Load GRPO flattened training examples from a JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_grpo_examples(examples: list, path: Path):
    """Save GRPO training examples to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def accumulate_grpo_data(iter_dir: Path, current_iter: int) -> list:
    """Load and merge all grpo_examples_iter_*.json from iterations 0..current_iter.

    Implements the same snowball accumulation as the DPO variant: iteration N
    trains on examples from iterations 0, 1, ..., N to prevent forgetting.
    """
    all_examples = []
    for i in range(current_iter + 1):
        examples_file = iter_dir / f"iter_{i}" / "grpo_examples.json"
        if examples_file.exists():
            examples = load_grpo_examples(examples_file)
            logger.info(f"  Loaded {len(examples)} GRPO examples from iteration {i}")
            all_examples.extend(examples)
        else:
            logger.warning(f"  Missing examples file for iteration {i}: {examples_file}")
    return all_examples


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO Dataset Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_grpo_dataset(
    labeled_records: list,
    output_path: Path,
    min_reward_std: float = 0.05,
    iteration: int = -1,
) -> list:
    """Convert labeled records to GRPO advantage-weighted training examples.

    Unlike DPO which selects a single (chosen, rejected) pair, GRPO uses ALL
    rollouts in each group.  For each group:
      1. Collect all rollout completions and their critic scores (rewards).
      2. Normalize rewards to advantages: A_i = (r_i - mean_r) / (std_r + ε).
      3. Flatten to individual (prompt, completion, advantage) training examples.

    Groups with near-zero reward variance (all rollouts scored identically) are
    discarded because they produce zero-gradient updates.

    Args:
        labeled_records:  Output of run_labeling() – list of dicts with keys:
                          id, question, image_paths, gt_answer, rollouts, scores
        output_path:      JSON file to save the flattened examples.
        min_reward_std:   Minimum std of rewards in a group to keep it.
        iteration:        Current iteration index (stored in metadata).

    Returns:
        List of flattened GRPO training example dicts, each with keys:
          image, conversations, advantage, group_id
    """
    # Import the system prompt used during rollout (must match training prompt)
    from generate_image_instructions import SYSTEM_PROMPT

    examples = []
    skipped_no_rollouts = 0
    skipped_low_variance = 0

    for rec in labeled_records:
        scores = rec.get("scores", [])
        rollouts = rec.get("rollouts", [])

        if len(scores) < 2 or len(rollouts) < 2:
            skipped_no_rollouts += 1
            continue

        # Build score lookup: rollout_idx → score
        score_by_idx = {s["rollout_idx"]: s["score"] for s in scores}

        # Match rollouts to their scores (skip rollouts with no score or empty output)
        matched = []
        for ri, rollout in enumerate(rollouts):
            raw_output = rollout.get("raw_output", "").strip()
            if not raw_output:
                continue
            ri_key = rollout.get("rollout_idx", ri)
            if ri_key in score_by_idx:
                matched.append({
                    "raw_output": raw_output,
                    "score": score_by_idx[ri_key],
                    "rollout_idx": ri_key,
                })

        if len(matched) < 2:
            skipped_no_rollouts += 1
            continue

        # Compute reward statistics within the group
        reward_values = [m["score"] for m in matched]
        mean_r = sum(reward_values) / len(reward_values)
        variance = sum((r - mean_r) ** 2 for r in reward_values) / len(reward_values)
        std_r = max(variance ** 0.5, 1e-8)

        if std_r < min_reward_std:
            # All rollouts scored similarly → zero gradient signal
            skipped_low_variance += 1
            continue

        # Compute normalized advantages
        advantages = [(r - mean_r) / std_r for r in reward_values]

        # Build image-annotated user prompt (one <image> token per scene image)
        n_images = len(rec["image_paths"])
        image_tokens = "<image>" * n_images
        user_content = f"{image_tokens}\n{rec['question']}"

        group_id = f"{rec['id']}_iter{iteration}"

        # Flatten: one training example per rollout
        for m, adv in zip(matched, advantages):
            example = {
                "image": rec["image_paths"],
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": user_content},
                    {"from": "gpt",   "value": m["raw_output"]},
                ],
                "advantage": adv,
                "group_id": group_id,
                "metadata": {
                    "id": rec["id"],
                    "question": rec["question"],
                    "gt_answer": rec.get("gt_answer", ""),
                    "rollout_idx": m["rollout_idx"],
                    "raw_reward": m["score"],
                    "mean_reward": mean_r,
                    "reward_std": std_r,
                    "iteration": iteration,
                },
            }
            examples.append(example)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    n_groups_kept = len(labeled_records) - skipped_no_rollouts - skipped_low_variance
    logger.info(
        f"Built {len(examples)} GRPO examples from {n_groups_kept} groups "
        f"({skipped_no_rollouts} skipped/no-rollouts, "
        f"{skipped_low_variance} skipped/low-variance)"
    )
    logger.info(f"Saved GRPO examples → {output_path}")
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  GRPO Training
# ══════════════════════════════════════════════════════════════════════════════

def run_grpo_training(
    model_id: str,
    data_path: str,
    output_dir: str,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    beta: float = 0.04,
    num_gpus: int = -1,
    lora_enable: bool = True,
    extra_args: Optional[List[str]] = None,
):
    """Launch GRPO training via train_planner_grpo.py subprocess.

    Uses torchrun for multi-GPU DDP (same pattern as DPO training).
    The GRPO training script applies advantage-weighted NLL loss with optional
    KL penalty from the frozen base (reference) model.

    Args:
        model_id:    Base model path (used as both policy init and reference).
        data_path:   Path to GRPO accumulated examples JSON.
        output_dir:  Directory to save LoRA adapter weights.
        beta:        KL penalty coefficient (0 = pure policy gradient).
        lora_enable: If True, trains LoRA adapter only. If False, full-model.
    """
    train_script = str(SCRIPT_DIR / "train_planner_grpo.py")
    if not Path(train_script).exists():
        raise FileNotFoundError(
            f"train_planner_grpo.py not found at {train_script}. "
            "Please ensure it exists alongside grpo.py."
        )

    # Determine number of GPUs
    n_available = torch.cuda.device_count()
    if num_gpus <= 0:
        num_gpus = n_available
    num_gpus = min(num_gpus, n_available)

    # Cap GPUs to dataset size
    try:
        with open(data_path, "r") as f:
            n_samples = len(json.load(f))
        num_gpus = min(num_gpus, max(1, n_samples))
        logger.info(f"GRPO training with {n_samples} examples on {num_gpus} GPU(s)")
    except Exception:
        pass

    # Build command
    if num_gpus > 1:
        cmd = [
            sys.executable, "-u", "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", str(29600 + os.getpid() % 1000),
            train_script,
        ]
    else:
        cmd = [sys.executable, "-u", train_script]

    # ── LoRA vs Full-model ────────────────────────────────────────────────
    if lora_enable:
        lora_args = [
            "--lora_enable", "True",
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--lora_dropout", "0.05",
            "--freeze_vision_tower", "True",
            "--freeze_llm", "True",
            "--freeze_merger", "True",
        ]
    else:
        lora_args = [
            "--lora_enable", "False",
            "--freeze_vision_tower", "False",
            "--freeze_llm", "False",
            "--freeze_merger", "False",
        ]

    cmd.extend([
        "--model_id", model_id,
        "--data_path", data_path,
        "--image_folder", "",
        "--remove_unused_columns", "False",
        "--beta", str(beta),
    ])
    cmd.extend(lora_args)
    cmd.extend([
        "--bf16", "True",
        "--fp16", "False",
        "--disable_flash_attn2", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--image_min_pixels", str(512 * 28 * 28),
        "--image_max_pixels", str(1280 * 28 * 28),
        "--learning_rate", str(learning_rate),
        "--weight_decay", "0.1",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--gradient_checkpointing", "True",
        "--report_to", "none",
        "--lazy_preprocess", "True",
        "--save_strategy", "steps",
        "--save_steps", "200",
        "--save_total_limit", "3",
        "--dataloader_num_workers", "4",
    ])
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"GRPO training command:\n  {' '.join(cmd)}")

    env = os.environ.copy()
    # Point Triton's autotune cache to a local /tmp directory so that
    # multiple DDP workers don't race on the NFS-backed ~/.triton/autotune,
    # which causes SIGSEGV during Flash Attention 2 kernel compilation.
    triton_cache = f"/tmp/triton_cache_{os.getpid()}"
    os.makedirs(triton_cache, exist_ok=True)
    env.setdefault("TRITON_CACHE_DIR", triton_cache)
    # NCCL/CUDA env vars for Blackwell GPU stability
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"GRPO training failed with return code {result.returncode}")

    logger.info(f"GRPO training complete → {output_dir}")


def merge_lora_weights(base_model_id: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter weights back into base model for the next iteration's
    on-policy rollout.  Mirrors the same function in iterative_dpo.py."""
    merge_script = str(
        SCRIPT_DIR.parent / "Qwen-VL-Series-Finetune" / "src" / "merge_lora_weights.py"
    )

    if Path(merge_script).exists():
        cmd = [
            sys.executable, merge_script,
            "--model-path", adapter_path,
            "--model-base", base_model_id,
            "--save-model-path", output_path,
        ]
        logger.info(f"Merging LoRA: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"LoRA merge failed with return code {result.returncode}")
    else:
        # Fallback: merge in-process
        logger.info("merge_lora_weights.py not found, merging in-process...")
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id, dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        processor = AutoProcessor.from_pretrained(base_model_id)
        processor.save_pretrained(output_path)

    logger.info(f"Merged model saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative GRPO training for spatial Planning Model."
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="mmsibench",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples (-1=all). Use 2000 for pilot.")

    # ── Models ────────────────────────────────────────────────────────────
    parser.add_argument("--planner_model_path", type=str, required=True,
                        help="Base Planning Model (Qwen3-VL-4B-Instruct).")
    parser.add_argument("--critic_model_path", type=str, default=None,
                        help="Frozen Critic MLLM (Qwen3-VL-8B). Defaults to planner.")
    parser.add_argument("--flux_ckpt", type=str, default="checkpoints/flux2-klein-4B",
                        help="Frozen Executor (Flux2Klein-4B).")

    # ── Iteration control ─────────────────────────────────────────────────
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of GRPO iterations (shards).")
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="On-policy rollouts per question per iteration (group size G).")
    parser.add_argument("--scoring_method", type=str, default="gt_similarity",
                        choices=["gt_similarity", "explicit", "both"])
    parser.add_argument("--min_reward_std", type=float, default=0.05,
                        help="Discard groups whose reward std is below this threshold "
                             "(all-same scores → zero gradient). Set 0.0 to keep all.")

    # ── GRPO hyperparams ──────────────────────────────────────────────────
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient for GRPO (0 = pure policy gradient). "
                             "Analogous to beta in DPO.")

    # ── Training hyperparams ──────────────────────────────────────────────
    parser.add_argument("--lora_enable", type=lambda x: x.lower() != "false", default=True,
                        help="LoRA mode (default True). Set False for full-model GRPO.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Epochs per iteration (1 recommended for iterative).")
    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="Per-device batch size for GRPO training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # ── Image generation ──────────────────────────────────────────────────
    parser.add_argument("--num_inference_steps", type=int, default=28)

    # ── Multi-GPU ─────────────────────────────────────────────────────────
    parser.add_argument("--num_gpus", type=int, default=-1)

    # ── Output & resume ───────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="results/grpo")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from existing run directory.")
    parser.add_argument("--resume_iter", type=int, default=0,
                        help="Resume from this iteration index (0-based).")

    # ── Seed ──────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main iterative loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    def resolve(p):
        q = Path(p)
        return q if q.is_absolute() else SCRIPT_DIR / q

    # ── Output directory ──────────────────────────────────────────────────
    if args.resume_from:
        run_dir = resolve(args.resume_from)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = resolve(args.output_dir) / f"{args.dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log_file = run_dir / "grpo.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GRPO] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )

    if args.critic_model_path is None:
        args.critic_model_path = args.planner_model_path

    # ── Load dataset ──────────────────────────────────────────────────────
    data_path = resolve(args.data_path)
    image_root = resolve(args.image_root)
    loader = DATASET_LOADERS[args.dataset]
    all_samples = loader(str(data_path), str(image_root), args.max_samples, 0)
    logger.info(f"Loaded {len(all_samples)} samples from {args.dataset}.")

    # ── Split into shards ─────────────────────────────────────────────────
    shards = split_dataset(all_samples, args.num_iterations, seed=args.seed)
    shard_sizes = [len(s) for s in shards]
    logger.info(f"Split into {args.num_iterations} shards: {shard_sizes}")

    # Save shards for reproducibility
    shards_file = run_dir / "shards.json"
    if not shards_file.exists():
        shard_ids = [[s["id"] for s in shard] for shard in shards]
        with open(shards_file, "w") as f:
            json.dump(shard_ids, f)

    # Save run config
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Print overview ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  ITERATIVE GRPO — SNOWBALL TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Planning Model : {args.planner_model_path}")
    logger.info(f"  Critic Model   : {args.critic_model_path}")
    logger.info(f"  Executor       : {args.flux_ckpt}")
    logger.info(f"  Total samples  : {len(all_samples)}")
    logger.info(f"  Iterations     : {args.num_iterations}")
    logger.info(f"  Shard sizes    : {shard_sizes}")
    logger.info(f"  Rollouts/q (G) : {args.num_rollouts}")
    logger.info(f"  Scoring        : {args.scoring_method}")
    logger.info(f"  KL beta        : {args.beta}")
    logger.info(f"  LoRA rank      : {args.lora_rank}")
    logger.info(f"  Learning rate  : {args.learning_rate}")
    logger.info(f"  Output         : {run_dir}")
    logger.info("=" * 70)

    # ── Base model path (frozen reference for GRPO KL penalty) ────────────
    base_model_path = str(resolve(args.planner_model_path))

    # Current model = model used for on-policy rollouts
    # Iteration 0: base model; Iteration N: merged model from iteration N-1
    current_model_path = base_model_path

    # ══════════════════════════════════════════════════════════════════════
    #  ITERATION LOOP
    # ══════════════════════════════════════════════════════════════════════
    for iteration in range(args.resume_iter, args.num_iterations):
        iter_dir = run_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info(
            f"║  ITERATION {iteration + 1} / {args.num_iterations}"
            f"  ({len(shards[iteration])} samples)"
            + " " * max(0, 68 - 30 - len(str(iteration + 1)) - len(str(args.num_iterations)) - len(str(len(shards[iteration]))))
            + "║"
        )
        logger.info(
            f"║  Current model: {Path(current_model_path).name}"
            + " " * max(0, 68 - 18 - len(Path(current_model_path).name))
            + "║"
        )
        logger.info("╚" + "═" * 68 + "╝")

        shard_samples = shards[iteration]

        # ── Step A: On-policy Rollout ─────────────────────────────────────
        rollout_path = iter_dir / "rollouts.jsonl"
        if rollout_path.exists():
            logger.info(f"[Iter {iteration}] Step A: Rollouts exist, loading...")
            rollout_records = []
            with open(rollout_path) as f:
                for line in f:
                    if line.strip():
                        rollout_records.append(json.loads(line))
        else:
            logger.info(
                f"[Iter {iteration}] Step A: On-policy rollout "
                f"(G={args.num_rollouts}) with {current_model_path}"
            )
            rollout_records, _ = run_rollouts(
                shard_samples,
                current_model_path,
                args.num_rollouts,
                iter_dir,
                args.num_gpus,
            )

        logger.info(
            f"[Iter {iteration}] Step A done: {len(rollout_records)} samples rolled out."
        )

        # ── Step B: Image Generation (Execution) ──────────────────────────
        gen_dir = iter_dir / "generated_images"
        if gen_dir.exists() and any(gen_dir.iterdir()):
            logger.info(f"[Iter {iteration}] Step B: Generated images exist, skipping.")
        else:
            logger.info(
                f"[Iter {iteration}] Step B: Executing instructions with Flux2Klein"
            )
            gen_dir = run_execution(
                rollout_records,
                str(resolve(args.flux_ckpt)),
                iter_dir,
                args.num_inference_steps,
                args.num_gpus,
            )

        # ── Step C: Labeling (Critic Scoring) ────────────────────────────
        labeled_path = iter_dir / "labeled.jsonl"
        if labeled_path.exists():
            logger.info(f"[Iter {iteration}] Step C: Labels exist, loading...")
            labeled_records = []
            with open(labeled_path) as f:
                for line in f:
                    if line.strip():
                        labeled_records.append(json.loads(line))
        else:
            logger.info(
                f"[Iter {iteration}] Step C: Labeling with Critic ({args.critic_model_path})"
            )
            labeled_records, _ = run_labeling(
                rollout_records,
                str(resolve(args.critic_model_path)),
                gen_dir,
                args.scoring_method,
                iter_dir,
                args.num_gpus,
            )

        # ── Step D: Build GRPO Dataset ────────────────────────────────────
        # All G rollouts contribute (not just best/worst pair)
        iter_examples_path = iter_dir / "grpo_examples.json"
        if iter_examples_path.exists():
            logger.info(f"[Iter {iteration}] Step D: GRPO examples exist, loading...")
            iter_examples = load_grpo_examples(iter_examples_path)
        else:
            logger.info(
                f"[Iter {iteration}] Step D: Building GRPO advantage dataset"
            )
            iter_examples = build_grpo_dataset(
                labeled_records,
                iter_examples_path,
                min_reward_std=args.min_reward_std,
                iteration=iteration,
            )

        logger.info(
            f"[Iter {iteration}] This iteration: {len(iter_examples)} GRPO examples "
            f"(from {len(labeled_records)} samples × up to {args.num_rollouts} rollouts)"
        )

        # ── Strict on-policy: use ONLY current iteration's examples ────────
        # The behavior policy that generated iter_examples IS current_model_path,
        # so training data and learning model are always aligned (on-policy).
        if len(iter_examples) == 0:
            logger.warning(
                f"[Iter {iteration}] No GRPO examples available! Skipping training."
            )
            continue

        # ── Step E: GRPO Training ─────────────────────────────────────────
        # On-policy: train from current_model_path (= behavior policy that generated
        # this iter's rollouts). The reference model loaded inside train_planner_grpo.py
        # is also current_model_path, so pi_ref == pi_behavior. LoRA is applied on top
        # of current_model_path and merged back at Step F.
        train_output_dir = str(iter_dir / "model_lora")

        _lora_dir = iter_dir / "model_lora"
        _lora_done = _lora_dir.exists() and any(_lora_dir.iterdir())
        if _lora_done:
            logger.info(
                f"[Iter {iteration}] Step E: Trained adapter exists, skipping training."
            )
        else:
            logger.info(
                f"[Iter {iteration}] Step E: GRPO training (on-policy) "
                f"with {len(iter_examples)} examples from current iter"
            )
            run_grpo_training(
                model_id=current_model_path,
                data_path=str(iter_examples_path),
                output_dir=train_output_dir,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                num_train_epochs=args.num_train_epochs,
                per_device_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                beta=args.beta,
                num_gpus=args.num_gpus,
                lora_enable=args.lora_enable,
            )

        # ── Step F: Merge LoRA for next iteration's rollout ───────────────
        merged_model_path = str(iter_dir / "model_merged")
        if Path(merged_model_path).exists():
            logger.info(f"[Iter {iteration}] Merged model exists, skipping merge.")
        else:
            logger.info(
                f"[Iter {iteration}] Step F: Merging LoRA weights for next iteration..."
            )
            merge_lora_weights(current_model_path, train_output_dir, merged_model_path)

        # Update current model for next iteration's on-policy rollout
        current_model_path = merged_model_path

        # ── Iteration summary ─────────────────────────────────────────────
        summary = {
            "iteration": iteration,
            "shard_size": len(shard_samples),
            "rollout_count": len(rollout_records),
            "examples_this_iter": len(iter_examples),
            "current_model": current_model_path,
        }
        with open(iter_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"[Iter {iteration}] ✓ Complete: "
            f"{len(iter_examples)} examples trained on-policy"
        )
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #  Final summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("  ITERATIVE GRPO COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Iterations completed : {args.num_iterations}")
    logger.info(f"  Final model          : {current_model_path}")
    logger.info(f"  Run directory        : {run_dir}")
    logger.info("")
    logger.info("  Next step: evaluate the fine-tuned planner:")
    logger.info(f"    python generate_image_instructions.py \\")
    logger.info(f"      --model_path {current_model_path} \\")
    logger.info(f"      --dataset mmsibench ...")
    logger.info("=" * 70)

    # Compute total examples generated
    total_examples = 0
    for i in range(args.num_iterations):
        ep = run_dir / f"iter_{i}" / "grpo_examples.json"
        if ep.exists():
            total_examples += len(load_grpo_examples(ep))

    # Save final status
    final_status = {
        "status": "complete",
        "num_iterations": args.num_iterations,
        "final_model": current_model_path,
        "base_model": base_model_path,
        "total_examples": total_examples,
    }
    with open(run_dir / "final_status.json", "w") as f:
        json.dump(final_status, f, indent=2)


if __name__ == "__main__":
    main()
