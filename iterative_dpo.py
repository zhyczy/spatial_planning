"""
Iterative DPO: 5-round snowball training for the spatial Planning Model.

Architecture:
  - Planning Model : Qwen3-VL-4B-Instruct  (trainable, LoRA)
  - Executor       : Flux2Klein-4B          (frozen)
  - Critic / Judge : Qwen3-VL-8B            (frozen)

Each iteration executes a closed loop:
  Step A  – On-policy rollout: sample 8 instruction sets per question (T=0.9)
  Step B  – Execution + Labeling: generate images → MLLM scoring → build pairs
  Step C  – DPO training with accumulated data from all prior iterations

Data is split into N shards. Each iteration processes one shard and accumulates
preference pairs across iterations to prevent catastrophic forgetting.

Usage:
  # Quick pilot on 1700 subset (5 iters × ~340 samples)
  python iterative_dpo.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --critic_model_path  checkpoints/Qwen3-VL-8B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --max_samples 1700 \
    --num_iterations 5

  # Full scale (175K, after pilot succeeds)
  python iterative_dpo.py \
    --dataset mmsibench \
    --data_path datasets/training/spatial_planning_175k.json \
    --image_root datasets/training \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --critic_model_path  checkpoints/Qwen3-VL-8B-Instruct \
    --max_samples -1 \
    --num_iterations 5

  # Resume from iteration 3
  python iterative_dpo.py --resume_from results/iterative_dpo/mmsibench_xxx --resume_iter 3
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
    build_preference_pairs,
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


def load_json_pairs(path: Path) -> list:
    """Load preference pairs from a JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_pairs(pairs: list, path: Path):
    """Save preference pairs to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


def accumulate_pairs(iter_dir: Path, current_iter: int) -> list:
    """Load and merge all dpo_pairs_iter_*.json from iterations 0..current_iter.

    This implements data accumulation: iteration N trains on pairs from
    iterations 0, 1, ..., N to prevent catastrophic forgetting.
    """
    all_pairs = []
    for i in range(current_iter + 1):
        pairs_file = iter_dir / f"iter_{i}" / "dpo_pairs.json"
        if pairs_file.exists():
            pairs = load_json_pairs(pairs_file)
            logger.info(f"  Loaded {len(pairs)} pairs from iteration {i}")
            all_pairs.extend(pairs)
        else:
            logger.warning(f"  Missing pairs file for iteration {i}: {pairs_file}")
    return all_pairs


def run_dpo_training(
    model_id: str,
    data_path: str,
    output_dir: str,
    deepspeed_config: str,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    num_gpus: int = -1,
    extra_args: Optional[List[str]] = None,
):
    """Launch DPO training via deepspeed subprocess.

    We invoke train_planner.py as a subprocess rather than importing it
    because DeepSpeed's distributed launcher needs to own the process.
    """
    train_script = str(SCRIPT_DIR / "train_planner.py")

    # Determine number of GPUs for training
    n_available = torch.cuda.device_count()
    if num_gpus <= 0:
        num_gpus = n_available
    num_gpus = min(num_gpus, n_available)

    # Read dataset size and cap GPUs so each rank gets >= 1 sample
    try:
        with open(data_path, "r") as f:
            n_samples = len(json.load(f))
        # Each rank needs at least 1 sample
        num_gpus = min(num_gpus, max(1, n_samples))
        logger.info(f"Training with {n_samples} samples on {num_gpus} GPU(s)")
    except Exception:
        pass

    cmd = [
        "deepspeed", "--num_gpus", str(num_gpus), train_script,
        "--dpo_loss", "sigmoid",
        "--precompute_ref_log_probs", "False",
        "--beta", str(beta),
        "--use_liger_loss", "True",
        "--deepspeed", deepspeed_config,
        "--model_id", model_id,
        "--data_path", data_path,
        "--image_folder", "",
        "--remove_unused_columns", "False",
        "--lora_enable", "True",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", "0.05",
        "--freeze_vision_tower", "True",
        "--freeze_llm", "True",
        "--freeze_merger", "True",
        "--bf16", "True",
        "--fp16", "False",
        "--disable_flash_attn2", "False",
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
        "--report_to", "tensorboard",
        "--lazy_preprocess", "True",
        "--save_strategy", "steps",
        "--save_steps", "200",
        "--save_total_limit", "3",
        "--dataloader_num_workers", "4",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Training command:\n  {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"DPO training failed with return code {result.returncode}")

    logger.info(f"DPO training complete → {output_dir}")


def merge_lora_weights(base_model_id: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter weights back into base model for next iteration's
    on-policy rollout."""
    merge_script = str(
        SCRIPT_DIR.parent / "Qwen-VL-Series-Finetune" / "src" / "merge_lora_weights.py"
    )

    if Path(merge_script).exists():
        cmd = [
            sys.executable, merge_script,
            "--model_id", base_model_id,
            "--adapter_path", adapter_path,
            "--output_path", output_path,
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
#  Main iterative loop
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative DPO training for spatial Planning Model."
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="mmsibench",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples (-1=all). Use 1700 for pilot.")

    # ── Models ────────────────────────────────────────────────────────────
    parser.add_argument("--planner_model_path", type=str, required=True,
                        help="Base Planning Model (Qwen3-VL-4B-Instruct).")
    parser.add_argument("--critic_model_path", type=str, default=None,
                        help="Frozen Critic MLLM (Qwen3-VL-8B). Defaults to planner.")
    parser.add_argument("--flux_ckpt", type=str, default="checkpoints/flux2-klein-4B",
                        help="Frozen Executor (Flux2Klein-4B).")

    # ── Iteration control ─────────────────────────────────────────────────
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="Number of DPO iterations (shards).")
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="On-policy rollouts per question per iteration.")
    parser.add_argument("--scoring_method", type=str, default="confidence",
                        choices=["confidence", "explicit", "both"])
    parser.add_argument("--min_score_gap", type=float, default=0.3,
                        help="Min score gap for pair selection (lower than single-shot "
                             "since iterative refinement produces tighter distributions).")

    # ── Training hyperparams ──────────────────────────────────────────────
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO KL penalty coefficient.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Epochs per iteration (1 recommended for iterative).")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="DeepSpeed config. Auto-resolved if None.")

    # ── Image generation ──────────────────────────────────────────────────
    parser.add_argument("--num_inference_steps", type=int, default=28)

    # ── Multi-GPU ─────────────────────────────────────────────────────────
    parser.add_argument("--num_gpus", type=int, default=-1)

    # ── Output & resume ───────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="results/iterative_dpo")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from existing run directory.")
    parser.add_argument("--resume_iter", type=int, default=0,
                        help="Resume from this iteration index (0-based).")

    # ── Seed ──────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


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
    log_file = run_dir / "iterative_dpo.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ITER-DPO] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )

    if args.critic_model_path is None:
        args.critic_model_path = args.planner_model_path

    # ── Resolve DeepSpeed config ──────────────────────────────────────────
    if args.deepspeed_config is None:
        ds_config = (
            SCRIPT_DIR.parent
            / "Qwen-VL-Series-Finetune" / "scripts" / "zero2.json"
        )
        if not ds_config.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {ds_config}")
        args.deepspeed_config = str(ds_config)

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
    logger.info("  ITERATIVE DPO — SNOWBALL TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Planning Model : {args.planner_model_path}")
    logger.info(f"  Critic Model   : {args.critic_model_path}")
    logger.info(f"  Executor       : {args.flux_ckpt}")
    logger.info(f"  Total samples  : {len(all_samples)}")
    logger.info(f"  Iterations     : {args.num_iterations}")
    logger.info(f"  Shard sizes    : {shard_sizes}")
    logger.info(f"  Rollouts/q     : {args.num_rollouts}")
    logger.info(f"  Scoring        : {args.scoring_method}")
    logger.info(f"  LoRA rank      : {args.lora_rank}")
    logger.info(f"  Learning rate  : {args.learning_rate}")
    logger.info(f"  Beta (DPO)     : {args.beta}")
    logger.info(f"  Output         : {run_dir}")
    logger.info("=" * 70)

    # ── Base model path (frozen, always used as DPO reference) ────────────
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
        logger.info(f"║  ITERATION {iteration + 1} / {args.num_iterations}"
                     f"  ({len(shards[iteration])} samples)"
                     + " " * (68 - 30 - len(str(iteration+1)) - len(str(args.num_iterations)) - len(str(len(shards[iteration])))) + "║")
        logger.info(f"║  Current model: {Path(current_model_path).name}"
                     + " " * max(0, 68 - 18 - len(Path(current_model_path).name)) + "║")
        logger.info("╚" + "═" * 68 + "╝")

        shard_samples = shards[iteration]

        # ── Step A: On-policy Rollout ─────────────────────────────────────
        rollout_path = iter_dir / "rollouts.jsonl"
        if rollout_path.exists():
            logger.info(f"[Iter {iteration}] Step A: Rollouts already exist, loading...")
            rollout_records = []
            with open(rollout_path) as f:
                for line in f:
                    if line.strip():
                        rollout_records.append(json.loads(line))
        else:
            logger.info(f"[Iter {iteration}] Step A: On-policy rollout with {current_model_path}")
            rollout_records, _ = run_rollouts(
                shard_samples,
                current_model_path,
                args.num_rollouts,
                iter_dir,
                args.num_gpus,
            )
            # Rename merged to expected path
            src_rollout = iter_dir / "rollouts.jsonl"
            if not src_rollout.exists():
                logger.error(f"Rollout output not found at {src_rollout}")
                raise FileNotFoundError(src_rollout)

        logger.info(f"[Iter {iteration}] Step A done: {len(rollout_records)} samples rolled out.")

        # ── Step B-1: Execution ───────────────────────────────────────────
        gen_dir = iter_dir / "generated_images"
        if gen_dir.exists() and any(gen_dir.iterdir()):
            logger.info(f"[Iter {iteration}] Step B-1: Generated images exist, skipping.")
        else:
            logger.info(f"[Iter {iteration}] Step B-1: Executing instructions with Flux2Klein")
            gen_dir = run_execution(
                rollout_records,
                str(resolve(args.flux_ckpt)),
                iter_dir,
                args.num_inference_steps,
                args.num_gpus,
            )

        # ── Step B-2: Labeling ────────────────────────────────────────────
        labeled_path = iter_dir / "labeled.jsonl"
        if labeled_path.exists():
            logger.info(f"[Iter {iteration}] Step B-2: Labels exist, loading...")
            labeled_records = []
            with open(labeled_path) as f:
                for line in f:
                    if line.strip():
                        labeled_records.append(json.loads(line))
        else:
            logger.info(f"[Iter {iteration}] Step B-2: Labeling with Critic ({args.critic_model_path})")
            labeled_records, _ = run_labeling(
                rollout_records,
                str(resolve(args.critic_model_path)),
                gen_dir,
                args.scoring_method,
                iter_dir,
                args.num_gpus,
            )

        # ── Step B-3: Build pairs for this iteration ──────────────────────
        iter_pairs_path = iter_dir / "dpo_pairs.json"
        if iter_pairs_path.exists():
            logger.info(f"[Iter {iteration}] Step B-3: Pairs exist, loading...")
            iter_pairs = load_json_pairs(iter_pairs_path)
        else:
            logger.info(f"[Iter {iteration}] Step B-3: Building preference pairs")
            iter_pairs = build_preference_pairs(
                labeled_records, iter_pairs_path,
                min_score_gap=args.min_score_gap,
                iteration=iteration,
            )

        logger.info(f"[Iter {iteration}] This iteration: {len(iter_pairs)} pairs")

        # ── Accumulate data from all iterations ───────────────────────────
        accumulated_pairs = accumulate_pairs(run_dir, iteration)
        accumulated_path = iter_dir / "dpo_accumulated.json"
        save_json_pairs(accumulated_pairs, accumulated_path)

        logger.info(f"[Iter {iteration}] Accumulated total: {len(accumulated_pairs)} pairs "
                     f"(from iterations 0..{iteration})")

        if len(accumulated_pairs) == 0:
            logger.warning(f"[Iter {iteration}] No preference pairs available! Skipping training.")
            continue

        # ── Step C: DPO Training ──────────────────────────────────────────
        # Always train from BASE model with accumulated data
        # This ensures:
        #   1. Clean LoRA initialization each time
        #   2. All historical data is seen (no forgetting)
        #   3. Reference model = base model (correct DPO semantics)
        train_output_dir = str(iter_dir / "model_lora")

        if (iter_dir / "model_lora").exists():
            logger.info(f"[Iter {iteration}] Step C: Trained model exists, skipping training.")
        else:
            logger.info(f"[Iter {iteration}] Step C: DPO training with {len(accumulated_pairs)} accumulated pairs")
            run_dpo_training(
                model_id=base_model_path,
                data_path=str(accumulated_path),
                output_dir=train_output_dir,
                deepspeed_config=args.deepspeed_config,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                num_train_epochs=args.num_train_epochs,
                per_device_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                beta=args.beta,
                num_gpus=args.num_gpus,
            )

        # ── Merge LoRA for next iteration's on-policy rollout ─────────────
        merged_model_path = str(iter_dir / "model_merged")
        if Path(merged_model_path).exists():
            logger.info(f"[Iter {iteration}] Merged model exists, skipping merge.")
        else:
            logger.info(f"[Iter {iteration}] Merging LoRA weights for next iteration...")
            merge_lora_weights(base_model_path, train_output_dir, merged_model_path)

        # Update current model for next iteration's on-policy rollout
        current_model_path = merged_model_path

        # ── Iteration summary ─────────────────────────────────────────────
        summary = {
            "iteration": iteration,
            "shard_size": len(shard_samples),
            "rollout_count": len(rollout_records),
            "pairs_this_iter": len(iter_pairs),
            "pairs_accumulated": len(accumulated_pairs),
            "current_model": current_model_path,
        }
        with open(iter_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[Iter {iteration}] ✓ Complete: "
                     f"{len(iter_pairs)} new pairs, "
                     f"{len(accumulated_pairs)} total accumulated")
        logger.info("")

    # ══════════════════════════════════════════════════════════════════════
    #  Final summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("  ITERATIVE DPO COMPLETE")
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

    # Save final status
    final_status = {
        "status": "complete",
        "num_iterations": args.num_iterations,
        "final_model": current_model_path,
        "base_model": base_model_path,
        "total_pairs": sum(
            len(load_json_pairs(run_dir / f"iter_{i}" / "dpo_pairs.json"))
            for i in range(args.num_iterations)
        ),
    }
    with open(run_dir / "final_status.json", "w") as f:
        json.dump(final_status, f, indent=2)


if __name__ == "__main__":
    main()
