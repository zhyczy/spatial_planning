"""
Step 0: Automatically generate DPO preference pairs for the Planning Model.

Pipeline overview:
  1. Rollout    – Sample K diverse instructions per (images, question) from the
                  current Planning Model (using high temperature / nucleus sampling).
  2. Execution  – Feed each instruction to the frozen image generator (Flux2Klein)
                  to produce a candidate image.
  3. Labeling   – Score each candidate with an MLLM judge:
                  (A) Explicit: "Which generated image helps most?" 
                  (B) Implicit: measure answer-logit confidence boost.
  4. Selection  – Pick the best-scoring candidate as Chosen, worst as Rejected.

Output:
  A JSON file where each entry has:
    { "image": [...], "prompt": "...", "chosen": "...", "rejected": "..." }
  Ready for DPO training via train_planner.py.

Usage:
  # End-to-end (rollout → generate → label → build pairs)
  python generate_dpo_data.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --judge_model_path  checkpoints/Qwen3-VL-4B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B

  # Resume from existing rollout results
  python generate_dpo_data.py --resume_from results/dpo_data/rollout_xxx/

  # Quick test with 10 samples
  python generate_dpo_data.py --dataset mmsibench ... --max_samples 10
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Add parent to path so we can reuse dataset loaders from generate_image_instructions.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_image_instructions import (
    DATASET_LOADERS,
    SYSTEM_PROMPT,
    build_message,
    chunk_dataset,
    parse_instructions,
)


# ── Constants ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

NUM_ROLLOUTS_DEFAULT = 8       # How many instr sets to sample per question
ROLLOUT_TEMPERATURE  = 0.9     # High temp for diversity
ROLLOUT_TOP_P        = 0.95

# Judge system prompts
JUDGE_EXPLICIT_PROMPT = """\
You are a spatial reasoning judge. You will be given:
1. One or more original scene images.
2. A spatial reasoning question about the scene.
3. Several candidate generated images (labeled Candidate 1, 2, …).

Your task: rank the candidate images by how much they help answer the spatial
reasoning question. Consider:
- Does the candidate reveal NEW spatial information not visible in the originals?
- Does the candidate show the correct spatial relationships (no hallucination)?
- Is the candidate high quality and physically plausible?

Output your ranking as a JSON object:
{"ranking": [<best_idx>, ..., <worst_idx>], "scores": {<idx>: <score_1_to_10>, ...}, "reason": "..."}

IMPORTANT: indices are 0-based (matching Candidate 0, 1, …). Every candidate
must appear exactly once in the ranking."""

JUDGE_CONFIDENCE_INSTRUCTION = """\
Look at the images and answer the following spatial reasoning question.
Think step by step and give your final answer.

Question: {question}

Respond with ONLY your answer."""


# ── Model loading helpers ─────────────────────────────────────────────────────

def load_model_for_inference(model_path: str, device: str):
    """Load a Qwen VL model for inference (planner or judge)."""
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, TypeError):
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def generate_text(model, processor, messages, max_new_tokens=512,
                  temperature=0.7, top_p=0.9, do_sample=True):
    """Run inference and return generated text."""
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def get_answer_logits(model, processor, messages, max_new_tokens=64):
    """Run inference and return (answer_text, mean_logprob) for confidence
    scoring.  Higher mean_logprob = model is more confident."""
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

    generated_ids = outputs.sequences
    scores = outputs.scores  # tuple of (vocab_size,) per step

    # Compute mean log-prob of generated tokens
    trimmed = generated_ids[0][len(inputs["input_ids"][0]):]
    total_logprob = 0.0
    n_tokens = 0
    for step_idx, token_id in enumerate(trimmed):
        if step_idx >= len(scores):
            break
        logprobs = torch.log_softmax(scores[step_idx][0], dim=-1)
        total_logprob += logprobs[token_id].item()
        n_tokens += 1

    mean_logprob = total_logprob / max(n_tokens, 1)
    answer_text = processor.batch_decode(
        [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return answer_text, mean_logprob


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Rollout — sample K instruction sets from the planner
# ══════════════════════════════════════════════════════════════════════════════

def rollout_worker(
    gpu_id: str,
    samples: list,
    model_path: str,
    num_rollouts: int,
    output_path: str,
    log_file: Optional[str] = None,
):
    """Sample num_rollouts instruction sets per sample on one GPU."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    model, processor = load_model_for_inference(model_path, device)
    log.info(f"Planner loaded on {device}. Processing {len(samples)} samples × {num_rollouts} rollouts.")

    results = []
    with open(output_path, "w") as f:
        for si, sample in enumerate(samples):
            rollouts = []
            for ri in range(num_rollouts):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    build_message(sample["question"], sample["image_paths"]),
                ]
                try:
                    raw_output = generate_text(
                        model, processor, messages,
                        max_new_tokens=512,
                        temperature=ROLLOUT_TEMPERATURE,
                        top_p=ROLLOUT_TOP_P,
                        do_sample=True,
                    )
                    instructions = parse_instructions(raw_output)
                except Exception as e:
                    log.error(f"Sample {si} rollout {ri}: {e}")
                    raw_output = ""
                    instructions = []

                rollouts.append({
                    "raw_output": raw_output,
                    "instructions": instructions,
                })

            record = {
                "id": sample["id"],
                "question": sample["question"],
                "image_paths": sample["image_paths"],
                "gt_answer": sample["gt_answer"],
                "meta": sample.get("meta", {}),
                "rollouts": rollouts,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            log.info(
                f"[{si+1}/{len(samples)}] id={sample['id']}  "
                f"rollouts: {[len(r['instructions']) for r in rollouts]}"
            )

    log.info(f"Rollout worker done. {len(results)} samples → {output_path}")


def run_rollouts(samples, model_path, num_rollouts, output_dir, num_gpus):
    """Shard samples across GPUs and run rollouts in parallel."""
    n_available = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]
    if num_gpus > 0:
        all_gpu_ids = all_gpu_ids[:num_gpus]

    log_file = str(output_dir / "rollout.log")
    chunks = chunk_dataset(samples, len(all_gpu_ids))
    processes = []

    shard_paths = []
    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = str(output_dir / f"rollout_shard_{idx}.jsonl")
        shard_paths.append(shard_path)
        p = mp.Process(
            target=rollout_worker,
            args=(gpu_id, chunk, model_path, num_rollouts, shard_path, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(f"Rollout worker {idx} → GPU {gpu_id} ({len(chunk)} samples)")

    for p in processes:
        p.join()

    # Merge shards
    all_records = []
    for path in shard_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))

    merged_path = output_dir / "rollouts.jsonl"
    with open(merged_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Rollouts merged: {len(all_records)} samples → {merged_path}")
    return all_records, merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Execution — generate images via Flux2Klein
# ══════════════════════════════════════════════════════════════════════════════

def execution_worker(
    gpu_id: str,
    tasks: list,
    flux_ckpt: str,
    output_root: str,
    num_inference_steps: int,
    log_file: Optional[str] = None,
):
    """Generate images for a list of (sample_id, rollout_idx, instruction_idx,
    instruction_text, image_paths) tasks."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    from diffusers import Flux2KleinPipeline
    log.info(f"Loading Flux2KleinPipeline on {device} ...")
    pipe = Flux2KleinPipeline.from_pretrained(flux_ckpt, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    log.info("Generator loaded.")

    out_root = Path(output_root)

    for ti, task in enumerate(tasks):
        sample_id = task["sample_id"]
        rollout_idx = task["rollout_idx"]
        instr_idx = task["instr_idx"]
        instruction = task["instruction"]
        image_paths = task["image_paths"]

        out_dir = out_root / str(sample_id) / f"rollout_{rollout_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"img_{instr_idx}.png"

        if out_path.exists():
            continue

        # Load source images
        src_images = []
        for p in image_paths:
            if os.path.isfile(p):
                try:
                    src_images.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass

        if not src_images:
            log.warning(f"No source images for sample {sample_id}, skip.")
            continue

        try:
            result = pipe(image=src_images, prompt=instruction,
                          num_inference_steps=num_inference_steps)
            result.images[0].save(out_path)
            log.info(f"[{ti+1}/{len(tasks)}] {sample_id}/rollout_{rollout_idx}/img_{instr_idx} saved.")
        except Exception as e:
            log.error(f"Generation failed for {sample_id}/r{rollout_idx}/i{instr_idx}: {e}")


def run_execution(rollout_records, flux_ckpt, output_dir, num_inference_steps, num_gpus):
    """Build task list from rollout records and generate images in parallel."""
    gen_dir = output_dir / "generated_images"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Build flat task list
    tasks = []
    for rec in rollout_records:
        for ri, rollout in enumerate(rec["rollouts"]):
            for ii, instruction in enumerate(rollout["instructions"]):
                tasks.append({
                    "sample_id": rec["id"],
                    "rollout_idx": ri,
                    "instr_idx": ii,
                    "instruction": instruction,
                    "image_paths": rec["image_paths"],
                })

    if not tasks:
        logger.warning("No tasks to execute (all rollouts produced empty instructions).")
        return gen_dir

    logger.info(f"Execution: {len(tasks)} image generation tasks.")

    # Shard across GPUs
    n_available = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]
    if num_gpus > 0:
        all_gpu_ids = all_gpu_ids[:num_gpus]

    log_file = str(output_dir / "execution.log")
    chunks = chunk_dataset(tasks, len(all_gpu_ids))
    processes = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        p = mp.Process(
            target=execution_worker,
            args=(gpu_id, chunk, str(flux_ckpt), str(gen_dir),
                  num_inference_steps, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(f"Execution worker {idx} → GPU {gpu_id} ({len(chunk)} tasks)")

    for p in processes:
        p.join()

    logger.info(f"Execution done. Images saved under {gen_dir}")
    return gen_dir


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Labeling — MLLM scores each rollout's generated images
# ══════════════════════════════════════════════════════════════════════════════

def label_worker(
    gpu_id: str,
    samples: list,
    judge_model_path: str,
    gen_dir: str,
    scoring_method: str,
    output_path: str,
    log_file: Optional[str] = None,
):
    """Score rollouts for each sample using the MLLM judge."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    model, processor = load_model_for_inference(judge_model_path, device)
    log.info(f"Judge loaded on {device}. Scoring {len(samples)} samples ({scoring_method}).")

    gen_root = Path(gen_dir)
    results = []

    with open(output_path, "w") as f:
        for si, sample in enumerate(samples):
            rollout_scores = []

            for ri, rollout in enumerate(sample["rollouts"]):
                instructions = rollout["instructions"]

                if not instructions:
                    # Empty instructions — this is a valid "no generation needed" response
                    rollout_scores.append({
                        "rollout_idx": ri,
                        "score": 0.0,  # Neutral score
                        "method": scoring_method,
                        "detail": "empty_instructions",
                    })
                    continue

                # Collect generated images for this rollout
                rollout_dir = gen_root / str(sample["id"]) / f"rollout_{ri}"
                gen_images = []
                if rollout_dir.is_dir():
                    gen_images = sorted(
                        str(p) for p in rollout_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                    )

                if not gen_images:
                    # Generator failed entirely — negative signal
                    rollout_scores.append({
                        "rollout_idx": ri,
                        "score": -1.0,
                        "method": scoring_method,
                        "detail": "no_generated_images",
                    })
                    continue

                # ── Score this rollout ─────────────────────────────────────
                if scoring_method in ("confidence", "both"):
                    # Implicit scoring: measure answer confidence with generated images
                    aug_images = sample["image_paths"] + gen_images
                    conf_msg = [
                        {"role": "user", "content": (
                            [{"type": "image", "image": p} for p in aug_images if os.path.isfile(p)]
                            + [{"type": "text", "text": JUDGE_CONFIDENCE_INSTRUCTION.format(
                                question=sample["question"]
                            )}]
                        )},
                    ]

                    try:
                        _, mean_logprob = get_answer_logits(
                            model, processor, conf_msg, max_new_tokens=64
                        )
                    except Exception as e:
                        log.error(f"Confidence scoring failed for sample {sample['id']} rollout {ri}: {e}")
                        mean_logprob = -10.0

                    score = mean_logprob  # Higher = more confident

                if scoring_method in ("explicit", "both"):
                    # Explicit scoring: ask judge directly
                    content = []
                    # Original images
                    for p in sample["image_paths"]:
                        if os.path.isfile(p):
                            content.append({"type": "image", "image": p})
                    # Generated images
                    for gi, gp in enumerate(gen_images):
                        content.append({"type": "image", "image": gp})

                    content.append({"type": "text", "text": (
                        f"Question: {sample['question']}\n\n"
                        f"The first {len(sample['image_paths'])} images are the original scene. "
                        f"The remaining {len(gen_images)} images are generated candidates.\n"
                        f"Rate how helpful the generated images are for answering the question. "
                        f"Give a score from 1-10."
                    )})

                    explicit_msg = [
                        {"role": "system", "content": JUDGE_EXPLICIT_PROMPT},
                        {"role": "user", "content": content},
                    ]

                    try:
                        raw_judge = generate_text(
                            model, processor, explicit_msg,
                            max_new_tokens=256, temperature=0.1, do_sample=False,
                        )
                        # Try to parse score from JSON or raw text
                        score_match = re.search(r'"score[s]?"[:\s]*\{[^}]*\}|(\d+)\s*/\s*10|score[:\s]+(\d+)', raw_judge, re.IGNORECASE)
                        if score_match:
                            found = score_match.group(1) or score_match.group(2) or "5"
                            explicit_score = float(found)
                        else:
                            # Fallback: look for any number 1-10
                            nums = re.findall(r'\b(\d+)\b', raw_judge)
                            nums = [int(n) for n in nums if 1 <= int(n) <= 10]
                            explicit_score = float(nums[0]) if nums else 5.0
                    except Exception as e:
                        log.error(f"Explicit scoring failed for sample {sample['id']} rollout {ri}: {e}")
                        explicit_score = 5.0

                    if scoring_method == "explicit":
                        score = explicit_score
                    elif scoring_method == "both":
                        # Combine: normalize confidence to 0-10 range then average
                        conf_normalized = max(0, min(10, (mean_logprob + 5) * 2))
                        score = 0.5 * explicit_score + 0.5 * conf_normalized

                rollout_scores.append({
                    "rollout_idx": ri,
                    "score": score,
                    "method": scoring_method,
                })

            record = {
                "id": sample["id"],
                "question": sample["question"],
                "image_paths": sample["image_paths"],
                "gt_answer": sample["gt_answer"],
                "rollouts": sample["rollouts"],
                "scores": rollout_scores,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            score_summary = [(s["rollout_idx"], round(s["score"], 3)) for s in rollout_scores]
            log.info(f"[{si+1}/{len(samples)}] id={sample['id']} scores={score_summary}")

    log.info(f"Labeling done. {len(results)} samples → {output_path}")


def run_labeling(rollout_records, judge_model_path, gen_dir, scoring_method,
                 output_dir, num_gpus):
    """Shard and score in parallel."""
    n_available = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]
    if num_gpus > 0:
        all_gpu_ids = all_gpu_ids[:num_gpus]

    log_file = str(output_dir / "labeling.log")
    chunks = chunk_dataset(rollout_records, len(all_gpu_ids))
    processes = []
    shard_paths = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = str(output_dir / f"labeled_shard_{idx}.jsonl")
        shard_paths.append(shard_path)
        p = mp.Process(
            target=label_worker,
            args=(gpu_id, chunk, judge_model_path, str(gen_dir),
                  scoring_method, shard_path, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(f"Labeling worker {idx} → GPU {gpu_id} ({len(chunk)} samples)")

    for p in processes:
        p.join()

    # Merge
    all_records = []
    for path in shard_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))

    merged_path = output_dir / "labeled.jsonl"
    with open(merged_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Labeling merged: {len(all_records)} samples → {merged_path}")
    return all_records, merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Build DPO preference pairs
# ══════════════════════════════════════════════════════════════════════════════

def build_preference_pairs(labeled_records: list, output_path: Path,
                           min_score_gap: float = 0.5,
                           iteration: int = -1):
    """From labeled records, extract (chosen, rejected) pairs.

    Selection strategy:
      - chosen  = rollout with the highest score
      - rejected = rollout with the lowest score
      - If all scores are identical (all-correct or all-wrong):
          * Use logits spread: take the rollout with highest score as chosen,
            lowest as rejected, even if gap < min_score_gap.
          * If scores are truly all the same, discard the sample.
      - Otherwise skip if score gap < min_score_gap (ambiguous pair).
    """
    pairs = []
    skipped = 0
    all_same_count = 0

    for rec in labeled_records:
        scores = rec.get("scores", [])
        if len(scores) < 2:
            skipped += 1
            continue

        # Sort by score descending
        sorted_scores = sorted(scores, key=lambda s: s["score"], reverse=True)
        best = sorted_scores[0]
        worst = sorted_scores[-1]

        gap = best["score"] - worst["score"]

        # Edge case: all rollouts scored identically
        unique_scores = set(s["score"] for s in scores)
        if len(unique_scores) == 1:
            # Truly all same — discard (no signal)
            all_same_count += 1
            skipped += 1
            continue

        if gap < min_score_gap:
            # Low gap but not identical — still use as weak signal
            # (useful for on-policy iterative DPO where near-ties carry info)
            if gap > 0:
                # Keep the pair but mark it as low-confidence
                pass
            else:
                skipped += 1
                continue

        best_rollout = rec["rollouts"][best["rollout_idx"]]
        worst_rollout = rec["rollouts"][worst["rollout_idx"]]

        pair = {
            "image": rec["image_paths"],
            "prompt": (
                f"<image>" * len(rec["image_paths"]) + "\n" + rec["question"]
            ),
            "chosen": best_rollout["raw_output"],
            "rejected": worst_rollout["raw_output"],
            "metadata": {
                "id": rec["id"],
                "question": rec["question"],
                "gt_answer": rec["gt_answer"],
                "chosen_score": best["score"],
                "rejected_score": worst["score"],
                "score_gap": gap,
                "chosen_rollout_idx": best["rollout_idx"],
                "rejected_rollout_idx": worst["rollout_idx"],
                "chosen_instructions": best_rollout["instructions"],
                "rejected_instructions": worst_rollout["instructions"],
                "iteration": iteration,
            },
        }
        pairs.append(pair)

    # Save as JSON (compatible with Qwen-VL-Series-Finetune DPO format)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Built {len(pairs)} preference pairs "
        f"({skipped} skipped, {all_same_count} all-same-score)"
    )
    logger.info(f"Saved to {output_path}")

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference data for the spatial Planning Model."
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="mmsibench",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)

    # Models
    parser.add_argument("--planner_model_path", type=str, required=True,
                        help="Path to the current Planning Model checkpoint.")
    parser.add_argument("--judge_model_path", type=str, default=None,
                        help="Path to the MLLM judge. Defaults to planner_model_path.")
    parser.add_argument("--flux_ckpt", type=str, default="checkpoints/flux2-klein-4B",
                        help="Path to the Flux2Klein checkpoint.")

    # Rollout settings
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS_DEFAULT,
                        help="Number of instruction sets to sample per question.")

    # Scoring
    parser.add_argument("--scoring_method", type=str, default="confidence",
                        choices=["confidence", "explicit", "both"],
                        help="How to score rollouts: implicit confidence, explicit judge, or both.")
    parser.add_argument("--min_score_gap", type=float, default=0.5,
                        help="Minimum score gap to keep a preference pair.")

    # Image generation
    parser.add_argument("--num_inference_steps", type=int, default=28)

    # Multi-GPU
    parser.add_argument("--num_gpus", type=int, default=-1)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/dpo_data",
                        help="Root output directory.")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from existing run directory (skip completed steps).")

    # Step control
    parser.add_argument("--skip_rollout", action="store_true",
                        help="Skip rollout, use existing rollouts.jsonl.")
    parser.add_argument("--skip_execution", action="store_true",
                        help="Skip image generation, use existing images.")
    parser.add_argument("--skip_labeling", action="store_true",
                        help="Skip labeling, use existing labeled.jsonl.")

    return parser.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    script_dir = Path(__file__).parent
    def resolve(p):
        q = Path(p)
        return q if q.is_absolute() else script_dir / q

    # Output directory
    if args.resume_from:
        output_dir = resolve(args.resume_from)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = resolve(args.output_dir) / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_file = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    if args.judge_model_path is None:
        args.judge_model_path = args.planner_model_path

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_path = resolve(args.data_path)
    image_root = resolve(args.image_root)
    loader = DATASET_LOADERS[args.dataset]
    samples = loader(str(data_path), str(image_root), args.max_samples, args.start_idx)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}.")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Step 1: Rollout ───────────────────────────────────────────────────────
    rollout_path = output_dir / "rollouts.jsonl"
    if args.skip_rollout and rollout_path.exists():
        logger.info(f"Skipping rollout, loading from {rollout_path}")
        rollout_records = []
        with open(rollout_path) as f:
            for line in f:
                if line.strip():
                    rollout_records.append(json.loads(line))
    else:
        logger.info("=" * 60)
        logger.info("STEP 1: Rollout — Sampling diverse instructions")
        logger.info("=" * 60)
        rollout_records, _ = run_rollouts(
            samples, args.planner_model_path, args.num_rollouts,
            output_dir, args.num_gpus,
        )

    # ── Step 2: Execution ─────────────────────────────────────────────────────
    gen_dir = output_dir / "generated_images"
    if args.skip_execution and gen_dir.exists():
        logger.info(f"Skipping execution, using existing images in {gen_dir}")
    else:
        logger.info("=" * 60)
        logger.info("STEP 2: Execution — Generating images with Flux2Klein")
        logger.info("=" * 60)
        gen_dir = run_execution(
            rollout_records,
            str(resolve(args.flux_ckpt)),
            output_dir,
            args.num_inference_steps,
            args.num_gpus,
        )

    # ── Step 3: Labeling ──────────────────────────────────────────────────────
    labeled_path = output_dir / "labeled.jsonl"
    if args.skip_labeling and labeled_path.exists():
        logger.info(f"Skipping labeling, loading from {labeled_path}")
        labeled_records = []
        with open(labeled_path) as f:
            for line in f:
                if line.strip():
                    labeled_records.append(json.loads(line))
    else:
        logger.info("=" * 60)
        logger.info("STEP 3: Labeling — MLLM scoring of generated images")
        logger.info("=" * 60)
        labeled_records, _ = run_labeling(
            rollout_records, args.judge_model_path, gen_dir,
            args.scoring_method, output_dir, args.num_gpus,
        )

    # ── Step 4: Build preference pairs ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building DPO preference pairs")
    logger.info("=" * 60)
    dpo_data_path = output_dir / "dpo_train.json"
    pairs = build_preference_pairs(labeled_records, dpo_data_path, args.min_score_gap)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total samples processed : {len(samples)}")
    logger.info(f"  Rollouts per sample     : {args.num_rollouts}")
    logger.info(f"  Scoring method          : {args.scoring_method}")
    logger.info(f"  Preference pairs        : {len(pairs)}")
    logger.info(f"  Output directory        : {output_dir}")
    logger.info(f"  DPO training data       : {dpo_data_path}")
    logger.info(f"")
    logger.info(f"Next step: run DPO training with:")
    logger.info(f"  python train_planner.py --data_path {dpo_data_path} ...")


if __name__ == "__main__":
    main()
