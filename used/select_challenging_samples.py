"""
Training-free pipeline that identifies "both-wrong" SAT training samples for SFT.

A sample is considered "challenging" when:
  1. Baseline  – Qwen3-VL answers incorrectly using only the original scene images.
  2. Augmented – After the Planner generates instructions and Flux produces extra
                 images, Qwen3-VL still answers incorrectly with all images.

The final 2000 samples are balanced across all question-type categories.

Pipeline stages (each stage checkpoints so the run is resumable):
  Stage 1 – Baseline VQA      : Qwen3-VL on original images → answer
  Stage 2 – Planning          : Planner generates image instructions for wrong cases
  Stage 3 – Execution         : Flux2Klein generates images from instructions
  Stage 4 – Augmented VQA     : Qwen3-VL on original + generated images → answer
  Stage 5 – Selection         : filter both-wrong, balance, save SFT JSON

Usage:
  python select_challenging_samples.py \\
    --data_path datasets/evaluation/SAT/train.json \\
    --image_root datasets/evaluation/SAT \\
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \\
    --flux_ckpt checkpoints/flux2-klein-4B \\
    --output_dir results/challenging_sat \\
    --target_total 2000 \\
    --max_candidates_per_type 3000 \\
    --num_gpus -1
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from generate_image_instructions import (
    SYSTEM_PROMPT,
    build_message,
    chunk_dataset,
    parse_instructions,
)

logger = logging.getLogger(__name__)

# ── SAT prompt constants ──────────────────────────────────────────────────────
_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)

# ══════════════════════════════════════════════════════════════════════════════
#  Answer parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_answer_tag(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1] if matches else text


def clean_text(text: str) -> str:
    """Normalise model output for exact-match comparison."""
    cleaned = _extract_answer_tag(text)
    cleaned = re.sub(r"\s+", " ", cleaned.replace("\n", " ").replace("\r", " "))
    return cleaned.strip().rstrip(".").lower()


def is_correct(model_output: str, correct_answer: str) -> bool:
    """Return True if model_output matches correct_answer (case-insensitive)."""
    pred = clean_text(model_output)
    gt   = clean_text(correct_answer)
    # Exact match
    if pred == gt:
        return True
    # Model might output the choice letter (A/B/C/D/E); also accept that
    if len(pred) == 1 and pred.isalpha():
        return False  # will be rechecked by caller after mapping to text
    return False


def map_letter_to_answer(letter: str, answer_choices: List[str]) -> str:
    """Convert a single-letter answer (A/B/C/D/E) to the actual choice text."""
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(answer_choices):
        return answer_choices[idx]
    return letter


def check_correct(model_output: str, correct_answer: str,
                  answer_choices: List[str]) -> bool:
    """
    Check if the model's answer matches the ground truth.
    Handles both free-text answers and single-letter option selections.
    """
    pred_raw = clean_text(model_output)
    gt       = clean_text(correct_answer)

    # Direct match
    if pred_raw == gt:
        return True

    # If model output a single letter, map it to the choice text
    if len(pred_raw) == 1 and pred_raw.isalpha():
        mapped = clean_text(map_letter_to_answer(pred_raw, answer_choices))
        return mapped == gt

    # If model output "a. some text" or "a) some text" pattern
    m = re.match(r"^([a-e])[.):\s]", pred_raw)
    if m:
        mapped = clean_text(map_letter_to_answer(m.group(1), answer_choices))
        return mapped == gt

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading (shared by Stage 1 & 4)
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_vl(model_path: str, device: str):
    """Load Qwen3-VL (or compatible) model and processor."""
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


def run_vqa(model, processor, image_paths: List[str], question_text: str,
            max_new_tokens: int = 64) -> str:
    """Run a single VQA forward pass and return generated text."""
    from qwen_vl_utils import process_vision_info

    content = []
    for p in image_paths:
        if os.path.isfile(p):
            content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": question_text})

    messages = [{"role": "user", "content": content}]
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
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def build_sat_question(item: dict) -> str:
    """Format the SAT question + choices into a prompt string."""
    q = item["question"]
    choices = item.get("answer_choices", [])
    if choices:
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"{q}\nOptions:\n{opts}\n{_ANSWER_INSTRUCTION}"
    return f"{q}\n{_ANSWER_INSTRUCTION}"


def resolve_image_paths(item: dict, image_root: str) -> List[str]:
    """Convert relative img_paths to absolute paths."""
    root = Path(image_root)
    paths = []
    for p in item.get("img_paths", []):
        full = Path(p) if Path(p).is_absolute() else root / p
        paths.append(str(full.resolve()))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Baseline VQA
# ══════════════════════════════════════════════════════════════════════════════

def _stage1_worker(gpu_id: str, samples: list, model_path: str,
                   image_root: str, output_path: str, log_file: Optional[str]):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [S1 GPU{gpu_id}] %(message)s",
        handlers=handlers, force=True,
    )
    log = logging.getLogger(__name__)
    torch.cuda.set_device(int(gpu_id))
    model, processor = load_qwen_vl(model_path, f"cuda:{gpu_id}")
    log.info(f"Model loaded. Processing {len(samples)} samples.")

    results = []
    with open(output_path, "w") as f:
        for i, item in enumerate(samples):
            image_paths = resolve_image_paths(item, image_root)
            question    = build_sat_question(item)
            try:
                output = run_vqa(model, processor, image_paths, question)
            except Exception as e:
                log.error(f"VQA failed for idx {item['database_idx']}: {e}")
                output = ""

            correct = check_correct(output, item["correct_answer"],
                                    item.get("answer_choices", []))
            rec = {
                "database_idx":    item["database_idx"],
                "question_type":   item["question_type"],
                "question":        item["question"],
                "answer_choices":  item.get("answer_choices", []),
                "correct_answer":  item["correct_answer"],
                "img_paths":       item["img_paths"],
                "image_paths_abs": image_paths,
                "baseline_output": output,
                "baseline_correct": correct,
            }
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                log.info(f"[{i+1}/{len(samples)}] done, last correct={correct}")

    log.info(f"Stage 1 worker done → {output_path}")


def run_stage1(samples: list, model_path: str, image_root: str,
               stage_dir: Path, num_gpus: int) -> List[dict]:
    """Run baseline VQA across all GPUs."""
    merged = stage_dir / "baseline_results.jsonl"
    if merged.exists():
        recs = _load_jsonl(merged)
        logger.info(f"Stage 1: loaded {len(recs)} cached baseline records.")
        return recs

    gpu_ids = _get_gpu_ids(num_gpus)
    log_file = str(stage_dir / "stage1.log")
    chunks = chunk_dataset(samples, len(gpu_ids))
    procs = []
    shard_paths = []

    for idx, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        sp = str(stage_dir / f"s1_shard_{idx}.jsonl")
        shard_paths.append(sp)
        p = mp.Process(target=_stage1_worker,
                       args=(gid, chunk, model_path, image_root, sp, log_file))
        p.start()
        procs.append(p)
        logger.info(f"Stage 1 worker {idx} → GPU {gid} ({len(chunk)} samples)")

    for p in procs:
        p.join()

    recs = _merge_jsonl_shards(shard_paths, merged)
    logger.info(f"Stage 1 done: {len(recs)} records, "
                f"{sum(1 for r in recs if not r['baseline_correct'])} wrong.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2: Planning (Planner generates image instructions for wrong cases)
# ══════════════════════════════════════════════════════════════════════════════

def _stage2_worker(gpu_id: str, samples: list, model_path: str,
                   output_path: str, log_file: Optional[str]):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [S2 GPU{gpu_id}] %(message)s",
        handlers=handlers, force=True,
    )
    log = logging.getLogger(__name__)
    torch.cuda.set_device(int(gpu_id))

    # Load planner with same helper used in generate_dpo_data.py
    from generate_image_instructions import load_model_and_processor, Config
    cfg = Config(model_path=model_path, max_new_tokens=1024,
                 temperature=0.7, top_p=0.9, do_sample=True)
    model, processor = load_model_and_processor(model_path, "qwen-vl", f"cuda:{gpu_id}")
    model.eval()
    log.info(f"Planner loaded. Processing {len(samples)} samples.")

    from generate_image_instructions import run_inference as _run_inference

    results = []
    with open(output_path, "w") as f:
        for i, rec in enumerate(samples):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                build_message(rec["question"], rec["image_paths_abs"]),
            ]
            try:
                raw_output  = _run_inference(model, processor, messages, cfg)
                instructions = parse_instructions(raw_output)
            except Exception as e:
                log.error(f"Planning failed for idx {rec['database_idx']}: {e}")
                raw_output   = ""
                instructions = []

            out = dict(rec)
            out["plan_raw_output"]  = raw_output
            out["plan_instructions"] = instructions
            results.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                log.info(f"[{i+1}/{len(samples)}] planned {len(instructions)} instr.")

    log.info(f"Stage 2 worker done → {output_path}")


def run_stage2(wrong_recs: list, model_path: str,
               stage_dir: Path, num_gpus: int) -> List[dict]:
    """Run planner on baseline-wrong samples."""
    merged = stage_dir / "planning_results.jsonl"
    if merged.exists():
        recs = _load_jsonl(merged)
        logger.info(f"Stage 2: loaded {len(recs)} cached planning records.")
        return recs

    gpu_ids = _get_gpu_ids(num_gpus)
    log_file = str(stage_dir / "stage2.log")
    chunks = chunk_dataset(wrong_recs, len(gpu_ids))
    procs = []
    shard_paths = []

    for idx, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        sp = str(stage_dir / f"s2_shard_{idx}.jsonl")
        shard_paths.append(sp)
        p = mp.Process(target=_stage2_worker,
                       args=(gid, chunk, model_path, sp, log_file))
        p.start()
        procs.append(p)
        logger.info(f"Stage 2 worker {idx} → GPU {gid} ({len(chunk)} samples)")

    for p in procs:
        p.join()

    recs = _merge_jsonl_shards(shard_paths, merged)
    n_with_instr = sum(1 for r in recs if r.get("plan_instructions"))
    logger.info(f"Stage 2 done: {len(recs)} records, {n_with_instr} with instructions.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3: Execution (Flux2Klein generates images from instructions)
# ══════════════════════════════════════════════════════════════════════════════

def _stage3_worker(gpu_id: str, tasks: list, flux_ckpt: str,
                   output_root: str, num_inference_steps: int,
                   log_file: Optional[str]):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [S3 GPU{gpu_id}] %(message)s",
        handlers=handlers, force=True,
    )
    log = logging.getLogger(__name__)
    torch.cuda.set_device(int(gpu_id))

    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(
        flux_ckpt, torch_dtype=torch.bfloat16
    ).to(f"cuda:{gpu_id}")
    pipe.set_progress_bar_config(disable=True)
    log.info(f"Flux loaded. Processing {len(tasks)} tasks.")

    out_root = Path(output_root)
    for ti, task in enumerate(tasks):
        out_dir = out_root / str(task["database_idx"]) / f"instr_{task['instr_idx']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "generated.png"
        if out_path.exists():
            continue

        src_images = []
        for p in task["image_paths_abs"]:
            if os.path.isfile(p):
                try:
                    src_images.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
        if not src_images:
            log.warning(f"No source images for idx {task['database_idx']}, skip.")
            continue

        try:
            result = pipe(image=src_images, prompt=task["instruction"],
                          num_inference_steps=num_inference_steps)
            result.images[0].save(out_path)
        except Exception as e:
            log.error(f"Flux failed task {ti}: {e}")

        if (ti + 1) % 20 == 0:
            log.info(f"[{ti+1}/{len(tasks)}] generated.")

    log.info(f"Stage 3 worker done.")


def run_stage3(plan_recs: list, flux_ckpt: str,
               stage_dir: Path, num_inference_steps: int,
               num_gpus: int) -> Path:
    """Generate images for all planned instructions."""
    gen_root = stage_dir / "generated_images"
    if gen_root.exists() and any(gen_root.rglob("*.png")):
        logger.info(f"Stage 3: generated images exist, skipping.")
        return gen_root

    gen_root.mkdir(parents=True, exist_ok=True)

    # Build flat task list
    tasks = []
    for rec in plan_recs:
        for ii, instr in enumerate(rec.get("plan_instructions", [])):
            tasks.append({
                "database_idx":   rec["database_idx"],
                "instr_idx":      ii,
                "instruction":    instr,
                "image_paths_abs": rec["image_paths_abs"],
            })

    if not tasks:
        logger.warning("Stage 3: no tasks (all planners produced empty instructions).")
        return gen_root

    logger.info(f"Stage 3: {len(tasks)} image generation tasks.")
    gpu_ids  = _get_gpu_ids(num_gpus)
    log_file = str(stage_dir / "stage3.log")
    chunks   = chunk_dataset(tasks, len(gpu_ids))
    procs    = []

    for idx, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        p = mp.Process(target=_stage3_worker,
                       args=(gid, chunk, str(flux_ckpt), str(gen_root),
                             num_inference_steps, log_file))
        p.start()
        procs.append(p)
        logger.info(f"Stage 3 worker {idx} → GPU {gid} ({len(chunk)} tasks)")

    for p in procs:
        p.join()

    n_generated = len(list(gen_root.rglob("*.png")))
    logger.info(f"Stage 3 done: {n_generated} images under {gen_root}")
    return gen_root


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4: Augmented VQA
# ══════════════════════════════════════════════════════════════════════════════

def _stage4_worker(gpu_id: str, samples: list, model_path: str,
                   gen_root: str, output_path: str, log_file: Optional[str]):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [S4 GPU{gpu_id}] %(message)s",
        handlers=handlers, force=True,
    )
    log = logging.getLogger(__name__)
    torch.cuda.set_device(int(gpu_id))
    model, processor = load_qwen_vl(model_path, f"cuda:{gpu_id}")
    log.info(f"Model loaded. Processing {len(samples)} samples.")

    gen_root_path = Path(gen_root)
    results = []
    with open(output_path, "w") as f:
        for i, rec in enumerate(samples):
            # Collect generated images for this sample
            sample_gen_dir = gen_root_path / str(rec["database_idx"])
            gen_images = []
            if sample_gen_dir.is_dir():
                gen_images = sorted(
                    str(p) for p in sample_gen_dir.rglob("*.png")
                )

            # Augmented input = original images + generated images
            all_images = rec["image_paths_abs"] + gen_images
            question   = build_sat_question(rec)

            try:
                output = run_vqa(model, processor, all_images, question)
            except Exception as e:
                log.error(f"Augmented VQA failed for idx {rec['database_idx']}: {e}")
                output = ""

            correct = check_correct(output, rec["correct_answer"],
                                    rec.get("answer_choices", []))
            out = dict(rec)
            out["generated_images"]   = gen_images
            out["augmented_output"]   = output
            out["augmented_correct"]  = correct
            out["both_wrong"]         = (not rec["baseline_correct"]) and (not correct)
            results.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                log.info(f"[{i+1}/{len(samples)}] aug_correct={correct}, both_wrong={out['both_wrong']}")

    log.info(f"Stage 4 worker done → {output_path}")


def run_stage4(plan_recs: list, model_path: str, gen_root: Path,
               stage_dir: Path, num_gpus: int) -> List[dict]:
    """Run augmented VQA on all baseline-wrong samples."""
    merged = stage_dir / "augmented_results.jsonl"
    if merged.exists():
        recs = _load_jsonl(merged)
        logger.info(f"Stage 4: loaded {len(recs)} cached augmented records.")
        return recs

    gpu_ids  = _get_gpu_ids(num_gpus)
    log_file = str(stage_dir / "stage4.log")
    chunks   = chunk_dataset(plan_recs, len(gpu_ids))
    procs    = []
    shard_paths = []

    for idx, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        sp = str(stage_dir / f"s4_shard_{idx}.jsonl")
        shard_paths.append(sp)
        p = mp.Process(target=_stage4_worker,
                       args=(gid, chunk, model_path, str(gen_root), sp, log_file))
        p.start()
        procs.append(p)
        logger.info(f"Stage 4 worker {idx} → GPU {gid} ({len(chunk)} samples)")

    for p in procs:
        p.join()

    recs = _merge_jsonl_shards(shard_paths, merged)
    both_wrong = sum(1 for r in recs if r.get("both_wrong"))
    logger.info(f"Stage 4 done: {len(recs)} records, {both_wrong} both-wrong.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 5: Balanced selection
# ══════════════════════════════════════════════════════════════════════════════

def select_balanced(all_recs: list, target_total: int, seed: int = 42) -> List[dict]:
    """
    From 'both_wrong' records, select target_total samples balanced across
    question_type categories.

    Strategy:
      - Group by question_type.
      - Target per category = target_total // n_categories, with remainder
        distributed to largest categories.
      - If a category has fewer both-wrong cases than its target, take all of
        them and redistribute the gap to other categories.
    """
    rng = random.Random(seed)

    # Pool: only both-wrong samples
    pool_by_type: Dict[str, List[dict]] = defaultdict(list)
    for rec in all_recs:
        if rec.get("both_wrong"):
            pool_by_type[rec["question_type"]].append(rec)

    categories = sorted(pool_by_type.keys())
    n_cats = len(categories)
    if n_cats == 0:
        logger.warning("No both-wrong samples found!")
        return []

    logger.info("Both-wrong pool per category:")
    for cat in categories:
        logger.info(f"  {cat}: {len(pool_by_type[cat])}")

    # Iterative fair allocation
    quota: Dict[str, int] = {cat: target_total // n_cats for cat in categories}
    # Distribute remainder to largest-pool categories
    remainder = target_total - sum(quota.values())
    for cat in sorted(categories, key=lambda c: -len(pool_by_type[c])):
        if remainder <= 0:
            break
        quota[cat] += 1
        remainder -= 1

    # Iteratively handle under-supplied categories
    changed = True
    while changed:
        changed = False
        shortfall = 0
        fulfilled: List[str] = []
        unfulfilled: List[str] = []
        for cat in categories:
            avail = len(pool_by_type[cat])
            if avail < quota[cat]:
                shortfall += quota[cat] - avail
                quota[cat] = avail
                fulfilled.append(cat)
                changed = True
            else:
                unfulfilled.append(cat)

        # Redistribute shortfall to unfulfilled categories proportionally
        if shortfall > 0 and unfulfilled:
            total_unfulfilled = sum(len(pool_by_type[c]) - quota[c]
                                    for c in unfulfilled)
            for cat in unfulfilled:
                headroom = len(pool_by_type[cat]) - quota[cat]
                if total_unfulfilled > 0:
                    extra = round(shortfall * headroom / total_unfulfilled)
                    quota[cat] += min(extra, headroom)
            # Correct rounding drift
            diff = target_total - sum(quota.values())
            for cat in sorted(unfulfilled,
                               key=lambda c: -(len(pool_by_type[c]) - quota[c])):
                if diff == 0:
                    break
                headroom = len(pool_by_type[cat]) - quota[cat]
                add = max(-quota[cat], min(diff, headroom))
                quota[cat] += add
                diff -= add

    logger.info("Final quota per category:")
    total_selected = 0
    for cat in categories:
        logger.info(f"  {cat}: {quota[cat]} / {len(pool_by_type[cat])} available")
        total_selected += quota[cat]
    logger.info(f"  TOTAL: {total_selected}")

    # Sample
    selected = []
    for cat in categories:
        pool = pool_by_type[cat]
        rng.shuffle(pool)
        selected.extend(pool[: quota[cat]])

    rng.shuffle(selected)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _get_gpu_ids(num_gpus: int) -> List[str]:
    n_available = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_ids = [str(i) for i in range(n_available)]
    if num_gpus > 0:
        all_ids = all_ids[:num_gpus]
    return all_ids


def _load_jsonl(path: Path) -> List[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _merge_jsonl_shards(shard_paths: List[str], merged_path: Path) -> List[dict]:
    all_recs = []
    for sp in shard_paths:
        if os.path.exists(sp):
            all_recs.extend(_load_jsonl(Path(sp)))
        else:
            logger.warning(f"Shard not found: {sp}")
    with open(merged_path, "w") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return all_recs


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Find balanced 'both-wrong' challenging SAT samples (training-free)."
    )
    p.add_argument("--data_path", required=True,
                   help="Path to SAT train.json")
    p.add_argument("--image_root", required=True,
                   help="Root dir for resolving relative img_paths in SAT JSON")
    p.add_argument("--planner_model_path", required=True,
                   help="Qwen3-VL checkpoint (used for both baseline VQA and planning)")
    p.add_argument("--flux_ckpt", required=True,
                   help="Flux2Klein checkpoint dir")
    p.add_argument("--output_dir", default="results/challenging_sat",
                   help="Directory for intermediate files and final output")
    p.add_argument("--target_total", type=int, default=2000,
                   help="Total number of challenging samples to select")
    p.add_argument("--max_candidates_per_type", type=int, default=3000,
                   help="Max samples to evaluate per question_type in Stage 1 "
                        "(caps total processing; -1 = all)")
    p.add_argument("--num_inference_steps", type=int, default=28,
                   help="Flux denoising steps")
    p.add_argument("--num_gpus", type=int, default=-1,
                   help="Number of GPUs (-1 = all available / CUDA_VISIBLE_DEVICES)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else SCRIPT_DIR / q

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file = out_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )

    # ── Load SAT train.json ───────────────────────────────────────────────────
    data_path  = resolve(args.data_path)
    image_root = resolve(args.image_root)

    with open(data_path) as f:
        all_data = json.load(f)

    logger.info(f"Loaded {len(all_data)} SAT training samples from {data_path}")

    # ── Sample candidates per question type ───────────────────────────────────
    rng = random.Random(args.seed)
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for item in all_data:
        by_type[item["question_type"]].append(item)

    candidates: List[dict] = []
    cap = args.max_candidates_per_type
    for qtype, items in by_type.items():
        rng.shuffle(items)
        subset = items if cap <= 0 else items[:cap]
        candidates.extend(subset)
        logger.info(f"  {qtype}: {len(subset)} candidates (pool: {len(items)})")
    rng.shuffle(candidates)
    logger.info(f"Total candidates for Stage 1: {len(candidates)}")

    # ── Stage 1: Baseline VQA ─────────────────────────────────────────────────
    stage1_dir = out_dir / "stage1"
    stage1_dir.mkdir(exist_ok=True)
    baseline_recs = run_stage1(candidates, str(resolve(args.planner_model_path)),
                               str(image_root), stage1_dir, args.num_gpus)

    wrong_recs = [r for r in baseline_recs if not r["baseline_correct"]]
    logger.info(
        f"Stage 1 complete: {len(baseline_recs)} total, "
        f"{len(wrong_recs)} baseline-wrong "
        f"({100*len(wrong_recs)/max(len(baseline_recs),1):.1f}%)"
    )

    # Per-type breakdown
    wrong_by_type = Counter(r["question_type"] for r in wrong_recs)
    logger.info("Baseline-wrong per type: " + str(dict(wrong_by_type)))

    # ── Stage 2: Planning ─────────────────────────────────────────────────────
    stage2_dir = out_dir / "stage2"
    stage2_dir.mkdir(exist_ok=True)
    plan_recs = run_stage2(wrong_recs, str(resolve(args.planner_model_path)),
                           stage2_dir, args.num_gpus)

    # ── Stage 3: Execution ────────────────────────────────────────────────────
    stage3_dir = out_dir / "stage3"
    stage3_dir.mkdir(exist_ok=True)
    gen_root = run_stage3(plan_recs, resolve(args.flux_ckpt),
                          stage3_dir, args.num_inference_steps, args.num_gpus)

    # ── Stage 4: Augmented VQA ────────────────────────────────────────────────
    stage4_dir = out_dir / "stage4"
    stage4_dir.mkdir(exist_ok=True)
    aug_recs = run_stage4(plan_recs, str(resolve(args.planner_model_path)),
                          gen_root, stage4_dir, args.num_gpus)

    both_wrong_recs = [r for r in aug_recs if r.get("both_wrong")]
    logger.info(
        f"Stage 4 complete: {len(aug_recs)} samples, "
        f"{len(both_wrong_recs)} both-wrong "
        f"({100*len(both_wrong_recs)/max(len(aug_recs),1):.1f}%)"
    )
    bw_by_type = Counter(r["question_type"] for r in both_wrong_recs)
    logger.info("Both-wrong per type: " + str(dict(bw_by_type)))

    # ── Stage 5: Balanced selection ───────────────────────────────────────────
    selected = select_balanced(aug_recs, args.target_total, seed=args.seed)
    logger.info(f"Selected {len(selected)} challenging samples.")

    # Build clean SFT output (only original SAT fields + diagnostics)
    sft_records = []
    for rec in selected:
        sft_records.append({
            # Original SAT fields
            "database_idx":    rec["database_idx"],
            "question_type":   rec["question_type"],
            "question":        rec["question"],
            "answer_choices":  rec["answer_choices"],
            "correct_answer":  rec["correct_answer"],
            "img_paths":       rec["img_paths"],
            # Diagnostics
            "baseline_output":   rec["baseline_output"],
            "baseline_correct":  rec["baseline_correct"],
            "plan_instructions": rec.get("plan_instructions", []),
            "generated_images":  rec.get("generated_images", []),
            "augmented_output":  rec.get("augmented_output", ""),
            "augmented_correct": rec.get("augmented_correct", False),
            "both_wrong":        rec["both_wrong"],
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_json = out_dir / "challenging_samples_2k.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sft_records, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("=" * 60)
    logger.info("  SELECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Output  : {out_json}")
    logger.info(f"  Total   : {len(sft_records)}")
    sel_by_type = Counter(r["question_type"] for r in sft_records)
    for cat, cnt in sorted(sel_by_type.items()):
        logger.info(f"  {cat:25s}: {cnt}")
    logger.info("=" * 60)

    # Per-type summary JSON
    summary = {
        "total_candidates": len(candidates),
        "baseline_wrong":   len(wrong_recs),
        "both_wrong_pool":  len(both_wrong_recs),
        "selected":         len(sft_records),
        "per_type": {cat: cnt for cat, cnt in sel_by_type.items()},
        "output_path": str(out_json),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
