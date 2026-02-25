"""
Dual-method reasoning evaluation on MMSIBench.

For every sample, runs inference with TWO methods in a single pass:

  Method A – Baseline
      Input: original dataset images + question

  Method B – Augmented
      Input: original dataset images + generated images (from --gen_dir)
             + question
      If no generated images exist for a sample (empty/missing folder),
      the augmented method falls back to the same input as baseline.

Both methods share the same model instance per GPU worker.

Usage
-----
python evaluation.py \\
    --model_type  qwen3-vl \\
    --model_path  checkpoints/Qwen3-VL-4B-Instruct \\
    --data_dir    datasets/evaluation/MMSIBench \\
    --gen_dir     generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B

Multi-GPU (auto-shards across all visible GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py \\
        --model_type qwen3-vl \\
        --model_path checkpoints/Qwen3-VL-4B-Instruct \\
        --data_dir   datasets/evaluation/MMSIBench \\
        --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B

Smoke test:
    python evaluation.py --limit 12 \\
        --model_type qwen3-vl \\
        --model_path checkpoints/Qwen3-VL-4B-Instruct \\
        --data_dir   datasets/evaluation/MMSIBench \\
        --gen_dir    generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt templates (same convention as Spatial_RAI eval)
# ---------------------------------------------------------------------------
QUESTION_TEMPLATE = "{Question}"
ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)


# ===========================================================================
# Dataset utilities
# ===========================================================================

def load_dataset(data_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MMSIBench dataset from *test_data_final.json*.

    Image paths stored in the JSON (`local_images`) are relative to the
    dataset root (the directory that contains the ``data/`` sub-folder, i.e.
    ``data_dir`` itself).  We resolve them to absolute paths here so that
    worker processes don't need to know the working directory.

    Args:
        data_dir: Root of the MMSIBench dataset directory.
        limit:    Optional cap on the number of samples to load.

    Returns:
        List of dicts with keys: index, image (list[str]), question, answer,
        category, thought.
    """
    json_file = data_dir / "data" / "test_data_final.json"
    if not json_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {json_file}\n"
            "Run datasets/evaluation/MMSIBench/download.py first."
        )

    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if limit is not None:
        raw = raw[:limit]

    dataset: List[Dict[str, Any]] = []
    for item in raw:
        local_images = item.get("local_images", [])
        # Paths in the JSON are relative to data_dir (the MMSIBench root).
        image_paths = [str((data_dir / p).resolve()) for p in local_images]

        dataset.append({
            "index": item.get("id", len(dataset)),
            "image": image_paths,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "category": item.get("type", "unknown"),
            "thought": item.get("thought_gt", ""),
        })

    return dataset


def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [dataset[s : s + chunk_size] for s in range(0, len(dataset), chunk_size)]


def get_generated_images(sample_index: int, gen_dir: Optional[Path]) -> List[str]:
    """Return sorted list of generated image paths for *sample_index*.

    Returns an empty list if gen_dir is None, the sub-folder doesn't exist,
    or the sub-folder is empty (meaning no image generation was needed).
    """
    if gen_dir is None:
        return []
    sample_dir = gen_dir / str(sample_index)
    if not sample_dir.is_dir():
        return []
    imgs = sorted(
        p for p in sample_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    return [str(p) for p in imgs]


# ===========================================================================
# Answer extraction
# ===========================================================================

def extract_answer_letter(text: str) -> str:
    """Extract the answer letter (A-D) from raw model output.

    Priority order:
    1. Content between double back-ticks ``X``
    2. Content between single back-ticks `X`
    3. Isolated A/B/C/D token (word boundary, not an article like "A bike")
    """
    if not text or not isinstance(text, str):
        return ""

    # ``X``
    m = re.search(r"``([^`]*)``", text)
    if m:
        text = m.group(1)

    # `X`
    m = re.search(r"`([^`]*)`", text)
    if m:
        text = m.group(1)

    # Bare letter with word boundary (not followed immediately by a lowercase word)
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", text)
    if m:
        return m.group()

    return ""


# ===========================================================================
# Model loading
# ===========================================================================

def load_model_and_processor(
    model_type: str,
    model_path: str,
    device: str = "cuda:0",
) -> Tuple[Any, Any]:
    """Load model and processor for the given *model_type*.

    Supported model types
    ---------------------
    ``qwen2.5-vl``   – Qwen2.5-VL family (baseline)
    ``qwen3-vl``     – Qwen3-VL family (uses same API as Qwen2.5-VL)

    The model is first loaded on CPU and then moved to *device* explicitly so
    that multi-GPU workers can each own a single GPU without relying on
    ``CUDA_VISIBLE_DEVICES`` tricks.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model type='{model_type}' from '{model_path}' onto {device}")

    if model_type in ("qwen2.5-vl", "qwen3-vl"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        model_device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {model_device}")
        return model, processor

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Supported: 'qwen2.5-vl', 'qwen3-vl'"
    )


# ===========================================================================
# Inference helpers
# ===========================================================================

def build_user_message(item: Dict[str, Any], extra_images: List[str]) -> Dict:
    """Build a Qwen-VL chat message.

    Image order: original dataset images → extra (generated) images → question text.
    """
    all_images = item["image"] + extra_images
    image_contents = [{"type": "image", "image": p} for p in all_images]
    text = f"{QUESTION_TEMPLATE.format(Question=item['question'])}\n{ANSWER_INSTRUCTION}"
    return {"role": "user", "content": image_contents + [{"type": "text", "text": text}]}


def prepare_batch(
    batch_data: List[Dict],
    extra_images_list: List[List[str]],
    processor: Any,
) -> Tuple[Dict, List[str]]:
    """Tokenise a batch. *extra_images_list[i]* are generated images for batch_data[i]."""
    from qwen_vl_utils import process_vision_info

    batch_messages = [
        [build_user_message(item, extra)]
        for item, extra in zip(batch_data, extra_images_list)
    ]

    prompts_text = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    all_image_inputs: List[Any] = []
    all_video_inputs: List[Any] = []
    for msgs in batch_messages:
        imgs, vids = process_vision_info(msgs)
        all_image_inputs.extend(imgs or [])
        all_video_inputs.extend(vids or [])

    batch_inputs = processor(
        text=prompts_text,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    return batch_inputs, prompts_text


def run_inference(
    batch_inputs: Dict,
    model: Any,
    processor: Any,
    max_new_tokens: int = 128,
) -> List[str]:
    """Run generation and return decoded strings for each sample in the batch."""
    batch_inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Trim input tokens from output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(batch_inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def postprocess(
    batch_data: List[Dict],
    outputs: List[str],
    prompts: List[str],
    extra_images_list: List[List[str]],
    method: str,
) -> List[Dict]:
    """Package raw outputs into result dicts."""
    results = []
    for item, output, prompt, extra in zip(batch_data, outputs, prompts, extra_images_list):
        results.append({
            "method": method,
            "index": item.get("index", ""),
            "category": item.get("category", "unknown"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "prediction": extract_answer_letter(output),
            "output": output,
            "thought_gt": item.get("thought", ""),
            "original_images": item["image"],
            "generated_images": extra,
            "prompt": prompt,
        })
    return results


def _error_result(item: Dict, exc: Exception, method: str, extra: List[str]) -> Dict:
    return {
        "method": method,
        "index": item.get("index", ""),
        "category": item.get("category", "unknown"),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "prediction": "",
        "output": f"ERROR: {exc}",
        "thought_gt": item.get("thought", ""),
        "original_images": item.get("image", []),
        "generated_images": extra,
        "prompt": "",
    }


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute overall and per-category accuracy from *results*."""
    total = len(results)
    correct = 0
    cat_correct: dict = defaultdict(int)
    cat_total: dict = defaultdict(int)

    for r in results:
        pred = r.get("prediction", "").lower().strip()
        gt = r.get("answer", "").lower().strip()
        cat = r.get("category", "unknown")
        cat_total[cat] += 1
        if pred == gt:
            correct += 1
            cat_correct[cat] += 1

    cat_accuracy = {cat: cat_correct[cat] / cat_total[cat] for cat in cat_total}

    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": correct,
        "category_accuracy": cat_accuracy,
        "category_counts": dict(cat_total),
    }


def log_metrics(metrics: Dict, label: str, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info(f"RESULTS — {label}")
    logger.info("=" * 60)
    logger.info(f"  Total   : {metrics['total_samples']}")
    logger.info(f"  Correct : {metrics['correct_samples']}")
    logger.info(f"  Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info("")
    logger.info("  Per-category accuracy:")
    for cat, acc in sorted(metrics["category_accuracy"].items()):
        n = metrics["category_counts"].get(cat, 0)
        logger.info(f"    {cat:35s}: {acc:6.2%}  ({n} samples)")
    logger.info("=" * 60)


# ===========================================================================
# Core evaluation function (runs on one GPU)
# ===========================================================================

def evaluate(
    data: List[Dict],
    model_type: str,
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    gen_dir: Optional[Path],
    output_path_baseline: Path,
    output_path_augmented: Path,
    device: str = "cuda:0",
) -> Tuple[List[Dict], List[Dict]]:
    """Run both methods for every sample in *data*.

    The model is loaded once and reused for both inference passes.
    Returns (baseline_results, augmented_results).
    """
    logger = logging.getLogger(__name__)

    if batch_size != 1:
        logger.warning("batch_size > 1 not supported; forcing to 1.")
        batch_size = 1

    model, processor = load_model_and_processor(model_type, model_path, device)

    baseline_results: List[Dict] = []
    augmented_results: List[Dict] = []

    n_with_gen = sum(1 for item in data if get_generated_images(item["index"], gen_dir))
    logger.info(
        f"[{device}] {len(data)} samples total, "
        f"{n_with_gen} have generated images, "
        f"{len(data) - n_with_gen} will use baseline images for both methods."
    )

    n_batches = math.ceil(len(data) / batch_size)
    for i in tqdm(range(0, len(data), batch_size), total=n_batches, desc=f"[{device}]"):
        batch_data = data[i : i + batch_size]

        # ---- Method A: Baseline (original images only) ----
        try:
            no_extra = [[] for _ in batch_data]
            b_inputs, b_prompts = prepare_batch(batch_data, no_extra, processor)
            b_outputs = run_inference(b_inputs, model, processor, max_new_tokens)
            baseline_results.extend(
                postprocess(batch_data, b_outputs, b_prompts, no_extra, "baseline")
            )
        except Exception as exc:
            logger.error(
                f"[baseline] Error on idx={batch_data[0].get('index','?')}: {exc}",
                exc_info=True,
            )
            for item in batch_data:
                baseline_results.append(_error_result(item, exc, "baseline", []))

        # ---- Method B: Augmented (original + generated images) ----
        try:
            gen_extra = [
                get_generated_images(item["index"], gen_dir) for item in batch_data
            ]
            a_inputs, a_prompts = prepare_batch(batch_data, gen_extra, processor)
            a_outputs = run_inference(a_inputs, model, processor, max_new_tokens)
            augmented_results.extend(
                postprocess(batch_data, a_outputs, a_prompts, gen_extra, "augmented")
            )
        except Exception as exc:
            logger.error(
                f"[augmented] Error on idx={batch_data[0].get('index','?')}: {exc}",
                exc_info=True,
            )
            for item, extra in zip(batch_data, gen_extra):
                augmented_results.append(_error_result(item, exc, "augmented", extra))

    for path, results in [
        (output_path_baseline, baseline_results),
        (output_path_augmented, augmented_results),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(results)} results → {path}")

    return baseline_results, augmented_results


# ===========================================================================
# Multi-GPU worker
# ===========================================================================

def _worker(
    gpu_id: str,
    data_shard: List[Dict],
    model_type: str,
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    gen_dir: Optional[str],
    out_baseline: Path,
    out_augmented: Path,
    log_file: Optional[str],
) -> None:
    """Entry point for each subprocess (one per GPU)."""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )

    logger = logging.getLogger(__name__)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    logger.info(f"[Worker {gpu_id}] Starting – {len(data_shard)} samples on {device}")
    evaluate(
        data_shard,
        model_type,
        model_path,
        batch_size,
        max_new_tokens,
        Path(gen_dir) if gen_dir else None,
        out_baseline,
        out_augmented,
        device,
    )
    logger.info(f"[Worker {gpu_id}] Done.")


# ===========================================================================
# Per-sample change analysis
# ===========================================================================

def analyze_changes(
    baseline_results: List[Dict],
    augmented_results: List[Dict],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Categorise every sample into one of five mutually exclusive groups.

    Groups
    ------
    degraded
        Baseline correct, augmented wrong.
        Generated images hurt performance on this sample.

    improved
        Baseline wrong, augmented correct.
        Generated images helped performance on this sample.

    correct_no_gen
        Baseline correct, no generated images were produced for this sample
        (augmented input == baseline input, augmented prediction == baseline).

    correct_with_gen
        Baseline correct, generated images existed, augmented also correct.

    always_wrong
        Both baseline and augmented wrong.

    Returns a dict with per-group sample lists, counts, and proportions.
    """
    # Index augmented results by sample index for O(1) lookup
    aug_by_idx = {r["index"]: r for r in augmented_results}

    groups: Dict[str, List[Dict]] = {
        "degraded": [],
        "improved": [],
        "correct_no_gen": [],
        "correct_with_gen": [],
        "always_wrong": [],
    }

    for b in baseline_results:
        idx = b["index"]
        a = aug_by_idx.get(idx)
        if a is None:
            logger.warning(f"No augmented result found for index {idx}, skipping.")
            continue

        b_correct = b.get("prediction", "").lower().strip() == b.get("answer", "").lower().strip()
        a_correct = a.get("prediction", "").lower().strip() == a.get("answer", "").lower().strip()
        has_gen = bool(a.get("generated_images"))  # non-empty list → images were added

        entry = {
            "index": idx,
            "category": b.get("category", "unknown"),
            "question": b.get("question", ""),
            "answer": b.get("answer", ""),
            "baseline_prediction": b.get("prediction", ""),
            "augmented_prediction": a.get("prediction", ""),
            "baseline_output": b.get("output", ""),
            "augmented_output": a.get("output", ""),
            "generated_images": a.get("generated_images", []),
        }

        if b_correct and not a_correct:
            groups["degraded"].append(entry)
        elif not b_correct and a_correct:
            groups["improved"].append(entry)
        elif b_correct and not has_gen:
            groups["correct_no_gen"].append(entry)
        elif b_correct and has_gen and a_correct:
            groups["correct_with_gen"].append(entry)
        else:
            # covers: baseline wrong + augmented wrong (with or without gen images)
            groups["always_wrong"].append(entry)

    total = sum(len(v) for v in groups.values())

    counts = {k: len(v) for k, v in groups.items()}
    proportions = {k: len(v) / total if total else 0.0 for k, v in groups.items()}

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PER-SAMPLE CHANGE ANALYSIS")
    logger.info("=" * 60)
    descriptions = {
        "degraded":       "Baseline ✓ → Augmented ✗  (gen images hurt)",
        "improved":       "Baseline ✗ → Augmented ✓  (gen images helped)",
        "correct_no_gen": "Baseline ✓, no gen images, Augmented ✓",
        "correct_with_gen":"Baseline ✓, has gen images, Augmented ✓",
        "always_wrong":   "Baseline ✗ & Augmented ✗  (both wrong)",
    }
    for key, desc in descriptions.items():
        n = counts[key]
        pct = proportions[key]
        logger.info(f"  {desc:<50s}: {n:4d}  ({pct:.1%})")
    logger.info(f"  {'Total':<50s}: {total:4d}")
    logger.info("=" * 60)

    return {
        "total": total,
        "counts": counts,
        "proportions": proportions,
        "descriptions": descriptions,
        "samples": {k: v for k, v in groups.items()},
    }


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-method QwenVL evaluation on MMSIBench (baseline vs augmented)."
    )
    # Model
    parser.add_argument("--model_type", type=str, default="qwen2.5-vl",
                        choices=["qwen2.5-vl", "qwen3-vl"])
    parser.add_argument("--model_path", type=str, required=True)
    # Data
    parser.add_argument("--data_dir", type=str,
                        default="datasets/evaluation/MMSIBench")
    parser.add_argument("--gen_dir", type=str, default=None,
                        help="Root of generated images, e.g. "
                             "generated_images/mmsibench/Qwen3-VL-4B-Instruct/flux2-klein-4B. "
                             "Sub-folder per sample index containing img_*.png files.")
    parser.add_argument("--limit", type=int, default=None)
    # Inference
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    # Output
    parser.add_argument("--output_dir", type=str, default="results/mmsibench")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Sub-folder name for this run (default: model_type).")
    parser.add_argument("--gen_model_name", type=str, default=None,
                        help="Human-readable name for the image generation model "
                             "(defaults to the last path component of --gen_dir).")

    args = parser.parse_args()

    # ---- Setup ----
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or args.model_type
    experiment_name = f"{model_name}_{timestamp}"
    output_dir = Path(args.output_dir).resolve() / model_name / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # ---- Load dataset ----
    data_dir = Path(args.data_dir).resolve()
    dataset = load_dataset(data_dir, limit=args.limit)

    # ---- Resolve gen_dir ----
    gen_dir: Optional[Path] = None
    if args.gen_dir:
        gen_dir = Path(args.gen_dir)
        if not gen_dir.is_absolute():
            gen_dir = Path.cwd() / gen_dir
        gen_dir = gen_dir.resolve()
        if not gen_dir.exists():
            logger.warning(f"gen_dir does not exist: {gen_dir}. Augmented method = baseline.")

    # ---- GPU setup ----
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("At least one CUDA device is required.")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_ids = (
        [x.strip() for x in cuda_visible.split(",") if x.strip()]
        if cuda_visible
        else [str(i) for i in range(n_gpu)]
    )

    n_with_gen = sum(1 for item in dataset if get_generated_images(item["index"], gen_dir))

    # Resolve generation model display name
    if args.gen_model_name:
        gen_model_display = args.gen_model_name
    elif gen_dir is not None:
        gen_model_display = gen_dir.name  # last path component
    else:
        gen_model_display = "(none — augmented = baseline)"

    logger.info("=" * 60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info("")
    planning_model_path = Path(args.model_path).resolve()
    planning_model_display = args.model_name or planning_model_path.name

    logger.info("  [Planning model — VQA inference]")
    logger.info(f"    model_name     : {planning_model_display}")
    logger.info(f"    model_type     : {args.model_type}")
    logger.info(f"    model_path     : {planning_model_path}")
    logger.info(f"    max_new_tokens : {args.max_new_tokens}")
    logger.info(f"    batch_size     : {args.batch_size}")
    logger.info("")
    logger.info("  [Generation model — augmented images]")
    logger.info(f"    gen_model_name : {gen_model_display}")
    logger.info(f"    gen_dir        : {gen_dir}")
    logger.info("")
    logger.info("  [Dataset]")
    logger.info(f"    data_dir       : {data_dir}")
    logger.info(f"    samples        : {len(dataset)}")
    logger.info(f"    w/ gen images  : {n_with_gen} / {len(dataset)}")
    logger.info("")
    logger.info("  [Runtime]")
    logger.info(f"    GPUs           : {gpu_ids}")
    logger.info(f"    output_dir     : {output_dir}")
    logger.info(f"    log_file       : {log_file}")
    logger.info("")
    logger.info("=" * 60)

    # ---- Save configuration.json ----
    config_record = {
        "planning_model": {
            "model_name": planning_model_display,
            "model_type": args.model_type,
            "model_path": str(planning_model_path),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "reasoning_model": {
            "model_name": planning_model_display,
            "model_type": args.model_type,
            "model_path": str(planning_model_path),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "generation_model": {
            "gen_model_name": gen_model_display,
            "gen_dir": str(gen_dir) if gen_dir else None,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "limit": args.limit,
            "total_samples": len(dataset),
            "samples_with_gen_images": n_with_gen,
        },
        "runtime": {
            "gpus": gpu_ids,
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "timestamp": timestamp,
        },
    }
    config_path = output_dir / "configuration.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False)
    logger.info(f"Configuration saved → {config_path}")

    # ---- Launch workers ----
    shards = chunk_dataset(dataset, len(gpu_ids))
    out_baseline_paths: List[Path] = []
    out_augmented_paths: List[Path] = []
    processes: List[mp.Process] = []

    for gpu_id, shard in zip(gpu_ids, shards):
        ob = output_dir / f"baseline_gpu{gpu_id}.json"
        oa = output_dir / f"augmented_gpu{gpu_id}.json"
        out_baseline_paths.append(ob)
        out_augmented_paths.append(oa)

        p = mp.Process(
            target=_worker,
            args=(
                gpu_id, shard,
                args.model_type, args.model_path,
                args.batch_size, args.max_new_tokens,
                str(gen_dir) if gen_dir else None,
                ob, oa,
                str(log_file),
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ---- Merge results ----
    logger.info("Merging results from all workers…")

    def merge(paths: List[Path]) -> List[Dict]:
        merged = []
        for path in paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
            else:
                logger.warning(f"Missing worker output: {path}")
        merged.sort(key=lambda r: r.get("index", 0))
        return merged

    baseline_all = merge(out_baseline_paths)
    augmented_all = merge(out_augmented_paths)

    # ---- Compute & log metrics ----
    metrics_baseline = compute_metrics(baseline_all)
    metrics_augmented = compute_metrics(augmented_all)

    log_metrics(metrics_baseline, "BASELINE  (original images only)", logger)
    log_metrics(metrics_augmented, "AUGMENTED (original + generated images)", logger)

    # Side-by-side summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  {'Method':<45} {'Accuracy':>8}  {'Correct':>8} / Total")
    logger.info(f"  {'-'*45}  {'-'*8}  {'-'*14}")
    logger.info(
        f"  {'Baseline  (original images only)':<45} "
        f"{metrics_baseline['overall_accuracy']:>8.2%}  "
        f"{metrics_baseline['correct_samples']:>8} / {metrics_baseline['total_samples']}"
    )
    logger.info(
        f"  {'Augmented (original + generated images)':<45} "
        f"{metrics_augmented['overall_accuracy']:>8.2%}  "
        f"{metrics_augmented['correct_samples']:>8} / {metrics_augmented['total_samples']}"
    )
    logger.info("=" * 60)

    # ---- Save outputs ----
    with open(output_dir / "eval_result_baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline_all, f, ensure_ascii=False, indent=2)
    with open(output_dir / "eval_result_augmented.json", "w", encoding="utf-8") as f:
        json.dump(augmented_all, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics_baseline, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics_augmented.json", "w", encoding="utf-8") as f:
        json.dump(metrics_augmented, f, ensure_ascii=False, indent=2)

    comparison = {
        "baseline": metrics_baseline,
        "augmented": metrics_augmented,
        "delta_overall_accuracy": metrics_augmented["overall_accuracy"] - metrics_baseline["overall_accuracy"],
    }
    with open(output_dir / "metrics_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # ---- Per-sample change analysis ----
    analysis = analyze_changes(baseline_all, augmented_all, logger)
    with open(output_dir / "analysis_changes.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
