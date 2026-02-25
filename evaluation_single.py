"""
Baseline reasoning evaluation on MMSIBench.

Loads a QwenVL model and answers every question using all provided input images,
then computes per-category and overall accuracy.

Usage
-----
Single-GPU (auto):
    python evaluation.py \\
        --model_type qwen2.5-vl \\
        --model_path checkpoints/Qwen2.5-VL-3B-Instruct \\
        --data_dir datasets/evaluation/MMSIBench

Multi-GPU (auto-shards dataset across all visible GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py \\
        --model_type qwen3-vl \\
        --model_path checkpoints/Qwen3-VL-4B-Instruct \\
        --data_dir datasets/evaluation/MMSIBench

Limit samples (quick test):
    python evaluation.py --model_type qwen2.5-vl \\
        --model_path checkpoints/Qwen2.5-VL-3B-Instruct \\
        --data_dir datasets/evaluation/MMSIBench --limit 20
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
    """Split *dataset* into *num_shards* roughly equal parts."""
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [
        dataset[start : start + chunk_size]
        for start in range(0, len(dataset), chunk_size)
    ]


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

def build_user_message(item: Dict[str, Any]) -> Dict:
    """Build a Qwen-VL chat message from a dataset *item*.

    All images for the sample are prepended before the question text, which
    mimics the multi-image setup required by MMSIBench.
    """
    image_contents = [
        {"type": "image", "image": img_path}
        for img_path in item["image"]
    ]
    question_text = QUESTION_TEMPLATE.format(Question=item["question"])
    text_content = {"type": "text", "text": f"{question_text}\n{ANSWER_INSTRUCTION}"}

    return {"role": "user", "content": image_contents + [text_content]}


def prepare_batch(
    batch_data: List[Dict],
    processor: Any,
) -> Tuple[Dict, List[str]]:
    """Tokenise a batch of samples for inference.

    Returns
    -------
    batch_inputs : dict of tensors ready for ``model.generate``
    prompts_text : list of raw prompt strings (for logging / output)
    """
    from qwen_vl_utils import process_vision_info

    batch_messages = [[build_user_message(item)] for item in batch_data]

    prompts_text = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    # Collect all images across the batch
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
) -> List[Dict]:
    """Package raw outputs into result dicts."""
    results = []
    for item, output, prompt in zip(batch_data, outputs, prompts):
        prediction = extract_answer_letter(output)
        results.append({
            "index": item.get("index", ""),
            "category": item.get("category", "unknown"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "prediction": prediction,
            "output": output,
            "thought_gt": item.get("thought", ""),
            "prompt": prompt,
        })
    return results


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


def log_metrics(metrics: Dict, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total   : {metrics['total_samples']}")
    logger.info(f"Correct : {metrics['correct_samples']}")
    logger.info(f"Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info("")
    logger.info("Per-category accuracy:")
    for cat, acc in sorted(metrics["category_accuracy"].items()):
        n = metrics["category_counts"].get(cat, 0)
        logger.info(f"  {cat:35s}: {acc:6.2%}  ({n} samples)")
    logger.info("=" * 60)


# ===========================================================================
# Core evaluation function (runs on one GPU)
# ===========================================================================

def evaluate(
    data: List[Dict],
    model_type: str,
    model_path: str,
    batch_size: int,
    output_path: Path,
    device: str = "cuda:0",
) -> List[Dict]:
    """Evaluate *model_type* on *data*, save to *output_path*, return results."""
    logger = logging.getLogger(__name__)

    # Batch size must be 1 for multi-image samples (variable image count)
    if batch_size != 1:
        logger.warning("batch_size > 1 not supported for multi-image samples; forcing to 1.")
        batch_size = 1

    model, processor = load_model_and_processor(model_type, model_path, device)

    all_results: List[Dict] = []
    n_batches = math.ceil(len(data) / batch_size)

    for i in tqdm(range(0, len(data), batch_size), total=n_batches, desc=f"[{device}] inference"):
        batch_data = data[i : i + batch_size]
        try:
            batch_inputs, prompts = prepare_batch(batch_data, processor)
            outputs = run_inference(batch_inputs, model, processor)
            all_results.extend(postprocess(batch_data, outputs, prompts))
        except Exception as exc:
            logger.error(f"Error on sample idx={batch_data[0].get('index','?')}: {exc}", exc_info=True)
            # Append placeholder so index alignment is preserved
            for item in batch_data:
                all_results.append({
                    "index": item.get("index", ""),
                    "category": item.get("category", "unknown"),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "prediction": "",
                    "output": f"ERROR: {exc}",
                    "thought_gt": item.get("thought", ""),
                    "prompt": "",
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")

    return all_results


# ===========================================================================
# Multi-GPU worker
# ===========================================================================

def _worker(
    gpu_id: str,
    data_shard: List[Dict],
    model_type: str,
    model_path: str,
    batch_size: int,
    output_path: Path,
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
    evaluate(data_shard, model_type, model_path, batch_size, output_path, device)
    logger.info(f"[Worker {gpu_id}] Done.")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline QwenVL reasoning evaluation on MMSIBench."
    )
    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2.5-vl",
        choices=["qwen2.5-vl", "qwen3-vl"],
        help="Model family to use (default: qwen2.5-vl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/evaluation/MMSIBench",
        help="Root directory of the MMSIBench dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of samples to evaluate (useful for smoke tests)",
    )
    # Inference
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size (forced to 1 for multi-image samples)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per sample",
    )
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_single/mmsibench",
        help="Root directory for saving results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name used for the results sub-folder (default: model_type)",
    )

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
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_dataset(data_dir, limit=args.limit)
    logger.info(f"Loaded {len(dataset)} samples")

    # ---- GPU setup ----
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("At least one CUDA device is required.")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]

    # ---- Log configuration ----
    logger.info("=" * 60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  model_type   : {args.model_type}")
    logger.info(f"  model_path   : {args.model_path}")
    logger.info(f"  model_name   : {model_name}")
    logger.info(f"  data_dir     : {data_dir}")
    logger.info(f"  samples      : {len(dataset)}")
    logger.info(f"  batch_size   : {args.batch_size}")
    logger.info(f"  max_new_tok  : {args.max_new_tokens}")
    logger.info(f"  GPUs         : {gpu_ids}")
    logger.info(f"  output_dir   : {output_dir}")
    logger.info("=" * 60)

    # ---- Shard dataset across GPUs ----
    shards = chunk_dataset(dataset, len(gpu_ids))
    output_paths: List[Path] = []
    processes: List[mp.Process] = []

    for idx, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        out_path = output_dir / f"results_gpu{gpu_id}.json"
        output_paths.append(out_path)

        p = mp.Process(
            target=_worker,
            args=(
                gpu_id,
                shard,
                args.model_type,
                args.model_path,
                args.batch_size,
                out_path,
                str(log_file),
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ---- Merge worker outputs ----
    logger.info("Merging results from all workers…")
    all_results: List[Dict] = []
    for path in output_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
        else:
            logger.warning(f"Missing worker output: {path}")

    # Sort by index to ensure deterministic order
    all_results.sort(key=lambda r: r.get("index", 0))

    # ---- Compute and log metrics ----
    metrics = compute_metrics(all_results)
    log_metrics(metrics, logger)

    # ---- Save final outputs ----
    with open(output_dir / "eval_result.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
