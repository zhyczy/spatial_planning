"""
Step 2: Image Generation with FLUX2-klein-4B (multi-GPU)

For each planning-model result in predicted_instructions/{planning_model}/results.jsonl:
  - Load the input images recorded in image_paths.
  - For each instruction, call Flux2KleinPipeline once (conditioned on all input images
    + the instruction text) to generate one output image.
  - If a sample has no instructions, create an empty output folder.

Samples are sharded evenly across all available GPUs (or those in
CUDA_VISIBLE_DEVICES). Each GPU worker loads its own copy of Flux2Pipeline.
No merge step is needed because outputs are image files written directly to disk.

Output layout:
  generated_images/
    {dataset}/
      {planning_model}/
        {generation_model}/
          {question_id}/
            img_0.png
            img_1.png
            ...

Usage:
  # Single planning-model
  python image_generation.py \
    --planning_model Qwen3-VL-4B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B

  # All planning-models found under predicted_instructions/
  python image_generation.py --all_planning_models

  # Limit to 2 GPUs
  python image_generation.py --all_planning_models --num_gpus 2

  # Specific GPUs via environment variable
  CUDA_VISIBLE_DEVICES=0,2 python image_generation.py --all_planning_models

  # Re-generate everything (ignore existing outputs)
  python image_generation.py --planning_model Qwen3-VL-4B-Instruct --no_skip_existing
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helper: split list into N roughly equal chunks (mirrors generate_image_instructions.py)
# ---------------------------------------------------------------------------
def chunk_dataset(samples: list, n: int) -> List[list]:
    size = max(1, len(samples))
    k, rem = divmod(size, n)
    chunks, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < rem else 0)
        if start < size:
            chunks.append(samples[start:end])
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Helper: load PIL images from paths, skip missing/broken files
# ---------------------------------------------------------------------------
def _resolve_image_path(p: str) -> str:
    """Try the original path first; if missing, remap dataset/ → datasets/."""
    if os.path.isfile(p):
        return p
    # Handle directory rename: dataset/ → datasets/
    remapped = p.replace("/dataset/", "/datasets/")
    if remapped != p and os.path.isfile(remapped):
        return remapped
    return p  # return original so the caller can log the not-found path


def load_images(image_paths: List[str], logger: logging.Logger) -> List[Image.Image]:
    images = []
    for p in image_paths:
        resolved = _resolve_image_path(p)
        if not os.path.isfile(resolved):
            logger.warning(f"Image not found, skipping: {p}")
            continue
        try:
            images.append(Image.open(resolved).convert("RGB"))
        except Exception as e:
            logger.warning(f"Failed to open {resolved}: {e}")
    return images


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------
def run_worker(
    gpu_id: str,
    samples: list,           # list of dicts with keys: planning_model, id, dataset,
                             #   image_paths, instructions
    flux_ckpt: str,
    output_root: str,
    num_inference_steps: int,
    height: Optional[int],
    width: Optional[int],
    skip_existing: bool,
    log_file: Optional[str] = None,
):
    """Subprocess entry point: load Flux2Pipeline on one GPU and process a shard."""
    # ── Per-process logging ──────────────────────────────────────────────────
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    logger.info("=" * 60)
    logger.info(f"Device        : {device}")
    logger.info(f"Samples       : {len(samples)}")
    logger.info(f"Checkpoint    : {flux_ckpt}")
    logger.info("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    from diffusers import Flux2KleinPipeline  # noqa: PLC0415

    logger.info(f"Loading Flux2KleinPipeline ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        flux_ckpt,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("Model loaded.")

    out_root = Path(output_root)
    gen_model = Path(flux_ckpt).name   # e.g. "flux2-klein-4B"
    t_start = time.time()

    # ── Process each sample ──────────────────────────────────────────────────
    for i, sample in enumerate(samples):
        sample_id      = sample["id"]
        dataset        = sample["dataset"]
        planning_model = sample["planning_model"]
        image_paths    = sample.get("image_paths", [])
        instructions   = sample.get("instructions", [])

        # generated_images/{dataset}/{planning_model}/{gen_model}/{id}/
        out_dir = out_root / dataset / planning_model / gen_model / str(sample_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # No instructions → empty folder, skip generation
        if not instructions:
            logger.info(f"[{i+1}/{len(samples)}] Sample {sample_id} ({dataset}/{planning_model}): no instructions.")
            continue

        # Load source conditioning images
        src_images = load_images(image_paths, logger)
        if not src_images:
            logger.warning(
                f"[{i+1}/{len(samples)}] Sample {sample_id}: no valid source images, skipping."
            )
            continue

        # One generation call per instruction
        for idx, instruction in enumerate(instructions):
            out_path = out_dir / f"img_{idx}.png"

            if skip_existing and out_path.is_file():
                logger.info(f"  Skip existing: {out_path.name}")
                continue

            t0 = time.time()
            logger.info(
                f"[{i+1}/{len(samples)}] Sample {sample_id} | instr {idx}: "
                f"{instruction[:90]}{'...' if len(instruction) > 90 else ''}"
            )

            try:
                call_kwargs: dict = dict(
                    image=src_images,
                    prompt=instruction,
                    num_inference_steps=num_inference_steps,
                )
                if height is not None:
                    call_kwargs["height"] = height
                if width is not None:
                    call_kwargs["width"] = width

                result = pipe(**call_kwargs)
                result.images[0].save(out_path)
                logger.info(f"    Saved ({time.time()-t0:.1f}s) → {out_path}")

            except Exception as e:
                logger.error(f"  Sample {sample_id}, instr {idx} FAILED: {e}")

    elapsed = time.time() - t_start
    logger.info(f"Worker done — {len(samples)} samples in {elapsed:.1f}s")
    logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with FLUX2-dev conditioned on predicted instructions (multi-GPU)."
    )

    # --- Planning model selection ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--planning_model",
        type=str,
        help="Name of the planning model (folder under predicted_instructions/).",
    )
    group.add_argument(
        "--all_planning_models",
        action="store_true",
        help="Process all planning-model folders found under predicted_instructions/.",
    )

    # --- Paths ---
    parser.add_argument(
        "--flux_ckpt",
        type=str,
        default="checkpoints/flux2-klein-4B",
        help="Path to the Flux2Klein checkpoint directory.",
    )
    parser.add_argument(
        "--predicted_instructions_root",
        type=str,
        default="predicted_instructions",
        help="Root folder containing planning-model result subfolders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="generated_images",
        help="Root folder for generated images.",
    )

    # --- Generation hyperparameters ---
    parser.add_argument(
        "--num_inference_steps", type=int, default=28,
        help="Number of denoising steps (default: 28).",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Output image height in pixels (default: model default).",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Output image width in pixels (default: model default).",
    )

    # --- Multi-GPU ---
    parser.add_argument(
        "--num_gpus", type=int, default=-1,
        help="Number of GPUs to use. -1 = all available (respects CUDA_VISIBLE_DEVICES).",
    )

    # --- Misc ---
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-generate even if the output image already exists.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Process only first N samples (useful for quick testing). -1 = all.",
    )

    return parser.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    script_dir = Path(__file__).parent

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else script_dir / q

    flux_ckpt  = resolve(args.flux_ckpt)
    instr_root = resolve(args.predicted_instructions_root)
    out_root   = resolve(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Logging (main process) ───────────────────────────────────────────────
    log_file = out_root / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    # ── Collect planning models ──────────────────────────────────────────────
    if args.all_planning_models:
        planning_models = sorted([d.name for d in instr_root.iterdir() if d.is_dir()])
        if not planning_models:
            logger.error(f"No subfolders found under {instr_root}")
            sys.exit(1)
    else:
        planning_models = [args.planning_model]
    logger.info(f"Planning models : {planning_models}")

    # ── Read all samples from results.jsonl files ────────────────────────────
    all_samples: list = []
    for pm in planning_models:
        results_jsonl = instr_root / pm / "results.jsonl"
        if not results_jsonl.is_file():
            logger.warning(f"results.jsonl not found: {results_jsonl}")
            continue
        with open(results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["planning_model"] = pm   # tag for output path
                    all_samples.append(rec)

    logger.info(f"Total samples   : {len(all_samples)}")

    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
        logger.info(f"max_samples     : {args.max_samples} (truncated to {len(all_samples)})")

    # ── Determine GPU IDs ────────────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    if n_available == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]

    if args.num_gpus > 0:
        all_gpu_ids = all_gpu_ids[: args.num_gpus]

    n_gpu = len(all_gpu_ids)

    logger.info("=" * 60)
    logger.info("IMAGE GENERATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Checkpoint      : {flux_ckpt}")
    logger.info(f"Instructions    : {instr_root}")
    logger.info(f"Output root     : {out_root}")
    logger.info(f"Samples         : {len(all_samples)}")
    logger.info(f"GPUs in use     : {n_gpu}  {all_gpu_ids}")
    logger.info(f"Steps           : {args.num_inference_steps}")
    logger.info(f"Skip existing   : {not args.no_skip_existing}")
    logger.info("=" * 60)

    # ── Shard and spawn ──────────────────────────────────────────────────────
    chunks    = chunk_dataset(all_samples, n_gpu)
    processes = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                chunk,
                str(flux_ckpt),
                str(out_root),
                args.num_inference_steps,
                args.height,
                args.width,
                not args.no_skip_existing,
                str(log_file),
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {idx} on GPU {gpu_id} ({len(chunk)} samples)")

    for p in processes:
        p.join()

    logger.info("All workers finished. Done.")

    # ── Write per-question image-count statistics ────────────────────────────
    from collections import defaultdict  # noqa: PLC0415
    gen_model = flux_ckpt.name

    # Group samples by (dataset, planning_model) to write one stats file each
    dest_dirs: list = []
    group_map: dict = defaultdict(list)
    for sample in all_samples:
        group_map[(sample["dataset"], sample["planning_model"])].append(sample)

    for (dataset, pm), samples in sorted(group_map.items()):
        dest_dir = out_root / dataset / pm / gen_model
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_dirs.append(dest_dir)

        count_to_ids: dict = defaultdict(list)   # n_generated → [qid, ...]
        needed_to_ids: dict = defaultdict(list)  # n_needed    → [qid, ...]
        total_needed = 0
        total_generated = 0

        for sample in samples:
            qid   = sample["id"]
            n_need = len(sample.get("instructions", []))
            q_dir  = dest_dir / str(qid)
            n_gen  = len(list(q_dir.glob("img_*.png"))) if q_dir.is_dir() else 0

            count_to_ids[n_gen].append(qid)
            needed_to_ids[n_need].append(qid)
            total_needed    += n_need
            total_generated += n_gen

        n_total = len(samples)

        summary = {}
        buckets = {}
        for k, v in sorted(count_to_ids.items()):
            label = f"{k} generation"
            summary[f"number of {label}"] = len(v)
            summary[f"ratio of {label}"]  = round(len(v) / n_total, 4)
            buckets[label] = sorted(v)

        stats = {"summary": summary, **buckets}

        stats_path = dest_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"[{dataset}/{pm}/{gen_model}] "
            f"questions={n_total} | needed={total_needed} | generated={total_generated} | "
            + " | ".join(f"{k}: {v}" for k, v in summary.items())
            + f" → {stats_path}"
        )

    # ── Copy run.log into each dest_dir ─────────────────────────────────────
    for handler in logging.getLogger().handlers:
        handler.flush()
    for dest_dir in dest_dirs:
        dst_log = dest_dir / "run.log"
        shutil.copy2(log_file, dst_log)
        logger.info(f"Log copied      : {dst_log}")


if __name__ == "__main__":
    main()