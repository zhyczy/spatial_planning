"""
Step 2 (Video): Generate videos with Wan2.1-VACE-14B (multi-GPU)

For each planning-model result in predicted_instructions/{planning_model}/results.jsonl:
  - Load the input images recorded in image_paths (as reference frames for R2V).
  - For each instruction/description, call WanVace once (conditioned on ALL valid
    reference images + the instruction text) to generate one output video.
  - If a sample has no instructions, create an empty output folder.

This script replaces image_generation.py's Flux2Klein image generator with
Wan2.1-VACE-14B video generator.  The generation task is R2V (Reference-to-Video):
all original QA images are passed as reference images and the instruction
(description from the planner model) is used as the text prompt.

Samples are sharded evenly across all available GPUs (or those in
CUDA_VISIBLE_DEVICES). Each GPU worker loads its own copy of WanVace.

Output layout:
  generated_videos/
    {dataset}/
      {planning_model}/
        {generation_model}/
          {question_id}/
            vid_0.mp4
            vid_1.mp4
            ...

Usage:
  # Single planning-model
  python video_generation.py \\
    --planning_model Qwen3-VL-4B-Instruct \\
    --vace_ckpt checkpoints/Wan2.1-VACE-14B

  # All planning-models found under predicted_instructions/
  python video_generation.py --all_planning_models

  # Limit to 2 GPUs
  python video_generation.py --all_planning_models --num_gpus 2

  # Specific GPUs via environment variable
  CUDA_VISIBLE_DEVICES=0,2 python video_generation.py --all_planning_models

  # Re-generate everything (ignore existing outputs)
  python video_generation.py --planning_model Qwen3-VL-4B-Instruct --no_skip_existing

  # Override video resolution
  python video_generation.py --planning_model Qwen3-VL-4B-Instruct --size 480p
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.io as tvio
from PIL import Image


# ---------------------------------------------------------------------------
# Helper: Add VACE to import path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
VACE_DIR = SCRIPT_DIR / "external" / "VACE" / "vace"
WAN21_DIR = SCRIPT_DIR / "external" / "Wan21"


def _add_vace_to_path():
    """Add the VACE vace/ directory to sys.path so models can be imported."""
    vace_str = str(VACE_DIR)
    if vace_str not in sys.path:
        sys.path.insert(0, vace_str)
    # Add VACE repository root so its relative imports resolve.
    wan_root = SCRIPT_DIR / "external" / "VACE"
    wan_str = str(wan_root)
    if wan_str not in sys.path:
        sys.path.insert(0, wan_str)

    # Prefer Wan2.1 runtime package for VACE-Wan2.1 models.
    if WAN21_DIR.is_dir():
        wan21_str = str(WAN21_DIR)
        if wan21_str not in sys.path:
            sys.path.insert(0, wan21_str)


# ---------------------------------------------------------------------------
# Helper: split list into N roughly equal chunks
# ---------------------------------------------------------------------------
def chunk_dataset(samples: list, n: int) -> List[list]:
    size = len(samples)
    if size == 0 or n <= 0:
        return []
    k, rem = divmod(size, n)
    chunks, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < rem else 0)
        if start < size:
            chunks.append(samples[start:end])
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Helper: resolve image path, remapping dataset/ → datasets/ if needed
# ---------------------------------------------------------------------------
def _resolve_image_path(p: str) -> str:
    if os.path.isfile(p):
        return p
    remapped = p.replace("/dataset/", "/datasets/")
    if remapped != p and os.path.isfile(remapped):
        return remapped
    return p


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------
def run_worker(
    gpu_id: str,
    samples: list,           # list of dicts with keys: planning_model, id, dataset,
                             #   image_paths, instructions
    vace_ckpt: str,
    model_name: str,
    output_root: str,
    size: str,
    frame_num: int,
    sample_steps: int,
    sample_shift: float,
    guide_scale: float,
    skip_existing: bool,
    log_file: Optional[str] = None,
):
    """Subprocess entry point: load WanVace on one GPU and process a shard."""
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

    device_id = int(gpu_id)
    torch.cuda.set_device(device_id)

    logger.info("=" * 60)
    logger.info(f"Device        : cuda:{gpu_id}")
    logger.info(f"Samples       : {len(samples)}")
    logger.info(f"Checkpoint    : {vace_ckpt}")
    logger.info(f"Model name    : {model_name}")
    logger.info(f"Size          : {size}")
    logger.info(f"Frame num     : {frame_num}")
    logger.info("=" * 60)

    if not samples:
        logger.info("No samples assigned to this worker; skip model loading.")
        return

    if not any(sample.get("instructions", []) for sample in samples):
        logger.info("All assigned samples have zero instructions; skip model loading.")
        return

    # ── Load VACE model ──────────────────────────────────────────────────────
    _add_vace_to_path()

    from models.wan import WanVace  # noqa: PLC0415
    from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS  # noqa: PLC0415

    wan_import_error = None
    wan_loaded = False
    for candidate in [WAN21_DIR, None]:
        try:
            if candidate is not None and candidate.is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)

            import wan  # noqa: PLC0415
            logger.info(f"Using wan package from: {Path(wan.__file__).resolve()}")
            wan_loaded = True
            break
        except ImportError as e:
            wan_import_error = e

    if not wan_loaded:
        raise ImportError(
            "Unable to import 'wan' for VACE-Wan2.1. Tried external/Wan21 and current sys.path."
        ) from wan_import_error

    cfg = WAN_CONFIGS[model_name]
    size_hw = SIZE_CONFIGS[size]  # (height, width)

    logger.info(f"Loading WanVace ({model_name}) ...")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=vace_ckpt,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    logger.info("WanVace model loaded.")

    out_root = Path(output_root)
    gen_model_name = Path(vace_ckpt).name   # e.g. "Wan2.1-VACE-14B"
    t_start = time.time()

    # ── Process each sample ──────────────────────────────────────────────────
    for i, sample in enumerate(samples):
        sample_id      = sample["id"]
        dataset        = sample["dataset"]
        planning_model = sample["planning_model"]
        image_paths    = sample.get("image_paths", [])
        instructions   = sample.get("instructions", [])

        # generated_videos/{dataset}/{planning_model}/{gen_model}/{id}/
        out_dir = out_root / dataset / planning_model / gen_model_name / str(sample_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        # No instructions → save meta and skip generation
        if not instructions:
            logger.info(f"[{i+1}/{len(samples)}] Sample {sample_id} ({dataset}/{planning_model}): no instructions.")
            meta_path = out_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as _f:
                json.dump({
                    "id": sample_id,
                    "question": sample.get("question", ""),
                    "image_paths": image_paths,
                    "instructions": [],
                }, _f, indent=2, ensure_ascii=False)
            continue

        # Resolve all valid QA images as reference images
        ref_image_paths: List[str] = []
        for p in image_paths:
            resolved = _resolve_image_path(p)
            if os.path.isfile(resolved):
                ref_image_paths.append(resolved)

        if not ref_image_paths:
            logger.warning(
                f"[{i+1}/{len(samples)}] Sample {sample_id}: no valid source images, skipping."
            )
            continue

        logger.info(
            f"[{i+1}/{len(samples)}] Sample {sample_id}: using {len(ref_image_paths)} reference image(s)."
        )

        # One generation call per instruction
        for idx, instruction in enumerate(instructions):
            out_path = out_dir / f"vid_{idx}.mp4"

            if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                logger.info(f"  Skip existing: {out_path.name}")
                continue

            t0 = time.time()
            logger.info(
                f"[{i+1}/{len(samples)}] Sample {sample_id} | instr {idx}: "
                f"{instruction[:90]}{'...' if len(instruction) > 90 else ''}"
            )

            try:
                # R2V: src_video=None (zero-initialized), src_mask=None (full mask),
                #       src_ref_images=[ref_image_paths] (all original QA images)
                src_video, src_mask, src_ref_images = wan_vace.prepare_source(
                    [None],               # no source video → zeros
                    [None],               # no source mask → full generation mask
                    [ref_image_paths],    # all valid reference images from original QA
                    frame_num,
                    size_hw,
                    f"cuda:{gpu_id}",
                )

                video = wan_vace.generate(
                    instruction,
                    src_video,
                    src_mask,
                    src_ref_images,
                    size=size_hw,
                    frame_num=frame_num,
                    shift=sample_shift,
                    sample_solver="unipc",
                    sampling_steps=sample_steps,
                    guide_scale=guide_scale,
                    seed=2025,
                    offload_model=True,
                )

                # video: [C, T, H, W], values in (-1, 1) → [T, H, W, C] uint8
                frames_uint8 = (
                    (video.clamp(-1, 1) + 1) / 2 * 255
                ).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu()
                tvio.write_video(str(out_path), frames_uint8, fps=cfg.sample_fps, video_codec="h264")
                logger.info(f"    Saved ({time.time()-t0:.1f}s) → {out_path}")

                # ── Downsample: one middle frame per 10-frame segment ────────
                # video shape: [C, T, H, W], values in (-1, 1)
                frames_dir = out_dir / f"vid_{idx}_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                num_frames = video.shape[1]
                segment_size = 10
                saved_frame_count = 0
                for seg_start in range(0, num_frames, segment_size):
                    seg_end = min(seg_start + segment_size, num_frames)
                    mid_idx = (seg_start + seg_end - 1) // 2
                    frame = video[:, mid_idx, :, :]          # [C, H, W]
                    frame = (frame.clamp(-1, 1) + 1) / 2 * 255
                    frame_np = frame.byte().permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                    img = Image.fromarray(frame_np)
                    img.save(str(frames_dir / f"frame_{saved_frame_count:03d}.png"))
                    saved_frame_count += 1
                logger.info(f"    Saved {saved_frame_count} frames → {frames_dir}")

                # ── Save per-video metadata ──────────────────────────────────
                vid_meta_path = out_dir / f"vid_{idx}_meta.json"
                with open(vid_meta_path, "w", encoding="utf-8") as _f:
                    json.dump({
                        "id": sample_id,
                        "question": sample.get("question", ""),
                        "image_paths": image_paths,
                        "instruction": instruction,
                    }, _f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"  Sample {sample_id}, instr {idx} FAILED: {e}", exc_info=True)

    elapsed = time.time() - t_start
    logger.info(f"Worker done — {len(samples)} samples in {elapsed:.1f}s")
    logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos with Wan2.1-VACE-14B conditioned on predicted instructions (multi-GPU)."
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
        "--vace_ckpt",
        type=str,
        default="checkpoints/Wan2.1-VACE-14B",
        help="Path to the Wan2.1-VACE-14B checkpoint directory.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-14B",
        choices=["vace-1.3B", "vace-14B"],
        help="VACE model variant to use (default: vace-14B).",
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
        default="generated_videos",
        help="Root folder for generated videos.",
    )

    # --- Video generation hyperparameters ---
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        choices=["720*1280", "1280*720", "480*832", "832*480", "480p", "720p"],
        help="Video resolution (default: 480p = 480×832).",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=80,
        help="Number of frames to generate (default: 80 = 5s at 16fps).",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Number of diffusion sampling steps (default: 50).",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=16.0,
        help="Noise schedule shift for flow matching (default: 16.0).",
    )
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0).",
    )

    # --- Multi-GPU ---
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use. -1 = all available (respects CUDA_VISIBLE_DEVICES).",
    )

    # --- Misc ---
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-generate even if the output video already exists.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
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

    vace_ckpt  = resolve(args.vace_ckpt)
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

    gen_model_name = vace_ckpt.name   # e.g. "Wan2.1-VACE-14B"

    # ── Read all samples from results.jsonl files ────────────────────────────
    all_samples: list = []
    for pm in planning_models:
        results_jsonl = instr_root / pm / "results.jsonl"
        if not results_jsonl.is_file():
            logger.warning(f"results.jsonl not found: {results_jsonl}")
            dataset_name = instr_root.name
            pre_gen_dir = out_root / dataset_name / pm / gen_model_name
            pre_gen_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Pre-created output dir: {pre_gen_dir}")
            continue
        with open(results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["planning_model"] = pm
                    all_samples.append(rec)

    logger.info(f"Total samples   : {len(all_samples)}")

    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
        logger.info(f"max_samples     : {args.max_samples} (truncated to {len(all_samples)})")

    # Pre-create empty output folders for samples with no instructions, then
    # remove them from generation tasks so we don't load model unnecessarily.
    samples_to_generate: list = []
    no_instruction_count = 0
    for rec in all_samples:
        instructions = rec.get("instructions", [])
        sample_id = rec["id"]
        dataset = rec["dataset"]
        planning_model = rec["planning_model"]
        out_dir = out_root / dataset / planning_model / gen_model_name / str(sample_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        if instructions:
            samples_to_generate.append(rec)
        else:
            no_instruction_count += 1
            meta_path = out_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "id": sample_id,
                    "question": rec.get("question", ""),
                    "image_paths": rec.get("image_paths", []),
                    "instructions": [],
                }, f, indent=2, ensure_ascii=False)

    logger.info(f"Samples with instructions    : {len(samples_to_generate)}")
    logger.info(f"Samples with zero instruction: {no_instruction_count}")

    if len(samples_to_generate) == 0:
        logger.info("No instructions to generate. Skip loading video generation model.")

        # Still write per-question video-count statistics.
        group_map: dict = defaultdict(list)
        for sample in all_samples:
            group_map[(sample["dataset"], sample["planning_model"])].append(sample)

        for (dataset, pm), samples in sorted(group_map.items()):
            dest_dir = out_root / dataset / pm / gen_model_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            count_to_ids: dict = defaultdict(list)
            for rec in samples:
                sample_id = rec["id"]
                vid_dir = dest_dir / str(sample_id)
                n_vids = len(list(vid_dir.glob("vid_*.mp4"))) if vid_dir.is_dir() else 0
                count_to_ids[n_vids].append(sample_id)

            stats_path = dest_dir / "video_counts.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in sorted(count_to_ids.items())},
                    f,
                    indent=2,
                )
            logger.info(
                f"Stats written → {stats_path}  "
                f"(total {len(samples)} samples, "
                f"distribution: {{ {', '.join([f'{k}: {len(v)}' for k, v in sorted(count_to_ids.items())])} }})"
            )
        return

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
    logger.info("VIDEO GENERATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"VACE checkpoint : {vace_ckpt}")
    logger.info(f"Model name      : {args.model_name}")
    logger.info(f"Instructions    : {instr_root}")
    logger.info(f"Output root     : {out_root}")
    logger.info(f"Samples         : {len(samples_to_generate)}")
    logger.info(f"GPUs in use     : {n_gpu}  {all_gpu_ids}")
    logger.info(f"Resolution      : {args.size}")
    logger.info(f"Frames          : {args.frame_num}")
    logger.info(f"Steps           : {args.sample_steps}")
    logger.info(f"Guide scale     : {args.guide_scale}")
    logger.info(f"Skip existing   : {not args.no_skip_existing}")
    logger.info("=" * 60)

    # ── Shard and spawn ──────────────────────────────────────────────────────
    chunks    = chunk_dataset(samples_to_generate, n_gpu)
    processes = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        if not chunk:
            logger.info(f"Skip worker {idx} on GPU {gpu_id}: no samples in shard")
            continue
        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                chunk,
                str(vace_ckpt),
                args.model_name,
                str(out_root),
                args.size,
                args.frame_num,
                args.sample_steps,
                args.sample_shift,
                args.guide_scale,
                not args.no_skip_existing,
                str(log_file),
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {idx} on GPU {gpu_id} ({len(chunk)} samples)")

    for p in processes:
        p.join()

    failed_workers = [p.pid for p in processes if p.exitcode not in (0, None)]
    if failed_workers:
        logger.error(f"Worker failures detected (PIDs): {failed_workers}")
        raise SystemExit(1)

    logger.info("All workers finished. Done.")

    # ── Write per-question video-count statistics ────────────────────────────
    group_map: dict = defaultdict(list)
    for sample in all_samples:
        group_map[(sample["dataset"], sample["planning_model"])].append(sample)

    for (dataset, pm), samples in sorted(group_map.items()):
        dest_dir = out_root / dataset / pm / gen_model_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        count_to_ids: dict = defaultdict(list)
        for rec in samples:
            sample_id = rec["id"]
            vid_dir = dest_dir / str(sample_id)
            n_vids = len(list(vid_dir.glob("vid_*.mp4"))) if vid_dir.is_dir() else 0
            count_to_ids[n_vids].append(sample_id)

        stats_path = dest_dir / "video_counts.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in sorted(count_to_ids.items())},
                f, indent=2,
            )
        logger.info(
            f"Stats written → {stats_path}  "
            f"(total {len(samples)} samples, "
            f"distribution: { {k: len(v) for k, v in sorted(count_to_ids.items())} })"
        )


if __name__ == "__main__":
    main()
