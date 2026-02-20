"""
Main program: Load spatial reasoning questions + images from datasets,
call Qwen VL locally, and output image-generation instructions.

Usage examples:
  # MindCube dataset
  python generate_image_instructions.py \
    --dataset mindcube \
    --data_path datasets/evaluation/MindCube/MindCube_tinybench.jsonl \
    --image_root datasets/evaluation/MindCube \
    --model_path checkpoints/Qwen3-VL-4B-Instruct

  # SAT dataset
  python generate_image_instructions.py \
    --dataset sat \
    --data_path datasets/evaluation/SAT/test.json \
    --image_root datasets/evaluation/SAT \
    --model_path checkpoints/Qwen3-VL-4B-Instruct

  # VSIBench dataset (images loaded by scene from scannet/arkitscenes subfolders)
  python generate_image_instructions.py \
    --dataset vsibench \
    --data_path datasets/evaluation/vsibench/test.jsonl \
    --image_root datasets/evaluation/vsibench \
    --model_path checkpoints/Qwen3-VL-4B-Instruct

  # MMSIBench dataset
  python generate_image_instructions.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --model_path checkpoints/Qwen3-VL-4B-Instruct
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a visual planning assistant for spatial reasoning tasks.

You will be given:
  1. One or more images of a scene.
  2. A spatial reasoning question about that scene.

Your job is to decide whether the existing images are sufficient to answer the
question, or whether additional images need to be generated to provide missing
visual information.

Think step by step inside <thinking>...</thinking> tags:
  - What spatial information is needed to answer the question?
  - What information is already visible in the provided images?
  - What is still missing or ambiguous?
  - Would additional images help? If so, what kind?

Then output your instructions inside <instructions>...</instructions> tags.
Each individual instruction goes inside an <instruction>...</instruction> tag.

Rules for the instructions:
  - If the existing images are sufficient, output empty <instructions></instructions>.
  - Otherwise output 1–5 instructions (no more than 5).
  - Each instruction describes ONE specific additional image to generate.
  - Focus on what visual information is missing. Typical needs include:
      * Viewpoint change: a different camera angle or position to reveal hidden geometry
      * Occlusion removal: showing an object from a direction where it is not blocked
      * Object-state change: showing the object/scene from a clearer distance or zoom
      * Surrounding context: widening the field of view to reveal spatial relationships
  - Each instruction must be self-contained and directly usable as an image-generation prompt.
  - Do NOT include question text, answer choices, or meta-commentary.

Example output format:
<thinking>
The question asks what is behind the sofa. Images 1–4 show the front and sides
but not the back wall. I need a view from behind the sofa facing toward the room.
</thinking>
<instructions>
<instruction>A wide-angle view from behind the sofa, facing toward the center of the living room, revealing all objects between the sofa and the far wall.</instruction>
<instruction>A top-down overhead view of the entire living room, showing the positions of all furniture relative to each other.</instruction>
</instructions>"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def chunk_dataset(samples: list, n: int) -> List[list]:
    """Split *samples* into *n* roughly equal-sized sublists."""
    size = max(1, len(samples))
    k, rem = divmod(size, n)
    chunks, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < rem else 0)
        if start < size:
            chunks.append(samples[start:end])
        start = end
    return chunks


# ── CLI config ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Dataset
    dataset: str = "mindcube"
    """Dataset name: mindcube | sat | vsibench | mmsibench"""
    data_path: str = "datasets/evaluation/MindCube/MindCube_tinybench.jsonl"
    """Path to the dataset file (jsonl or json)."""
    image_root: str = "datasets/evaluation/MindCube"
    """Root directory for resolving relative image paths."""

    # Model
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    """HuggingFace model path or local checkpoint directory."""
    model_type: str = "qwen-vl"
    """Model family: qwen-vl | qwen3-vl  (both use the same loading logic)."""
    max_new_tokens: int = 512
    """Max tokens for the generation instruction output."""
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Processing
    max_samples: int = -1
    """Maximum number of samples to process; -1 means all."""
    start_idx: int = 0
    """Start processing from this sample index."""

    # Output
    output_path: str = "results/image_instructions.jsonl"
    """Base output path (a timestamped sub-directory will be created beside it)."""

    # Multi-GPU
    num_gpus: int = -1
    """Number of GPUs to use. -1 = use all available (or those in CUDA_VISIBLE_DEVICES)."""

    # Misc
    device: str = "cuda"
    """Ignored in multi-GPU mode (each worker picks cuda:{gpu_id} automatically)."""
    verbose: bool = True


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_mindcube(data_path: str, image_root: str, max_samples: int, start_idx: int):
    """
    Each line: {id, question, images: [rel_path, ...], gt_answer, ...}
    images are relative to image_root.
    """
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    samples = samples[start_idx:]
    if max_samples > 0:
        samples = samples[:max_samples]

    result = []
    for s in samples:
        image_paths = [os.path.join(image_root, p) for p in s["images"]]
        result.append({
            "id": s["id"],
            "question": s["question"],
            "image_paths": image_paths,
            "gt_answer": s.get("gt_answer", ""),
            "meta": {k: v for k, v in s.items() if k not in ("id", "question", "images", "gt_answer")},
        })
    return result


def load_sat(data_path: str, image_root: str, max_samples: int, start_idx: int):
    """
    JSON list: [{database_idx, question, answer_choices, correct_answer, img_paths, ...}]
    img_paths are relative to image_root.
    """
    with open(data_path) as f:
        raw = json.load(f)

    raw = raw[start_idx:]
    if max_samples > 0:
        raw = raw[:max_samples]

    result = []
    for i, s in enumerate(raw):
        # Build full multiple-choice question text
        question = s["question"]
        if s.get("answer_choices"):
            choices = "\n".join(
                f"{chr(65 + j)}. {c}" for j, c in enumerate(s["answer_choices"])
            )
            question = f"{question}\n{choices}"

        image_paths = [os.path.join(image_root, p) for p in s.get("img_paths", [])]
        result.append({
            "id": s.get("database_idx", i),
            "question": question,
            "image_paths": image_paths,
            "gt_answer": s.get("correct_answer", ""),
            "meta": {k: v for k, v in s.items()
                     if k not in ("question", "answer_choices", "correct_answer", "img_paths", "database_idx")},
        })
    return result


def load_vsibench(data_path: str, image_root: str, max_samples: int, start_idx: int):
    """
    Each line: {id, dataset, scene_name, question_type, question, ground_truth, options}
    Images are assumed to live in image_root/<dataset>/<scene_name>/*.jpg
    (or the scannet/arkitscenes subdirectory structure).
    """
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    samples = samples[start_idx:]
    if max_samples > 0:
        samples = samples[:max_samples]

    result = []
    for s in samples:
        question = s["question"]
        if s.get("options"):
            opts = s["options"]
            if isinstance(opts, list):
                choices = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(opts))
                question = f"{question}\n{choices}"

        # Try to find scene images under image_root/<dataset>/<scene_name>/
        scene_dir = os.path.join(image_root, s.get("dataset", ""), s.get("scene_name", ""))
        image_paths = []
        if os.path.isdir(scene_dir):
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = sorted(
                [os.path.join(scene_dir, fn) for fn in os.listdir(scene_dir)
                 if os.path.splitext(fn)[1].lower() in exts]
            )[:8]  # cap at 8 frames to avoid exceeding context

        result.append({
            "id": s.get("id", ""),
            "question": question,
            "image_paths": image_paths,
            "gt_answer": s.get("ground_truth", ""),
            "meta": {k: v for k, v in s.items()
                     if k not in ("id", "question", "options", "ground_truth")},
        })
    return result


def load_mmsibench(data_path: str, image_root: str, max_samples: int, start_idx: int):
    """
    JSON / JSONL for MMSIBench.

    Our download.py stores images in 'local_images' with paths relative to the
    MMSIBench root directory (parent of data/), e.g.:
      ./data/images/sample_0_img_0.jpg  ->  <mmsibench_root>/data/images/...

    Spatial_RAI's download.py uses 'image' with the same relative-path convention.
    Both are handled below.
    """
    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".jsonl",):
        raw = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    else:
        with open(data_path) as f:
            raw = json.load(f)
    if isinstance(raw, dict):
        raw = list(raw.values())

    raw = raw[start_idx:]
    if max_samples > 0:
        raw = raw[:max_samples]

    # MMSIBench root = parent of data/ (i.e. two levels up from the json file
    # when data_path is  .../MMSIBench/data/test_data_final.json)
    mmsibench_root = Path(data_path).parent.parent  # .../MMSIBench/

    def _resolve(rel_path: str) -> str:
        p = Path(rel_path)
        if p.is_absolute():
            return str(p)
        # Strip leading ./ or ../ and join to MMSIBench root
        return str((mmsibench_root / rel_path).resolve())

    result = []
    for i, s in enumerate(raw):
        question = s.get("question", s.get("Question", ""))
        # Support both our 'local_images' key and Spatial_RAI's 'image' key
        images_field = s.get("local_images",
                      s.get("image",
                      s.get("images",
                      s.get("Image", []))))
        if isinstance(images_field, str):
            images_field = [images_field]
        image_paths = [_resolve(p) for p in images_field]

        result.append({
            "id": s.get("id", s.get("index", i)),
            "question": question,
            "image_paths": image_paths,
            "gt_answer": s.get("answer", s.get("Answer", s.get("gt_answer", ""))),
            "meta": {k: v for k, v in s.items()
                     if k not in ("id", "index", "question", "Question",
                                  "local_images", "image", "images", "Image",
                                  "answer", "Answer", "gt_answer")},
        })
    return result


DATASET_LOADERS = {
    "mindcube": load_mindcube,
    "sat": load_sat,
    "vsibench": load_vsibench,
    "mmsibench": load_mmsibench,
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str, model_type: str, device: str):
    """
    Load Qwen3-VL or Qwen2.5-VL model and processor.

    Qwen3-VL (transformers >= 4.52) uses AutoModelForImageTextToText and
    the `dtype` kwarg instead of the deprecated `torch_dtype`.
    Qwen2.5-VL falls back to Qwen2_5_VLForConditionalGeneration.
    """
    from transformers import AutoProcessor

    # AutoModelForImageTextToText is the correct class for Qwen3-VL
    # (and also works for Qwen2.5-VL with recent transformers).
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,          # new API (transformers >= 4.52)
            device_map=device,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, TypeError):
        # Older transformers: fall back to Qwen2.5-VL class with torch_dtype
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print(f"[INFO] Model class  : {type(model).__name__}")
    print(f"[INFO] Model path   : {model_path}")
    print(f"[INFO] Device map   : {model.hf_device_map if hasattr(model, 'hf_device_map') else device}")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def parse_instructions(model_output: str) -> List[str]:
    """Extract the list of image-generation instructions from model output.

    Parses <instruction>...</instruction> tags inside <instructions>...</instructions>.
    Returns an empty list when the model decides no additional images are needed.
    Silently caps the list at 5 entries.
    """
    import re

    # Locate the <instructions> block
    block_match = re.search(
        r"<instructions>(.*?)</instructions>",
        model_output,
        re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return []

    block = block_match.group(1)

    # Extract individual <instruction> tags
    items = re.findall(
        r"<instruction>(.*?)</instruction>",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    # Clean and cap
    instructions = [item.strip() for item in items if item.strip()]
    return instructions[:5]


def build_message(question: str, image_paths: List[str]) -> dict:
    """Build a single-turn user message with images + question."""
    content = []
    for img_path in image_paths:
        if os.path.isfile(img_path):
            content.append({"type": "image", "image": img_path})
        else:
            print(f"[WARN] Image not found, skipping: {img_path}")

    content.append({"type": "text", "text": question})
    return {"role": "user", "content": content}


def run_inference(
    model,
    processor,
    messages: List[dict],
    config: Config,
) -> str:
    """Run model inference and return generated text."""
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

    generation_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        use_cache=True,
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # Strip input tokens from output
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


# ── Per-GPU worker ────────────────────────────────────────────────────────────

def run_worker(
    gpu_id: str,
    samples: list,
    config: Config,
    output_path: str,
    log_file: Optional[str] = None,
):
    """Worker function executed in a subprocess on one GPU.

    Args:
        gpu_id:       Physical GPU id string (e.g. "0", "2").
        samples:      Data shard assigned to this worker.
        config:       Shared Config object (model path, generation params …).
        output_path:  Where this worker writes its JSONL shard.
        log_file:     Shared log file path (optional).
    """
    # ── Per-process logging ──────────────────────────────────────────────────
    handlers: List[logging.Handler] = [logging.StreamHandler()]
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

    logger.info(f"======================================")
    logger.info(f"Assigned device : {device}")
    logger.info(f"current_device  : {torch.cuda.current_device()}")
    logger.info(f"device_count    : {torch.cuda.device_count()}")
    logger.info(f"Samples         : {len(samples)}")
    logger.info(f"======================================")

    # ── Load model on this GPU ───────────────────────────────────────────────
    model, processor = load_model_and_processor(config.model_path, config.model_type, device)

    # ── Inference loop ───────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    with open(output_path, "w") as out_f:
        for i, sample in enumerate(samples):
            t0 = time.time()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                build_message(sample["question"], sample["image_paths"]),
            ]

            try:
                raw_output = run_inference(model, processor, messages, config)
                instructions = parse_instructions(raw_output)
            except Exception as e:
                logger.error(f"Sample {i} (id={sample['id']}): {e}")
                raw_output = ""
                instructions = []

            elapsed = time.time() - t0

            record = {
                "id": sample["id"],
                "dataset": config.dataset,
                "question": sample["question"],
                "image_paths": sample["image_paths"],
                "gt_answer": sample["gt_answer"],
                "raw_output": raw_output,
                "instructions": instructions,   # List[str], empty = no extra images needed
                "num_instructions": len(instructions),
                "meta": sample["meta"],
            }
            results.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if config.verbose:
                instr_display = (
                    "  (none — existing images sufficient)"
                    if not instructions
                    else "\n".join(f"  [{j+1}] {ins[:150]}" for j, ins in enumerate(instructions))
                )
                logger.info(
                    f"[{i+1}/{len(samples)}] id={sample['id']}  ({elapsed:.1f}s)\n"
                    f"  Question     : {sample['question'][:100]}...\n"
                    f"  Images       : {len(sample['image_paths'])} file(s)\n"
                    f"  Instructions : {len(instructions)}\n{instr_display}"
                )

    total_time = time.time() - t_start
    logger.info(f"Worker done — {len(results)} samples in {total_time:.1f}s")
    logger.info(f"Peak GPU memory : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import tyro

    # Must be called before any CUDA usage in the main process
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    config = tyro.cli(
        Config,
        description="Generate image-generation instructions from spatial QA datasets using Qwen VL (multi-GPU).",
    )

    # ── Resolve paths ──────────────────────────────────────────────────────────
    # File lives at spatial_planning/generate_image_instructions.py, so parent = spatial_planning/
    script_dir = Path(__file__).parent
    data_path  = Path(config.data_path)  if Path(config.data_path).is_absolute()  else script_dir / config.data_path
    image_root = Path(config.image_root) if Path(config.image_root).is_absolute() else script_dir / config.image_root

    # Derive a short model name from the checkpoint path (last directory component)
    model_name = Path(config.model_path).name or "model"

    # Output layout: results/<dataset>/<model_name>/<dataset>_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results = (
        Path(config.output_path).parent
        if Path(config.output_path).is_absolute()
        else script_dir / Path(config.output_path).parent
    )
    out_dir = base_results / config.dataset / model_name / f"{config.dataset}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # ── Validate ───────────────────────────────────────────────────────────────
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if config.dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {config.dataset}. Available: {list(DATASET_LOADERS)}")

    # ── Load dataset in main process ───────────────────────────────────────────
    logger.info(f"Loading dataset '{config.dataset}' from {data_path}")
    loader  = DATASET_LOADERS[config.dataset]
    samples = loader(str(data_path), str(image_root), config.max_samples, config.start_idx)
    logger.info(f"Loaded {len(samples)} samples.")

    # ── Determine GPU IDs ──────────────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    if n_available == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]

    # Optionally limit to --num_gpus
    if config.num_gpus > 0:
        all_gpu_ids = all_gpu_ids[: config.num_gpus]

    n_gpu = len(all_gpu_ids)
    logger.info("=" * 60)
    logger.info("GENERATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Dataset      : {config.dataset}")
    logger.info(f"Data path    : {data_path}")
    logger.info(f"Model path   : {config.model_path}")
    logger.info(f"Model type   : {config.model_type}")
    logger.info(f"Samples      : {len(samples)}")
    logger.info(f"GPUs in use  : {n_gpu}  {all_gpu_ids}")
    logger.info(f"Output dir   : {out_dir}")
    logger.info(f"Timestamp    : {timestamp}")
    logger.info("=" * 60)

    # ── Shard data and spawn one process per GPU ───────────────────────────────
    chunks       = chunk_dataset(samples, n_gpu)
    processes    = []
    output_paths = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = out_dir / f"shard_{idx}_gpu{gpu_id}.jsonl"
        output_paths.append(shard_path)

        p = mp.Process(
            target=run_worker,
            args=(gpu_id, chunk, config, str(shard_path), str(log_file)),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {idx} on GPU {gpu_id} ({len(chunk)} samples) → {shard_path.name}")

    for p in processes:
        p.join()

    # ── Merge shards ───────────────────────────────────────────────────────────
    logger.info("Merging shard results …")
    all_records: list = []
    for path in output_paths:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
        else:
            logger.warning(f"Shard not found: {path}")

    final_path = out_dir / "results.jsonl"
    with open(final_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Merged {len(all_records)} records → {final_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
