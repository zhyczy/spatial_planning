import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Annotated

import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm

# add Spatial-MLLM repo root to sys.path (for qwen_vl_utils and src.qwenvl.*)
_REPO_ROOT = Path(__file__).resolve().parent / "repo" / "Spatial-MLLM"
sys.path.insert(0, str(_REPO_ROOT))

from qwen_vl_utils import process_vision_info

# ─── constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/checkpoints/Spatial-MLLM-v1.1-Instruct-135K"
SFT_TYPE_TEMPLATE = "Answer with the option's letter from the given choices directly."

MINDCUBE_DATA_JSONL = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MindCube/MindCube_tinybench.jsonl"
MINDCUBE_DATA_DIR = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MindCube"

# ─── model helpers ────────────────────────────────────────────────────────────

def prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs):
    """
        Prepare inputs for Spatial MLLM model.
        Batch: Dict return by the processor
        video_input and image_inputs is returned by process_vision_info

        video_inputs: List[torch.Tensor[Int]] | List[torch.Tensor[Float]] | List[List[PIL.Image]]
        image_inputs: List[PIL.Image]
    """
    video_tchw = []
    image_tchw = []

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0  # Normalize to [0, 1]
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                # Convert list of PIL Images to tensor
                video_input = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
            else:
                raise ValueError("Unsupported video input format.")
            video_tchw.append(video_input)

    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                # Single image → treat as 1-frame video [1, C, H, W]
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                raise ValueError("Unsupported image input format.")
            image_tchw.append(image_input)

    batch.update({
        "video_tchw": video_tchw if video_tchw else None,
        "image_tchw": image_tchw if image_tchw else None,
    })

    return batch


def load_model_and_processor(model_type: str, model_path: str):
    """Load model and processor based on type."""
    if "spatial-mllm" in model_type:
        from transformers import Qwen2_5_VLProcessor

        from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration

        config = SpatialMLLMConfig.from_pretrained(model_path)
        model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        return model, processor

    if "qwen2.5-vl" in model_type:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        return model, processor

    raise ValueError(f"Unknown model type: {model_type}")


# ─── MindCube utilities ──────────────────────────────────────────────────────

def _extract_answer(text: str) -> str:
    """Return content inside the last <answer>...</answer> block if it exists."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1] if matches else text


def clean_text(text: str) -> str:
    """Normalize model output to a simple, comparable string."""
    cleaned = _extract_answer(text)
    for char in ("\n", "\r"):
        cleaned = cleaned.replace(char, " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().rstrip(".").upper()


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter (A, B, C, D) from model output."""
    cleaned = clean_text(text)

    matches = re.findall(r'\b([A-D])\b', cleaned)
    if matches:
        return matches[-1]

    matches = re.findall(r'(?:ANSWER|OPTION)(?:\s+IS)?\s*([A-D])', cleaned)
    if matches:
        return matches[-1]

    for char in cleaned:
        if char in 'ABCD':
            return char

    return None


def mindcube_reward(clean_ans_gt: str, clean_ans_pred: str) -> float:
    """Calculate reward (exact match) for MindCube dataset."""
    gt_letter = extract_answer_letter(clean_ans_gt)
    pred_letter = extract_answer_letter(clean_ans_pred)

    if gt_letter is None or pred_letter is None:
        return 0.0

    return 1.0 if pred_letter == gt_letter else 0.0


def calculate_mindcube_metrics(results: List[Dict]) -> Dict:
    """Calculate per-category, per-type, and overall accuracy."""
    if not results:
        return {"per_category": {}, "per_type": {}, "overall": {"accuracy": 0.0, "count": 0}}

    import pandas as pd

    df = pd.DataFrame([
        {
            "reward": res.get("reward", 0.0),
            "category": res["sample"].get("category", ["unknown"])[0] if res["sample"].get("category") else "unknown",
            "type": res["sample"].get("type", "unknown"),
        }
        for res in results
    ])

    per_category = {
        cat: {"score": float(grp["reward"].mean()), "count": int(len(grp))}
        for cat, grp in df.groupby("category")
    }
    per_type = {
        typ: {"score": float(grp["reward"].mean()), "count": int(len(grp))}
        for typ, grp in df.groupby("type")
    }

    return {
        "per_category": per_category,
        "per_type": per_type,
        "overall": {"accuracy": float(df["reward"].mean()), "count": len(df)},
    }


# ─── MindCube evaluation helpers ─────────────────────────────────────────────

def load_mindcube_dataset(jsonl_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load MindCube dataset from a JSONL file."""
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if limit and len(data) >= limit:
                    break
    return data


def build_user_message(item: Dict, data_dir: Path) -> Dict:
    """Create the chat-style message payload for a single MindCube sample."""
    question = f"{item.get('question', '')}\n{SFT_TYPE_TEMPLATE}"

    img_paths = item.get("images", [])
    if not img_paths:
        raise ValueError(f"Sample {item.get('id')} has no image paths.")

    image_paths = []
    for img_path in img_paths:
        if Path(img_path).is_absolute():
            image_paths.append(str(Path(img_path).resolve()))
        else:
            image_paths.append(str((data_dir / img_path).resolve()))

    # Treat multiple images as video frames so the spatial encoder (VGGT) can
    # do cross-frame attention across views rather than processing each image
    # independently.  process_vision_info returns a List[PIL.Image] for this
    # format, which the video branch of prepare_spatial_mllm_inputs stacks into
    # a single [T, C, H, W] tensor.
    video_content = {"type": "video", "video": image_paths}

    return {
        "role": "user",
        "content": [video_content, {"type": "text", "text": question}],
    }


def prepare_chat_batch(
    batch_data: List[Dict],
    processor: Any,
    model_type: str,
    data_dir: Path,
) -> Tuple[Dict, List[str]]:
    """Tokenize a batch of MindCube samples."""
    batch_messages = [[build_user_message(item, data_dir)] for item in batch_data]

    prompts_text = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    video_inputs: List = []
    image_inputs: List = []
    for msgs in batch_messages:
        imgs, vids = process_vision_info(msgs)
        if imgs:
            image_inputs.extend(imgs)
        elif vids:
            video_inputs.extend(vids)

    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on a prepared batch."""
    batch_inputs.to(model.device)
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [t.to(model.device) for t in batch_inputs["image_tchw"]]
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [t.to(model.device) for t in batch_inputs["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs, **generation_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def postprocess_batch(
    batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]
) -> List[Dict]:
    """Clean outputs, compute rewards, and package results."""
    results = []
    for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
        clean_ans = clean_text(model_output)
        clean_ans_gt = clean_text(sample.get("gt_answer", ""))
        reward = mindcube_reward(clean_ans_gt, clean_ans)
        results.append({
            "sample": sample,
            "prompt": prompt,
            "model_output": model_output,
            "cleaned_model_output": clean_ans,
            "cleaned_gt_answer": clean_ans_gt,
            "reward": reward,
            "correct": reward == 1.0,
        })
    return results


# ─── shared batch helper ──────────────────────────────────────────────────────

def _prepare_batch_from_messages(
    batch_messages: List[List[Dict]],
    processor: Any,
    model_type: str,
) -> Tuple[Dict, List[str]]:
    """Tokenise pre-built chat messages into model inputs."""
    prompts_text = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    video_inputs: List = []
    image_inputs: List = []
    for msgs in batch_messages:
        imgs, vids = process_vision_info(msgs)
        if imgs:
            image_inputs.extend(imgs)
        elif vids:
            video_inputs.extend(vids)

    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text


# ─── MMSIBench utilities ──────────────────────────────────────────────────────

def load_mmsibench_dataset(json_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load MMSIBench from test_data_final.json; resolve image paths."""
    json_file = Path(json_path)
    mmsibench_root = json_file.parent.parent  # .../MMSIBench/data/test_data_final.json
    with open(json_file) as f:
        raw = json.load(f)
    if limit:
        raw = raw[:limit]
    dataset = []
    for item in raw:
        image_paths = [
            str((mmsibench_root / p).resolve())
            for p in item.get("local_images", [])
        ]
        dataset.append({
            "index": item.get("id", ""),
            "image": image_paths,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "category": item.get("type", "unknown"),
        })
    return dataset


def build_mmsibench_message(item: Dict) -> Dict:
    """Video-format message for an MMSIBench sample (multi-image → video frames)."""
    question = f"{item['question']}\n{SFT_TYPE_TEMPLATE}"
    return {
        "role": "user",
        "content": [
            {"type": "video", "video": item["image"]},
            {"type": "text", "text": question},
        ],
    }


def extract_mmsibench_answer(text: str) -> str:
    """Extract A/B/C/D from MMSIBench output using backtick then word-boundary regex."""
    m = re.search(r'``([^`]*)``', text)
    if m:
        text = m.group(1)
    else:
        m = re.search(r'`([^`]*)`', text)
        if m:
            text = m.group(1)
    m = re.search(r'\b[A-D]\b(?!\s[a-zA-Z])', text)
    return m.group().upper() if m else ""


def calculate_mmsibench_metrics(results: List[Dict]) -> Dict:
    """Overall + per-category accuracy for MMSIBench."""
    from collections import defaultdict
    total = len(results)
    correct = sum(r["reward"] for r in results)
    cat_correct: Dict = defaultdict(float)
    cat_total: Dict = defaultdict(int)
    for r in results:
        cat = r["sample"].get("category", "unknown")
        cat_total[cat] += 1
        cat_correct[cat] += r["reward"]
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_samples": total,
        "correct_samples": int(correct),
        "category_accuracy": {c: cat_correct[c] / cat_total[c] for c in cat_total},
        "category_counts": dict(cat_total),
    }


# ─── SAT utilities ────────────────────────────────────────────────────────────

def load_sat_dataset(json_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load SAT test.json."""
    with open(json_path) as f:
        data = json.load(f)
    return data[:limit] if limit else data


def build_sat_message(item: Dict, data_dir: Path) -> Dict:
    """Video-format message for a SAT sample with lettered answer choices."""
    answer_choices = item.get("answer_choices", [])
    options_text = "Options:\n" + "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(answer_choices)
    )
    question = f"{item.get('question', '')}\n{options_text}\n{SFT_TYPE_TEMPLATE}"

    img_paths = item.get("img_paths", [])
    if not img_paths:
        raise ValueError(f"SAT sample {item.get('database_idx')} has no image paths.")
    resolved = [
        str((data_dir / p).resolve()) if not Path(p).is_absolute() else p
        for p in img_paths
    ]
    return {
        "role": "user",
        "content": [
            {"type": "video", "video": resolved},
            {"type": "text", "text": question},
        ],
    }


def _sat_gt_letter(item: Dict) -> str:
    """Return the answer letter (A/B/…) for correct_answer in a SAT item."""
    correct = item.get("correct_answer", "").strip().lower()
    for i, choice in enumerate(item.get("answer_choices", [])):
        if choice.strip().lower() == correct:
            return chr(65 + i)
    return correct  # fallback


def calculate_sat_metrics(results: List[Dict]) -> Dict:
    """Per-question-type + overall accuracy for SAT."""
    if not results:
        return {"per_question_type": {}, "overall": {"accuracy": 0.0, "count": 0}}
    import pandas as pd
    df = pd.DataFrame([
        {"reward": r["reward"], "question_type": r["sample"].get("question_type", "unknown")}
        for r in results
    ])
    per_qtype = {
        qt: {"score": float(g["reward"].mean()), "count": int(len(g))}
        for qt, g in df.groupby("question_type")
    }
    return {
        "per_question_type": per_qtype,
        "overall": {"accuracy": float(df["reward"].mean()), "count": len(df)},
    }


# ─── SPARBench utilities ──────────────────────────────────────────────────────

def load_sparbench_dataset(json_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load SPARBench JSON (images are base64-encoded JPEG strings)."""
    with open(json_path) as f:
        data = json.load(f)
    return data[:limit] if limit else data


def build_sparbench_message(item: Dict) -> Dict:
    """Video-format message for a SPARBench sample; decode base64 images on the fly."""
    import base64
    import io as _io
    question = f"{item.get('question', '')}\n{SFT_TYPE_TEMPLATE}"
    pil_images = []
    for b64_str in item.get("images", []):
        img_bytes = base64.b64decode(b64_str)
        pil_images.append(Image.open(_io.BytesIO(img_bytes)).convert("RGB"))
    if not pil_images:
        raise ValueError(f"SPARBench sample {item.get('id')} has no images.")
    return {
        "role": "user",
        "content": [
            {"type": "video", "video": pil_images},
            {"type": "text", "text": question},
        ],
    }


def calculate_sparbench_metrics(results: List[Dict]) -> Dict:
    """Per-task + overall accuracy for SPARBench."""
    if not results:
        return {"per_task": {}, "overall": {"accuracy": 0.0, "count": 0}}
    import pandas as pd
    df = pd.DataFrame([
        {"reward": r["reward"], "task": r["sample"].get("task", "unknown")}
        for r in results
    ])
    per_task = {
        t: {"score": float(g["reward"].mean()), "count": int(len(g))}
        for t, g in df.groupby("task")
    }
    return {
        "per_task": per_task,
        "overall": {"accuracy": float(df["reward"].mean()), "count": len(df)},
    }


# ─── CLI configs and entry points ────────────────────────────────────────────

@dataclass
class DemoArgs:
    """Run a single-sample inference (video + text prompt)."""
    video_path: str = "assets/arkitscenes_41069025.mp4"
    text: str = "How many chair(s) are in this room?\nPlease answer the question using a single word or phrase."
    model_type: str = "spatial-mllm"
    model_path: str = DEFAULT_MODEL_PATH


@dataclass
class MindCubeArgs:
    """Evaluate Spatial-MLLM on the MindCube benchmark."""
    data_jsonl: str = MINDCUBE_DATA_JSONL
    data_dir: str = MINDCUBE_DATA_DIR
    output_dir: str = "eval_results"
    model_type: str = "spatial-mllm"
    model_path: str = DEFAULT_MODEL_PATH
    model_name: Optional[str] = None
    batch_size: int = 1
    limit: Optional[int] = None


def run_demo(args: DemoArgs) -> None:
    torch.cuda.empty_cache()

    model, processor = load_model_and_processor(args.model_type, args.model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video_path, "nframes": 16},
                {"type": "text", "text": args.text},
            ],
        }
    ]

    prompts_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[prompts_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in args.model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    batch.to(model.device)
    if "image_tchw" in batch and batch["image_tchw"] is not None:
        batch["image_tchw"] = [t.to(model.device) for t in batch["image_tchw"]]
    if "video_tchw" in batch and batch["video_tchw"] is not None:
        batch["video_tchw"] = [t.to(model.device) for t in batch["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    time_0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**batch, **generation_kwargs)
    time_taken = time.time() - time_0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
    ]
    num_generated_tokens = sum(len(ids) for ids in generated_ids_trimmed)

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"Time taken for inference: {time_taken:.2f} seconds")
    print(f"GPU Memory taken for inference: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Number of generated tokens: {num_generated_tokens}")
    print(f"Time taken per token: {time_taken / num_generated_tokens:.4f} seconds/token")
    print(f"Output: {output_text}")


def run_mindcube(args: MindCubeArgs) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Build output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or args.model_type
    output_dir = Path(args.output_dir) / "MindCube" / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redirect logging to file as well
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Model path:   {args.model_path}")
    logger.info(f"Model type:   {args.model_type}")
    logger.info(f"Data JSONL:   {args.data_jsonl}")
    logger.info(f"Data dir:     {args.data_dir}")
    logger.info(f"Output dir:   {output_dir}")
    logger.info(f"Batch size:   {args.batch_size}")
    logger.info(f"Limit:        {args.limit}")

    mindcube_data = load_mindcube_dataset(args.data_jsonl, limit=args.limit)
    logger.info(f"Loaded {len(mindcube_data)} samples.")

    torch.cuda.empty_cache()
    model, processor = load_model_and_processor(args.model_type, args.model_path)
    data_dir = Path(args.data_dir).resolve()

    final_output: List[Dict] = []
    eval_result_path = output_dir / "eval_result.json"

    for i in tqdm(range(0, len(mindcube_data), args.batch_size), desc="Evaluating MindCube"):
        batch_data = mindcube_data[i : i + args.batch_size]

        batch_inputs, prompts_text = prepare_chat_batch(batch_data, processor, args.model_type, data_dir)
        batch_output_text = inference_batch(batch_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # Checkpoint every 10 batches
        if (i // args.batch_size + 1) % 10 == 0 or (i + args.batch_size) >= len(mindcube_data):
            with open(eval_result_path, "w") as f:
                json.dump(final_output, f, indent=2)

    metrics = calculate_mindcube_metrics(final_output)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Results saved to: {output_dir}")


# ─── MMSIBench run ────────────────────────────────────────────────────────────

@dataclass
class MMSIBenchArgs:
    """Evaluate Spatial-MLLM on MMSIBench."""
    data_json: str = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/MMSIBench/data/test_data_final.json"
    output_dir: str = "eval_results"
    model_type: str = "spatial-mllm"
    model_path: str = DEFAULT_MODEL_PATH
    model_name: Optional[str] = None
    batch_size: int = 1
    limit: Optional[int] = None


def run_mmsibench(args: MMSIBenchArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or args.model_type
    output_dir = Path(args.output_dir) / "MMSIBench" / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Model path:  {args.model_path}")
    logger.info(f"Data JSON:   {args.data_json}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Limit:       {args.limit}")

    data = load_mmsibench_dataset(args.data_json, limit=args.limit)
    logger.info(f"Loaded {len(data)} samples.")

    torch.cuda.empty_cache()
    model, processor = load_model_and_processor(args.model_type, args.model_path)

    final_output: List[Dict] = []
    eval_result_path = output_dir / "eval_result.json"

    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating MMSIBench"):
        batch_data = data[i : i + args.batch_size]
        batch_messages = [[build_mmsibench_message(item)] for item in batch_data]
        batch_inputs, prompts_text = _prepare_batch_from_messages(batch_messages, processor, args.model_type)
        batch_output_text = inference_batch(batch_inputs, model, processor)

        for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
            pred_letter = extract_mmsibench_answer(model_output)
            gt_letter = sample.get("answer", "").strip().upper()
            reward = 1.0 if pred_letter == gt_letter and pred_letter != "" else 0.0
            final_output.append({
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "reward": reward,
                "correct": reward == 1.0,
            })

        if (i // args.batch_size + 1) % 10 == 0 or (i + args.batch_size) >= len(data):
            with open(eval_result_path, "w") as f:
                json.dump(final_output, f, indent=2)

    metrics = calculate_mmsibench_metrics(final_output)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Results saved to: {output_dir}")


# ─── SAT run ─────────────────────────────────────────────────────────────────

@dataclass
class SATArgs:
    """Evaluate Spatial-MLLM on SAT (sat_real)."""
    data_json: str = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SAT/test.json"
    data_dir: str = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SAT"
    output_dir: str = "eval_results"
    model_type: str = "spatial-mllm"
    model_path: str = DEFAULT_MODEL_PATH
    model_name: Optional[str] = None
    batch_size: int = 1
    limit: Optional[int] = None


def run_sat(args: SATArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or args.model_type
    output_dir = Path(args.output_dir) / "SAT" / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Model path:  {args.model_path}")
    logger.info(f"Data JSON:   {args.data_json}")
    logger.info(f"Data dir:    {args.data_dir}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Limit:       {args.limit}")

    data = load_sat_dataset(args.data_json, limit=args.limit)
    logger.info(f"Loaded {len(data)} samples.")

    torch.cuda.empty_cache()
    model, processor = load_model_and_processor(args.model_type, args.model_path)
    data_dir = Path(args.data_dir).resolve()

    final_output: List[Dict] = []
    eval_result_path = output_dir / "eval_result.json"

    for i in tqdm(range(0, len(data), args.batch_size), desc="Evaluating SAT"):
        batch_data = data[i : i + args.batch_size]
        batch_messages = [[build_sat_message(item, data_dir)] for item in batch_data]
        batch_inputs, prompts_text = _prepare_batch_from_messages(batch_messages, processor, args.model_type)
        batch_output_text = inference_batch(batch_inputs, model, processor)

        for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
            pred_letter = extract_answer_letter(model_output)
            gt_letter = _sat_gt_letter(sample)
            reward = 1.0 if pred_letter is not None and pred_letter == gt_letter else 0.0
            final_output.append({
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "reward": reward,
                "correct": reward == 1.0,
            })

        if (i // args.batch_size + 1) % 10 == 0 or (i + args.batch_size) >= len(data):
            with open(eval_result_path, "w") as f:
                json.dump(final_output, f, indent=2)

    metrics = calculate_sat_metrics(final_output)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Results saved to: {output_dir}")


# ─── SPARBench run ────────────────────────────────────────────────────────────

@dataclass
class SPARBenchArgs:
    """Evaluate Spatial-MLLM on SPARBench (multi_view or single_view)."""
    data_json: str = "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/datasets/evaluation/SPARBench/sparbench_multi_view.json"
    output_dir: str = "eval_results"
    model_type: str = "spatial-mllm"
    model_path: str = DEFAULT_MODEL_PATH
    model_name: Optional[str] = None
    batch_size: int = 1
    limit: Optional[int] = None


def run_sparbench(args: SPARBenchArgs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or args.model_type
    dataset_label = Path(args.data_json).stem  # e.g. "sparbench_multi_view"
    output_dir = Path(args.output_dir) / "SPARBench" / dataset_label / model_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Model path:  {args.model_path}")
    logger.info(f"Data JSON:   {args.data_json}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Limit:       {args.limit}")

    data = load_sparbench_dataset(args.data_json, limit=args.limit)
    logger.info(f"Loaded {len(data)} samples.")

    torch.cuda.empty_cache()
    model, processor = load_model_and_processor(args.model_type, args.model_path)

    final_output: List[Dict] = []
    eval_result_path = output_dir / "eval_result.json"

    for i in tqdm(range(0, len(data), args.batch_size), desc=f"Evaluating {dataset_label}"):
        batch_data = data[i : i + args.batch_size]
        batch_messages = [[build_sparbench_message(item)] for item in batch_data]
        batch_inputs, prompts_text = _prepare_batch_from_messages(batch_messages, processor, args.model_type)
        batch_output_text = inference_batch(batch_inputs, model, processor)

        for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
            pred_letter = extract_answer_letter(model_output)
            gt_letter = sample.get("answer", "").strip().upper()
            reward = 1.0 if pred_letter is not None and pred_letter == gt_letter else 0.0
            # Don't store base64 images in results
            sample_out = {k: v for k, v in sample.items() if k != "images"}
            final_output.append({
                "sample": sample_out,
                "prompt": prompt,
                "model_output": model_output,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "reward": reward,
                "correct": reward == 1.0,
            })

        if (i // args.batch_size + 1) % 10 == 0 or (i + args.batch_size) >= len(data):
            with open(eval_result_path, "w") as f:
                json.dump(final_output, f, indent=2)

    metrics = calculate_sparbench_metrics(final_output)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Results saved to: {output_dir}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = tyro.cli(
        Union[
            Annotated[DemoArgs,       tyro.conf.subcommand("demo",       default=DemoArgs())],
            Annotated[MindCubeArgs,   tyro.conf.subcommand("mindcube",   default=MindCubeArgs())],
            Annotated[MMSIBenchArgs,  tyro.conf.subcommand("mmsibench",  default=MMSIBenchArgs())],
            Annotated[SATArgs,        tyro.conf.subcommand("sat",        default=SATArgs())],
            Annotated[SPARBenchArgs,  tyro.conf.subcommand("sparbench",  default=SPARBenchArgs())],
        ]
    )
    if isinstance(args, DemoArgs):
        run_demo(args)
    elif isinstance(args, MindCubeArgs):
        run_mindcube(args)
    elif isinstance(args, MMSIBenchArgs):
        run_mmsibench(args)
    elif isinstance(args, SATArgs):
        run_sat(args)
    elif isinstance(args, SPARBenchArgs):
        run_sparbench(args)


if __name__ == "__main__":
    main()
