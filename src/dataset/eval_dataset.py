import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class Eval_Dataset(Dataset):
    """Wraps raw eval samples into a PyTorch Dataset that produces
    processor-encoded batches with LM labels for computing eval loss.

    Each sample is a dict with keys: image (list of paths), question, answer.
    """

    def __init__(self, samples: List[Dict[str, Any]], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_paths = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        # Load images
        images = []
        content = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": question})

        # Full conversation with assistant answer (for computing LM loss)
        text_full = self.processor.apply_chat_template(
            [{"role": "user", "content": content},
             {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )

        # Build labels: mask everything except the answer tokens
        suffix_ids = self.processor.tokenizer(
            answer + "<|im_end|>\n", add_special_tokens=False
        )["input_ids"]
        suffix_len = len(suffix_ids)
        labels = proc_out["input_ids"].clone()
        labels[0, :-suffix_len] = -100

        return {
            **proc_out,
            "labels": labels,
        }


def load_testing_dataset(
    data_dir: Path,
    limit: Optional[int] = None,
    dataset: str = "mmsibench",
) -> List[Dict[str, Any]]:
    """Load evaluation dataset.

    Supports: mmsibench | mindcube | sat | vsibench
    Image paths are resolved to absolute paths.
    """
    data_dir = Path(data_dir)
    samples: List[Dict[str, Any]] = []

    if dataset == "mmsibench":
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
        for item in raw:
            local_images = item.get("local_images", [])
            image_paths = [str((data_dir / p).resolve()) for p in local_images]
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("type", "unknown"),
                "thought": item.get("thought_gt", ""),
                "data_dir": str(data_dir),
            })

    elif dataset == "mindcube":
        jsonl_file = data_dir / "MindCube_tinybench.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            image_paths = [str((data_dir / p).resolve()) for p in item.get("images", [])]
            category = item.get("category", [])
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("gt_answer", ""),
                "category": category[0] if category else "unknown",
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset in ("sat", "sat_real"):
        json_file = data_dir / "test.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        _letters = "ABCDEFGHIJ"
        for item in raw:
            img_paths = item.get("img_paths", item.get("images", []))
            image_paths = [str((data_dir / p).resolve()) for p in img_paths]
            choices = item.get("answer_choices", [])
            correct = item.get("correct_answer", item.get("answer", ""))
            if choices:
                formatted = "\n".join(
                    f"{_letters[i]}. {c}" for i, c in enumerate(choices)
                )
                question_text = item.get("question", "") + "\n" + formatted
                try:
                    answer_letter = _letters[choices.index(correct)]
                except ValueError:
                    answer_letter = correct
            else:
                question_text = item.get("question", "")
                answer_letter = correct
            samples.append({
                "index": item.get("database_idx", item.get("id", len(samples))),
                "image": image_paths,
                "question": question_text,
                "answer": answer_letter,
                "category": item.get("question_type", item.get("type", "unknown")),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset == "vsibench":
        jsonl_file = data_dir / "test.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            image_paths = [str((data_dir / p).resolve()) for p in item.get("images", [])]
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("gt_answer", "")),
                "category": item.get("type", "unknown"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset in ("sparbench_multi_view", "sparbench_single_view", "sparbench_mv"):
        import base64, tempfile
        if dataset == "sparbench_mv":
            suffix = "mv"
        elif dataset == "sparbench_multi_view":
            suffix = "multi_view"
        else:
            suffix = "single_view"
        json_file = data_dir / f"sparbench_{suffix}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        _tmp_dir = Path(tempfile.mkdtemp(prefix=f"sparbench_{suffix}_"))
        for item in raw:
            b64_images = item.get("images", [])
            image_paths = []
            item_id = item.get("id", len(samples))
            for img_idx, b64 in enumerate(b64_images):
                img_bytes = base64.b64decode(b64)
                img_path = _tmp_dir / f"{item_id}_{img_idx}.jpg"
                img_path.write_bytes(img_bytes)
                image_paths.append(str(img_path))
            samples.append({
                "index": item_id,
                "image": image_paths,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("task", "unknown"),
                "format_type": item.get("format_type", "select"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Choose: mmsibench | mindcube | sat | sat_real | vsibench | "
            "sparbench_multi_view | sparbench_single_view | sparbench_mv"
        )

    return samples


def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [dataset[s : s + chunk_size] for s in range(0, len(dataset), chunk_size)]
