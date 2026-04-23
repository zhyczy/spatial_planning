import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .train_dataset import resize_xyz


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


class Eval_Dataset_Coord(Dataset):
    """
    MindCube evaluation dataset in the same prompt format as MindCube_Train_Dataset_Coord.

    Produces batches with:
      - QA supervision labels
      - image_xyz, image_xyz_hires for coord-head loss computation

    The coord head reads LM hidden states at the <|image_pad|> vision-token
    positions directly; no dedicated per-patch text token is inserted.

    This lets the eval loop call model() directly and log all losses
    (lm_loss, coord_loss).
    """

    def __init__(
        self,
        jsonl_path:         str,
        results_dir:        str,
        processor,
        log,
        max_images:         int = 4,
        spatial_merge_size: int = 2,
        coord_upscale:      int = 4,
        max_samples:        int | None = None,
        question_key:       str = "question",
        answer_key:         str = "gt_answer",
    ):
        raw = []
        with open(jsonl_path) as fh:
            for line in fh:
                if line.strip():
                    raw.append(json.loads(line))

        self.samples = []
        for entry in raw:
            eid = entry.get("id", "")
            sample_dir = os.path.join(results_dir, eid)
            if not os.path.isdir(sample_dir):
                continue
            self.samples.append((entry, sample_dir))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.processor          = processor
        self.max_images         = max_images
        self.spatial_merge_size = spatial_merge_size
        self.coord_upscale      = coord_upscale
        self.question_key       = question_key
        self.answer_key         = answer_key
        self.log                = log
        log.info(
            f"Eval_Dataset_Coord: {len(self.samples)} valid entries "
            f"(out of {len(raw)} total) from {jsonl_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, sample_dir = self.samples[idx]

        # ── load images and per-pixel xyz ─────────────────────────────────────
        view_dirs = sorted(d for d in os.listdir(sample_dir) if d.startswith("view_"))
        images, xyz_raw_list, mask_raw_list = [], [], []
        for vd in view_dirs[: self.max_images]:
            img_path = os.path.join(sample_dir, vd, "image.png")
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except (FileNotFoundError, OSError):
                break
            pts3d_path = os.path.join(sample_dir, vd, "pts3d.npy")
            mask_path  = os.path.join(sample_dir, vd, "mask.npy")
            xyz_raw_list.append(
                np.load(pts3d_path).astype(np.float32)
                if os.path.exists(pts3d_path) else None
            )
            mask_raw_list.append(
                np.load(mask_path) if os.path.exists(mask_path) else None
            )

        N = len(images)

        # ── build prompt (images + QA) ───────────────────────────────────────
        content: list = [{"type": "image", "image": img} for img in images]

        _question = entry.get(self.question_key, "")
        _answer   = entry.get(self.answer_key, "")
        if not (_question and _answer):
            raise RuntimeError(
                f"Eval_Dataset_Coord sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        content.append({"type": "text", "text": _question})

        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": _answer}],
            tokenize=False, add_generation_prompt=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        suffix_ids = self.processor.tokenizer(
            _answer + "<|im_end|>\n", add_special_tokens=False
        )["input_ids"]
        labels = proc_out["input_ids"].clone()
        labels[0, :-len(suffix_ids)] = -100

        # ── 3D position maps (pts3d → patch-level + sub-pixel) ───────────────
        image_xyz = None
        image_xyz_hires = None
        try:
            thw_all = proc_out["image_grid_thw"]
            sms = self.spatial_merge_size
            up = self.coord_upscale
            xyz_list, xyz_hires_list = [], []
            for k in range(N):
                xyz_raw  = xyz_raw_list[k]
                mask_raw = mask_raw_list[k]
                thw_k    = thw_all[k]
                llm_h    = int(thw_k[1]) // sms
                llm_w    = int(thw_k[2]) // sms
                if xyz_raw is not None:
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=mask_raw))
                    xyz_hires_list.append(resize_xyz(xyz_raw, llm_h * up, llm_w * up, valid=mask_raw))
                else:
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                    xyz_hires_list.append(torch.zeros(llm_h * up, llm_w * up, 3))
            image_xyz = xyz_list
            image_xyz_hires = xyz_hires_list
        except Exception as exc:
            self.log.debug(f"pts3d load failed for {sample_dir}: {exc}")

        return {
            **proc_out,
            "image_xyz":       image_xyz,
            "image_xyz_hires": image_xyz_hires,
            "labels":          labels,
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

    elif dataset == "spinbench":
        # SPINBench — multi-view object spatial reasoning.
        # Uses test.jsonl which includes an 'id' field matching
        # the 3d_results/<id>/ directory for precomputed XYZ maps.
        # Keys: problem, answer, images, metadata, id
        jsonl_file = data_dir / "test.jsonl"
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            raw = raw[:limit]
        for item in raw:
            image_paths = [str((data_dir / p).resolve()) for p in item.get("images", [])]
            meta = item.get("metadata", {})
            samples.append({
                "index": item.get("id", len(samples)),
                "image": image_paths,
                "question": item.get("problem", ""),
                "answer": item.get("answer", ""),
                "category": meta.get("task_type", meta.get("dataset_type", "unknown")),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset == "robospatial":
        # RoboSpatial — robot spatial reasoning with embedded images.
        # Parquet files under data/ with columns: category, question, answer,
        # img (bytes dict), depth_image (bytes dict), mask (bytes dict or None).
        # Images and masks are extracted to data_dir/images/ and data_dir/masks/ for caching.
        # 3d_results/{category}_{idx}/ directories are used for precomputed XYZ maps.
        import io
        import pandas as pd

        parquet_dir = data_dir / "data"
        if not parquet_dir.exists():
            raise FileNotFoundError(f"RoboSpatial data dir not found: {parquet_dir}")

        images_cache_dir = data_dir / "images"
        masks_cache_dir = data_dir / "masks"
        images_cache_dir.mkdir(exist_ok=True)
        masks_cache_dir.mkdir(exist_ok=True)

        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        all_rows = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            all_rows.extend(df.to_dict("records"))

        if limit is not None:
            all_rows = all_rows[:limit]

        for idx, row in enumerate(all_rows):
            category = row.get("category", "unknown")
            sample_id = f"{category}_{idx}"

            # Extract and cache image to disk
            img_data = row.get("img")
            if img_data is None:
                continue
            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
            img_path = images_cache_dir / f"{sample_id}.jpg"
            if not img_path.exists():
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path, "JPEG")

            # Extract and cache mask to disk (mask naming matches RoboSpatial-Eval convention)
            mask_rel_path = None
            mask_data = row.get("mask")
            if mask_data is not None:
                mask_bytes = mask_data.get("bytes") if isinstance(mask_data, dict) else mask_data
                if mask_bytes is not None:
                    mask_filename = f"mask_{category}_{idx}.png"
                    mask_path = masks_cache_dir / mask_filename
                    if not mask_path.exists():
                        Image.open(io.BytesIO(mask_bytes)).save(mask_path, "PNG")
                    mask_rel_path = f"masks/{mask_filename}"

            samples.append({
                "index": sample_id,
                "image": [str(img_path)],
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": category,
                "thought": "",
                "data_dir": str(data_dir),
                "mask": mask_rel_path,
                "format_type": "robospatial",
            })

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Choose: mmsibench | mindcube | sat | sat_real | vsibench | "
            "sparbench_multi_view | sparbench_single_view | sparbench_mv | spinbench | robospatial"
        )

    return samples


def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [dataset[s : s + chunk_size] for s in range(0, len(dataset), chunk_size)]
