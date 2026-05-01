"""Eval datasets for Qwen3.5-VL.

Eval_Dataset_Coord mirrors MindCube_Train_Dataset_Coord — both run each
loaded view through `_qwen_align_view` (PIL LANCZOS for image, bilinear for
pts3d, nearest for mask) before the Qwen processor and resize_xyz. This
keeps train and per-epoch eval on the same pixel/geometry grid.

The autoregressive generation eval (`evaluation.py`) loads images directly
from `load_testing_dataset` paths — it must call `_qwen_align_view` itself
on each PIL image so test images go through the same LANCZOS path as
training (Qwen's image_processor uses BICUBIC internally, so skipping the
pre-align would route test images through a different interpolation kernel
than training).
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .answer_format import build_interleaved_content
from .train_dataset_qwen35 import _load_and_align_views, resize_xyz


class Eval_Dataset_Coord(Dataset):
    """
    MindCube evaluation dataset in the same prompt format as
    MindCube_Train_Dataset_Coord.

    Produces batches with QA-supervision labels and image_xyz / image_xyz_hires
    for coord-head loss computation. The coord head reads LM hidden states at
    the <|image_pad|> vision-token positions directly; no dedicated per-patch
    text token is inserted.

    Each loaded view is Qwen-aligned (smart_resize-rounded) before the
    processor + resize_xyz run, so pts3d patches and Qwen vision tokens cover
    the same scene region pixel-for-pixel.
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

        images, xyz_raw_list, mask_raw_list, _ = _load_and_align_views(
            sample_dir, self.max_images,
        )
        N = len(images)

        # ── build prompt (images + QA) ───────────────────────────────────────
        _question = entry.get(self.question_key, "")
        _answer   = entry.get(self.answer_key, "")
        if not (_question and _answer):
            raise RuntimeError(
                f"Eval_Dataset_Coord sample {idx} (id={entry.get('id')}) has no QA pair."
            )

        content = build_interleaved_content(_question, images)

        from .answer_format import format_answer_with_text, IM_END_NEWLINE
        formatted_answer = format_answer_with_text(_answer, _question)
        text_full = self.processor.apply_chat_template(
            [{"role": "user",      "content": content},
             {"role": "assistant", "content": formatted_answer}],
            tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        proc_out = self.processor(
            text=[text_full], images=images,
            return_tensors="pt", padding=False,
        )
        suffix_ids = self.processor.tokenizer(
            formatted_answer + IM_END_NEWLINE, add_special_tokens=False
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
            "image_xyz":          image_xyz,
            "image_xyz_hires":    image_xyz_hires,
            "labels":             labels,
        }


def load_testing_dataset(
    data_dir: Path,
    limit: Optional[int] = None,
    dataset: str = "mmsibench",
) -> List[Dict[str, Any]]:
    """Load evaluation dataset.

    Supports: mmsibench | mindcube | sat | sat_real | sparbench_multi_view |
              sparbench_single_view | sparbench_mv | spinbench | robospatial |
              viewspatial | omnispatial_pt | embspatial
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

    elif dataset == "embspatial":
        # EmbSpatial-Bench (ACL 2024 Findings) — single-image egocentric
        # spatial relations on mp3d/ai2thor/scannet scenes. 3,640 Q across 6
        # relations: close/far/left/right/above/under.
        # Schema: data_source, question_id, question, relation, image (base64),
        # answer_options (list of 4), answer (int 0-3), objects (list of bbox).
        # Images are extracted to data_dir/images/ on first load (similar to
        # robospatial); subsequent runs reuse the cache.
        import base64, io
        json_file = data_dir / "embspatial_bench.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        images_cache_dir = data_dir / "images"
        images_cache_dir.mkdir(exist_ok=True)
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        _letters = "ABCDEFGHIJ"
        for item in raw:
            qid = item.get("question_id", "")
            img_path = images_cache_dir / f"{qid}.jpg"
            if not img_path.exists():
                img_b64 = item.get("image", "")
                img_bytes = base64.b64decode(img_b64)
                Image.open(io.BytesIO(img_bytes)).convert("RGB").save(img_path, "JPEG")
            options = item.get("answer_options", [])
            ans_idx = item.get("answer", -1)
            answer_letter = _letters[ans_idx] if 0 <= ans_idx < len(options) else ""
            choices_text = "\n".join(
                f"{_letters[i]}. {opt}" for i, opt in enumerate(options)
            )
            question_text = item.get("question", "") + "\n" + choices_text
            samples.append({
                "index": qid,
                "image": [str(img_path.resolve())],
                "question": question_text,
                "answer": answer_letter,
                "category": item.get("relation", "unknown"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset == "omnispatial_pt":
        # OmniSpatial — Perspective_Taking subset only (ICLR 2026).
        # data.json has 1533 entries across 4 task_types; we filter to
        # Perspective_Taking (561 Q: Allocentric 376 / Hypothetical 83 / Egocentric 102).
        # id format "{image_number}_{question_number}" → image at
        # Perspective_Taking/{image_number}.png. answer is int index into options.
        json_file = data_dir / "OmniSpatial-test" / "data.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw = [d for d in raw if d.get("task_type") == "Perspective_Taking"]
        if limit is not None:
            raw = raw[:limit]
        _letters = "ABCDEFGHIJ"
        for item in raw:
            qid = item.get("id", "")
            img_num = qid.split("_")[0] if qid else ""
            img_path = data_dir / "OmniSpatial-test" / "Perspective_Taking" / f"{img_num}.png"
            options = item.get("options", [])
            ans_idx = item.get("answer", -1)
            answer_letter = _letters[ans_idx] if 0 <= ans_idx < len(options) else ""
            choices_text = "\n".join(
                f"{_letters[i]}. {opt}" for i, opt in enumerate(options)
            )
            question_text = item.get("question", "") + "\n" + choices_text
            samples.append({
                "index": qid,
                "image": [str(img_path.resolve())],
                "question": question_text,
                "answer": answer_letter,
                "category": item.get("sub_task_type", "unknown"),
                "thought": "",
                "data_dir": str(data_dir),
            })

    elif dataset == "viewspatial":
        # ViewSpatial-Bench — perspective-taking benchmark on ScanNet+COCO.
        # JSON file: ViewSpatial-Bench.json with 5,712 entries.
        json_file = data_dir / "ViewSpatial-Bench.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if limit is not None:
            raw = raw[:limit]
        for idx, item in enumerate(raw):
            image_paths = []
            for p in item.get("image_path", []):
                rel = p.split("/", 1)[1] if p.startswith("ViewSpatial-Bench/") else p
                image_paths.append(str((data_dir / rel).resolve()))
            answer_raw = item.get("answer", "").strip()
            answer_letter = answer_raw.split(".", 1)[0].strip() if "." in answer_raw else answer_raw
            question_text = item.get("question", "") + "\n" + item.get("choices", "")
            samples.append({
                "index": idx,
                "image": image_paths,
                "question": question_text,
                "answer": answer_letter,
                "category": item.get("question_type", "unknown"),
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

            img_data = row.get("img")
            if img_data is None:
                continue
            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
            img_path = images_cache_dir / f"{sample_id}.jpg"
            if not img_path.exists():
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(img_path, "JPEG")

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
            "Choose: mmsibench | mindcube | sat | sat_real | "
            "sparbench_multi_view | sparbench_single_view | sparbench_mv | "
            "spinbench | robospatial | viewspatial | omnispatial_pt | embspatial"
        )

    return samples


def chunk_dataset(dataset: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [dataset]
    chunk_size = math.ceil(len(dataset) / num_shards)
    return [dataset[s : s + chunk_size] for s in range(0, len(dataset), chunk_size)]
