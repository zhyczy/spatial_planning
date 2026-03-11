"""SFT Data Annotation via Rollout for Challenging SAT Samples.

Full 6-step pipeline:

  1. Load      – Read hard cases from challenging_samples_2k.json (SAT format).

  2. Rollout   – Qwen3.5-4B (planner) generates exactly NUM_ROLLOUTS non-empty
                 instruction sets per hard case.  Empty-instruction slots are
                 retried up to MAX_RETRY_PER_SLOT times.

  3. Execution – Each instruction is fed to Flux2Klein to produce a candidate image.

  4. Critique  – Qwen3.5-9B (critic, thinking mode ON) scores each
                 (original_images, question, generated_image) triple:
                   Phase A – Scoring  : JSON score in [-2, -1, 0, 1, 2]
                   Phase B – Diagnosis: targeted diagnostic prompt based on score
                     low  (≤ -1) → "core failure" analysis – what went wrong & why
                     neutral (0) → "irrelevance" analysis – why it had no effect
                     high (≥  1) → "winning moves" analysis – what made it succeed
                 Score semantics:
                   -2 = completely wrong / misleading
                   -1 = partially wrong
                    0 = neutral / no useful information
                    1 = partially helpful
                    2 = completely helpful / decisive

  5. Rewrite   – A text-LM rewrites each critic diagnostic into a Planner <think>
                 block (perspective flip):
                     low     diagnosis → "anticipate & avoid" thinking + improved instruction
                     neutral diagnosis → "redirect focus" thinking + rewritten instruction
                     high    diagnosis → "consolidate lessons" thinking + original instruction
                 This produces the golden reference completions for SFT.

  6. SFT Build – Converts rewritten records into Qwen-VL-Series-Finetune SFT
                 format (sft_training.json) ready for 1-2 epoch SFT on
                 Qwen3-VL-4B-Instruct.

Output files (in timestamped run_dir):
  rollouts.jsonl          – raw rollout results
  generated_images/       – Flux2Klein generated images
  sft_annotations.jsonl   – per-instruction critique records
  sft_rewrites.jsonl      – per-instruction rewrite records
  sft_training.json       – final SFT training data (Qwen VL format)

Record schema (sft_annotations.jsonl):
  {
    "id":                    "<database_idx>_r<ri>_i<ii>",
    "database_idx":          int,
    "question":              str,
    "answer_choices":        [str, ...],
    "correct_answer":        str,
    "original_images":       [str, ...],
    "instruction":           str,
    "rollout_raw":           str,
    "rollout_idx":           int,
    "instr_idx":             int,
    "generated_image":       str | null,
    "critic_score":          int | null,      # -2..2
    "critic_thinking":       str | null,      # <think> content from Phase A
    "critic_summary":        str | null,      # post-</think> from Phase A
    "critic_full":           str | null,
    "critic_diagnosis":      str | null,      # Phase B targeted diagnosis text
    "critic_diagnosis_type": str | null,      # "low" | "high" | "neutral"
  }

SFT training record schema (sft_training.json):
  {
    "image":         [str, ...],
    "conversations": [
      {"from": "system", "value": SYSTEM_PROMPT},
      {"from": "human",  "value": "<image>...\\n{question}"},
      {"from": "gpt",   "value": "<think>...rewritten thinking...</think>\\n<instructions>...</instructions>"},
    ],
    "metadata": {...},
  }

Usage:
  python annotate_sft_rollout.py \\
      --data_path  data_annotation/challenging_sat/challenging_samples_2k.json \\
      --image_root datasets/evaluation/SAT \\
      --planner_model_path checkpoints/Qwen3.5-4B \\
      --critic_model_path  checkpoints/Qwen3.5-9B \\
      --flux_ckpt          checkpoints/flux2-klein-4B \\
      --output_dir         data_annotation/sft_rollout_annotations \\
      --num_rollouts 8 --num_gpus -1

  # Resume after execution, run critique + rewrite + build only
  python annotate_sft_rollout.py \\
      --resume_from data_annotation/sft_rollout_annotations/run_20260310_120000 \\
      --skip_rollout --skip_execution
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SPATIAL_PLANNING_DIR = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(SPATIAL_PLANNING_DIR))

from generate_image_instructions import (
    SYSTEM_PROMPT,
    build_message,
    chunk_dataset,
    parse_instructions,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_ROLLOUTS_DEFAULT = 8
MAX_RETRY_PER_SLOT   = 5   # max extra calls to fill one empty-instruction slot
ROLLOUT_TEMPERATURE  = 0.9
ROLLOUT_TOP_P        = 0.95
ROLLOUT_MAX_TOKENS   = 2048

# ── Critic system prompt ───────────────────────────────────────────────────────
CRITIC_SYSTEM_PROMPT = """\
You are a spatial reasoning critic evaluating the quality of image-generation \
instructions for a vision-language model.

You will be given:
  1. One or more ORIGINAL scene images from a spatial reasoning question.
  2. The spatial reasoning QUESTION (with answer choices if applicable).
  3. ONE NEWLY GENERATED image produced by following a specific instruction.
  4. The correct answer to the question.

Your task:
  (a) Think carefully in <think>…</think> about whether the generated image helps \
the model reach the correct answer.  Be specific:
      - Does the new image reveal spatial information that was missing from the \
originals?
      - Does it disambiguate the spatial relationships needed to correctly answer \
the question?
      - Is the generated image physically plausible and free of hallucinations?
      - Does it introduce misleading or incorrect spatial information?
      - Overall, does this instruction guide the model toward (+) or away from (-) \
the correct answer?

  (b) After </think>, output ONLY a JSON object with two keys:
      • "score"  : integer in [-2, -1, 0, 1, 2]
          -2 = completely wrong/misleading — directly contradicts the correct answer
          -1 = partially wrong — some elements are incorrect or confusing
           0 = neutral / no useful information
           1 = partially helpful — some relevant spatial information revealed
           2 = completely helpful — clearly reveals decisive spatial information
      • "reason" : one-sentence summary of your judgment

Example output:
<think>
The question asks about the relative position of the cabinet and the door.
The original two images only show the front of the cabinet.
The generated image shows a side view, revealing that the cabinet is to the
LEFT of the door — directly supporting the correct answer "moved left".
The image is visually coherent and plausible; no hallucinations detected.
This instruction is very helpful.
</think>
{"score": 2, "reason": "Side-view image clearly reveals the cabinet is left of the door, confirming the correct answer."}"""

# ── Diagnostic prompt for LOW-score instructions (score ≤ -1) ─────────────────
CRITIC_DIAGNOSTIC_LOW_PROMPT = """\
You are a spatial reasoning expert diagnosing a FAILED image-generation instruction.

Context:
  - The instruction was fed to an image generator (Flux2Klein).
  - The resulting image HURT a vision-language model's ability to answer the
    spatial reasoning question correctly.

Your diagnostic task — answer these three questions concisely:

  1. Core Failure: Exactly where does the generated image contradict or fail to
     support the correct answer? What spatial relationship went wrong?

  2. Instruction Defect: What ambiguous, missing, or incorrect phrasing in the
     instruction caused the generator to render the wrong spatial layout?
     (e.g., missing depth cues, ambiguous directional words, absent occlusion
     constraints, vague object size ratios)

  3. Improvement Direction: If you had to rewrite the instruction to fix these
     defects, what specific language changes would you make?

Be specific and concise.  Output plain text — no JSON, no extra formatting."""

# ── Diagnostic prompt for HIGH-score instructions (score ≥ 1) ─────────────────
CRITIC_DIAGNOSTIC_HIGH_PROMPT = """\
You are a spatial reasoning expert analysing a SUCCESSFUL image-generation instruction.

Context:
  - The instruction was fed to an image generator (Flux2Klein).
  - The resulting image HELPED a vision-language model answer the spatial
    reasoning question correctly.

Your analysis task — answer these three questions concisely:

  1. Winning Moves: Which specific phrases or descriptors in the instruction were
     most responsible for the generator rendering the correct spatial layout?
     (e.g., explicit coordinate anchors, named object size ratios, viewpoint
     specification, occlusion-free framing constraints)

  2. Spatial Logic: Why did these choices successfully disambiguate the spatial
     scene — what potential failure modes did they prevent?

  3. Reusable Lessons: Distil the instruction's success into 1-3 concise
     principles that a planner should apply when generating future instructions
     for similar spatial reasoning questions.

Be specific and concise.  Output plain text — no JSON, no extra formatting."""

# ── Diagnostic prompt for NEUTRAL-score instructions (score == 0) ──────────────
CRITIC_DIAGNOSTIC_NEUTRAL_PROMPT = """\
You are a spatial reasoning expert diagnosing a NEUTRAL image-generation instruction.

Context:
  - The instruction was fed to an image generator (Flux2Klein).
  - The resulting image had NO POSITIVE EFFECT on a vision-language model's ability
    to answer the spatial reasoning question (score = 0).

Your diagnostic task — answer these three questions concisely:

  1. Why It Didn't Help: What spatial information is actually needed to answer
     the question, and why did the generated image fail to provide it?
     (e.g., wrong viewpoint, irrelevant objects focused on, no new information
     compared to the original images, spatial relationship not captured)

  2. What the Instruction Missed: Which key spatial attributes or constraints
     were absent from the instruction that caused the generator to produce an
     unhelpful view?

  3. How to Make It Useful: Describe specifically what a better instruction
     should include to produce an image that genuinely helps answer the question.

Be specific and concise.  Output plain text — no JSON, no extra formatting."""

# ── Rewrite system prompt (perspective flip: critic → planner <think>) ─────────
REWRITE_SYSTEM_PROMPT = """\
You are a spatial planning trainer.  Your job is to convert a critic's diagnostic
report about an image-generation instruction into a high-quality Planner thinking
trace — the internal <think>...</think> reasoning the Planner SHOULD have produced
BEFORE writing that instruction (or an improved version of it).

The Planner's goal: given a spatial reasoning question and its original scene
images, decide what additional image to generate so that a VQA model can answer
correctly.

Output format (strict):
<think>
[The Planner's forward-looking spatial reasoning.  First-person, present tense.
Reference the question and spatial relationships explicitly.  Explain WHAT spatial
information is needed, WHY the chosen viewpoint/framing/description will provide it,
and HOW to avoid the failure mode(s) identified in the critique.
For LOW-score cases: end with "Therefore I will write the instruction to explicitly
specify ...".
For NEUTRAL-score cases: end with "The original instruction failed to capture X;
I must instead focus the generator on ..."
For HIGH-score cases: end with "This strategy of ... ensures the generator places
X correctly."]
</think>
<instructions>
<instruction>[Final, self-contained image-generation instruction in English.
For LOW-score cases: an improved instruction that fixes the diagnosed defects.
For NEUTRAL-score cases: a rewritten instruction that redirects the generator
  toward the missing spatial information.
For HIGH-score cases: the original successful instruction (possibly lightly polished).]
</instruction>
</instructions>

Rules:
- The <think> block must be in English.
- The <instruction> must be in English and directly usable as a Flux prompt.
- Do NOT include the question text, answer choices, or meta-commentary outside
  the specified tags.
- The <think> block should be 100-300 words.  Do not pad unnecessarily."""


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_challenging_sat(data_path: str, image_root: str,
                         max_samples: int = -1) -> List[dict]:
    """Load challenging_samples_2k.json (SAT format).

    Each record has keys: database_idx, question, answer_choices, correct_answer,
    img_paths, baseline_output, baseline_correct, ...
    """
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    if max_samples > 0:
        raw = raw[:max_samples]

    result = []
    for s in raw:
        question = s["question"]
        choices = s.get("answer_choices", [])
        if choices:
            choice_text = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(choices))
            question_with_choices = f"{question}\n{choice_text}"
        else:
            question_with_choices = question

        image_paths = [
            os.path.join(image_root, p) for p in s.get("img_paths", [])
        ]

        result.append({
            "id": s["database_idx"],
            "question": question_with_choices,
            "question_bare": question,
            "image_paths": image_paths,
            "answer_choices": choices,
            "correct_answer": s.get("correct_answer", ""),
            "meta": {
                k: v for k, v in s.items()
                if k not in ("database_idx", "question", "answer_choices",
                             "correct_answer", "img_paths")
            },
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(model_path: str, device: str,
                              enable_thinking: bool = False):
    """Load a Qwen VL model (any generation) on *device*.

    For Qwen3.5 (hybrid linear-attention): uses sdpa only for transformer
    blocks (FA2 crashes at generate() time despite loading cleanly).
    All other models: try FA2 first, fall back to sdpa.

    Returns (model, processor, enable_thinking_flag).
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    cfg = AutoConfig.from_pretrained(model_path)
    cfg_model_type = (cfg.model_type or "").lower()

    if cfg_model_type == "qwen3_5":
        attn_impls = ["sdpa"]
        try:
            import fla  # noqa: F401
        except Exception as fla_exc:
            logger.warning(
                f"flash-linear-attention not available ({fla_exc}). "
                "Qwen3.5 linear-attention will use slow torch fallback."
            )
    else:
        attn_impls = ["flash_attention_2", "sdpa"]
        enable_thinking = False  # Thinking mode only valid for Qwen3.5

    last_exc: Optional[Exception] = None
    model = None
    attn_impl_used = "sdpa"
    for attn_impl in attn_impls:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=device,
                attn_implementation=attn_impl,
            )
            attn_impl_used = attn_impl
            break
        except Exception as exc:
            last_exc = exc
            continue

    if model is None:
        raise RuntimeError(
            f"Failed to load model from {model_path}"
        ) from last_exc

    processor = AutoProcessor.from_pretrained(model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.eval()
    logger.info(
        f"Loaded {type(model).__name__} from {model_path} "
        f"on {device} [{attn_impl_used}, thinking={enable_thinking}]"
    )
    return model, processor, enable_thinking


def run_inference(model, processor, messages: List[dict],
                  max_new_tokens: int = 2048,
                  temperature: float = 0.7,
                  top_p: float = 0.9,
                  do_sample: bool = True,
                  enable_thinking: bool = False) -> str:
    """Run VL model inference and return generated text (decoded, stripped)."""
    from qwen_vl_utils import process_vision_info

    chat_kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        chat_kwargs["enable_thinking"] = True

    text_prompt = processor.apply_chat_template(messages, **chat_kwargs)
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
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    trimmed = [out[len(inp):]
               for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_thinking(text: str) -> tuple[str, str]:
    """Split model output into (thinking_content, post_think_text).

    Returns ('', text) if no <think> block is found.
    """
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def parse_critic_score(text: str) -> Optional[int]:
    """Extract integer score in [-2, 2] from critic output JSON."""
    m = re.search(r'"score"\s*:\s*(-?\d+)', text)
    if m:
        val = int(m.group(1))
        return max(-2, min(2, val))
    # Fallback: look for any integer in [-2, 2] near relevant keywords
    nums = re.findall(r'-?\d+', text)
    valid = [int(n) for n in nums if -2 <= int(n) <= 2]
    return valid[0] if valid else None


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Rollout — generate exactly num_rollouts non-empty instruction sets
# ══════════════════════════════════════════════════════════════════════════════

def rollout_worker(
    gpu_id: str,
    samples: list,
    model_path: str,
    num_rollouts: int,
    max_retry: int,
    output_path: str,
    log_file: Optional[str] = None,
):
    """Run rollouts for *samples* on a single GPU.

    For each sample we need exactly *num_rollouts* "slots", each slot
    containing at least one non-empty instruction.  If a slot yields an
    empty instruction list, we retry it up to *max_retry* times.
    """
    _setup_logging(gpu_id, log_file)
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    model, processor, _ = load_model_and_processor(model_path, device)
    log.info(
        f"Planner on {device}. "
        f"{len(samples)} samples × {num_rollouts} rollouts."
    )

    results = []
    with open(output_path, "w") as f:
        for si, sample in enumerate(samples):
            rollouts: list = []
            slot = 0
            while slot < num_rollouts:
                attempt = 0
                raw_output = ""
                instructions: List[str] = []
                while attempt <= max_retry:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        build_message(sample["question"], sample["image_paths"]),
                    ]
                    try:
                        raw_output = run_inference(
                            model, processor, messages,
                            max_new_tokens=ROLLOUT_MAX_TOKENS,
                            temperature=ROLLOUT_TEMPERATURE,
                            top_p=ROLLOUT_TOP_P,
                            do_sample=True,
                        )
                        instructions = parse_instructions(raw_output)
                    except Exception as exc:
                        log.error(
                            f"Sample {sample['id']} slot {slot} attempt {attempt}: {exc}"
                        )
                        raw_output = ""
                        instructions = []

                    if instructions:          # non-empty → accept this slot
                        break
                    attempt += 1
                    log.warning(
                        f"Sample {sample['id']} slot {slot}: empty instructions, "
                        f"retry {attempt}/{max_retry}"
                    )

                rollouts.append({
                    "rollout_idx": slot,
                    "raw_output": raw_output,
                    "instructions": instructions,
                    "retry_count": attempt,
                })
                slot += 1

            record = {
                "id": sample["id"],
                "question": sample["question"],
                "image_paths": sample["image_paths"],
                "correct_answer": sample["correct_answer"],
                "answer_choices": sample["answer_choices"],
                "meta": sample.get("meta", {}),
                "rollouts": rollouts,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            log.info(
                f"[{si + 1}/{len(samples)}] id={sample['id']}  "
                f"slots: {[len(r['instructions']) for r in rollouts]}"
            )

    log.info(f"Rollout done. {len(results)} samples → {output_path}")


def run_rollouts(
    samples: list,
    model_path: str,
    num_rollouts: int,
    max_retry: int,
    output_dir: Path,
    num_gpus: int,
) -> tuple[list, Path]:
    """Shard samples across GPUs and collect rollout results."""
    all_gpu_ids = _resolve_gpu_ids(num_gpus)
    log_file = str(output_dir / "rollout.log")
    chunks = chunk_dataset(samples, len(all_gpu_ids))
    shard_paths: List[str] = []
    processes: List[mp.Process] = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = str(output_dir / f"rollout_shard_{idx}.jsonl")
        shard_paths.append(shard_path)
        p = mp.Process(
            target=rollout_worker,
            args=(gpu_id, chunk, model_path, num_rollouts, max_retry,
                  shard_path, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(f"Rollout worker {idx} → GPU {gpu_id} ({len(chunk)} samples)")

    for p in processes:
        p.join()

    all_records = _merge_jsonl_shards(shard_paths)
    merged_path = output_dir / "rollouts.jsonl"
    _write_jsonl(all_records, merged_path)
    logger.info(f"Rollouts merged: {len(all_records)} samples → {merged_path}")
    return all_records, merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Execution — generate images via Flux2Klein
# ══════════════════════════════════════════════════════════════════════════════

def execution_worker(
    gpu_id: str,
    tasks: list,
    flux_ckpt: str,
    output_root: str,
    num_inference_steps: int,
    log_file: Optional[str] = None,
):
    """Generate images for a list of task dicts on one GPU."""
    _setup_logging(gpu_id, log_file)
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    from diffusers import Flux2KleinPipeline  # type: ignore
    log.info(f"Loading Flux2KleinPipeline on {device} ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        flux_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    log.info("Generator ready.")

    out_root = Path(output_root)
    for ti, task in enumerate(tasks):
        sample_id   = task["sample_id"]
        rollout_idx = task["rollout_idx"]
        instr_idx   = task["instr_idx"]
        instruction = task["instruction"]
        image_paths = task["image_paths"]

        out_dir = out_root / str(sample_id) / f"rollout_{rollout_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"img_{instr_idx}.png"

        if out_path.exists():
            log.info(f"[{ti + 1}/{len(tasks)}] Already exists: {out_path}")
            continue

        src_images: List[Image.Image] = []
        for p in image_paths:
            if os.path.isfile(p):
                try:
                    src_images.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass

        if not src_images:
            log.warning(f"No source images for sample {sample_id}, skipping.")
            continue

        try:
            result = pipe(
                image=src_images,
                prompt=instruction,
                num_inference_steps=num_inference_steps,
            )
            result.images[0].save(out_path)
            log.info(
                f"[{ti + 1}/{len(tasks)}] "
                f"{sample_id}/rollout_{rollout_idx}/img_{instr_idx} saved."
            )
        except Exception as exc:
            log.error(
                f"Generation failed {sample_id}/r{rollout_idx}/i{instr_idx}: {exc}"
            )


def run_execution(
    rollout_records: list,
    flux_ckpt: str,
    output_dir: Path,
    num_inference_steps: int,
    num_gpus: int,
) -> Path:
    """Build flat task list from rollout records and generate images in parallel."""
    gen_dir = output_dir / "generated_images"
    gen_dir.mkdir(parents=True, exist_ok=True)

    tasks: list = []
    for rec in rollout_records:
        for rollout in rec["rollouts"]:
            ri = rollout["rollout_idx"]
            for ii, instruction in enumerate(rollout["instructions"]):
                if instruction.strip():
                    tasks.append({
                        "sample_id":   rec["id"],
                        "rollout_idx": ri,
                        "instr_idx":   ii,
                        "instruction": instruction,
                        "image_paths": rec["image_paths"],
                    })

    if not tasks:
        logger.warning("No tasks to execute (all rollouts produced empty instructions).")
        return gen_dir

    logger.info(f"Execution: {len(tasks)} image generation tasks.")

    all_gpu_ids = _resolve_gpu_ids(num_gpus)
    log_file = str(output_dir / "execution.log")
    chunks = chunk_dataset(tasks, len(all_gpu_ids))
    processes = []
    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        p = mp.Process(
            target=execution_worker,
            args=(gpu_id, chunk, str(flux_ckpt), str(gen_dir),
                  num_inference_steps, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(
            f"Execution worker {idx} → GPU {gpu_id} ({len(chunk)} tasks)"
        )

    for p in processes:
        p.join()

    logger.info(f"Execution done. Images saved under {gen_dir}")
    return gen_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: Critique — Qwen3.5-9B (thinking) annotates each generated image
# ══════════════════════════════════════════════════════════════════════════════

def critique_worker(
    gpu_id: str,
    samples: list,
    critic_model_path: str,
    gen_dir: str,
    output_path: str,
    low_score_threshold: int = -1,
    high_score_threshold: int = 1,
    log_file: Optional[str] = None,
):
    """Annotate each (original_images, question, generated_image) triple.

    Scores are in [-2, 2].  The full <think>…</think> block from the critic
    is preserved as the SFT supervision signal.
    """
    _setup_logging(gpu_id, log_file)
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    model, processor, enable_thinking = load_model_and_processor(
        critic_model_path, device, enable_thinking=True
    )
    log.info(
        f"Critic on {device}. Annotating {len(samples)} samples "
        f"(thinking={enable_thinking})."
    )

    gen_root = Path(gen_dir)
    sft_records: list = []

    with open(output_path, "w") as f:
        for si, sample in enumerate(samples):
            sample_id    = sample["id"]
            question     = sample["question"]
            image_paths  = sample["image_paths"]
            correct_ans  = sample.get("correct_answer", "")
            answer_choices = sample.get("answer_choices", [])

            for rollout in sample["rollouts"]:
                ri           = rollout["rollout_idx"]
                raw_output   = rollout.get("raw_output", "")
                instructions = rollout.get("instructions", [])

                for ii, instruction in enumerate(instructions):
                    if not instruction.strip():
                        continue  # should not happen after enforced non-empty rollout

                    gen_path = gen_root / str(sample_id) / f"rollout_{ri}" / f"img_{ii}.png"
                    gen_path_str = str(gen_path) if gen_path.exists() else None

                    record_id = f"{sample_id}_r{ri}_i{ii}"

                    # Base record (filled even if critique fails)
                    sft_rec: dict = {
                        "id":               record_id,
                        "database_idx":     sample_id,
                        "question":         question,
                        "answer_choices":   answer_choices,
                        "correct_answer":   correct_ans,
                        "original_images":  image_paths,
                        "instruction":      instruction,
                        "rollout_raw":      raw_output,
                        "rollout_idx":      ri,
                        "instr_idx":        ii,
                        "generated_image":  gen_path_str,
                        "critic_score":     None,
                        "critic_thinking":  None,
                        "critic_summary":   None,
                        "critic_full":      None,
                        "critic_diagnosis":      None,  # Phase B diagnostic text
                        "critic_diagnosis_type": None,  # "low" | "high" | "neutral"
                    }

                    if gen_path_str is None:
                        log.warning(
                            f"{record_id}: generated image not found, skipping critique."
                        )
                        sft_records.append(sft_rec)
                        f.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")
                        f.flush()
                        continue

                    # ── Build critic message ───────────────────────────────
                    # Content: original images → generated image → question → correct answer
                    content: list = []
                    for op in image_paths:
                        if os.path.isfile(op):
                            content.append({"type": "image", "image": op})
                    content.append({"type": "image", "image": gen_path_str})
                    content.append({
                        "type": "text",
                        "text": (
                            f"Question: {question}\n\n"
                            f"Correct answer: {correct_ans}\n\n"
                            f"The last image above was generated by following "
                            f"this instruction:\n\"{instruction}\"\n\n"
                            f"Evaluate whether this generated image helps a "
                            f"vision-language model arrive at the correct answer. "
                            f"Think step by step in <think>…</think>, then output "
                            f"a JSON object with keys \"score\" (integer -2 to 2) "
                            f"and \"reason\" (brief explanation)."
                        ),
                    })

                    messages = [
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {"role": "user",   "content": content},
                    ]

                    try:
                        critic_out = run_inference(
                            model, processor, messages,
                            max_new_tokens=4096,
                            temperature=0.6,
                            top_p=0.95,
                            do_sample=True,
                            enable_thinking=enable_thinking,
                        )
                        thinking, summary = extract_thinking(critic_out)
                        score = parse_critic_score(critic_out)

                        sft_rec["critic_score"]    = score
                        sft_rec["critic_thinking"] = thinking
                        sft_rec["critic_summary"]  = summary
                        sft_rec["critic_full"]     = critic_out

                        log.info(
                            f"{record_id}  score={score}  "
                            f"thinking_len={len(thinking)}  "
                            f"summary={summary[:80]!r}"
                        )

                        # ── Phase B: targeted diagnosis based on score ─────
                        if score is not None and score <= low_score_threshold:
                            diag_type = "low"
                            diag_prompt = CRITIC_DIAGNOSTIC_LOW_PROMPT
                        elif score is not None and score >= high_score_threshold:
                            diag_type = "high"
                            diag_prompt = CRITIC_DIAGNOSTIC_HIGH_PROMPT
                        elif score == 0:
                            # Neutral: instruction is not helpful — still diagnose
                            diag_type = "neutral"
                            diag_prompt = CRITIC_DIAGNOSTIC_NEUTRAL_PROMPT
                        else:
                            diag_type = "neutral"
                            diag_prompt = CRITIC_DIAGNOSTIC_NEUTRAL_PROMPT

                        diag_content: list = []
                        for op in image_paths:
                            if os.path.isfile(op):
                                diag_content.append({"type": "image", "image": op})
                        diag_content.append({"type": "image", "image": gen_path_str})
                        diag_content.append({
                            "type": "text",
                            "text": (
                                f"Spatial reasoning question: {question}\n\n"
                                f"Correct answer: {correct_ans}\n\n"
                                f"Image-generation instruction that was used:\n"
                                f"\"{instruction}\"\n\n"
                                f"Critic score given: {score} / 2\n\n"
                                f"Now perform your diagnostic analysis as instructed."
                            ),
                        })
                        diag_messages = [
                            {"role": "system", "content": diag_prompt},
                            {"role": "user",   "content": diag_content},
                        ]
                        try:
                            diagnosis = run_inference(
                                model, processor, diag_messages,
                                max_new_tokens=1024,
                                temperature=0.5,
                                top_p=0.9,
                                do_sample=True,
                                enable_thinking=enable_thinking,
                            )
                            # Strip any stray <think> block the model might emit
                            _, diagnosis_clean = extract_thinking(diagnosis)
                            sft_rec["critic_diagnosis"]      = diagnosis_clean or diagnosis
                            sft_rec["critic_diagnosis_type"] = diag_type
                            log.info(
                                f"{record_id}  diagnosis_type={diag_type}  "
                                f"diagnosis_len={len(sft_rec['critic_diagnosis'])}"
                            )
                        except Exception as diag_exc:
                            sft_rec["critic_diagnosis_type"] = diag_type
                            log.error(
                                f"Diagnosis failed for {record_id}: {diag_exc}"
                            )
                    except Exception as exc:
                        log.error(f"Critique failed for {record_id}: {exc}")

                    sft_records.append(sft_rec)
                    f.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")
                    f.flush()

            log.info(f"[{si + 1}/{len(samples)}] id={sample_id} done.")

    log.info(f"Critique done. {len(sft_records)} records → {output_path}")


def run_critique(
    rollout_records: list,
    critic_model_path: str,
    gen_dir: Path,
    output_dir: Path,
    num_gpus: int,
    low_score_threshold: int = -1,
    high_score_threshold: int = 1,
) -> tuple[list, Path]:
    """Shard rollout records across GPUs and run critique in parallel."""
    all_gpu_ids = _resolve_gpu_ids(num_gpus)
    log_file = str(output_dir / "critique.log")
    chunks = chunk_dataset(rollout_records, len(all_gpu_ids))
    shard_paths: List[str] = []
    processes: List[mp.Process] = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = str(output_dir / f"critique_shard_{idx}.jsonl")
        shard_paths.append(shard_path)
        p = mp.Process(
            target=critique_worker,
            args=(gpu_id, chunk, critic_model_path, str(gen_dir),
                  shard_path, low_score_threshold, high_score_threshold,
                  log_file),
        )
        p.start()
        processes.append(p)
        logger.info(
            f"Critique worker {idx} → GPU {gpu_id} ({len(chunk)} samples)"
        )

    for p in processes:
        p.join()

    all_records = _merge_jsonl_shards(shard_paths)
    merged_path = output_dir / "sft_annotations.jsonl"
    _write_jsonl(all_records, merged_path)
    logger.info(
        f"Critique merged: {len(all_records)} SFT records → {merged_path}"
    )
    return all_records, merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: Rewrite — perspective-flip critic diagnoses into Planner <think>
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_worker(
    gpu_id: str,
    records: list,
    rewrite_model_path: str,
    output_path: str,
    log_file: Optional[str] = None,
):
    """Convert critic diagnostic reports into Planner <think> + instruction.

    Uses the rewrite model in text-only mode (no images needed for the flip).
    Each record must have non-null `critic_diagnosis` and `critic_diagnosis_type`
    in {"low", "high"}.  Neutral or failed records are passed through unchanged.
    """
    _setup_logging(gpu_id, log_file)
    log = logging.getLogger(__name__)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))

    # Load model — text-only inference, so we try CausalLM first, then VL.
    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(rewrite_model_path)
    model_type = (cfg.model_type or "").lower()
    has_vision = getattr(cfg, "vision_config", None) is not None
    attn_impls = ["sdpa"] if model_type == "qwen3_5" else ["flash_attention_2", "sdpa"]
    model = None
    last_exc = None
    for attn_impl in attn_impls:
        try:
            if has_vision:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    rewrite_model_path, dtype=torch.bfloat16,
                    device_map=device, attn_implementation=attn_impl,
                )
            else:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    rewrite_model_path, dtype=torch.bfloat16,
                    device_map=device, attn_implementation=attn_impl,
                )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        raise RuntimeError(f"Failed to load rewrite model") from last_exc
    processor = AutoProcessor.from_pretrained(rewrite_model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.eval()
    enable_thinking = model_type == "qwen3_5"
    log.info(
        f"Rewrite model on {device}. Rewriting {len(records)} records "
        f"(thinking={enable_thinking})."
    )

    results = []
    with open(output_path, "w") as f:
        for ri, rec in enumerate(records):
            out_rec = dict(rec)
            out_rec.update({
                "rewrite_think":       None,
                "rewrite_instruction": None,
                "rewrite_full":        None,
            })

            diagnosis      = rec.get("critic_diagnosis")
            diagnosis_type = rec.get("critic_diagnosis_type")

            if not diagnosis or diagnosis_type is None:
                # No diagnosis available — skip rewrite
                results.append(out_rec)
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                f.flush()
                continue

            score = rec.get("critic_score", 0) or 0

            # Label for the user prompt
            _label_map = {"low": "failure", "neutral": "irrelevance", "high": "success"}
            _label = _label_map.get(diagnosis_type, "analysis")

            # Build text-only rewrite prompt
            user_text = (
                f"## Context\n\n"
                f"**Spatial reasoning question:**\n{rec['question']}\n\n"
                f"**Correct answer:** {rec['correct_answer']}\n\n"
                f"**Image-generation instruction used (score {score}/2):**\n"
                f"\"{rec['instruction']}\"\n\n"
                f"**Critic's {_label} analysis:**\n"
                f"{diagnosis}\n\n"
                f"---\n\n"
            )
            if diagnosis_type == "low":
                user_text += (
                    "## Your task\n\n"
                    "This was a LOW-SCORE instruction.  Rewrite the critic's failure "
                    "analysis as the Planner's forward-looking <think> block that:\n"
                    "  1. Identifies what spatial information is needed to answer the question.\n"
                    "  2. Anticipates the exact failure mode the original instruction caused.\n"
                    "  3. Explains how an improved instruction explicitly avoids that failure.\n"
                    "  4. Concludes with an improved <instruction> that fixes the defects.\n\n"
                    "Output ONLY the <think>...</think><instructions><instruction>...</instruction>"
                    "</instructions> block.  No other text."
                )
            elif diagnosis_type == "neutral":
                user_text += (
                    "## Your task\n\n"
                    "This was a NEUTRAL-SCORE instruction (score 0) — the generated image "
                    "provided no useful spatial information.  Rewrite the critic's irrelevance "
                    "analysis as the Planner's forward-looking <think> block that:\n"
                    "  1. Identifies precisely what spatial information IS needed to answer the question.\n"
                    "  2. Explains why the original instruction failed to capture that information.\n"
                    "  3. Describes the key spatial constraints a useful instruction must include.\n"
                    "  4. Concludes with a rewritten <instruction> that redirects the generator "
                    "     toward the missing spatial information.\n\n"
                    "Output ONLY the <think>...</think><instructions><instruction>...</instruction>"
                    "</instructions> block.  No other text."
                )
            else:  # high
                user_text += (
                    "## Your task\n\n"
                    "This was a HIGH-SCORE instruction.  Rewrite the critic's success "
                    "analysis as the Planner's forward-looking <think> block that:\n"
                    "  1. Identifies what spatial information is needed to answer the question.\n"
                    "  2. Explains which specific strategies in the instruction guaranteed the "
                    "     correct spatial layout.\n"
                    "  3. Generalises the winning principles so they guide future instructions.\n"
                    "  4. Concludes with the original <instruction> (lightly polished if needed).\n\n"
                    "Output ONLY the <think>...</think><instructions><instruction>...</instruction>"
                    "</instructions> block.  No other text."
                )

            messages = [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ]

            try:
                # Text-only inference — use run_inference but no images in messages
                rewrite_out = run_inference(
                    model, processor, messages,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    enable_thinking=enable_thinking,
                )
                r_think, r_rest = extract_thinking(rewrite_out)

                # Extract the final instruction from <instruction>...</instruction>
                instr_match = re.search(
                    r"<instruction>(.*?)</instruction>",
                    r_rest, re.DOTALL | re.IGNORECASE,
                )
                r_instruction = instr_match.group(1).strip() if instr_match else ""

                out_rec["rewrite_think"]       = r_think
                out_rec["rewrite_instruction"] = r_instruction
                out_rec["rewrite_full"]        = rewrite_out

                log.info(
                    f"[{ri + 1}/{len(records)}] {rec['id']}  "
                    f"type={diagnosis_type}(score={score})  "
                    f"think_len={len(r_think)}  "
                    f"instr={r_instruction[:60]!r}"
                )
            except Exception as exc:
                log.error(f"Rewrite failed for {rec['id']}: {exc}")

            results.append(out_rec)
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            f.flush()

    log.info(f"Rewrite done. {len(results)} records → {output_path}")


def run_rewrite(
    sft_records: list,
    rewrite_model_path: str,
    output_dir: Path,
    num_gpus: int,
) -> tuple[list, Path]:
    """Shard sft_records across GPUs and run the perspective-flip rewrite."""
    # Shard all records across workers; records without a diagnosis are passed
    # through cheaply.  Neutral records now also receive a rewrite.
    all_gpu_ids = _resolve_gpu_ids(num_gpus)
    log_file = str(output_dir / "rewrite.log")

    # Split all records across shards (pass-through is cheap)
    chunks = chunk_dataset(sft_records, len(all_gpu_ids))
    shard_paths: List[str] = []
    processes: List[mp.Process] = []

    for idx, (gpu_id, chunk) in enumerate(zip(all_gpu_ids, chunks)):
        shard_path = str(output_dir / f"rewrite_shard_{idx}.jsonl")
        shard_paths.append(shard_path)
        p = mp.Process(
            target=rewrite_worker,
            args=(gpu_id, chunk, rewrite_model_path, shard_path, log_file),
        )
        p.start()
        processes.append(p)
        logger.info(
            f"Rewrite worker {idx} → GPU {gpu_id} ({len(chunk)} records)"
        )

    for p in processes:
        p.join()

    all_records = _merge_jsonl_shards(shard_paths)
    merged_path = output_dir / "sft_rewrites.jsonl"
    _write_jsonl(all_records, merged_path)
    logger.info(
        f"Rewrite merged: {len(all_records)} records → {merged_path}"
    )
    return all_records, merged_path


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: Build SFT training data (Qwen-VL-Series-Finetune format)
# ══════════════════════════════════════════════════════════════════════════════

def build_sft_training_data(rewrite_records: list, output_dir: Path) -> list:
    """Convert rewritten records into Qwen VL SFT training examples.

    Only records with a non-empty `rewrite_instruction` and `rewrite_think` are
    included.  Each example follows the Qwen-VL-Series-Finetune conversation
    format:

      system : SYSTEM_PROMPT
      human  : <image>×N + question
      gpt    : <think>{rewrite_think}</think>
               <instructions><instruction>{rewrite_instruction}</instruction></instructions>

    The gpt turn is the full model completion including the CoT thinking block,
    matching the training objective of CoT-GRPO / SFT on thinking models.
    """
    examples = []
    skipped = 0

    for rec in rewrite_records:
        think       = rec.get("rewrite_think") or ""
        instruction = rec.get("rewrite_instruction") or ""
        images      = rec.get("original_images", [])
        question    = rec.get("question", "")

        if not think.strip() or not instruction.strip():
            skipped += 1
            continue

        n_images = len(images)
        image_tokens = "<image>" * n_images
        human_content = f"{image_tokens}\n{question}" if n_images else question

        gpt_content = (
            f"<think>\n{think}\n</think>\n"
            f"<instructions>\n"
            f"<instruction>{instruction}</instruction>\n"
            f"</instructions>"
        )

        examples.append({
            "image": images,
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human",  "value": human_content},
                {"from": "gpt",    "value": gpt_content},
            ],
            "metadata": {
                "id":                    rec.get("id"),
                "database_idx":          rec.get("database_idx"),
                "question":              question,
                "correct_answer":        rec.get("correct_answer"),
                "critic_score":          rec.get("critic_score"),
                "critic_diagnosis_type": rec.get("critic_diagnosis_type"),
                "original_instruction":  rec.get("instruction"),
            },
        })

    output_path = output_dir / "sft_training.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(
        f"SFT training data: {len(examples)} examples "
        f"({skipped} skipped / no rewrite) → {output_path}"
    )

    # Score-type breakdown
    by_type: dict = {"low": 0, "neutral": 0, "high": 0, None: 0}
    for ex in examples:
        t = ex["metadata"].get("critic_diagnosis_type")
        by_type[t] = by_type.get(t, 0) + 1
    logger.info(
        f"  SFT breakdown — low: {by_type.get('low', 0)}, "
        f"high: {by_type.get('high', 0)}"
    )
    return examples


# ══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging(gpu_id: str, log_file: Optional[str] = None):
    handlers: list = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {gpu_id}] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _resolve_gpu_ids(num_gpus: int) -> List[str]:
    n_available = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        all_gpu_ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    else:
        all_gpu_ids = [str(i) for i in range(n_available)]
    if num_gpus > 0:
        all_gpu_ids = all_gpu_ids[:num_gpus]
    if not all_gpu_ids:
        raise RuntimeError(
            "No CUDA GPUs available. "
            "Set CUDA_VISIBLE_DEVICES or ensure GPUs are present "
            f"(torch.cuda.device_count()={n_available})."
        )
    return all_gpu_ids


def _merge_jsonl_shards(shard_paths: List[str]) -> list:
    records = []
    for path in shard_paths:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def _write_jsonl(records: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Statistics summary
# ══════════════════════════════════════════════════════════════════════════════

def print_statistics(sft_records: list, output_dir: Path):
    """Print and save a statistics summary of the SFT annotation run."""
    total = len(sft_records)
    scored = [r for r in sft_records if r["critic_score"] is not None]
    no_gen = [r for r in sft_records if r["generated_image"] is None]
    no_score = [r for r in sft_records if r["generated_image"] and r["critic_score"] is None]

    score_dist: dict = {s: 0 for s in range(-2, 3)}
    for r in scored:
        sc = r["critic_score"]
        if sc in score_dist:
            score_dist[sc] += 1

    avg_score = (sum(r["critic_score"] for r in scored) / len(scored)) if scored else float("nan")

    lines = [
        "=" * 60,
        "SFT Annotation Statistics",
        "=" * 60,
        f"  Total SFT records   : {total}",
        f"  Scored records      : {len(scored)}",
        f"  No generated image  : {len(no_gen)}",
        f"  Critique failed     : {len(no_score)}",
        f"  Average critic score: {avg_score:.3f}",
        "",
        "  Score distribution:",
    ]
    for sc in range(-2, 3):
        pct = 100 * score_dist[sc] / max(len(scored), 1)
        lines.append(f"    {sc:+d} : {score_dist[sc]:5d}  ({pct:.1f}%)")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)

    stat_path = output_dir / "statistics.txt"
    stat_path.write_text(summary, encoding="utf-8")
    logger.info(f"Statistics saved → {stat_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT data annotation via rollout for challenging SAT samples."
    )

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_path", type=str,
        default="data_annotation/challenging_sat/challenging_samples_2k.json",
        help="Path to challenging_samples_2k.json (relative to spatial_planning/).",
    )
    parser.add_argument(
        "--image_root", type=str,
        default="datasets/evaluation/SAT",
        help="Root directory for resolving SAT img_paths.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Max hard cases to process (-1 = all).",
    )

    # ── Models ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--planner_model_path", type=str, required=True,
        help="Qwen3.5-4B checkpoint for instruction generation.",
    )
    parser.add_argument(
        "--critic_model_path", type=str, required=True,
        help="Qwen3.5-9B checkpoint for thinking-mode critique.",
    )
    parser.add_argument(
        "--flux_ckpt", type=str, default="checkpoints/flux2-klein-4B",
        help="Flux2Klein checkpoint for image generation.",
    )

    # ── Rollout settings ─────────────────────────────────────────────────
    parser.add_argument(
        "--num_rollouts", type=int, default=NUM_ROLLOUTS_DEFAULT,
        help="Number of non-empty instruction rollouts per hard case.",
    )
    parser.add_argument(
        "--max_retry", type=int, default=MAX_RETRY_PER_SLOT,
        help="Max retries to fill one empty-instruction slot.",
    )

    # ── Execution settings ────────────────────────────────────────────────
    parser.add_argument(
        "--num_inference_steps", type=int, default=28,
        help="Flux2Klein denoising steps.",
    )

    # ── Multi-GPU ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--num_gpus", type=int, default=-1,
        help="Number of GPUs (-1 = all available).",
    )

    # ── Output & resume ───────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str,
        default="data_annotation/sft_rollout_annotations",
        help="Root output directory; a timestamped sub-folder is created inside.",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Resume from an existing run directory (skip completed steps).",
    )
    parser.add_argument(
        "--skip_rollout", action="store_true",
        help="Skip rollout step (load rollouts.jsonl from resume_from).",
    )
    parser.add_argument(
        "--skip_execution", action="store_true",
        help="Skip image generation step.",
    )
    parser.add_argument(
        "--skip_critique", action="store_true",
        help="Skip critique step (only run rollout + execution).",
    )
    parser.add_argument(
        "--skip_rewrite", action="store_true",
        help="Skip perspective-flip rewrite step.",
    )
    parser.add_argument(
        "--skip_sft_build", action="store_true",
        help="Skip final SFT training data build.",
    )

    # ── Critique thresholds ───────────────────────────────────────────────
    parser.add_argument(
        "--low_score_threshold", type=int, default=-1,
        help="Score ≤ this triggers low-score ('core failure') diagnostic (default: -1).",
    )
    parser.add_argument(
        "--high_score_threshold", type=int, default=1,
        help="Score ≥ this triggers high-score ('winning moves') diagnostic (default: 1).",
    )

    # ── Rewrite model ─────────────────────────────────────────────────────
    parser.add_argument(
        "--rewrite_model_path", type=str, default=None,
        help="Model for perspective-flip rewrite. Defaults to critic_model_path.",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────────────
    if args.resume_from:
        run_dir = Path(args.resume_from).resolve()
        logger.info(f"Resuming from {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (SPATIAL_PLANNING_DIR / args.output_dir / f"run_{timestamp}").resolve()

    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(run_dir / "pipeline.log")),
        ],
        force=True,
    )

    rewrite_model = args.rewrite_model_path or args.critic_model_path

    logger.info("=" * 60)
    logger.info("SFT Rollout Annotation Pipeline (6 steps)")
    logger.info(f"  Run dir            : {run_dir}")
    logger.info(f"  Data path          : {args.data_path}")
    logger.info(f"  Image root         : {args.image_root}")
    logger.info(f"  Planner model      : {args.planner_model_path}")
    logger.info(f"  Critic model       : {args.critic_model_path}")
    logger.info(f"  Rewrite model      : {rewrite_model}")
    logger.info(f"  Flux ckpt          : {args.flux_ckpt}")
    logger.info(f"  Num rollouts       : {args.num_rollouts}")
    logger.info(f"  Max retry/slot     : {args.max_retry}")
    logger.info(f"  Low score thresh   : {args.low_score_threshold}")
    logger.info(f"  High score thresh  : {args.high_score_threshold}")
    logger.info("=" * 60)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Resolve paths relative to spatial_planning directory
    data_path  = str(SPATIAL_PLANNING_DIR / args.data_path)
    image_root = str(SPATIAL_PLANNING_DIR / args.image_root)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Step 1/6: Loading challenging samples ...")
    samples = load_challenging_sat(data_path, image_root, args.max_samples)
    logger.info(f"  Loaded {len(samples)} hard cases.")

    # ── Step 2: Rollout ────────────────────────────────────────────────────
    logger.info("─" * 60)
    rollouts_path = run_dir / "rollouts.jsonl"

    if args.skip_rollout and rollouts_path.exists():
        logger.info("Step 2/6: Loading existing rollouts (--skip_rollout).")
        rollout_records = []
        with open(rollouts_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rollout_records.append(json.loads(line))
        logger.info(f"  Loaded {len(rollout_records)} rollout records.")
    else:
        logger.info(
            f"Step 2/6: Running rollouts ({args.num_rollouts} non-empty slots "
            f"per sample, max_retry={args.max_retry}) ..."
        )
        mp.set_start_method("spawn", force=True)
        rollout_records, rollouts_path = run_rollouts(
            samples=samples,
            model_path=args.planner_model_path,
            num_rollouts=args.num_rollouts,
            max_retry=args.max_retry,
            output_dir=run_dir,
            num_gpus=args.num_gpus,
        )
        logger.info(
            f"  Rollouts complete: {len(rollout_records)} samples × "
            f"{args.num_rollouts} rollouts."
        )

    # ── Step 3: Execution ──────────────────────────────────────────────────
    logger.info("─" * 60)
    gen_dir = run_dir / "generated_images"

    if args.skip_execution:
        logger.info("Step 3/6: Skipping image generation (--skip_execution).")
    else:
        logger.info("Step 3/6: Generating images via Flux2Klein ...")
        gen_dir = run_execution(
            rollout_records=rollout_records,
            flux_ckpt=str(SPATIAL_PLANNING_DIR / args.flux_ckpt),
            output_dir=run_dir,
            num_inference_steps=args.num_inference_steps,
            num_gpus=args.num_gpus,
        )
        logger.info(f"  Image generation complete. Dir: {gen_dir}")

    # ── Step 4: Critique (Phase A: score + Phase B: diagnosis) ────────────
    logger.info("─" * 60)
    sft_annotations_path = run_dir / "sft_annotations.jsonl"

    if args.skip_critique and sft_annotations_path.exists():
        logger.info("Step 4/6: Loading existing critique (--skip_critique).")
        sft_records = []
        with open(sft_annotations_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sft_records.append(json.loads(line))
        logger.info(f"  Loaded {len(sft_records)} annotated records.")
    else:
        logger.info(
            "Step 4/6: Running Qwen3.5-9B critic "
            "(Phase A: score+thinking, Phase B: diagnosis) ..."
        )
        sft_records, sft_annotations_path = run_critique(
            rollout_records=rollout_records,
            critic_model_path=args.critic_model_path,
            gen_dir=gen_dir,
            output_dir=run_dir,
            num_gpus=args.num_gpus,
            low_score_threshold=args.low_score_threshold,
            high_score_threshold=args.high_score_threshold,
        )
        logger.info(
            f"  Critique complete: {len(sft_records)} SFT records → "
            f"{sft_annotations_path}"
        )
        print_statistics(sft_records, run_dir)

    # ── Step 5: Rewrite (perspective flip) ────────────────────────────────
    logger.info("─" * 60)
    sft_rewrites_path = run_dir / "sft_rewrites.jsonl"

    if args.skip_rewrite and sft_rewrites_path.exists():
        logger.info("Step 5/6: Loading existing rewrites (--skip_rewrite).")
        rewrite_records = []
        with open(sft_rewrites_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rewrite_records.append(json.loads(line))
        logger.info(f"  Loaded {len(rewrite_records)} rewrite records.")
    else:
        n_diagnosable = sum(
            1 for r in sft_records
            if r.get("critic_diagnosis_type") in ("low", "neutral", "high")
        )
        logger.info(
            f"Step 5/6: Perspective-flip rewrite for {n_diagnosable} "
            f"diagnosable records (of {len(sft_records)} total) ..."
        )
        rewrite_records, sft_rewrites_path = run_rewrite(
            sft_records=sft_records,
            rewrite_model_path=rewrite_model,
            output_dir=run_dir,
            num_gpus=args.num_gpus,
        )
        logger.info(
            f"  Rewrite complete: {len(rewrite_records)} records → "
            f"{sft_rewrites_path}"
        )

    # ── Step 6: Build SFT training data ───────────────────────────────────
    logger.info("─" * 60)
    if args.skip_sft_build:
        logger.info("Step 6/6: Skipping SFT build (--skip_sft_build).")
    else:
        logger.info("Step 6/6: Building SFT training data ...")
        sft_examples = build_sft_training_data(rewrite_records, run_dir)
        logger.info(
            f"  SFT build complete: {len(sft_examples)} training examples "
            f"→ {run_dir / 'sft_training.json'}"
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
