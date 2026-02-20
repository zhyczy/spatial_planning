"""
SPR (Spatial Perception Reasoning) model.

Uses a standard Qwen2.5-VL to analyze input questions and frames, producing
text-based reasoning about what visual/spatial information from the scene is
needed to answer the question.  The calling convention mirrors
SpatialMLLMForConditionalGeneration so it can be used as a drop-in module.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from qwen_vl_utils import process_vision_info


# ──────────────────────────────────────────────────────────────────────
# Default system prompt – instructs the VLM to perform perception analysis
# ──────────────────────────────────────────────────────────────────────
DEFAULT_SPR_SYSTEM_PROMPT = (
    "You are a spatial perception reasoning assistant. "
    "Given a set of video frames from a 3D scene and a question about that scene, "
    "your task is to analyze what visual and spatial information is needed to answer "
    "the question. You should identify:\n"
    "1. The key objects mentioned or implied in the question.\n"
    "2. The spatial relationships (relative positions, distances, directions) that "
    "must be understood.\n"
    "3. Any viewpoint or perspective considerations.\n"
    "4. What specific visual evidence in the frames supports the reasoning.\n\n"
    "Do NOT answer the question itself. Instead, provide a structured analysis of "
    "the visual information requirements."
)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
class SPRConfig(Qwen2_5_VLConfig):
    """Configuration for SPR – identical to Qwen2_5_VLConfig with an extra
    ``spr_system_prompt`` field so the analysis prompt can be customised from
    the config file."""

    model_type = "spr"

    def __init__(self, spr_system_prompt: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.spr_system_prompt = spr_system_prompt or DEFAULT_SPR_SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────
class SPRForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Spatial Perception Reasoning model.

    Wraps a vanilla Qwen2.5-VL checkpoint and exposes two additional
    high-level helpers (`analyze` / `analyze_batch`) that accept raw
    frames + question(s) and return text analysis.  The low-level
    ``forward`` / ``generate`` remain fully compatible with
    ``Qwen2_5_VLForConditionalGeneration``, so existing training and
    inference pipelines work unchanged.
    """

    config_class = SPRConfig

    def __init__(self, config: SPRConfig):
        super().__init__(config)
        self.spr_system_prompt = getattr(config, "spr_system_prompt", DEFAULT_SPR_SYSTEM_PROMPT)
        # processor will be lazily loaded when needed via high-level API
        self._processor: Optional[Qwen2_5_VLProcessor] = None
        self.post_init()

    # ── lazy processor loader ────────────────────────────────────────
    def _get_processor(self) -> Qwen2_5_VLProcessor:
        """Lazily load the processor from the same directory as the model
        weights (works when loaded via ``from_pretrained``)."""
        if self._processor is None:
            self._processor = Qwen2_5_VLProcessor.from_pretrained(
                self.config._name_or_path
            )
        return self._processor

    def set_processor(self, processor: Qwen2_5_VLProcessor):
        """Manually set the processor (useful when the model path is not
        available, e.g. after ``deepspeed`` wrapping)."""
        self._processor = processor

    # ── message building ─────────────────────────────────────────────
    @staticmethod
    def _build_messages(
        question: str,
        frames: Optional[List] = None,
        video_path: Optional[str] = None,
        nframes: int = 16,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Build the ``messages`` list accepted by
        ``Qwen2_5_VLProcessor.apply_chat_template``.

        Accepts *either* a list of PIL frames *or* a video path.
        """
        user_content: List[Dict] = []

        # --- visual input ---
        if frames is not None:
            # Frames provided as a list of PIL Images
            user_content.append({
                "type": "video",
                "video": frames,
                "nframes": len(frames),
            })
        elif video_path is not None:
            user_content.append({
                "type": "video",
                "video": video_path,
                "nframes": nframes,
            })

        # --- textual input (wrap the original question) ---
        analysis_query = (
            f"Question about this scene: {question}\n\n"
            "Based on the video frames above, analyze what visual and spatial "
            "information from this scene is needed to answer the question. "
            "Identify key objects, spatial relationships, viewpoint considerations, "
            "and relevant visual evidence. Do NOT answer the question itself."
        )
        user_content.append({"type": "text", "text": analysis_query})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── single-sample analysis ───────────────────────────────────────
    @torch.no_grad()
    def analyze(
        self,
        question: str,
        frames: Optional[List] = None,
        video_path: Optional[str] = None,
        nframes: int = 16,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.001,
        **generate_kwargs,
    ) -> str:
        """Run perception analysis for a single question + video.

        Parameters
        ----------
        question : str
            The spatial question to analyse.
        frames : list[PIL.Image], optional
            Pre-extracted video frames.
        video_path : str, optional
            Path to a video file (used if *frames* is ``None``).
        nframes : int
            Number of frames to sample when *video_path* is given.
        system_prompt : str, optional
            Override the default SPR system prompt.
        max_new_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        str
            The generated perception analysis text.
        """
        system_prompt = system_prompt or self.spr_system_prompt
        processor = self._get_processor()

        messages = self._build_messages(
            question=question,
            frames=frames,
            video_path=video_path,
            nframes=nframes,
            system_prompt=system_prompt,
        )

        # Tokenize
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        batch = processor(
            text=[prompt_text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batch = batch.to(self.device)

        # Generate
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        gen_kwargs.update(generate_kwargs)

        generated_ids = self.generate(**batch, **gen_kwargs)

        # Decode – strip the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    # ── batched analysis ─────────────────────────────────────────────
    @torch.no_grad()
    def analyze_batch(
        self,
        questions: List[str],
        frames_list: Optional[List[List]] = None,
        video_paths: Optional[List[str]] = None,
        nframes: int = 16,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.001,
        **generate_kwargs,
    ) -> List[str]:
        """Batch version of :meth:`analyze`.

        Provide **either** ``frames_list`` (one list of PIL frames per
        sample) **or** ``video_paths``.  Returns a list of analysis strings,
        one per sample.
        """
        system_prompt = system_prompt or self.spr_system_prompt
        processor = self._get_processor()

        bsz = len(questions)
        if frames_list is not None:
            assert len(frames_list) == bsz
        if video_paths is not None:
            assert len(video_paths) == bsz

        all_prompt_texts = []
        all_image_inputs = []
        all_video_inputs = []

        for i in range(bsz):
            msgs = self._build_messages(
                question=questions[i],
                frames=frames_list[i] if frames_list else None,
                video_path=video_paths[i] if video_paths else None,
                nframes=nframes,
                system_prompt=system_prompt,
            )
            all_prompt_texts.append(
                processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            img_in, vid_in = process_vision_info(msgs)
            if img_in:
                all_image_inputs.extend(img_in)
            if vid_in:
                all_video_inputs.extend(vid_in)

        batch = processor(
            text=all_prompt_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batch = batch.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        gen_kwargs.update(generate_kwargs)

        generated_ids = self.generate(**batch, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts


# ──────────────────────────────────────────────────────────────────────
# Convenience loader (mirrors load_model_and_processor in inference.py)
# ──────────────────────────────────────────────────────────────────────
def load_spr_model(
    model_path: str = "checkpoints/Qwen2.5-VL-3B-Instruct",
    torch_dtype: str = "bfloat16",
    device_map: str = "cuda",
    attn_implementation: str = "flash_attention_2",
):
    """Load SPR model and processor from a Qwen2.5-VL checkpoint.

    Returns
    -------
    model : SPRForConditionalGeneration
    processor : Qwen2_5_VLProcessor
    """
    config = SPRConfig.from_pretrained(model_path)
    model = SPRForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model.set_processor(processor)
    return model, processor
