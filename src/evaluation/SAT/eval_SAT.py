import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch.multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parents[3]))
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from src.evaluation.utils.common_utils import (
    chunk_dataset,
    flatten,
    prepare_spatial_mllm_inputs,
    save_json,
    setup_logging,
)

from src.evaluation.SAT.dataset_utils import SAT_QUESTION_TYPES, clean_text, sat_reward, calculate_sat_metrics
from src.qwenvl.model.spatial_mllm_VACE import SpaMLLMVACEConfig, SpaMLLMVACEForConditionalGeneration
from src.qwenvl.model.QwenVACE import QwenVACEConfig, QwenVACEForConditionalGeneration
from external.VACE.vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# Constants
SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = "Answer with the option's letter from the given choices directly or describe the answer in a single phrase."


def load_sat_dataset(json_path: str) -> List[Dict]:
    """Load SAT dataset from JSON file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading SAT dataset from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from SAT dataset")
    return data


def load_model_and_processor(model_type: str, model_path: str, device: str = None, visualize_vace_videos: bool = False):
    """Load model and processor based on type.
    
    Args:
        model_type: Type of the model
        model_path: Path to the model
        device: Explicit device string like 'cuda:0', 'cuda:1', etc.
        visualize_vace_videos: Whether to enable VACE video visualization (decode and save)
    """
    logger = logging.getLogger(__name__)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"[Model Loader] Loading model on device={device}")
    logger.info(f"[Model Loader] Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"[Model Loader] VACE Visualization: {visualize_vace_videos}")
    
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

    if "spatial-VACE" in model_type:
        from transformers import Qwen2_5_VLProcessor
        config = SpaMLLMVACEConfig.from_pretrained(model_path)

        # Add VACE config to the model config
        # Use the Wan2.1 1.3B VACE config
        vace_cfg_obj = WAN_CONFIGS['vace-1.3B']
        
        # Convert EasyDict to regular dict for model config
        vace_cfg = dict(vace_cfg_obj)

        # VACE models are in the Spatial_RAI/external/VACE directory
        vace_checkpoint_dir = str(Path(__file__).resolve().parents[3] / "external" / "VACE" / "models" / "Wan2.1-VACE-1.3B")
        config.vace_config = vace_cfg
        config.vace_checkpoint_dir = vace_checkpoint_dir
        config.visualize_vace_videos = visualize_vace_videos

        logger.info(f"[Model Loader] Loading SpaMLLMVACEForConditionalGeneration on {device}...")
        model = SpaMLLMVACEForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        
        # Explicitly move model to the assigned GPU
        logger.info(f"[Model Loader] Moving SPI model to {device}")
        model = model.to(device)
        
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        
        model_device = next(model.parameters()).device
        logger.info(f"[Model Loader] SPI model loaded on device: {model_device}")
        
        return model, processor

    if "qwen-vace" in model_type:
        from transformers import Qwen2_5_VLProcessor

        # Build config from Qwen2.5-VL checkpoint + VACE config
        config = QwenVACEConfig.from_pretrained(model_path)

        vace_cfg_obj = WAN_CONFIGS['vace-1.3B']
        vace_cfg = dict(vace_cfg_obj)
        vace_checkpoint_dir = str(Path(__file__).resolve().parents[3] / "external" / "VACE" / "models" / "Wan2.1-VACE-1.3B")
        config.vace_config = vace_cfg
        config.vace_checkpoint_dir = vace_checkpoint_dir
        config.visualize_vace_videos = visualize_vace_videos

        logger.info(f"[Model Loader] Loading QwenVACEForConditionalGeneration on {device}...")
        model = QwenVACEForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )

        logger.info(f"[Model Loader] Moving QwenVACE model to {device}")
        model = model.to(device)

        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

        model_device = next(model.parameters()).device
        logger.info(f"[Model Loader] QwenVACE model loaded on device: {model_device}")

        return model, processor

    if "qwen2.5-vl" in model_type:
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

        logger.info(f"[Model Loader] Loading Qwen2.5-VL on {device}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        
        logger.info(f"[Model Loader] Moving Qwen2.5-VL model to {device}")
        model = model.to(device)
        
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        
        model_device = next(model.parameters()).device
        logger.info(f"[Model Loader] Qwen2.5-VL model loaded on device: {model_device}")
        
        return model, processor

    raise ValueError(f"Unknown model type: {model_type}")


def build_user_message(item: Dict, data_dir: Path) -> Dict:
    """Create the chat-style message payload for a single SAT sample."""
    # Build question with answer choices
    raw_question = SFT_QUESTION_TEMPLATE.format(Question=item["question"])
    
    answer_choices = item.get("answer_choices", [])
    if answer_choices:
        # Format answer choices as A, B, C, D...
        options_text = "Options:\n" + "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(answer_choices)]
        )
        question = f"{raw_question}\n{options_text}\n{SFT_TYPE_TEMPLATE}"
    else:
        question = f"{raw_question}\n{SFT_TYPE_TEMPLATE}"

    text_content = {"type": "text", "text": question}
    
    # Load images from img_paths
    img_paths = item.get("img_paths", [])
    if not img_paths:
        raise ValueError(f"Sample {item.get('database_idx')} has no image paths.")
    
    # Convert relative paths to absolute paths
    image_paths = []
    for img_path in img_paths:
        # Handle both absolute and relative paths
        if Path(img_path).is_absolute():
            image_paths.append(str(Path(img_path).resolve()))
        else:
            # Assume paths are relative to data_dir
            image_paths.append(str((data_dir / img_path).resolve()))
    
    # Create image content entries
    image_contents = [{"type": "image", "image": img_path} for img_path in image_paths]

    return {
        "role": "user",
        "content": image_contents + [text_content],
    }


def prepare_chat_batch(
    batch_data: List[Dict],
    processor: Any,
    model_type: str,
    data_dir: Path,
) -> Tuple[Dict, List[str]]:
    """Prepare batch for inference: build prompts, process images, and tokenize."""
    batch_messages = [[build_user_message(item, data_dir)] for item in batch_data]

    prompts_text = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in batch_messages
    ]
    prompts_text_copy = prompts_text.copy()

    video_inputs = []
    image_inputs = []

    # Process images from batch messages
    for example in batch_messages:
        images, videos = process_vision_info(example)
        if images:
            image_inputs.extend(images)
        elif videos:
            video_inputs.extend(videos)

    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    if "spatial-mllm" in model_type or "spatial-VACE" in model_type or "qwen-vace" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text_copy


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on the batch inputs."""
    batch_inputs.to(model.device)
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [image_tchw_i.to(model.device) for image_tchw_i in batch_inputs["image_tchw"]]
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [video_tchw_i.to(model.device) for video_tchw_i in batch_inputs["video_tchw"]]

    generation_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**batch_inputs, **generation_kwargs)

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def postprocess_batch(
    batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]
) -> List[Dict]:
    """Post-process outputs: clean text, calculate rewards, and structure results."""
    batch_results = []
    for sample, model_output, prompt in zip(batch_data, batch_output_text, prompts_text):
        clean_ans = clean_text(model_output)
        clean_ans_gt = clean_text(sample.get("correct_answer", ""))
        reward = sat_reward(clean_ans_gt, clean_ans)

        batch_results.append(
            {
                "sample": sample,
                "prompt": prompt,
                "model_output": model_output,
                "cleaned_model_output": clean_ans,
                "cleaned_gt_answer": clean_ans_gt,
                "reward": reward,
                "correct": reward == 1.0,
            }
        )

    return batch_results


def evaluate_sat(sat_data, model_type, model_path, batch_size, data_dir, output_path, device=None, log_config=False, visualize_vace_videos=False):
    """Evaluate model on SAT dataset."""

    setup_logging()
    model, processor = load_model_and_processor(model_type, model_path, device=device, visualize_vace_videos=visualize_vace_videos)
    
    # Log model configuration only on the first GPU (cuda:0)
    if log_config:
        logger = logging.getLogger(__name__)
        logger.info("\n" + "="*60)
        logger.info("MODEL CONFIGURATION")
        logger.info("="*60)
        
        if hasattr(model, 'config'):
            config = model.config
            logger.info(f"Model Type: {type(model).__name__}")
            logger.info(f"Config Type: {type(config).__name__}")
            
            # Log main config attributes
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
            for key, value in sorted(config_dict.items()):
                # Skip very large nested structures, just show their type
                if isinstance(value, (dict, list)) and len(str(value)) > 200:
                    logger.info(f"{key}: {type(value).__name__} (size: {len(value)})")
                else:
                    logger.info(f"{key}: {value}")
            
            # Special handling for VACE config if present
            if hasattr(config, 'vace_config') and config.vace_config:
                logger.info("\n" + "-"*40)
                logger.info("VACE Configuration:")
                logger.info("-"*40)
                vace_cfg = config.vace_config
                if isinstance(vace_cfg, dict):
                    for key, value in sorted(vace_cfg.items()):
                        logger.info(f"vace.{key}: {value}")
                else:
                    logger.info(f"VACE Config: {vace_cfg}")
            
            if hasattr(config, 'vace_checkpoint_dir'):
                logger.info(f"VACE Checkpoint Dir: {config.vace_checkpoint_dir}")
        else:
            logger.info("Model does not have a 'config' attribute")
        
        logger.info("="*60 + "\n")
    
    final_output = []

    for i in tqdm(range(0, len(sat_data), batch_size), desc="Evaluating SAT"):
        batch_data = sat_data[i : i + batch_size]
        
        # Set video metadata for VACE output organization (if model supports it)
        if hasattr(model, 'set_video_metadata') and len(batch_data) > 0:
            item = batch_data[0]  # batch_size=1
            video_name = f"sat_{item.get('database_idx', i)}"
            
            # SAT uses images, not videos
            img_paths = item.get('img_paths', [])
            video_path = img_paths[0] if img_paths else None
            
            # Extract question and answer information
            question = item.get('question', '')
            ground_truth = item.get('correct_answer', '')
            question_type = item.get('question_type', '')
            answer_choices = item.get('answer_choices', [])
            
            model.set_video_metadata(
                video_name=video_name, 
                video_path=video_path,
                question=question,
                ground_truth=ground_truth,
                question_type=question_type,
                options=answer_choices,
                dataset='SAT'
            )
        
        batch_llm_inputs, prompts_text = prepare_chat_batch(batch_data, processor, model_type, data_dir)
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)

        # Checkpoint partial results every 10 batches or at the end
        if (i + 1) % 10 == 0 or (i + 1) == len(sat_data):
            save_json(output_path, final_output)

    return final_output


def run_worker(gpu_id, sat_data, model_type, model_path, batch_size, data_dir, output_path, log_config=False, log_file=None, visualize_vace_videos=False):
    """Worker function to run evaluation on a specific GPU."""
    # Setup logging in worker process to write to the same log file
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
    
    logger = logging.getLogger(__name__)
    
    # Use explicit device string - no CUDA_VISIBLE_DEVICES needed
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(int(gpu_id))
    
    logger.info(f"\n[Worker {gpu_id}] ========================================")
    logger.info(f"[Worker {gpu_id}] Assigned device: {device}")
    logger.info(f"[Worker {gpu_id}] torch.cuda.current_device()={torch.cuda.current_device()}")
    logger.info(f"[Worker {gpu_id}] torch.cuda.device_count()={torch.cuda.device_count()}")
    logger.info(f"[Worker {gpu_id}] ========================================\n")
    
    evaluate_sat(sat_data, model_type, model_path, batch_size, data_dir, output_path, device=device, log_config=log_config, visualize_vace_videos=visualize_vace_videos)


def main(args):
    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Create output directory structure: results/dataset_name/model_name/experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use provided model_name, or default to model_type
    model_name = args.model_name if hasattr(args, 'model_name') and args.model_name else args.model_type
    dataset_name = "SAT"
    experiment_name = f"{model_name}_{timestamp}"
    
    output_dir = Path(args.output_dir).resolve() / dataset_name / model_name / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file in the output directory
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Print configuration parameters (will be logged on rank 0/cuda:0)
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Data JSON: {args.data_json}")
    logger.info(f"Data Dir: {args.data_dir}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Visualize VACE Videos: {args.visualize_vace_videos}")
    logger.info("="*60)

    # Load SAT dataset
    sat_data = load_sat_dataset(args.data_json)
    
    data_dir = Path(args.data_dir).resolve()
    
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("SAT evaluation requires at least one CUDA device.")

    logger.info(f"Starting evaluation on {n_gpu} GPUs...")

    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]

    processes = []
    output_paths = []

    for idx, data_chunk in enumerate(chunk_dataset(sat_data, n_gpu)):
        output_path_gpu = output_dir / f"results_{args.model_type}_{idx}.json"
        output_paths.append(output_path_gpu)

        # Select GPU ID
        gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else str(idx)
        
        # Only log config on the first worker (gpu_id 0)
        log_config = (str(gpu_id) == "0")

        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                data_chunk,
                args.model_type,
                args.model_path,
                args.batch_size,
                data_dir,
                output_path_gpu,
                log_config,
                str(log_file),
                args.visualize_vace_videos,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final_output = []
    for path in output_paths:
        if path.exists():
            with open(path, "r") as f:
                final_output.extend(json.load(f))
        else:
            logger.info(f"Warning: Output file {path} not found.")

    # Compute the overall metrics across shards.
    final_metrics = calculate_sat_metrics(final_output)
    
    # Save with simple names: eval_result.json and metrics.json
    save_json(
        output_dir / "eval_result.json",
        final_output,
    )
    save_json(
        output_dir / "metrics.json",
        final_metrics,
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Finished evaluation for SAT.")
    logger.info(f"Final Metrics: {final_metrics}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on SAT dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm", help="Type of the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (forced to 1).")
    parser.add_argument(
        "--data_json", type=str, required=True, help="Path to the SAT JSON file (e.g., test.json or val.json)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the SAT image files.",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Root directory to save evaluation results.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name for organizing results. If not provided, uses model_type.",
    )
    parser.add_argument(
        "--visualize_vace_videos",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="Whether to visualize and save VACE decoded videos (default: False).",
    )
    args = parser.parse_args()

    main(args)
