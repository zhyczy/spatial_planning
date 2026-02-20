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

from src.evaluation.MMSIBench.dataset_utils import (
    load_mmsibench_dataset,
    load_mmsibench_dataset_json,
    extract_answer_letter,
    get_mmsibench_metrics,
    MMSIBENCH_CATEGORIES,
)
from src.qwenvl.model.spatial_mllm_VACE import SpaMLLMVACEConfig, SpaMLLMVACEForConditionalGeneration
from src.qwenvl.model.QwenVACE import QwenVACEConfig, QwenVACEForConditionalGeneration
from external.VACE.vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# Constants
SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."


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
        vace_cfg_obj = WAN_CONFIGS['vace-1.3B']
        vace_cfg = dict(vace_cfg_obj)

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
        return model, processor
    
    if "qwen-vace" in model_type:
        from transformers import Qwen2_5_VLProcessor
        config = QwenVACEConfig.from_pretrained(model_path)

        # Add VACE config to the model config
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
            local_files_only=True,
        )
        
        # Explicitly move model to the assigned GPU
        logger.info(f"[Model Loader] Moving Qwen2.5-VL model to {device}")
        model = model.to(device)
            
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        logger.info(f"[Model Loader] Qwen2.5-VL model loaded on device: {model_device}")
        
        return model, processor

    raise ValueError(f"Unknown model type: {model_type}")


def build_user_message(item: Dict, data_dir: Path) -> Dict:
    """Build user message with multi-image content following Qwen-VL format.
    
    Args:
        item: Dataset item containing multi-image paths and question
        data_dir: Base directory for resolving relative image paths
        
    Returns:
        User message dict with role='user' and content=[image1, image2, ..., text]
    """
    # Get multiple image paths (already a list from dataset loading)
    img_paths = item.get("image", [])
    
    # Resolve paths relative to data_dir
    image_paths = []
    for img_path in img_paths:
        if Path(img_path).is_absolute():
            image_paths.append(str(Path(img_path).resolve()))
        else:
            image_paths.append(str((data_dir / img_path).resolve()))
    
    # Build text content with post-prompt for answer extraction
    question_text = item.get("question", "")
    text_content = SFT_QUESTION_TEMPLATE.format(Question=question_text)
    text_content = f"{text_content}\n{SFT_TYPE_TEMPLATE}"
    
    # Build multi-image content
    image_contents = [{"type": "image", "image": img_path} for img_path in image_paths]
    text_content_dict = {"type": "text", "text": text_content}
    
    return {
        "role": "user", 
        "content": image_contents + [text_content_dict]
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

    # Process multi-image inputs
    for example in batch_messages:
        images, videos = process_vision_info(example)
        if images:
            image_inputs.extend(images)
        elif videos:
            video_inputs.extend(videos)
        else:
            raise ValueError("No images found in message")
    
    # Debug: Count total images
    logger = logging.getLogger(__name__)
    logger.info(f"[DEBUG] Total images extracted from batch: {len(image_inputs)}")

    # Tokenize and prepare inputs
    batch = processor(
        text=prompts_text,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    # Add spatial MLLM-specific inputs for models that need them
    if "spatial-mllm" in model_type or "spatial-VACE" in model_type or "qwen-vace" in model_type:
        batch = prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs)

    return batch, prompts_text_copy


def inference_batch(batch_inputs: Dict, model: Any, processor: Any) -> List[str]:
    """Run inference on the batch and return decoded text outputs."""
    # Move inputs to model's device
    batch_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

    # Move extra visual data to GPU if present
    if "image_tchw" in batch_inputs and batch_inputs["image_tchw"] is not None:
        batch_inputs["image_tchw"] = [img.to(model.device) for img in batch_inputs["image_tchw"]]
    
    if "video_tchw" in batch_inputs and batch_inputs["video_tchw"] is not None:
        batch_inputs["video_tchw"] = [vid.to(model.device) for vid in batch_inputs["video_tchw"]]
    
    generated_ids = model.generate(
        **batch_inputs,
        max_new_tokens=128,
        do_sample=False,
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs["input_ids"], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text


def postprocess_batch(batch_data: List[Dict], batch_output_text: List[str], prompts_text: List[str]) -> List[Dict]:
    """Post-process batch outputs to extract answers and format results."""
    batch_results = []
    
    for item, output, prompt in zip(batch_data, batch_output_text, prompts_text):
        # Extract answer letter from model output
        prediction = extract_answer_letter(output)
        
        result = {
            "index": item.get("index", ""),
            "category": item.get("category", "unknown"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "prediction": prediction,
            "output": output,
            "thought": item.get("thought", ""),
            "difficulty": item.get("difficulty", "unknown"),
            "prompt": prompt,
        }
        batch_results.append(result)
    
    return batch_results


def evaluate_mmsibench(mmsibench_data, model_type, model_path, batch_size, data_dir, output_path, device=None, visualize_vace_videos=False):
    """Evaluate model on MMSI-Bench dataset. Forces batch size to 1."""
    logger = logging.getLogger(__name__)
    
    # Force batch size to 1 for consistency
    if batch_size != 1:
        logger.warning(f"Batch size {batch_size} not supported. Forcing batch size to 1.")
        batch_size = 1
    
    # Load model and processor
    model, processor = load_model_and_processor(model_type, model_path, device=device, visualize_vace_videos=visualize_vace_videos)
    
    # Process data in batches
    final_output = []
    total_batches = (len(mmsibench_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(mmsibench_data), batch_size), desc="Evaluating MMSIBench", total=total_batches):
        batch_data = mmsibench_data[batch_idx : batch_idx + batch_size]
        
        # Set video metadata for VACE models (use first sample's info)
        if hasattr(model, 'set_video_metadata'):
            item = batch_data[0]
            category = item.get('category', 'unknown')
            question = item.get('question', '')
            ground_truth = item.get('answer', '')
            
            # Get all image paths for this sample
            img_paths = item.get('image', [])
            logger.info(f"[DEBUG] Sample {item.get('index', batch_idx)} has {len(img_paths)} images: {img_paths}")
            
            # Use index as identifier
            sample_id = str(item.get('index', batch_idx))
            
            # Use first image path as video_path (for backward compatibility)
            # Note: For multi-image samples, all images are passed via input_frames in the forward pass
            video_path = img_paths[0] if img_paths else None
            
            model.set_video_metadata(
                video_name=f"mmsibench_{sample_id}",
                video_path=video_path,
                question=question,
                ground_truth=ground_truth,
                question_type=category,
                options=[],  # Options embedded in question text
                dataset='MMSIBench'
            )
        
        batch_llm_inputs, prompts_text = prepare_chat_batch(batch_data, processor, model_type, data_dir)
        batch_output_text = inference_batch(batch_llm_inputs, model, processor)
        batch_results = postprocess_batch(batch_data, batch_output_text, prompts_text)
        final_output.extend(batch_results)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, final_output)
    logger.info(f"Saved predictions to {output_path}")
    
    # Calculate and log metrics
    metrics = get_mmsibench_metrics(final_output)
    
    logger.info("=" * 50)
    logger.info("MMSI-Bench Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Correct Samples: {metrics['correct_samples']}")
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info("")
    logger.info("Category-wise Accuracy:")
    for category, acc in sorted(metrics['category_accuracy'].items()):
        count = metrics['category_counts'].get(category, 0)
        logger.info(f"  {category:30s}: {acc:6.2%} ({count:4d} samples)")
    logger.info("=" * 50)
    
    # Save metrics
    metrics_path = output_path.parent / "metrics.json"
    save_json(metrics_path, metrics)
    logger.info(f"Saved metrics to {metrics_path}")
    
    return final_output, metrics


def run_worker(gpu_id, mmsibench_data, model_type, model_path, batch_size, data_dir, output_path, log_file=None, visualize_vace_videos=False):
    """Worker function to run evaluation on a specific GPU.
    
    Args:
        gpu_id: GPU device ID to use (0-based index, e.g., 0, 1, 2, ...)
        log_file: Path to the log file to write to
        visualize_vace_videos: Whether to enable VACE video visualization
    """
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
    logger.info(f"[Worker {gpu_id}] Processing {len(mmsibench_data)} samples")
    logger.info(f"[Worker {gpu_id}] ========================================\n")
    
    evaluate_mmsibench(mmsibench_data, model_type, model_path, batch_size, data_dir, output_path, device=device, visualize_vace_videos=visualize_vace_videos)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MMSI-Bench dataset")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["qwen2.5-vl", "spatial-mllm", "spatial-VACE", "qwen-vace"],
                       help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model checkpoint")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing MMSI-Bench data (MMSI_bench.tsv and images)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to evaluate (for testing)")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (forced to 1 for consistency)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Root directory to save evaluation results")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for organizing results. If not provided, uses model_type")
    
    # VACE arguments
    parser.add_argument("--visualize_vace_videos",
                       type=lambda x: (str(x).lower() == 'true'),
                       default=False,
                       help="Whether to visualize and save VACE decoded videos (default: False)")
    
    args = parser.parse_args()
    
    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Create output directory structure: results/dataset_name/model_name/experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name if args.model_name else args.model_type
    dataset_name = "MMSIBench"
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
    
    logger = logging.getLogger(__name__)
    
    # Load data in main process
    data_dir = Path(args.data_dir).resolve()

    # Prefer the pre-processed JSON (generated by download.py) which has
    # resolved local image paths for multi-image samples.
    json_file = data_dir / "data" / "test_data_final.json"
    tsv_file = data_dir / "data" / "MMSI_bench.tsv"

    if json_file.exists():
        logger.info(f"Loading dataset from JSON: {json_file}")
        mmsibench_data = load_mmsibench_dataset_json(json_file, limit=args.limit)
    elif tsv_file.exists():
        logger.info(f"JSON not found, falling back to TSV: {tsv_file}")
        mmsibench_data = load_mmsibench_dataset(tsv_file, limit=args.limit)
    else:
        logger.error(f"Data file not found. Looked for:\n  {json_file}\n  {tsv_file}")
        logger.error("Please run datasets/evaluation/MMSIBench/download.py first")
        return
    
    n_gpu = torch.cuda.device_count()
    if n_gpu <= 0:
        raise RuntimeError("MMSI-Bench evaluation requires at least one CUDA device.")
    
    logger.info("=" * 60)
    logger.info("MMSI-BENCH EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Data Dir: {args.data_dir}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Number of GPUs: {n_gpu}")
    logger.info(f"Sample Limit: {args.limit}")
    logger.info(f"Visualize VACE Videos: {args.visualize_vace_videos}")
    logger.info("=" * 60)
    
    logger.info(f"Starting evaluation on {n_gpu} GPUs...")
    
    # Parse CUDA_VISIBLE_DEVICES to handle specific GPU selection
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [x.strip() for x in cuda_visible_devices.split(",") if x.strip()]
    else:
        gpu_ids = [str(i) for i in range(n_gpu)]
    
    processes = []
    output_paths = []
    
    for idx, data_chunk in enumerate(chunk_dataset(mmsibench_data, n_gpu)):
        output_path_gpu = output_dir / f"results_{args.model_type}_{idx}.json"
        output_paths.append(output_path_gpu)
        
        # Select GPU ID
        gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else str(idx)
        
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
                str(log_file),
                args.visualize_vace_videos,
            ),
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Merge results from all workers
    logger.info("Merging results from all workers...")
    all_results = []
    for path in output_paths:
        if path.exists():
            with open(path, "r") as f:
                all_results.extend(json.load(f))
        else:
            logger.warning(f"Output file {path} not found.")
    
    # Calculate overall metrics
    metrics = get_mmsibench_metrics(all_results)
    
    logger.info("=" * 60)
    logger.info("FINAL MMSI-BENCH RESULTS (All GPUs)")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Correct Samples: {metrics['correct_samples']}")
    logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    logger.info("")
    logger.info("Category-wise Accuracy:")
    for category, acc in sorted(metrics['category_accuracy'].items()):
        count = metrics['category_counts'].get(category, 0)
        logger.info(f"  {category:30s}: {acc:6.2%} ({count:4d} samples)")
    logger.info("=" * 60)
    
    # Save results with simple names: eval_result.json and metrics.json
    save_json(output_dir / "eval_result.json", all_results)
    save_json(output_dir / "metrics.json", metrics)
    logger.info(f"Results saved to: {output_dir}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
