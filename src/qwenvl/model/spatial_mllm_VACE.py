from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from easydict import EasyDict as edict

from src.qwenvl.model.connector import get_connector
from src.qwenvl.model.spatial_encoder import VGGTSpatialEncoderConfig, VGGTSpatialEncoderPreTrainedModel

# Optional VACE import will be attempted when config provides vace settings
from external.VACE.vace.models.wan.wan_vace import WanVace

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from datetime import datetime


class SpaMLLMVACEConfig(Qwen2_5_VLConfig):
    model_type = "spatial-mllm"

    def __init__(self, spatial_config=None, connector_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs["spatial_config"] = VGGTSpatialEncoderConfig
        if isinstance(spatial_config, dict):
            self.spatial_config = self.sub_configs["spatial_config"](**spatial_config)
        elif spatial_config is None:
            self.spatial_config = self.sub_configs["spatial_config"]()

        self.connector_config = connector_config if connector_config is not None else {}


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)


class SpaMLLMVACEForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(config.spatial_config)
        self.connector = get_connector(config)

        # Store VACE config for lazy initialization (defer to first forward to avoid meta tensor issues)
        self.vace_config = getattr(config, "vace_config", None)
        self.vace_checkpoint_dir = getattr(config, "vace_checkpoint_dir", None)
        self.vace = None
        self.vace_projector = None
        self._vace_loaded = False
        self.visualize_vace_videos = getattr(config, "visualize_vace_videos", False)
        # Track video metadata for VACE output organization
        self._current_video_metadata = {}
        
        print("[SPI] VACE will be loaded on first forward (lazy initialization)")
        print(f"[SPI] vace_config provided: {self.vace_config is not None}")
        print(f"[SPI] vace_checkpoint_dir: {self.vace_checkpoint_dir}")

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_video_metadata(self, video_name: str = None, video_path: str = None, 
                          question: str = None, ground_truth: str = None, 
                          question_type: str = None, options: list = None,
                          dataset: str = None):
        """Set metadata for the current video being processed.
        
        Args:
            video_name: Identifier/name for the video
            video_path: Full path to the original video file
            question: The question being asked about the video
            ground_truth: The ground truth answer
            question_type: Type of question (e.g., 'object_rel_direction_hard')
            options: Multiple choice options if applicable
            dataset: Dataset name (e.g., 'SAT', 'MindCube', 'MMSIBench')
        """
        self._current_video_metadata = {
            'video_name': video_name,
            'video_path': video_path,
            'question': question,
            'ground_truth': ground_truth,
            'question_type': question_type,
            'options': options,
            'dataset': dataset
        }
    
    def _load_vace(self):
        """Lazy load VACE on first forward to avoid meta tensor initialization issues."""
        if self._vace_loaded:
            return
        
        print(f"[SPI] _load_vace() called. vace_config={self.vace_config is not None}, checkpoint_dir={self.vace_checkpoint_dir}")
        
        if self.vace_config is None or self.vace_checkpoint_dir is None:
            print("[SPI] ERROR: No VACE config provided, skipping VACE loading")
            self._vace_loaded = True
            return
        
        try:
            print("[SPI] Loading WanVace (lazy initialization)...")
            import os
            vace_cfg_obj = DictToObj(self.vace_config)
            
            # CRITICAL: Get device from the model itself, not from cuda.current_device()
            # The model has been moved to the correct device by device_map
            model_device = self.device  # This is the device where MLLM already is
            
            # Extract device_id from device string (e.g., "cuda:2" -> 2)
            if isinstance(model_device, torch.device):
                if model_device.type == 'cuda':
                    device_id = model_device.index if model_device.index is not None else 0
                else:
                    device_id = 0
            else:
                device_id = 0
            
            target_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            
            print(f"[SPI] Model device: {model_device}")
            print(f"[SPI] Extracted device_id: {device_id}")
            
            # Disable HuggingFace meta device initialization to avoid meta tensor issues
            old_env = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', None)
            try:
                # Prevent diffusers/transformers from using meta device
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                print(f"[SPI] Initializing WanVace on device {target_device} (device_id={device_id})...")
                self.vace = WanVace(
                    vace_cfg_obj,
                    self.vace_checkpoint_dir,
                    device_id=device_id,
                    dit_fsdp=False
                )
                print(f"[SPI] WanVace initialized successfully on {target_device}")
                
            finally:
                if old_env is not None:
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = old_env
                else:
                    os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
            
            # Initialize projector
            try:
                latent_dim = int(self.vace.vae.model.z_dim)
                self.vace_projector = nn.Linear(latent_dim, self.config.hidden_size)
                # Move to device and convert to the same dtype as the model (bfloat16)
                self.vace_projector = self.vace_projector.to(device=self.device, dtype=torch.bfloat16)
                print(f"[SPI] VACE loaded. Projector: {latent_dim} -> {self.config.hidden_size}")
            except Exception as e:
                print(f"[SPI] Warning: Could not initialize projector: {e}")
                self.vace_projector = None
            
            self._vace_loaded = True
        except Exception as e:
            import traceback
            print(f"[SPI] ERROR loading VACE: {e}")
            print(f"[SPI] Traceback: {traceback.format_exc()}")
            print("[SPI] Continuing without VACE support")
            self.vace = None
            self.vace_projector = None
            self._vace_loaded = True
    
    def visualize_vace_latents(self, vace_latents: List[torch.Tensor], output_dir: str = "./vace_outputs"):
        """Visualize VACE latent features and save statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"[SPI] Visualizing {len(vace_latents)} VACE latent tensors...")
            for idx, latent in enumerate(vace_latents):
                if not isinstance(latent, torch.Tensor):
                    continue
                
                print(f"[SPI]   Latent #{idx}: shape={latent.shape}, dtype={latent.dtype}, "
                      f"min={latent.min():.4f}, max={latent.max():.4f}, mean={latent.mean():.4f}")
                
            print(f"[SPI] VACE latents saved statistics in {output_dir}")
        except Exception as e:
            print(f"[SPI] Warning: Could not visualize VACE latents: {e}")
    
    def decode_vace_latents_to_video(
        self,
        vace_latents: List[torch.Tensor],
        output_path: str = "./vace_outputs/decoded_video.mp4",
        fps: int = 8,
        video_name: str = None,
        original_video_path: str = None,
        temporal_interpolation: int = 4,
        input_frames: Optional[List[torch.Tensor]] = None,
        dataset: str = None,
        is_image_input: bool = False
    ) -> Optional[str]:
        """Decode VACE latent features to video using VAE decoder.
        
        Args:
            vace_latents: List of latent tensors from VACE encoder
            output_path: Base output path (used if video_name is None)
            fps: Frames per second for output video
            video_name: Name/identifier for the video (creates subfolder)
            original_video_path: Path to original video (will be copied to subfolder)
            temporal_interpolation: Factor to interpolate decoded frames (compensates VACE temporal downsampling)
            input_frames: List of input frame tensors to save alongside decoded video
            dataset: Dataset name (e.g., 'SAT', 'MindCube', 'MMSIBench') for organizing output
            is_image_input: True if input is images, False if input is video
        
        Returns:
            Path to the decoded video file
            
        Note:
            VACE VAE has 4x temporal downsampling (vae_stride[0]=4). This means:
            - Input 16 frames → 4 latent frames → 4 decoded frames
            - temporal_interpolation=4 will interpolate to 16 frames using frame blending
        """
        if self.vace is None or not hasattr(self.vace, 'vae'):
            print("[SPI] ERROR: VACE or VAE decoder not available")
            return None
        
        # If video_name is provided, create a subfolder structure
        if video_name is not None:
            # Build path: vace_outputs/{dataset}/spatial_VACE/{video_name}
            base_dir = Path(output_path).parent.parent  # Get vace_outputs/
            if dataset:
                video_subfolder = base_dir / dataset / "spatial_VACE" / video_name
            else:
                video_subfolder = base_dir / "spatial_VACE" / video_name
            video_subfolder.mkdir(parents=True, exist_ok=True)
            
            output_path = video_subfolder / "decoded_video.mp4"
            
            # Copy original video to subfolder if provided (only for video input)
            if not is_image_input and original_video_path is not None and Path(original_video_path).exists():
                import shutil
                original_dest = video_subfolder / f"original_{Path(original_video_path).name}"
                try:
                    shutil.copy2(original_video_path, original_dest)
                    print(f"[SPI] Copied original video to {original_dest}")
                except Exception as e:
                    print(f"[SPI] Warning: Could not copy original video: {e}")
            
            # Save question and answer metadata to text file
            metadata = self._current_video_metadata
            if metadata.get('question') or metadata.get('ground_truth'):
                try:
                    qa_file = video_subfolder / "question_answer.txt"
                    with open(qa_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"Video: {video_name}\n")
                        f.write("=" * 80 + "\n\n")
                        
                        if metadata.get('question_type'):
                            f.write(f"Question Type: {metadata['question_type']}\n\n")
                        
                        if metadata.get('question'):
                            f.write("Question:\n")
                            f.write("-" * 80 + "\n")
                            f.write(f"{metadata['question']}\n\n")
                        
                        if metadata.get('options'):
                            f.write("Options:\n")
                            f.write("-" * 80 + "\n")
                            for opt in metadata['options']:
                                f.write(f"{opt}\n")
                            f.write("\n")
                        
                        if metadata.get('ground_truth'):
                            f.write("Ground Truth Answer:\n")
                            f.write("-" * 80 + "\n")
                            f.write(f"{metadata['ground_truth']}\n")
                    
                    print(f"[SPI] Saved question/answer to {qa_file}")
                except Exception as e:
                    print(f"[SPI] Warning: Could not save question/answer: {e}")
            
            # Save input frames if provided (always for image input, optional for video input)
            if input_frames is not None and len(input_frames) > 0:
                try:
                    input_images_dir = video_subfolder / "input_images"
                    input_images_dir.mkdir(parents=True, exist_ok=True)
                    
                    input_type_str = "images" if is_image_input else "video frames"
                    print(f"[SPI] Saving {len(input_frames)} input {input_type_str} to {input_images_dir}")
                    
                    for frame_idx, frame_tensor in enumerate(input_frames):
                        # Handle different tensor formats
                        if frame_tensor.ndim == 3:  # [C, H, W]
                            frame_to_save = frame_tensor
                        elif frame_tensor.ndim == 4:  # [T, C, H, W] or [C, T, H, W]
                            # Take the first frame
                            if frame_tensor.shape[1] == 3 or (frame_tensor.shape[0] > 10 and frame_tensor.shape[1] <= 10):
                                # [T, C, H, W]
                                frame_to_save = frame_tensor[0]  # [C, H, W]
                            else:
                                # [C, T, H, W]
                                frame_to_save = frame_tensor[:, 0, :, :]  # [C, H, W]
                        else:
                            print(f"[SPI] Warning: Unexpected frame dimension {frame_tensor.shape}, skipping")
                            continue
                        
                        # Convert tensor to numpy array and save as image
                        frame_np = frame_to_save.cpu().float().numpy()  # [C, H, W]
                        frame_np = frame_np.transpose(1, 2, 0)  # [H, W, C]
                        
                        # Normalize to [0, 255] range
                        if frame_np.min() < 0:
                            frame_np = (frame_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
                        frame_np = (frame_np * 255).astype(np.uint8)
                        frame_np = np.clip(frame_np, 0, 255)
                        
                        # Convert RGB to BGR for OpenCV
                        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame_np
                        
                        # Save image
                        image_path = input_images_dir / f"input_frame_{frame_idx:04d}.png"
                        cv2.imwrite(str(image_path), frame_bgr)
                    
                    print(f"[SPI] ✓ Saved {len(input_frames)} input images to {input_images_dir}")
                except Exception as e:
                    print(f"[SPI] Warning: Could not save input frames: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import numpy as np
            import cv2
            
            print(f"[SPI] Decoding {len(vace_latents)} VACE latent tensors to video...")
            
            decoded_frames = []
            
            for idx, latent in enumerate(vace_latents):
                if not isinstance(latent, torch.Tensor):
                    continue
                
                try:
                    # latent shape: [C, T, H, W] where C is channel, T is time/frames
                    if latent.ndim == 4:
                        # [C, T, H, W] -> process each frame
                        T = latent.shape[1]
                        print(f"[SPI]   Processing temporal latent with {T} frames")
                        for t in range(T):
                            frame_latent = latent[:, t:t+1, :, :]  # [C, 1, H, W]
                            with torch.no_grad():
                                decoded = self.vace.vae.decode([frame_latent.to(self.device)])
                            
                            # decoded is a list with one element, shape is usually [C, H, W] or [C, 1, H, W]
                            if isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                                decoded_tensor = decoded[0]  
                            else:
                                decoded_tensor = decoded
                            
                            # Convert to dense tensor if sparse
                            if hasattr(decoded_tensor, 'to_dense'):
                                decoded_tensor = decoded_tensor.to_dense()
                            
                            # Convert to float
                            decoded_tensor = decoded_tensor.float()
                            
                            # Handle various formats: [C, 1, H, W] -> [C, H, W]
                            # or [1, C, H, W] -> [C, H, W]
                            # or [C, H, W] (already correct)
                            if decoded_tensor.dim() == 4:
                                if decoded_tensor.shape[1] == 1 and decoded_tensor.shape[0] > 1 and decoded_tensor.shape[0] <= 4:
                                    # Format: [C, 1, H, W]
                                    decoded_tensor = decoded_tensor.squeeze(1)  # [C, H, W]
                                elif decoded_tensor.shape[0] == 1:
                                    # Format: [1, C, H, W]
                                    decoded_tensor = decoded_tensor.squeeze(0)  # [C, H, W]
                            
                            if decoded_tensor.dim() != 3:
                                print(f"[SPI]     WARNING: Cannot reshape to [C, H, W], tensor shape {decoded_tensor.shape}, skipping frame")
                                continue
                            
                            # Now decoded_tensor should be [C, H, W]
                            frame = decoded_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                            
                            # Normalize to [0, 255]
                            if frame.min() < 0:
                                frame = (frame + 1) / 2  # Convert from [-1, 1] to [0, 1]
                            frame = (frame * 255).astype(np.uint8)
                            frame = np.clip(frame, 0, 255)
                            
                            # OpenCV uses BGR, keep as RGB for now (will convert before write)
                            if frame.ndim == 3 and frame.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            else:
                                # Handle grayscale or other formats
                                frame_bgr = frame
                            
                            print(f"[SPI]     Frame {t}: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}, min={frame_bgr.min()}, max={frame_bgr.max()}")
                            decoded_frames.append(frame_bgr)
                    
                    elif latent.ndim == 3:
                        # Static frame: [C, H, W]
                        latent_4d = latent.unsqueeze(1)  # [C, 1, H, W]
                        with torch.no_grad():
                            decoded = self.vace.vae.decode([latent_4d.to(self.device)])
                        
                        if isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                            decoded_tensor = decoded[0]
                        else:
                            decoded_tensor = decoded
                        
                        # Convert to dense tensor if sparse
                        if hasattr(decoded_tensor, 'to_dense'):
                            decoded_tensor = decoded_tensor.to_dense()
                        
                        # Convert to float
                        decoded_tensor = decoded_tensor.float()
                        
                        # Handle formats: [C, 1, H, W] -> [C, H, W] or [1, C, H, W] -> [C, H, W]
                        if decoded_tensor.dim() == 4:
                            if decoded_tensor.shape[1] == 1 and decoded_tensor.shape[0] > 1 and decoded_tensor.shape[0] <= 4:
                                # Format: [C, 1, H, W]
                                decoded_tensor = decoded_tensor.squeeze(1)
                            elif decoded_tensor.shape[0] == 1:
                                # Format: [1, C, H, W]
                                decoded_tensor = decoded_tensor.squeeze(0)
                        
                        if decoded_tensor.dim() != 3:
                            print(f"[SPI]   WARNING: Cannot reshape to [C, H, W], tensor shape {decoded_tensor.shape}, skipping")
                            continue
                        
                        frame = decoded_tensor.permute(1, 2, 0).cpu().numpy()
                        
                        if frame.min() < 0:
                            frame = (frame + 1) / 2
                        frame = (frame * 255).astype(np.uint8)
                        frame = np.clip(frame, 0, 255)
                        
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame
                        
                        decoded_frames.append(frame_bgr)
                        decoded_frames.append(frame_bgr)  # Duplicate for video stability
                    
                except Exception as e:
                    print(f"[SPI] Warning: Could not decode latent #{idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not decoded_frames:
                print("[SPI] ERROR: No frames decoded from VACE latents")
                return None
            
            print(f"[SPI] Originally decoded {len(decoded_frames)} frames from VACE")
            print(f"[SPI] Note: VACE has 4x temporal downsampling (vae_stride[0]=4)")
            
            # Apply temporal interpolation to compensate for VACE's temporal downsampling
            if temporal_interpolation > 1 and len(decoded_frames) > 1:
                print(f"[SPI] Applying {temporal_interpolation}x temporal interpolation...")
                interpolated_frames = []
                
                for i in range(len(decoded_frames) - 1):
                    frame1 = decoded_frames[i].astype(np.float32)
                    frame2 = decoded_frames[i + 1].astype(np.float32)
                    
                    # Add the first frame
                    interpolated_frames.append(decoded_frames[i])
                    
                    # Generate interpolated frames
                    for j in range(1, temporal_interpolation):
                        alpha = j / temporal_interpolation
                        blended = (1 - alpha) * frame1 + alpha * frame2
                        interpolated_frames.append(blended.astype(np.uint8))
                
                # Add the last frame
                interpolated_frames.append(decoded_frames[-1])
                decoded_frames = interpolated_frames
                print(f"[SPI] After interpolation: {len(decoded_frames)} frames")
            
            # Write video using OpenCV
            first_frame = decoded_frames[0]
            height, width = first_frame.shape[:2]
            print(f"[SPI] Writing {len(decoded_frames)} frames to video...")
            print(f"[SPI]   Video size: {width}x{height}, fps={fps}")
            print(f"[SPI]   First frame: shape={first_frame.shape}, dtype={first_frame.dtype}, min={first_frame.min()}, max={first_frame.max()}")
            
            # Try different codecs in order of preference (H264 variants first for compatibility)
            codecs = ['avc1', 'mp4v', 'H264', 'X264', 'h264', 'XVID', 'MJPG']
            temp_output = None
            
            for codec_name in codecs:
                try:
                    print(f"[SPI]   Trying codec: {codec_name}")
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        print(f"[SPI]   Codec {codec_name} failed to open, trying next...")
                        continue
                    
                    print(f"[SPI]   Codec {codec_name} opened successfully")
                    
                    # Write all frames
                    write_failures = 0
                    for frame_idx, frame in enumerate(decoded_frames):
                        ret = out.write(frame)
                        if not ret:
                            write_failures += 1
                    
                    out.release()
                    
                    # Check if file was created
                    if output_path.exists() and output_path.stat().st_size > 1000:
                        file_size = output_path.stat().st_size / 1024
                        print(f"[SPI] ✓ Initial video saved with {codec_name} ({len(decoded_frames)} frames, {fps} fps, {file_size:.1f}KB)")
                        if write_failures > 0:
                            print(f"[SPI] Note: {write_failures}/{len(decoded_frames)} frames reported write failures")
                        temp_output = str(output_path)
                        break
                    else:
                        file_size = output_path.stat().st_size if output_path.exists() else 0
                        print(f"[SPI]   {codec_name}: File too small ({file_size} bytes), trying next codec")
                        continue
                    
                except Exception as e:
                    print(f"[SPI]   {codec_name} failed: {e}")
                    continue
            
            if temp_output is None:
                print(f"[SPI] ERROR: All video codecs failed to create valid file")
                return None
            
            # Re-encode to H264 using ffmpeg for maximum compatibility
            temp_path = Path(output_path).with_suffix('.temp.mp4')
            try:
                import subprocess
                output_path = Path(output_path)
                
                # Move current file to temp
                if output_path.exists():
                    output_path.rename(temp_path)
                
                print(f"[SPI] Re-encoding to H264 for better compatibility...")
                # Use ffmpeg to re-encode with H264
                cmd = [
                    'ffmpeg', '-y', '-i', str(temp_path),
                    '-c:v', 'libx264',  # Use H264 codec
                    '-preset', 'fast',   # Fast encoding
                    '-crf', '23',        # Good quality
                    '-pix_fmt', 'yuv420p',  # Compatibility
                    '-movflags', '+faststart',  # Enable streaming
                    str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    # Success - remove temp file
                    temp_path.unlink()
                    file_size = output_path.stat().st_size / 1024
                    print(f"[SPI] ✓ Video re-encoded to H264 successfully ({file_size:.1f}KB)")
                    return str(output_path)
                else:
                    # Ffmpeg failed - restore original
                    print(f"[SPI] Warning: ffmpeg re-encoding failed, using original codec")
                    if temp_path.exists():
                        temp_path.rename(output_path)
                    return str(output_path)
                    
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"[SPI] Warning: ffmpeg not available or timeout ({e}), using original codec")
                # Restore original file if needed
                if temp_path.exists() and not output_path.exists():
                    temp_path.rename(output_path)
                return str(output_path)
            except Exception as e:
                print(f"[SPI] Warning: Error during re-encoding ({e}), using original codec")
                # Restore original file if needed
                if temp_path.exists() and not output_path.exists():
                    temp_path.rename(output_path)
                return str(output_path)
        
        except ImportError:
            print("[SPI] ERROR: cv2 and numpy required for video decoding. Install with: pip install opencv-python numpy")
            return None
        except Exception as e:
            print(f"[SPI] ERROR decoding VACE latents to video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,                   # output sequence of tokenizer, the input number sequence
        attention_mask: Optional[torch.Tensor] = None,                  # differentiate real tokens with padding tokens for self-attention
        position_ids: Optional[torch.LongTensor] = None,                # 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # visual data does not enter LLM directly, but it comes in forms of tensor. 
        # It then will be transferred to tokens in forward function
        pixel_values: Optional[torch.Tensor] = None,                    #[batch, channel, height, width] pixel for images
        pixel_values_videos: Optional[torch.FloatTensor] = None,        #[batch, frame, channel, height, width] pixel for videos

        # These parameters represent the number of visual patches along the T (Time/Frames), H (Height), and W (Width) dimensions. 
        # The model relies on these grids to reconstruct the spatial-temporal order of the patches from a flattened sequence.
        # For example, given an input image, 128*256, patch size, 16*16
        # H: 128/16=8, W:256/16=16
        # image_grid_thw: [1, 8, 16]
        image_grid_thw: Optional[torch.LongTensor] = None,              # dynamic resolution for image
        video_grid_thw: Optional[torch.LongTensor] = None,              # dynamic resolution for video, 

        # Compare image_tchw with pixel_values:
        # One concrete E.g., for one element
        # pixel_values: torch.Size([12512, 1176])    one patch is: 14*14*3*2=1176, patch number: 12512
        # image_tchw[0]: torch.Size([16, 3, 476, 644])  16 frames, RGB, H:476, W:644
        image_tchw: Optional[List[torch.FloatTensor]] = None,           
        video_tchw: Optional[List[torch.FloatTensor]] = None,

        rope_deltas: Optional[torch.LongTensor] = None,       # Calculate RoPE, model understands the relative distance between patch A and patch B
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Lazy load VACE on first forward
        self._load_vace()

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # 【统计初始文本token数量】
            initial_text_tokens = input_ids.shape[1] if input_ids is not None else 0
            num_image_tokens = (input_ids == self.config.image_token_id).sum().item() if input_ids is not None else 0
            num_video_tokens = (input_ids == self.config.video_token_id).sum().item() if input_ids is not None else 0
            num_pure_text_tokens = initial_text_tokens - num_image_tokens - num_video_tokens
            
            # print(f"\n[SPI] ========== 初始TOKEN统计 ==========")
            # print(f"[SPI] input_ids总长度: {initial_text_tokens}")
            # print(f"[SPI]   - 纯文本token数: {num_pure_text_tokens}")
            # print(f"[SPI]   - image placeholder token数: {num_image_tokens}")
            # print(f"[SPI]   - video placeholder token数: {num_video_tokens}")
            # print(f"[SPI] 初始inputs_embeds shape: {inputs_embeds.shape}")
            
            # Store VACE latent embeddings to append later
            vace_latent_embeds = None
            
            # Process VACE encoding if available
            if self.vace is not None:
                # prefer image_tchw for images otherwise video_tchw
                frames_list = image_tchw if image_tchw is not None else video_tchw
                is_image_input = image_tchw is not None
                
                if frames_list is not None:
                    try:
                        # Prepare frames for VACE
                        frames_for_vace = []
                        ref_images_list = []
                        
                        for frame_data in frames_list:
                            if frame_data.ndim == 3:  # Single image [C, H, W]
                                # Image-to-Video: Create empty src_video, use image as reference only
                                C, H, W = frame_data.shape
                                frame_vace = torch.zeros(C, 1, H, W, dtype=frame_data.dtype, device=frame_data.device)  # Empty video
                                # Use this image as reference
                                ref_images_list.append([frame_data.unsqueeze(1)])  # [[C, 1, H, W]]
                            elif frame_data.ndim == 4:  # Video [T, C, H, W] or already [C, T, H, W]
                                if frame_data.shape[1] == 3 or (frame_data.shape[0] > 10 and frame_data.shape[1] <= 10):
                                    # Likely [T, C, H, W], convert to [C, T, H, W]
                                    frame_vace = frame_data.permute(1, 0, 2, 3)
                                else:
                                    # Already [C, T, H, W]
                                    frame_vace = frame_data
                                
                                # Extract first frame as reference
                                first_frame = frame_vace[:, 0:1, :, :]  # [C, 1, H, W]
                                ref_images_list.append([first_frame])
                            else:
                                raise ValueError(f"Unexpected frame dimension: {frame_data.shape}")
                            
                            frames_for_vace.append(frame_vace)
                        
                        # CRITICAL: Ensure frames are on the same device as VACE
                        if hasattr(self.vace, 'device'):
                            vace_device = self.vace.device
                            frames_for_vace = [f.to(vace_device) for f in frames_for_vace]
                            ref_images_list = [[ref.to(vace_device) for ref in refs] for refs in ref_images_list]
                            print(f"[SPI] Moving frames to VACE device: {vace_device}")
                        
                        # Log the mode
                        if is_image_input:
                            print(f"[SPI] 🖼️  Image-to-Video Mode: {len(frames_for_vace)} images as references")
                        else:
                            print(f"[SPI] 🎬 Reference-to-Video Mode: Using first frame as reference")
                        print(f"[SPI] Reference frames: {len(ref_images_list)}, shapes: {[r[0].shape for r in ref_images_list]}")
                        print(f"[SPI] Input frames shapes: {[f.shape for f in frames_for_vace]}")
                        
                        # vace expects list of frames per sample; our eval pipeline uses batch size 1
                        vace_latents = self.vace.vace_encode_frames(frames_for_vace, ref_images=ref_images_list, masks=None)

                        # 【可视化VACE latent输出】
                        # print(f"[SPI] VACE encoded {len(vace_latents)} samples")
                        # print(f"[SPI] Input frames shape: {[f.shape for f in frames_for_vace]}")
                        # print(f"[SPI] Output latents shape: {[l.shape for l in vace_latents]}")
                        # print(f"[SPI] Note: VACE has 4x temporal downsampling in VAE (vae_stride=(4,8,8))")
                        # self.visualize_vace_latents(vace_latents, output_dir="./vace_outputs")
                        
                        # Extract video metadata from stored metadata or generate
                        video_identifier = self._current_video_metadata.get('video_name')
                        original_video = self._current_video_metadata.get('video_path')
                        dataset = self._current_video_metadata.get('dataset', 'unknown')
                        
                        if video_identifier is None:
                            # Generate a unique identifier from timestamp
                            video_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        
                        if self.visualize_vace_videos:
                            # 【解码latent为video并保存】
                            #  使用4倍插值来补偿VACE的时间下采样
                            video_path = self.decode_vace_latents_to_video(
                                vace_latents, 
                                output_path="./vace_outputs/spatial_VACE/decoded_video.mp4",
                                fps=8,
                                video_name=video_identifier,
                                original_video_path=original_video,
                                temporal_interpolation=4,  # Compensate for VACE's 4x temporal downsampling
                                input_frames=frames_list,  # Save input frames alongside decoded video
                                dataset=dataset,  # Dataset name for organizing outputs
                                is_image_input=is_image_input  # Whether input is images or video
                            )

                        # 【关键：将每个VACE latent位置转换为独立token，保留细粒度信息】
                        if self.vace_projector is not None and len(vace_latents) > 0:
                            # Convert each spatial-temporal position to individual tokens
                            # vace_latents is a list of tensors, each with shape [C, T, H, W] 
                            #  [C: channels, feature dimensions, e.g.512
                            #   T: temporal dimension (number of frames, e.g.4 after VACE downsampling)
                            #   H, W: spatial dimensions (e.g. 16x16 or 8x8 depending on VACE config)]

                            # 1️⃣ 原始VACE输出: [C, T, H, W] 
                            # 示例: [512, 4, 28, 28] = 512维特征 × 4帧 × 28×28空间位置

                            # 2️⃣ 重新排列: [C, T, H, W] -> [T, H, W, C] 
                            # latent.permute(1, 2, 3, 0)  # [4, 28, 28, 512]
                            # 含义: 把特征维度放到最后，方便后续处理

                            # 3️⃣ Token化: [T, H, W, C] -> [T*H*W, C]
                            # latent_tokens = latent.view(T*H*W, C)  # [3136, 512] 
                            # 含义: 每个时空位置 = 一个512维的token
                            #      总共 4×28×28 = 3136 个独立tokens

                            # 4️⃣ 投影: [T*H*W, C] -> [T*H*W, hidden_size]  
                            # vace_latent_embeds = projector(latent_tokens)  # [3136, 4096]
                            # 含义: 每个token投影到LLM特征空间

                            # 5️⃣ 批次维度: [T*H*W, hidden_size] -> [1, T*H*W, hidden_size]
                            # vace_latent_embeds = latent_tokens.unsqueeze(0)  # [1, 3136, 4096]
                            # 含义: 可与text tokens拼接

                            vace_token_list = []
                            total_tokens = 0
                            
                            for latent_idx, latent in enumerate(vace_latents):
                                # Reshape: [C, T, H, W] -> [T*H*W, C] (每个时空位置成为一个token)
                                C, T, H, W = latent.shape
                                num_tokens = T * H * W
                                latent_tokens = latent.permute(1, 2, 3, 0).view(num_tokens, C)  # [T*H*W, C]
                                vace_token_list.append(latent_tokens)
                                total_tokens += num_tokens
                                print(f"[SPI] Latent #{latent_idx}: [C={C}, T={T}, H={H}, W={W}] -> {num_tokens} tokens of dim {C}")
                            
                            # Concatenate all tokens from all latents: [total_tokens, C]
                            if len(vace_token_list) > 1:
                                vace_features = torch.cat(vace_token_list, dim=0)  # [total_tokens, C]
                            else:
                                vace_features = vace_token_list[0]  # [total_tokens, C]
                            
                            # Project each token to hidden_size: [total_tokens, C] -> [total_tokens, hidden_size]
                            vace_features = vace_features.to(self.device, dtype=inputs_embeds.dtype)
                            vace_latent_embeds = self.vace_projector(vace_features)  # [total_tokens, hidden_size]
                            
                            # Add batch dimension: [total_tokens, hidden_size] -> [1, total_tokens, hidden_size]
                            vace_latent_embeds = vace_latent_embeds.unsqueeze(0)
                            
                            print(f"[SPI] VACE tokens preserved: {total_tokens} individual tokens")
                            print(f"[SPI] Final VACE embeds shape: {vace_latent_embeds.shape}")
                            print(f"[SPI] No information lost - each spatial-temporal position is a separate token!")
                        
                    except Exception as e:
                        print(f"[SPI] WARNING: VACE processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        vace_latent_embeds = None
        
            # Process standard visual features (images)
            if pixel_values is not None:
                assert image_tchw is not None, "`image_tchw` must be provided when `pixel_values` is not None."
                pixel_values = pixel_values.type(self.visual.dtype)
                image_tchw = [image_tchw_i.type(self.visual.dtype) for image_tchw_i in image_tchw]

                # get image embeddings
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                # get spatial embeddings
                spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw)

                # fuse video and spatial embeddings
                fused_embeds = self.connector(
                    image_embeds=image_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=image_grid_thw,
                )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_embeds)
                
                # print(f"[SPI] ========== 图像TOKEN处理后 ==========")
                # print(f"[SPI] 图像visual features数量: {n_image_features}")
                # print(f"[SPI] 替换后inputs_embeds shape: {inputs_embeds.shape}")

            # Process standard visual features (videos)
            if pixel_values_videos is not None:
                assert video_tchw is not None, "`video_tchw` must be provided when `pixel_values_videos` is not None."
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_tchw = [video_tchw_i.type(self.visual.dtype) for video_tchw_i in video_tchw]

                # get video embeddings
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                # get spatial embeddings
                spatial_embeds_list, patch_start_idx = self.spatial_encoder(video_tchw, grid_thw=video_grid_thw)

                # fuse video and spatial embeddings
                fused_embeds = self.connector(
                    video_embeds=video_embeds,
                    spatial_embeds_list=spatial_embeds_list,
                    patch_start_idx=patch_start_idx,
                    grid_thw=video_grid_thw,
                )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)
                
                # print(f"[SPI] ========== 视频TOKEN处理后 ==========")
                # print(f"[SPI] 视频visual features数量: {n_video_features}")
                # print(f"[SPI] 替换后inputs_embeds shape: {inputs_embeds.shape}")

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 【关键修复：先计算 RoPE，后拼接 VACE embeddings】
        # Calculate RoPE using original input_ids and attention_mask lengths before VACE concatenation
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # 【拼接 VACE latent embeddings 到 inputs_embeds 后面】- 在 RoPE 计算完成后
        # Now concatenate VACE embeddings after RoPE calculation is done
        if vace_latent_embeds is not None:
            # 【Token数量统计 - 拼接前】
            original_input_ids_len = input_ids.shape[1] if input_ids is not None else 0
            original_attention_mask_len = attention_mask.shape[1] if attention_mask is not None else 0
            original_inputs_embeds_len = inputs_embeds.shape[1]
            num_vace_tokens = vace_latent_embeds.shape[1]
            
            # 重新计算各部分token数量（因为可能有图像/视频处理）
            num_image_tokens_final = (input_ids == self.config.image_token_id).sum().item() if input_ids is not None else 0
            num_video_tokens_final = (input_ids == self.config.video_token_id).sum().item() if input_ids is not None else 0
            num_text_tokens = original_input_ids_len - num_image_tokens_final - num_video_tokens_final
            
            # print(f"\n[SPI] ========== TOKEN计数：VACE拼接前的完整统计 ==========")
            # print(f"[SPI] 当前inputs_embeds组成 (总长度={original_inputs_embeds_len}):")
            # print(f"[SPI]   1️⃣ 纯文本token: {num_text_tokens}")
            # print(f"[SPI]   2️⃣ 图像visual token: {num_image_tokens_final}")
            # print(f"[SPI]   3️⃣ 视频visual token: {num_video_tokens_final}")
            # print(f"[SPI]   📝 小计 (文本+图像+视频): {original_inputs_embeds_len}")
            # print(f"[SPI]")
            # print(f"[SPI] 即将添加的VACE tokens:")
            # print(f"[SPI]   4️⃣ VACE spatial-temporal tokens: {num_vace_tokens}")
            # print(f"[SPI]")
            # print(f"[SPI] Tensor shapes:")
            # print(f"[SPI]   - input_ids: {input_ids.shape if input_ids is not None else 'None'}")
            # print(f"[SPI]   - attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            # print(f"[SPI]   - inputs_embeds (当前): {inputs_embeds.shape}")
            # print(f"[SPI]   - vace_latent_embeds: {vace_latent_embeds.shape}")
            # print(f"[SPI] =============================================")
            
            # Concatenate VACE latent embeddings to the end of inputs_embeds
            inputs_embeds = torch.cat([inputs_embeds, vace_latent_embeds], dim=1)
            
            # 【Token数量统计 - 拼接后】
            new_inputs_embeds_len = inputs_embeds.shape[1]
            # print(f"\n[SPI] ========== TOKEN计数：VACE拼接后 ==========")
            # print(f"[SPI] 最终inputs_embeds组成 (总长度={new_inputs_embeds_len}):")
            # print(f"[SPI]   1️⃣ 纯文本token: {num_text_tokens}")
            # print(f"[SPI]   2️⃣ 图像visual token: {num_image_tokens_final}")
            # print(f"[SPI]   3️⃣ 视频visual token: {num_video_tokens_final}")
            # print(f"[SPI]   4️⃣ VACE spatial-temporal tokens: {num_vace_tokens}")
            # print(f"[SPI]   ═══════════════════════════════")
            # print(f"[SPI]   📊 总计输入MLLM的tokens: {new_inputs_embeds_len}")
            # print(f"[SPI]")
            # print(f"[SPI] 增加的token数量: {new_inputs_embeds_len - original_inputs_embeds_len}")
            # print(f"[SPI] inputs_embeds shape: {inputs_embeds.shape}")
            # print(f"[SPI] =============================================")
            
            # Extend attention_mask to cover the new VACE tokens
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                num_vace_tokens = vace_latent_embeds.shape[1]
                # Create attention mask for VACE tokens (all ones, meaning attend to them)
                vace_attention_mask = torch.ones(
                    (batch_size, num_vace_tokens), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                # Concatenate to existing attention mask
                attention_mask = torch.cat([attention_mask, vace_attention_mask], dim=1)
                
                # 【Attention Mask统计】
                # new_attention_mask_len = attention_mask.shape[1]
                # print(f"[SPI] ========== ATTENTION MASK扩展 ==========")
                # print(f"[SPI] 原始 attention_mask 长度: {original_attention_mask_len}")
                # print(f"[SPI] 新的 attention_mask 长度: {new_attention_mask_len}")
                # print(f"[SPI] 增加的attention mask长度: {new_attention_mask_len - original_attention_mask_len}")
                # print(f"[SPI] attention_mask shape: {attention_mask.shape}")
            
            # Extend position_ids to cover VACE tokens
            if position_ids is not None:
                # position_ids shape: [3, batch_size, seq_len] for 3D rotary
                vace_seq_len = vace_latent_embeds.shape[1]
                # For VACE tokens, use sequential position IDs (no special rotations)
                if position_ids.dim() == 3:
                    # Original: [3, batch_size, seq_len]
                    batch_size = position_ids.shape[1]
                    last_pos = position_ids[:, :, -1:] if position_ids.shape[2] > 0 else torch.zeros((3, batch_size, 1), device=position_ids.device, dtype=position_ids.dtype)
                    vace_pos_ids = last_pos + torch.arange(1, vace_seq_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(1, 1, -1)
                    vace_pos_ids = vace_pos_ids.expand(3, batch_size, -1)
                    position_ids = torch.cat([position_ids, vace_pos_ids], dim=2)
                elif position_ids.dim() == 2:
                    # [batch_size, seq_len]
                    batch_size = position_ids.shape[0]
                    last_pos = position_ids[:, -1:] if position_ids.shape[1] > 0 else torch.zeros((batch_size, 1), device=position_ids.device, dtype=position_ids.dtype)
                    vace_pos_ids = last_pos + torch.arange(1, vace_seq_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(1, -1)
                    position_ids = torch.cat([position_ids, vace_pos_ids], dim=1)
                
                # 【Position IDs统计】
                # print(f"[SPI] ========== POSITION IDS扩展 ==========")
                # print(f"[SPI] position_ids shape: {position_ids.shape}")
                # print(f"[SPI] position_ids 最后几个值: {position_ids[0, 0, -5:] if position_ids.dim() == 3 else position_ids[0, -5:]}")
            
            # 【最终总结】
            final_input_ids_len = input_ids.shape[1] if input_ids is not None else 0
            final_attention_mask_len = attention_mask.shape[1] if attention_mask is not None else 0
            final_inputs_embeds_len = inputs_embeds.shape[1]
            
            # print(f"\n[SPI] ========== 最终TENSOR维度验证 ==========")
            # print(f"[SPI] input_ids长度: {final_input_ids_len}")
            # print(f"[SPI] attention_mask长度: {final_attention_mask_len}")
            # print(f"[SPI] inputs_embeds长度: {final_inputs_embeds_len}")
            # print(f"[SPI]")
            # print(f"[SPI] ✅ 维度匹配检查:")
            # print(f"[SPI]   - attention_mask == inputs_embeds? {final_attention_mask_len == final_inputs_embeds_len} {'✅' if final_attention_mask_len == final_inputs_embeds_len else '❌'}")
            # print(f"[SPI]   - VACE tokens数量: {final_attention_mask_len - final_input_ids_len}")
            # print(f"[SPI]")
            # print(f"[SPI] 📋 完整Token流程:")
            # print(f"[SPI]   input_ids ({final_input_ids_len}) → embed_tokens → inputs_embeds")
            # print(f"[SPI]   → 替换image/video placeholders → inputs_embeds ({original_inputs_embeds_len})")
            # print(f"[SPI]   → 拼接VACE tokens (+{num_vace_tokens}) → final inputs_embeds ({final_inputs_embeds_len})")
            # print(f"[SPI] =============================================\n")
        else:
            # 即使没有VACE，也输出token统计
            if input_ids is not None:
                final_input_ids_len = input_ids.shape[1]
                final_inputs_embeds_len = inputs_embeds.shape[1]
                num_image_tokens_final = (input_ids == self.config.image_token_id).sum().item()
                num_video_tokens_final = (input_ids == self.config.video_token_id).sum().item()
                num_text_tokens = final_input_ids_len - num_image_tokens_final - num_video_tokens_final
                
                # print(f"\n[SPI] ========== 最终TOKEN统计 (无VACE) ==========")
                # print(f"[SPI] 输入MLLM的tokens组成 (总长度={final_inputs_embeds_len}):")
                # print(f"[SPI]   1️⃣ 纯文本token: {num_text_tokens}")
                # print(f"[SPI]   2️⃣ 图像visual token: {num_image_tokens_final}")
                # print(f"[SPI]   3️⃣ 视频visual token: {num_video_tokens_final}")
                # print(f"[SPI]   ═══════════════════════════════")
                # print(f"[SPI]   📊 总计: {final_inputs_embeds_len}")
                # print(f"[SPI] =============================================\n")

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None

        return model_inputs