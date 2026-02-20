"""
QwenVACE: Qwen2.5-VL + VACE (without Spatial Encoder/Connector)

This model uses vanilla Qwen2.5-VL as the MLLM backbone and integrates VACE
for video-conditioned auto-encoding. Unlike SPI which uses SpatialMLLM (with
spatial encoder + connector), this model uses the standard Qwen2.5-VL visual
pipeline directly.

Architecture:
    Qwen2.5-VL (pretrained) + VACE encoder/decoder + projector
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from easydict import EasyDict as edict

from external.VACE.vace.models.wan.wan_vace import WanVace

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from datetime import datetime


class QwenVACEConfig(Qwen2_5_VLConfig):
    """Configuration for Qwen2.5-VL + VACE model (no spatial encoder)."""
    model_type = "qwen-vace"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DictToObj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)


class QwenVACEForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL + VACE model without spatial encoder/connector.
    
    Uses vanilla Qwen2.5-VL visual pipeline for image/video understanding,
    and integrates VACE for video-conditioned auto-encoding features.
    The VACE interaction logic is identical to SPIForConditionalGeneration.
    """

    def __init__(self, config):
        super().__init__(config)

        # NO spatial encoder or connector — uses vanilla Qwen2.5-VL visual pipeline

        # Store VACE config for lazy initialization
        self.vace_config = getattr(config, "vace_config", None)
        self.vace_checkpoint_dir = getattr(config, "vace_checkpoint_dir", None)
        self.vace = None
        self.vace_projector = None
        self._vace_loaded = False
        self.visualize_vace_videos = getattr(config, "visualize_vace_videos", False)
        self._current_video_metadata = {}

        print("[QwenVACE] VACE will be loaded on first forward (lazy initialization)")
        print(f"[QwenVACE] vace_config provided: {self.vace_config is not None}")
        print(f"[QwenVACE] vace_checkpoint_dir: {self.vace_checkpoint_dir}")

        self.post_init()

    def set_video_metadata(self, video_name: str = None, video_path: str = None,
                           question: str = None, ground_truth: str = None,
                           question_type: str = None, options: list = None,
                           dataset: str = None):
        """Set metadata for the current video being processed."""
        self._current_video_metadata = {
            'video_name': video_name,
            'video_path': video_path,
            'question': question,
            'ground_truth': ground_truth,
            'question_type': question_type,
            'options': options,
            'dataset': dataset
        }

    # ------------------------------------------------------------------ #
    #                        VACE Loading (lazy)                          #
    # ------------------------------------------------------------------ #
    def _load_vace(self):
        """Lazy load VACE on first forward to avoid meta tensor initialization issues."""
        if self._vace_loaded:
            return

        print(f"[QwenVACE] _load_vace() called. vace_config={self.vace_config is not None}, "
              f"checkpoint_dir={self.vace_checkpoint_dir}")

        if self.vace_config is None or self.vace_checkpoint_dir is None:
            print("[QwenVACE] ERROR: No VACE config provided, skipping VACE loading")
            self._vace_loaded = True
            return

        try:
            import os
            print("[QwenVACE] Loading WanVace (lazy initialization)...")
            vace_cfg_obj = DictToObj(self.vace_config)

            model_device = self.device
            if isinstance(model_device, torch.device):
                device_id = model_device.index if model_device.type == 'cuda' and model_device.index is not None else 0
            else:
                device_id = 0

            target_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

            print(f"[QwenVACE] Model device: {model_device}")
            print(f"[QwenVACE] Extracted device_id: {device_id}")

            old_env = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', None)
            try:
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

                print(f"[QwenVACE] Initializing WanVace on device {target_device} (device_id={device_id})...")
                self.vace = WanVace(
                    vace_cfg_obj,
                    self.vace_checkpoint_dir,
                    device_id=device_id,
                    dit_fsdp=False,
                )
                print(f"[QwenVACE] WanVace initialized successfully on {target_device}")
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
                print(f"[QwenVACE] VACE loaded. Projector: {latent_dim} -> {self.config.hidden_size}")
            except Exception as e:
                print(f"[QwenVACE] Warning: Could not initialize projector: {e}")
                self.vace_projector = None

            self._vace_loaded = True
        except Exception as e:
            import traceback
            print(f"[QwenVACE] ERROR loading VACE: {e}")
            print(f"[QwenVACE] Traceback: {traceback.format_exc()}")
            print("[QwenVACE] Continuing without VACE support")
            self.vace = None
            self.vace_projector = None
            self._vace_loaded = True

    # ------------------------------------------------------------------ #
    #                    VACE Visualization helpers                       #
    # ------------------------------------------------------------------ #
    def visualize_vace_latents(self, vace_latents: List[torch.Tensor], output_dir: str = "./vace_outputs"):
        """Visualize VACE latent features and save statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[QwenVACE] Visualizing {len(vace_latents)} VACE latent tensors...")
            for idx, latent in enumerate(vace_latents):
                if not isinstance(latent, torch.Tensor):
                    continue
                print(f"[QwenVACE]   Latent #{idx}: shape={latent.shape}, dtype={latent.dtype}, "
                      f"min={latent.min():.4f}, max={latent.max():.4f}, mean={latent.mean():.4f}")
            print(f"[QwenVACE] VACE latents saved statistics in {output_dir}")
        except Exception as e:
            print(f"[QwenVACE] Warning: Could not visualize VACE latents: {e}")

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

        Identical to SPIForConditionalGeneration.decode_vace_latents_to_video.
        """
        if self.vace is None or not hasattr(self.vace, 'vae'):
            print("[QwenVACE] ERROR: VACE or VAE decoder not available")
            return None

        # Create subfolder structure if video_name provided
        if video_name is not None:
            # Build path: vace_outputs/{dataset}/QwenVACE/{video_name}
            base_dir = Path(output_path).parent.parent  # Get vace_outputs/
            if dataset:
                video_subfolder = base_dir / dataset / "QwenVACE" / video_name
            else:
                video_subfolder = base_dir / "QwenVACE" / video_name
            video_subfolder.mkdir(parents=True, exist_ok=True)
            output_path = video_subfolder / "decoded_video.mp4"

            # Copy original video (only for video input)
            if not is_image_input and original_video_path is not None and Path(original_video_path).exists():
                import shutil
                original_dest = video_subfolder / f"original_{Path(original_video_path).name}"
                try:
                    shutil.copy2(original_video_path, original_dest)
                    print(f"[QwenVACE] Copied original video to {original_dest}")
                except Exception as e:
                    print(f"[QwenVACE] Warning: Could not copy original video: {e}")

            # Save QA metadata
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
                            f.write("Question:\n" + "-" * 80 + "\n")
                            f.write(f"{metadata['question']}\n\n")
                        if metadata.get('options'):
                            f.write("Options:\n" + "-" * 80 + "\n")
                            for opt in metadata['options']:
                                f.write(f"{opt}\n")
                            f.write("\n")
                        if metadata.get('ground_truth'):
                            f.write("Ground Truth Answer:\n" + "-" * 80 + "\n")
                            f.write(f"{metadata['ground_truth']}\n")
                    print(f"[QwenVACE] Saved question/answer to {qa_file}")
                except Exception as e:
                    print(f"[QwenVACE] Warning: Could not save question/answer: {e}")
            
            # Save input frames if provided (always for image input, optional for video input)
            if input_frames is not None and len(input_frames) > 0:
                try:
                    input_images_dir = video_subfolder / "input_images"
                    input_images_dir.mkdir(parents=True, exist_ok=True)
                    
                    input_type_str = "images" if is_image_input else "video frames"
                    print(f"[QwenVACE] Saving {len(input_frames)} input {input_type_str} to {input_images_dir}")
                    
                    import numpy as np
                    import cv2
                    
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
                            print(f"[QwenVACE] Warning: Unexpected frame dimension {frame_tensor.shape}, skipping")
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
                    
                    print(f"[QwenVACE] ✓ Saved {len(input_frames)} input images to {input_images_dir}")
                except Exception as e:
                    print(f"[QwenVACE] Warning: Could not save input frames: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import cv2

            print(f"[QwenVACE] Decoding {len(vace_latents)} VACE latent tensors to video...")

            decoded_frames = []
            for idx, latent in enumerate(vace_latents):
                if not isinstance(latent, torch.Tensor):
                    continue
                try:
                    if latent.ndim == 4:
                        T = latent.shape[1]
                        print(f"[QwenVACE]   Processing temporal latent with {T} frames")
                        for t in range(T):
                            frame_latent = latent[:, t:t+1, :, :]
                            with torch.no_grad():
                                decoded = self.vace.vae.decode([frame_latent.to(self.device)])
                            decoded_tensor = decoded[0] if isinstance(decoded, (list, tuple)) and len(decoded) > 0 else decoded
                            if hasattr(decoded_tensor, 'to_dense'):
                                decoded_tensor = decoded_tensor.to_dense()
                            decoded_tensor = decoded_tensor.float()
                            if decoded_tensor.dim() == 4:
                                if decoded_tensor.shape[1] == 1 and decoded_tensor.shape[0] > 1 and decoded_tensor.shape[0] <= 4:
                                    decoded_tensor = decoded_tensor.squeeze(1)
                                elif decoded_tensor.shape[0] == 1:
                                    decoded_tensor = decoded_tensor.squeeze(0)
                            if decoded_tensor.dim() != 3:
                                continue
                            frame = decoded_tensor.permute(1, 2, 0).cpu().numpy()
                            if frame.min() < 0:
                                frame = (frame + 1) / 2
                            frame = np.clip((frame * 255).astype(np.uint8), 0, 255)
                            if frame.ndim == 3 and frame.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            else:
                                frame_bgr = frame
                            decoded_frames.append(frame_bgr)

                    elif latent.ndim == 3:
                        latent_4d = latent.unsqueeze(1)
                        with torch.no_grad():
                            decoded = self.vace.vae.decode([latent_4d.to(self.device)])
                        decoded_tensor = decoded[0] if isinstance(decoded, (list, tuple)) and len(decoded) > 0 else decoded
                        if hasattr(decoded_tensor, 'to_dense'):
                            decoded_tensor = decoded_tensor.to_dense()
                        decoded_tensor = decoded_tensor.float()
                        if decoded_tensor.dim() == 4:
                            if decoded_tensor.shape[1] == 1 and decoded_tensor.shape[0] > 1 and decoded_tensor.shape[0] <= 4:
                                decoded_tensor = decoded_tensor.squeeze(1)
                            elif decoded_tensor.shape[0] == 1:
                                decoded_tensor = decoded_tensor.squeeze(0)
                        if decoded_tensor.dim() != 3:
                            continue
                        frame = decoded_tensor.permute(1, 2, 0).cpu().numpy()
                        if frame.min() < 0:
                            frame = (frame + 1) / 2
                        frame = np.clip((frame * 255).astype(np.uint8), 0, 255)
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = frame
                        decoded_frames.append(frame_bgr)
                        decoded_frames.append(frame_bgr)

                except Exception as e:
                    print(f"[QwenVACE] Warning: Could not decode latent #{idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if not decoded_frames:
                print("[QwenVACE] ERROR: No frames decoded from VACE latents")
                return None

            print(f"[QwenVACE] Originally decoded {len(decoded_frames)} frames from VACE")

            # Temporal interpolation
            if temporal_interpolation > 1 and len(decoded_frames) > 1:
                print(f"[QwenVACE] Applying {temporal_interpolation}x temporal interpolation...")
                interpolated_frames = []
                for i in range(len(decoded_frames) - 1):
                    frame1 = decoded_frames[i].astype(np.float32)
                    frame2 = decoded_frames[i + 1].astype(np.float32)
                    interpolated_frames.append(decoded_frames[i])
                    for j in range(1, temporal_interpolation):
                        alpha = j / temporal_interpolation
                        blended = (1 - alpha) * frame1 + alpha * frame2
                        interpolated_frames.append(blended.astype(np.uint8))
                interpolated_frames.append(decoded_frames[-1])
                decoded_frames = interpolated_frames
                print(f"[QwenVACE] After interpolation: {len(decoded_frames)} frames")

            # Write video
            first_frame = decoded_frames[0]
            height, width = first_frame.shape[:2]

            codecs = ['avc1', 'mp4v', 'H264', 'X264', 'h264', 'XVID', 'MJPG']
            temp_output = None
            for codec_name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    if not out.isOpened():
                        continue
                    for frame in decoded_frames:
                        out.write(frame)
                    out.release()
                    if Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
                        temp_output = str(output_path)
                        break
                except Exception:
                    continue

            if temp_output is None:
                print(f"[QwenVACE] ERROR: All video codecs failed to create valid file")
                return None

            # Re-encode with ffmpeg
            temp_path = Path(output_path).with_suffix('.temp.mp4')
            try:
                import subprocess
                output_path = Path(output_path)
                if output_path.exists():
                    output_path.rename(temp_path)
                cmd = [
                    'ffmpeg', '-y', '-i', str(temp_path),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(output_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    temp_path.unlink()
                    return str(output_path)
                else:
                    if temp_path.exists():
                        temp_path.rename(output_path)
                    return str(output_path)
            except Exception:
                if temp_path.exists() and not Path(output_path).exists():
                    temp_path.rename(output_path)
                return str(output_path)

        except ImportError:
            print("[QwenVACE] ERROR: cv2 and numpy required. Install with: pip install opencv-python numpy")
            return None
        except Exception as e:
            print(f"[QwenVACE] ERROR decoding VACE latents to video: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------ #
    #                              Forward                                #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # NOTE: image_tchw / video_tchw are passed through for VACE encoding
        image_tchw: Optional[List[torch.FloatTensor]] = None,
        video_tchw: Optional[List[torch.FloatTensor]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

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
            
            # print(f"\n[QwenVACE] ========== 初始TOKEN统计 ==========")
            # print(f"[QwenVACE] input_ids总长度: {initial_text_tokens}")
            # print(f"[QwenVACE]   - 纯文本token数: {num_pure_text_tokens}")
            # print(f"[QwenVACE]   - image placeholder token数: {num_image_tokens}")
            # print(f"[QwenVACE]   - video placeholder token数: {num_video_tokens}")
            # print(f"[QwenVACE] 初始inputs_embeds shape: {inputs_embeds.shape}")

            # ---- VACE encoding (same logic as SPI) ----
            vace_latent_embeds = None
            if self.vace is not None:
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
                        
                        # Move to VACE device
                        if hasattr(self.vace, 'device'):
                            vace_device = self.vace.device
                            frames_for_vace = [f.to(vace_device) for f in frames_for_vace]
                            ref_images_list = [[ref.to(vace_device) for ref in refs] for refs in ref_images_list]
                        
                        # Log the mode
                        if is_image_input:
                            print(f"[QwenVACE] 🖼️  Image-to-Video Mode: {len(frames_for_vace)} images as references")
                        else:
                            print(f"[QwenVACE] 🎬 Reference-to-Video Mode: Using first frame as reference")
                        print(f"[QwenVACE] Reference frames: {len(ref_images_list)}, shapes: {[r[0].shape for r in ref_images_list]}")
                        print(f"[QwenVACE] Input frames shapes: {[f.shape for f in frames_for_vace]}")
                        
                        vace_latents = self.vace.vace_encode_frames(frames_for_vace, ref_images=ref_images_list, masks=None)

                        video_identifier = self._current_video_metadata.get('video_name')
                        original_video = self._current_video_metadata.get('video_path')
                        dataset = self._current_video_metadata.get('dataset', 'unknown')
                        
                        if video_identifier is None:
                            video_identifier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                        if self.visualize_vace_videos:
                            video_path = self.decode_vace_latents_to_video(
                                vace_latents,
                                output_path="./vace_outputs/QwenVACE/decoded_video.mp4",
                                fps=8,
                                video_name=video_identifier,
                                original_video_path=original_video,
                                temporal_interpolation=4,
                                input_frames=frames_list,
                                dataset=dataset,
                                is_image_input=is_image_input
                            )

                        if self.vace_projector is not None and len(vace_latents) > 0:
                            # Convert each spatial-temporal position to individual tokens
                            vace_token_list = []
                            total_tokens = 0
                            
                            for latent_idx, latent in enumerate(vace_latents):
                                # Reshape: [C, T, H, W] -> [T*H*W, C] (每个时空位置成为一个token)
                                C, T, H, W = latent.shape
                                num_tokens = T * H * W
                                latent_tokens = latent.permute(1, 2, 3, 0).view(num_tokens, C)  # [T*H*W, C]
                                vace_token_list.append(latent_tokens)
                                total_tokens += num_tokens
                                print(f"[QwenVACE] Latent #{latent_idx}: [C={C}, T={T}, H={H}, W={W}] -> {num_tokens} tokens of dim {C}")
                            
                            # Concatenate all tokens: [total_tokens, C]
                            if len(vace_token_list) > 1:
                                vace_features = torch.cat(vace_token_list, dim=0)
                            else:
                                vace_features = vace_token_list[0]
                            
                            # Project each token: [total_tokens, C] -> [total_tokens, hidden_size]
                            vace_features = vace_features.to(self.device, dtype=inputs_embeds.dtype)
                            vace_latent_embeds = self.vace_projector(vace_features)
                            vace_latent_embeds = vace_latent_embeds.unsqueeze(0)  # [1, total_tokens, hidden_size]
                            
                            print(f"[QwenVACE] VACE tokens preserved: {total_tokens} individual tokens")
                            print(f"[QwenVACE] Final VACE embeds shape: {vace_latent_embeds.shape}")
                            print(f"[QwenVACE] No information lost - each spatial-temporal position is a separate token!")

                    except Exception as e:
                        print(f"[QwenVACE] WARNING: VACE processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        vace_latent_embeds = None

            # ---- Standard Qwen2.5-VL visual pipeline (NO spatial encoder) ----
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                
                # print(f"[QwenVACE] ========== 图像TOKEN处理后 ==========")
                # print(f"[QwenVACE] 图像visual features数量: {n_image_features}")
                # print(f"[QwenVACE] 替换后inputs_embeds shape: {inputs_embeds.shape}")

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                
                # print(f"[QwenVACE] ========== 视频TOKEN处理后 ==========")
                # print(f"[QwenVACE] 视频visual features数量: {n_video_features}")
                # print(f"[QwenVACE] 替换后inputs_embeds shape: {inputs_embeds.shape}")

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ---- Position IDs & RoPE - 在 VACE concatenation 之前计算 ----
        # Calculate RoPE using original input_ids and attention_mask lengths before VACE concatenation
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
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
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ---- Concatenate VACE latent embeddings - after RoPE calculation ----
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
            
            print(f"\n[QwenVACE] ========== TOKEN计数：VACE拼接前的完整统计 ==========")
            print(f"[QwenVACE] 当前inputs_embeds组成 (总长度={original_inputs_embeds_len}):")
            print(f"[QwenVACE]   1️⃣ 纯文本token: {num_text_tokens}")
            print(f"[QwenVACE]   2️⃣ 图像visual token: {num_image_tokens_final}")
            print(f"[QwenVACE]   3️⃣ 视频visual token: {num_video_tokens_final}")
            print(f"[QwenVACE]   📝 小计 (文本+图像+视频): {original_inputs_embeds_len}")
            print(f"[QwenVACE]")
            print(f"[QwenVACE] 即将添加的VACE tokens:")
            print(f"[QwenVACE]   4️⃣ VACE spatial-temporal tokens: {num_vace_tokens}")
            print(f"[QwenVACE]")
            print(f"[QwenVACE] Tensor shapes:")
            print(f"[QwenVACE]   - input_ids: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"[QwenVACE]   - attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            print(f"[QwenVACE]   - inputs_embeds (当前): {inputs_embeds.shape}")
            print(f"[QwenVACE]   - vace_latent_embeds: {vace_latent_embeds.shape}")
            print(f"[QwenVACE] =============================================")
            
            inputs_embeds = torch.cat([inputs_embeds, vace_latent_embeds], dim=1)
            
            # 【Token数量统计 - 拼接后】
            new_inputs_embeds_len = inputs_embeds.shape[1]
            print(f"\n[QwenVACE] ========== TOKEN计数：VACE拼接后 ==========")
            print(f"[QwenVACE] 最终inputs_embeds组成 (总长度={new_inputs_embeds_len}):")
            print(f"[QwenVACE]   1️⃣ 纯文本token: {num_text_tokens}")
            print(f"[QwenVACE]   2️⃣ 图像visual token: {num_image_tokens_final}")
            print(f"[QwenVACE]   3️⃣ 视频visual token: {num_video_tokens_final}")
            print(f"[QwenVACE]   4️⃣ VACE spatial-temporal tokens: {num_vace_tokens}")
            print(f"[QwenVACE]   ═══════════════════════════════")
            print(f"[QwenVACE]   📊 总计输入MLLM的tokens: {new_inputs_embeds_len}")
            print(f"[QwenVACE]")
            print(f"[QwenVACE] 增加的token数量: {new_inputs_embeds_len - original_inputs_embeds_len}")
            print(f"[QwenVACE] inputs_embeds shape: {inputs_embeds.shape}")
            print(f"[QwenVACE] =============================================")
            
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                num_vace_tokens = vace_latent_embeds.shape[1]
                vace_attention_mask = torch.ones(
                    (batch_size, num_vace_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, vace_attention_mask], dim=1)
                
                # 【Attention Mask统计】
                new_attention_mask_len = attention_mask.shape[1]
                # print(f"\n[QwenVACE] ========== ATTENTION MASK扩展 ==========")
                # print(f"[QwenVACE] 原始 attention_mask 长度: {original_attention_mask_len}")
                # print(f"[QwenVACE] 新的 attention_mask 长度: {new_attention_mask_len}")
                # print(f"[QwenVACE] 增加的attention mask长度: {new_attention_mask_len - original_attention_mask_len}")
                # print(f"[QwenVACE] attention_mask shape: {attention_mask.shape}")
            
            # Extend position_ids to cover VACE tokens
            if position_ids is not None:
                vace_seq_len = vace_latent_embeds.shape[1]
                if position_ids.dim() == 3:
                    # [3, batch_size, seq_len]
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
                # print(f"\n[QwenVACE] ========== POSITION IDS扩展 ==========")
                # print(f"[QwenVACE] position_ids shape: {position_ids.shape}")
                # print(f"[QwenVACE] position_ids 最后几个值: {position_ids[0, 0, -5:] if position_ids.dim() == 3 else position_ids[0, -5:]}")
            
            # 【最终总结】
            final_input_ids_len = input_ids.shape[1] if input_ids is not None else 0
            final_attention_mask_len = attention_mask.shape[1] if attention_mask is not None else 0
            final_inputs_embeds_len = inputs_embeds.shape[1]
            
            # print(f"\n[QwenVACE] ========== 最终TENSOR维度验证 ==========")
            # print(f"[QwenVACE] input_ids长度: {final_input_ids_len}")
            # print(f"[QwenVACE] attention_mask长度: {final_attention_mask_len}")
            # print(f"[QwenVACE] inputs_embeds长度: {final_inputs_embeds_len}")
            # print(f"[QwenVACE]")
            # print(f"[QwenVACE] ✅ 维度匹配检查:")
            # print(f"[QwenVACE]   - attention_mask == inputs_embeds? {final_attention_mask_len == final_inputs_embeds_len} {'✅' if final_attention_mask_len == final_inputs_embeds_len else '❌'}")
            # print(f"[QwenVACE]   - VACE tokens数量: {final_attention_mask_len - final_input_ids_len}")
            # print(f"[QwenVACE]")
            # print(f"[QwenVACE] 📋 完整Token流程:")
            # print(f"[QwenVACE]   input_ids ({final_input_ids_len}) → embed_tokens → inputs_embeds")
            # print(f"[QwenVACE]   → 替换image/video placeholders → inputs_embeds ({original_inputs_embeds_len})")
            # print(f"[QwenVACE]   → 拼接VACE tokens (+{num_vace_tokens}) → final inputs_embeds ({final_inputs_embeds_len})")
            # print(f"[QwenVACE] =============================================\n")
        else:
            # 即使没有VACE，也输出token统计
            if input_ids is not None:
                final_input_ids_len = input_ids.shape[1]
                final_inputs_embeds_len = inputs_embeds.shape[1]
                num_image_tokens_final = (input_ids == self.config.image_token_id).sum().item()
                num_video_tokens_final = (input_ids == self.config.video_token_id).sum().item()
                num_text_tokens = final_input_ids_len - num_image_tokens_final - num_video_tokens_final
                
                # print(f"\n[QwenVACE] ========== 最终TOKEN统计 (无VACE) ==========")
                # print(f"[QwenVACE] 输入MLLM的tokens组成 (总长度={final_inputs_embeds_len}):")
                # print(f"[QwenVACE]   1️⃣ 纯文本token: {num_text_tokens}")
                # print(f"[QwenVACE]   2️⃣ 图像visual token: {num_image_tokens_final}")
                # print(f"[QwenVACE]   3️⃣ 视频visual token: {num_video_tokens_final}")
                # print(f"[QwenVACE]   ═══════════════════════════════")
                # print(f"[QwenVACE]   📊 总计: {final_inputs_embeds_len}")
                # print(f"[QwenVACE] =============================================\n")

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
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
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

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None

        return model_inputs
