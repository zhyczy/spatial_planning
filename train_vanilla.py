import argparse
import json
import logging
import math
import os
import sys
from itertools import combinations


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None          # type: ignore[assignment]
    _WANDB_AVAILABLE = False


import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from models.spa_emb import SpaForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Re-use dataset detection logic from the data processing pipeline
sys.path.insert(0, os.path.join(_ROOT, "src", "data_process"))
from reconstruct_3d import detect_dataset, _scene_id_from_path  # noqa: E402

# Re-use evaluation utilities
from evaluation import (
    load_dataset as load_eval_dataset,
    prepare_batch_spa,
    run_inference_spa,
    extract_answer_letter,
    extract_answer_number,
    compute_metrics,
    log_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DDP helpers ───────────────────────────────────────────────────────────────

local_rank: int = 0
world_size: int = 1


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ── paths ─────────────────────────────────────────────────────────────────────
SPAR_ROOT = os.path.join(
    _ROOT, "datasets/train/SPAR_7M/spar"
)
RECONSTRUCT_DIR = os.path.join(SPAR_ROOT, "reconstruct")
POS3D_DIR       = os.path.join(SPAR_ROOT, "3D_pos")


class AnswerOnlyModel(nn.Module):
    """
    Ablation model: LoRA fine-tuning with only LM answer-prediction loss.
    No pose regression head, no <pose> tokens.

    Used for two ablation studies:
      - no_cam:  keeps 4D M-RoPE (image_xyz passed), removes pose prediction
      - vanilla: uses original 3D M-RoPE (no image_xyz), removes pose prediction
    """

    def __init__(self, spa_model: nn.Module, use_xyz: bool = True):
        super().__init__()
        self.spa_model = spa_model
        self.use_xyz = use_xyz

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        image_xyz:      list | None = None,
        coord_scale:    float = 100.0,
        labels:         torch.Tensor | None = None,
        **kwargs,
    ):
        fwd_kwargs = dict(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            pixel_values         = pixel_values,
            image_grid_thw       = image_grid_thw,
            output_hidden_states = False,
            return_dict          = True,
        )
        if self.use_xyz:
            fwd_kwargs["image_xyz"]   = image_xyz
            fwd_kwargs["coord_scale"] = coord_scale

        outputs = self.spa_model(**fwd_kwargs)

        if labels is None:
            return None, None, None

        logits = outputs.logits                          # (1, seq_len, V)
        shift_logits = logits[:, :-1, :]                   # (1, seq_len-1, V)
        shift_labels = labels[:, 1:].to(logits.device)     # (1, seq_len-1)

        # 只取有效 label 位置的 logits，避免整个 (seq_len, V) 留在显存里
        mask = shift_labels[0] != -100                     # (seq_len-1,)
        shift_logits = shift_logits[0, mask]               # (N_valid, V)
        shift_labels = shift_labels[0, mask]               # (N_valid,)
        lm_loss = F.cross_entropy(shift_logits, shift_labels)
        _ldict = {"lm_loss": lm_loss.item()}
        return None, lm_loss, _ldict


# ── dataset ───────────────────────────────────────────────────────────────────

class SPARDataset(Dataset):
    """
    One sample = one SPAR scene entry.

    For each entry with N images (capped at max_images):
      - Builds a multi-image chat prompt with A(N,2) = N*(N-1) <pose> tokens,
        one per ordered pair (i, j) with i≠j, enumerated as:
            (0,1), (0,2), ..., (0,N-1),
            (1,0), (1,2), ..., (1,N-1),
            ...
            (N-1,0), ..., (N-1,N-2)
      - Returns processor tensors + GT relative transforms T_{i→j}.

    GT transforms come from reconstruct/{entry_id}.npz:
        relative_transforms[i, j] = T_{i→j}
        = inv(poses_ff[j]) @ poses_ff[i]
        Transforms a 3-D point from camera-i frame to camera-j frame.
    """

    def __init__(
        self,
        json_path:           str,
        spar_root:           str,
        reconstruct_dir:     str,
        processor,
        pose_token_id:       int,
        max_images:          int = 4,
        pos3d_dir:           str | None = None,
        spatial_merge_size:  int = 2,
        max_samples:         int | None = None,
        plus:                bool = False,
        no_pose:             bool = False,
    ):
        import json
        with open(json_path) as fh:
            entries = json.load(fh)

        self.samples = []
        for e in entries:
            eid = e.get("id", "")
            npz = os.path.join(reconstruct_dir, f"{eid}.npz")
            if not (os.path.exists(npz) and e.get("image")):
                continue
            # also require 3D_pos file if a pos3d_dir is given
            if pos3d_dir is not None:
                p3d = os.path.join(pos3d_dir, f"{eid}.npz")
                if not os.path.exists(p3d):
                    continue
            else:
                p3d = None
            self.samples.append((e, npz, p3d))

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.spar_root           = spar_root
        self.processor           = processor
        self.pose_token_id       = pose_token_id
        self.max_images          = max_images
        self.pos3d_dir           = pos3d_dir
        self.spatial_merge_size  = spatial_merge_size
        self.plus                = plus
        self.no_pose             = no_pose
        log.info(f"SPARDataset: {len(self.samples)} valid entries "
                 f"(out of {len(entries)} total)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry, npz_path, p3d_path = self.samples[idx]

        # ── load images ───────────────────────────────────────────────────────
        # Paths in the JSON are relative to {dataset}/images/; resolve using
        # detect_dataset() (scannet / scannetpp / structured3d).
        images = []
        for rel in entry["image"]:
            scene_id = _scene_id_from_path(rel)
            dataset  = detect_dataset(scene_id)
            full = os.path.join(self.spar_root, dataset, "images", rel)
            try:
                images.append(Image.open(full).convert("RGB"))
            except (FileNotFoundError, OSError):
                break
            if len(images) == self.max_images:
                break

        # ── validate against GT ───────────────────────────────────────────────
        data     = np.load(npz_path)
        n_stored = data["relative_transforms"].shape[0]
        N        = min(len(images), n_stored)

        if N < 2:
            raise RuntimeError(
                f"Sample {idx} (id={entry.get('id')}) has only {N} valid "
                f"images after loading; skipping. Check image paths."
            )

        images = images[:N]

        # ── GT transforms (skipped in no_pose ablation mode) ─────────────────
        if self.no_pose:
            gt_transforms = None
        else:
            rel_transforms = torch.tensor(
                data["relative_transforms"][:N, :N], dtype=torch.float32
            )  # (N, N, 4, 4)

            # All ordered pairs (i, j) with i≠j — A(N,2) = N*(N-1) pairs
            pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

            # GT: T_{i→j} for every ordered pair
            gt_transforms = torch.stack(
                [rel_transforms[i, j] for (i, j) in pairs], dim=0
            )  # (N*(N-1), 4, 4)

        # ── build multi-image prompt ──────────────────────────────────────────
        content: list = []
        for img in images:
            content.append({"type": "image", "image": img})

        pose_sentences = []
        # ── build prompt (plus / no_pose modes append QA for LM loss) ────────
        # Extract question/answer from conversations list (ShareGPT format)
        # or from flat "question"/"answer" fields.
        _convs = entry.get("conversations", [])
        _question = (
            entry.get("question")
            or next((c["value"] for c in _convs if c.get("from") == "human"), None)
        )
        _answer = (
            entry.get("answer")
            or next((c["value"] for c in _convs if c.get("from") == "gpt"), None)
        )

        labels = None
        if (self.plus or self.no_pose) and _question and _answer:
            question = _question
            answer   = _answer
            qa_content = list(content)
            if self.no_pose:
                # No pose sentences — just append the question
                qa_content.append({"type": "text", "text": question})
            else:
                # Replace the last text element to include the question
                qa_content[-1] = {
                    "type": "text",
                    "text": " ".join(pose_sentences) + " " + question,
                }
            # Full conversation with assistant answer
            text_full = self.processor.apply_chat_template(
                [{"role": "user", "content": qa_content},
                 {"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False,
            )
            proc_out = self.processor(
                text=[text_full], images=images,
                return_tensors="pt", padding=False,
            )
            # Supervise only the answer tokens at the tail of the sequence.
            # Mask everything before (including the <think> block) so the
            # model's reasoning behaviour is not suppressed.
            # <|im_end|> is a special token so suffix tokenisation is stable.
            suffix_ids = self.processor.tokenizer(
                answer + "<|im_end|>\n", add_special_tokens=False
            )["input_ids"]
            suffix_len = len(suffix_ids)
            labels = proc_out["input_ids"].clone()
            labels[0, :-suffix_len] = -100          # mask all except answer tokens
        else:
            messages = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            # ── tokenise + encode images ──────────────────────────────────────
            proc_out = self.processor(
                text   = [prompt_text],
                images = images,
                return_tensors = "pt",
                padding        = False,
            )

        # ── 3D position maps ──────────────────────────────────────────────────
        # Load per-pixel XYZ from 3D_pos file and downsample to the LLM patch
        # grid resolution consumed by get_rope_index() as image_xyz.
        # Each element: (llm_H_i, llm_W_i, 3) where
        #   llm_H_i = grid_thw[i,1] // spatial_merge_size
        #   llm_W_i = grid_thw[i,2] // spatial_merge_size
        image_xyz = None
        if p3d_path is not None:
            try:
                d3d   = np.load(p3d_path)
                thw_all = proc_out["image_grid_thw"]          # (N, 3)
                sms     = self.spatial_merge_size
                xyz_list = []
                n_3d = int(d3d["n_frames"])
                for k in range(min(N, n_3d)):
                    xyz_raw   = d3d[f"frame_{k}_xyz"]         # (H_img, W_img, 3)
                    valid_raw = d3d.get(f"frame_{k}_valid")   # (H_img, W_img) bool | None
                    thw_k     = thw_all[k]                    # (T, H, W) patch units
                    llm_h     = int(thw_k[1]) // sms
                    llm_w     = int(thw_k[2]) // sms
                    xyz_list.append(resize_xyz(xyz_raw, llm_h, llm_w, valid=valid_raw))
                # Pad missing frames with zeros if needed
                for k in range(len(xyz_list), N):
                    thw_k = thw_all[k]
                    llm_h = int(thw_k[1]) // sms
                    llm_w = int(thw_k[2]) // sms
                    xyz_list.append(torch.zeros(llm_h, llm_w, 3))
                image_xyz = xyz_list                          # list of N tensors
            except Exception as exc:
                log.debug(f"3D_pos load failed for {p3d_path}: {exc}")
                image_xyz = None

        return {
            **proc_out,                      # input_ids, attention_mask,
                                             # pixel_values, image_grid_thw …
            "gt_transforms": gt_transforms,  # (N*(N-1), 4, 4)
            "image_xyz":     image_xyz,      # list of (llm_H, llm_W, 3) or None
            "labels":        labels,         # (1, seq_len) for plus mode, else None
        }


def collate_fn(batch):
    """
    Identity collation for batch_size=1.
    The processor already returns tensors with the correct shapes
    (input_ids: (1, seq_len), pixel_values: (total_patches, C, H, W), …).
    """
    assert len(batch) == 1, "Only batch_size=1 is supported"
    return batch[0]


# ── model building ────────────────────────────────────────────────────────────

def build_model(
    model_path:    str,
    pose_token_id: int,
    lora_rank:     int = 16,
    freeze_vision: bool = True,
    skip_layers:   tuple[int, ...] = (-1,),
    plus:          bool = False,
    answer_weight: float = 1.0,
    ablation:      str | None = None,
) -> nn.Module:
    """
    Load backbone, patch M-RoPE, apply LoRA, and attach task head.
    ablation="vanilla" → AnswerOnlyModel with original 3D M-RoPE (no image_xyz)
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_section = config.text_config.rope_scaling.get("mrope_section", [11, 11, 10])

    # ── vanilla: keep original 3D M-RoPE, use stock Qwen model ───────────
    log.info(f"mrope_section: {orig_section} (original 3D M-RoPE, vanilla ablation)")
    spa = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        config             = config,
        torch_dtype        = torch.bfloat16,
        attn_implementation= "sdpa",
    )
    

    # ── Freeze vision encoder ─────────────────────────────────────
    for p in spa.model.visual.parameters():
        p.requires_grad_(False)
    log.info("Vision encoder frozen.")

    # ── apply LoRA to the language model ──────────────────────────────────────
    lora_cfg = LoraConfig(
        r              = lora_rank,
        lora_alpha     = lora_rank * 2,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    spa = get_peft_model(spa, lora_cfg)
    spa.print_trainable_parameters()

    # Gradient checkpointing: trade ~20% speed for ~60% activation memory savings
    spa.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Verify checkpointing propagated to the language model
    lm = spa.model.model.language_model if hasattr(spa.model, 'model') else spa.model.language_model
    gc_flag = getattr(lm, 'gradient_checkpointing', False)
    log.info(f"Gradient checkpointing enabled. language_model.gradient_checkpointing={gc_flag}")
    if not gc_flag:
        lm.gradient_checkpointing = True
        log.info("Manually set gradient_checkpointing=True on language_model")

    # ── ablation: answer-only model (no pose head) ───────────────────────────
    use_xyz = (ablation != "vanilla")
    log.info(f"Ablation '{ablation}': AnswerOnlyModel (use_xyz={use_xyz})")
    return AnswerOnlyModel(spa, use_xyz=use_xyz)



# ── periodic evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_benchmarks(
    model: nn.Module,
    processor,
    device: torch.device,
    eval_datasets: dict[str, str],
    spatial_merge_size: int = 2,
    max_new_tokens: int = 512,
    local_rank: int = 0,
    world_size: int = 1,
    eval_limit: int | None = None,
) -> dict[str, dict]:
    """
    Run evaluation on the given benchmark datasets, distributed across all ranks.
    Each rank processes every world_size-th sample; rank 0 gathers and logs results.
    Returns the full metrics dict on rank 0, empty dict on other ranks.
    """
    from pathlib import Path

    # Disable blocking kernel launches during eval (CUDA_LAUNCH_BLOCKING is
    # checked per-launch, so toggling os.environ takes effect immediately)
    _prev_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    _model = model.module if hasattr(model, "module") else model
    _model.eval()
    # Access the underlying spa_model for generation
    spa = _model.spa_model
    # gradient_checkpointing_enable() sets use_cache=False; re-enable for eval
    # so generation uses KV cache (otherwise each decode step is O(n²))
    spa.config.use_cache = True

    all_metrics: dict[str, dict] = {}

    for ds_name, ds_dir in eval_datasets.items():
        ds_path = Path(ds_dir)
        try:
            samples = load_eval_dataset(ds_path, dataset=ds_name, limit=eval_limit)
        except Exception as exc:
            log.warning(f"[eval] Failed to load {ds_name} from {ds_dir}: {exc}")
            continue

        # Shard samples across ranks: rank r processes indices r, r+world_size, ...
        shard = samples[local_rank::world_size]
        if local_rank == 0:
            log.info(f"[eval] Running {ds_name}: {len(samples)} samples "
                     f"({len(shard)} per rank × {world_size} ranks)")

        results = []
        for i, item in enumerate(tqdm(shard, desc=f"[eval] {ds_name} rank{local_rank}",
                                      leave=False, disable=(local_rank != 0))):
            try:
                inputs, prompt_text, image_xyz = prepare_batch_spa(
                    item, processor,
                    spatial_merge_size=spatial_merge_size,
                    use_coord=False,   # vanilla: no 3D coords
                    coord_scale=100.0,
                    thinking=False,
                )
                output = run_inference_spa(
                    inputs, image_xyz, spa, processor,
                    max_new_tokens=max_new_tokens,
                    coord_scale=100.0,
                    vanilla=True,  # vanilla ablation
                )
                full_output = "<answer>" + output
                fmt = item.get("format_type", "select")
                if fmt == "fill":
                    prediction = extract_answer_number(full_output)
                else:
                    prediction = extract_answer_letter(full_output)
                results.append({
                    "prediction": prediction,
                    "answer": item.get("answer", ""),
                    "category": item.get("category", "unknown"),
                    "format_type": fmt,
                })
            except Exception as exc:
                log.warning(f"[eval] {ds_name} sample {item.get('index')}: {exc}",
                            exc_info=True)
                results.append({
                    "prediction": "",
                    "answer": item.get("answer", ""),
                    "category": item.get("category", "unknown"),
                    "format_type": item.get("format_type", "select"),
                })
            if local_rank == 0 and (i + 1) % 50 == 0:
                log.info(f"[eval] {ds_name}: {i + 1}/{len(shard)} samples done")

        # Gather results from all ranks to rank 0
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, results)
            if local_rank == 0:
                results = [r for shard_results in gathered for r in shard_results]

        if local_rank == 0:
            metrics = compute_metrics(results)
            all_metrics[ds_name] = metrics
            log_metrics(metrics, f"[eval] {ds_name}", log)

    spa.config.use_cache = False  # restore for gradient-checkpointed training
    os.environ["CUDA_LAUNCH_BLOCKING"] = _prev_blocking  # restore for training
    _model.train()
    return all_metrics


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    global local_rank, world_size

    # ── DDP initialisation ────────────────────────────────────────────────────
    _env_rank = os.environ.get("LOCAL_RANK")
    if _env_rank is not None:
        local_rank = int(_env_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        log.info(f"DDP: local_rank={local_rank}  world_size={world_size}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info("Single-GPU / CPU mode")

    # ── processor + tokeniser ─────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    pose_token_id = -1  # unused in ablation modes
    rank0_print(f"Ablation '{args.ablation}': <pose> token not added")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        args.model_path,
        pose_token_id  = pose_token_id,
        lora_rank      = args.lora_rank,
        freeze_vision  = not args.train_vision,
        skip_layers    = tuple(args.skip_layers),
        plus           = args.plus,
        answer_weight  = args.answer_weight,
        ablation       = args.ablation,
    )
    model = model.to(device)

    # vanilla ablation: disable 3D position maps (uses original 3D M-RoPE)
    args.pos3d_dir = None
    rank0_print(f"pos3d_dir = {args.pos3d_dir}")

    # ── resolve spatial_merge_size from vision config ─────────────────────────
    import json as _json
    _vcfg = _json.load(open(os.path.join(args.model_path, "config.json"))
                       ).get("vision_config", {})
    spatial_merge_size = int(_vcfg.get("spatial_merge_size", 2))
    rank0_print(f"spatial_merge_size = {spatial_merge_size}")

    # ── DDP wrapping ──────────────────────────────────────────────────────────
    # find_unused_parameters=False: DDP only tracks requires_grad=True params;
    # frozen backbone weights are invisible to it. All trainable params (LoRA +
    # PoseHead) participate in every forward pass, so the extra graph traversal
    # from True is unnecessary.
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)
        _model = model.module   # unwrapped reference for checkpointing
    else:
        _model = model

    # ── dataset / loader ──────────────────────────────────────────────────────
    _no_pose = args.ablation is not None
    dataset = SPARDataset(
        json_path           = args.json_path,
        spar_root           = SPAR_ROOT,
        reconstruct_dir     = RECONSTRUCT_DIR,
        processor           = processor,
        pose_token_id       = pose_token_id,
        max_images          = args.max_images,
        pos3d_dir           = args.pos3d_dir,
        spatial_merge_size  = spatial_merge_size,
        max_samples         = args.max_samples,
        plus                = args.plus or _no_pose,
        no_pose             = _no_pose,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size,
                           rank=local_rank, shuffle=True)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = (sampler is None),
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        sampler     = sampler,
    )

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        _log_path = os.path.join(args.output_dir, "train.log")
        _fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
        ))
        logging.getLogger().addHandler(_fh)
        log.info(f"Log file: {_log_path}")

    # ── evaluation datasets (all ranks) ─────────────────────────────────────
    # All ranks build the same dict so the eval condition is consistent across
    # ranks (needed for dist.barrier() and distributed sample sharding).
    eval_datasets: dict[str, str] = {}
    if args.eval_steps > 0:
        if os.path.isdir(args.mindcube_dir):
            eval_datasets["mindcube"] = args.mindcube_dir
            if local_rank == 0:
                log.info(f"Eval dataset: mindcube → {args.mindcube_dir}")
        elif local_rank == 0:
            log.warning(f"MindCube dir not found: {args.mindcube_dir} — skipping")
        if os.path.isdir(args.sparbench_dir):
            eval_datasets["sparbench_multi_view"] = args.sparbench_dir
            if local_rank == 0:
                log.info(f"Eval dataset: sparbench_multi_view → {args.sparbench_dir}")
        elif local_rank == 0:
            log.warning(f"SPARBench dir not found: {args.sparbench_dir} — skipping")

    # ── WandB (rank 0 only) ───────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and args.wandb_project and local_rank == 0
    if use_wandb:
        wandb.init(
            entity  = args.wandb_entity or None,
            project = args.wandb_project,
            name    = args.wandb_run_name or None,
            config  = vars(args),
            dir     = args.output_dir,
        )
        log.info(f"WandB run: {wandb.run.name}  project: {args.wandb_project}")
    elif args.wandb_project and not _WANDB_AVAILABLE and local_rank == 0:
        log.warning("wandb not installed — logging disabled. `pip install wandb`")

    global_step = 0
    running_loss = 0.0
    running_loss_dict: dict[str, float] = {}
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(loader):

            # ── move batch to device ──────────────────────────────────────────
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            gt_transforms  = batch.get("gt_transforms")
            if gt_transforms is not None:
                gt_transforms = gt_transforms.to(device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            # Move 3D position maps to device (list of tensors or None)
            image_xyz = batch.get("image_xyz")
            if image_xyz is not None:
                image_xyz = [xyz.to(device) for xyz in image_xyz]

            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            # ── forward + loss ────────────────────────────────────────────────
            try:
                _, loss, loss_dict = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    pixel_values   = pixel_values,
                    image_grid_thw = image_grid_thw,
                    gt_transforms  = gt_transforms,
                    image_xyz      = image_xyz,
                    cycle_weight   = args.cycle_weight,
                    labels         = labels,
                )
            except Exception as exc:
                log.warning(f"[rank{local_rank}] Step {step} skipped: {exc}")
                continue

            if loss is None:
                log.warning(f"[rank{local_rank}] Step {step}: no <pose> token found, skipping.")
                continue

            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            if loss_dict:
                for k, v in loss_dict.items():
                    running_loss_dict[k] = running_loss_dict.get(k, 0.0) + v

            # ── gradient accumulation ─────────────────────────────────────────
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if local_rank == 0:
                    avg_loss = running_loss / args.grad_accum
                    avg_loss_dict = {
                        k: v / args.grad_accum
                        for k, v in running_loss_dict.items()
                    }
                    running_loss = 0.0
                    running_loss_dict.clear()
                    current_lr = scheduler.get_last_lr()[0]
                    detail = "  ".join(
                        f"{k}={v:.4f}" for k, v in avg_loss_dict.items()
                    )
                    log.info(
                        f"epoch={epoch+1:02d}  step={global_step:05d}  "
                        f"loss={avg_loss:.4f}"
                        + (f"  ({detail})" if detail else "")
                        + f"  lr={current_lr:.2e}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr":   current_lr,
                                "epoch":      epoch + 1,
                                **{f"train/{k}": v for k, v in avg_loss_dict.items()},
                            },
                            step=global_step,
                        )

                    # ── checkpoint ────────────────────────────────────────────
                    if global_step % args.save_steps == 0:
                        _save_checkpoint(_model, tokenizer, args.output_dir,
                                         global_step)

                else:
                    running_loss = 0.0
                    running_loss_dict.clear()

                # ── periodic evaluation (all ranks participate) ───────────
                if (args.eval_steps > 0
                        and global_step % args.eval_steps == 0
                        and eval_datasets):
                    if local_rank == 0:
                        log.info(f"[eval] Starting evaluation at step {global_step}")
                    eval_metrics = evaluate_benchmarks(
                        model=_model,
                        processor=processor,
                        device=device,
                        eval_datasets=eval_datasets,
                        spatial_merge_size=spatial_merge_size,
                        max_new_tokens=args.eval_max_new_tokens,
                        local_rank=local_rank,
                        world_size=world_size,
                        eval_limit=args.eval_max_samples,
                    )
                    if local_rank == 0 and eval_metrics:
                        # Save eval results to JSON
                        eval_json_path = os.path.join(
                            args.output_dir,
                            f"eval_step_{global_step:05d}.json",
                        )
                        with open(eval_json_path, "w", encoding="utf-8") as _ef:
                            json.dump(
                                {"step": global_step, "results": {
                                    ds: {
                                        "accuracy": m["overall_accuracy"],
                                        "correct":  m["correct_samples"],
                                        "total":    m["total_samples"],
                                        "category_accuracy": m.get("category_accuracy", {}),
                                    }
                                    for ds, m in eval_metrics.items()
                                }},
                                _ef, indent=2,
                            )
                        log.info(f"[eval] Results saved → {eval_json_path}")
                        if use_wandb:
                            wandb_eval: dict = {}
                            for ds_name, m in eval_metrics.items():
                                wandb_eval[f"eval/{ds_name}/accuracy"] = m["overall_accuracy"]
                                for cat, cat_acc in m.get("category_accuracy", {}).items():
                                    wandb_eval[f"eval/{ds_name}/{cat}_accuracy"] = cat_acc
                            wandb.log(wandb_eval, step=global_step, commit=True)

    # Final checkpoint (rank 0 only)
    if local_rank == 0:
        _save_checkpoint(_model, tokenizer, args.output_dir, global_step,
                         suffix="final")
    log.info(f"[rank{local_rank}] Training complete.")
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


def _save_checkpoint(
    model: AnswerOnlyModel,
    tokenizer,
    output_dir: str,
    step: int,
    suffix: str = "",
) -> None:
    tag  = f"step_{step}" + (f"_{suffix}" if suffix else "")
    ckpt = os.path.join(output_dir, tag)
    os.makedirs(ckpt, exist_ok=True)

    model.spa_model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    if hasattr(model, "pose_head"):
        torch.save(
            model.pose_head.state_dict(),
            os.path.join(ckpt, "pose_head.pt"),
        )
    log.info(f"Checkpoint saved → {ckpt}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of SpaForConditionalGeneration "
                    "for relative camera pose prediction."
    )
    p.add_argument(
        "--model_path",
        default=os.path.join(_ROOT, "checkpoints/Qwen3.5-4B"),
        help="Path to Qwen3.5-VL checkpoint",
    )
    p.add_argument(
        "--json_path",
        default=os.path.join(SPAR_ROOT, "train_10k.json"),
        help="Path to SPAR training JSON",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(_ROOT, "checkpoints/spa_correspondence"),
    )
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lora_rank",    type=int,   default=16,
                   help="LoRA rank r")
    p.add_argument("--max_images",   type=int,   default=4,
                   help="Max images per scene (memory budget)")
    p.add_argument("--grad_accum",   type=int,   default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--save_steps",   type=int,   default=200)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Truncate dataset to this many samples (None = use all)")
    p.add_argument(
        "--train_vision",
        action="store_true",
        help="Also unfreeze the vision encoder (ViT) for fine-tuning",
    )
    p.add_argument(
        "--pos3d_dir",
        default=POS3D_DIR,
        help="Directory of 3D_pos *.npz files (per-pixel XYZ for 4D M-RoPE). "
             "Pass empty string to disable.",
    )
    p.add_argument(
        "--skip_layers",
        type=int, nargs="+", default=[-8, -4, -1],
        help="LLM layer indices whose hidden states are concatenated before "
             "PoseRegressionHead. e.g. --skip_layers -4 -1 (default: -8 -4 -1)",
    )
    p.add_argument(
        "--cycle_weight",
        type=float, default=0.1,
        help="Weight for rotation cycle-consistency loss (0 to disable). "
             "Active only when N >= 3 views.",
    )
    p.add_argument(
        "--plus",
        action="store_true",
        help="Enable correspondence_plus mode: the model also predicts the text "
             "answer (from entry['answer']) and its cross-entropy loss is added "
             "to the pose loss.",
    )
    p.add_argument(
        "--answer_weight",
        type=float, default=1.0,
        help="Weight applied to the LM answer-prediction loss in plus mode.",
    )
    p.add_argument(
        "--ablation",
        choices=["no_cam", "vanilla"],
        default="vanilla",
        help="Ablation study mode. "
             "no_cam: keeps 4D M-RoPE position embedding, removes pose prediction, "
             "only LM answer loss. "
             "vanilla: uses original Qwen 3D M-RoPE, removes pose prediction, "
             "only LM answer loss.",
    )
    # ── Periodic evaluation ──────────────────────────────────────────────────
    p.add_argument(
        "--eval_steps", type=int, default=2,
        help="Run evaluation every N optimizer steps (default: 2). "
             "Set to 0 to disable periodic evaluation.",
    )
    p.add_argument(
        "--mindcube_dir",
        default=os.path.join(_ROOT, "datasets/evaluation/MindCube"),
        help="Path to MindCube evaluation dataset directory.",
    )
    p.add_argument(
        "--sparbench_dir",
        default=os.path.join(_ROOT, "datasets/evaluation/SPARBench"),
        help="Path to SPARBench evaluation dataset directory.",
    )
    p.add_argument(
        "--eval_max_new_tokens", type=int, default=512,
        help="Max new tokens for evaluation generation.",
    )
    p.add_argument(
        "--eval_max_samples", type=int, default=None,
        help="Max samples per eval dataset (None = full dataset).",
    )
    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--wandb_project",
        default="",
        help="WandB project name. Leave empty to disable WandB logging.",
    )
    p.add_argument(
        "--wandb_entity",
        default="",
        help="WandB entity (username or team name). Leave empty to use default.",
    )
    p.add_argument(
        "--wandb_run_name",
        default="",
        help="WandB run name (optional; auto-generated when empty).",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
