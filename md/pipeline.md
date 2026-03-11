# Spatial Planning: 完整 Training & Evaluation Pipeline

## 概览

```
SAT 训练数据
     │
     ▼
【Step 1】grpo.py ← 迭代训练 Planner（两种算法可选）
     │
     │  ───GRPO─────────────────────────────────────────────────
     │  rollout → 生成图 → 打分 → 全K组 advantages → GRPO 策略梯度训练 → 合并 LoRA

     ▼
 trained planner (model_merged/)
     │
     ▼
【Step 2】instruction_generation.py        ← Planner 对测试集推断视频生成指令
     │  (MMSIBench test set)
     ▼
 results.jsonl  (每个问题→0或1条视频生成 prompt)
     │
     ▼
【Step 3】generate_video_data.py    ← Wan2.1-VACE-14B 生成辅助视频及下采样帧
     │
     ▼
 generated_videos/  (每个问题→若干生成视频 + 下采样帧)
     │
     ▼
【Step 4】evaluation.py             ← VQA 评估（baseline vs augmented）
     │
     ▼
 accuracy 对比报告
```

---

## 环境准备

```bash
conda activate spi        # Python 3.10, torch 2.11, transformers 4.57, trl 0.25

cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning
```

**模型检查点：**

| 角色 | 路径 |
|------|------|
| Planner (初始/微调目标) | `checkpoints/Qwen3.5-4B` |
| Critic (打分，训练阶段) | `checkpoints/Qwen3.5-9B` |
| Generator | `checkpoints/Wan2.1-VACE-14B` |
| Judge (VQA 评估) | `checkpoints/Qwen3.5-4B` |

---

## Step 1: 迭代 GRPO 训练（推荐）(`grpo.py`)

### 原理

- **GRPO**：所有 K 个 rollout 都参与训练，归一化奖励作为 advantage，policy gradient 更新

每轮迭代：
1. **Rollout** — 当前 Planner 以高温度对每个问题生成 G 组（默认 G=8）不同的 `<thinking>+<instructions>`（多卡并行）
2. **Execution** — Flux2Klein 将每组 instructions 渲染成图片（多卡并行）
3. **Labeling** — Critic 模型对每组生成图打分（多卡并行，打分逻辑与 DPO 完全相同）
4. **GRPO Dataset** — 将所有 G 组 rollout 的奖励归一化为 advantage：`A_i = (r_i - mean_r) / std_r`，展平为 (prompt, completion, advantage) 三元组；累积历史所有轮数据
5. **GRPO Training** — 用 advantage 加权策略梯度损失训练，可选 KL 正则（beta），torchrun 多卡 DDP
6. **LoRA Merge** — 合并 LoRA 到 base model，作为下一轮 Planner

**GRPO 损失函数：**
```
L = mean_i[ A_i × NLL(completion_i | prompt) ] + beta × mean_i[ KL(π_θ ∥ π_ref)_i ]
```

### 命令

```bash
cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning

# ── Pilot：SAT 2000 samples，5 iterations，LoRA ──
conda run -n spi python grpo.py \
    --dataset sat \
    --data_path datasets/evaluation/SAT/train_action_consequence.json \
    --image_root datasets/evaluation/SAT \
    --max_samples 2000 \
    --output_dir train_records/grpo_sat_pilot \
    --planner_model_path checkpoints/Qwen3.5-4B \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --critic_model_path checkpoints/Qwen3.5-9B \
    --num_iterations 5 \
    --num_rollouts 8 \
    --num_gpus -1 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --beta 0.04 \
    --min_reward_std 0.05 \
    --lora_enable True \
    --lora_rank 64 \
    2>&1 | tee /tmp/sat_grpo_pilot.log

# ── Full scale ──
conda run -n spi python grpo.py \
    --dataset sat \
    --data_path datasets/evaluation/SAT/train_action_consequence.json \
    --image_root datasets/evaluation/SAT \
    --max_samples -1 \
    --output_dir train_records/grpo_sat_full \
    --planner_model_path checkpoints/Qwen3.5-4B \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --critic_model_path checkpoints/Qwen3.5-9B \
    --num_iterations 5 \
    --num_rollouts 8 \
    --num_gpus -1 \
    --learning_rate 5e-5 \
    --beta 0.04 \
    --lora_enable True \
    2>&1 | tee /tmp/sat_grpo_full.log

# ── 续跑 ──
conda run -n spi python grpo.py \
    --resume_from train_records/grpo_sat_full/sat_<timestamp> \
    --resume_iter 2 \
    ... (其他参数与首次相同)
```

### GRPO 参数说明

| 参数 | 默认值 | pilot 推荐 | full 推荐 | 说明 |
|------|--------|-----------|----------|------|
| `--num_rollouts` | `8` | `8` | `8` | 每问题采样组数 G；G 越大 advantage 估计越准，显存/时间也越多 |
| `--beta` | `0.04` | `0.04` | `0.04` | KL 惩罚系数；0 = 纯策略梯度；增大则更保守 |
| `--min_reward_std` | `0.05` | `0.05` | `0.05` | 奖励方差过小的组被丢弃（梯度为零）；0 保留全部 |
| `--learning_rate` | `1e-5` | `1e-4` | `5e-5` | 学习率 |
| `--num_train_epochs` | `1` | `1` | `1` | 每轮训练 epoch 数 |
| `--per_device_batch_size` | `1` | `1` | `1` | 每卡 batch size |
| `--gradient_accumulation_steps` | `8` | `4` | `8` | 梯度累积步数 |

### GRPO 输出结构

```
train_records/grpo_sat_pilot/
└── sat_<timestamp>/
    ├── config.json
    ├── grpo.log
    ├── shards.json
    ├── iter_0/
    │   ├── rollouts.jsonl            # Planner G 组原始输出
    │   ├── generated_images/         # Flux2Klein 生成图
    │   ├── labeled.jsonl             # 带评分的 rollout（打分逻辑同 DPO）
    │   ├── grpo_examples.json        # 本轮 GRPO 训练样本（含 advantage）
    │   ├── grpo_accumulated.json     # 累积训练样本
    │   ├── model_lora/               # LoRA 适配器
    │   └── model_merged/             # 合并模型（→ iter_1 的 Planner）
    └── iter_N/
        └── model_merged/             # ← 最终模型
```

---

## Step 2: 生成视频指令 (`instruction_generation.py`)

Planner 对测试集每个问题推断「是否需要生成视频以及最有价值的视频描述」，输出 0 或 1 条视频生成 prompt。

### 命令

```bash
TRAINED_MODEL="train_records/iterative_dpo_sat_pilot/sat_<timestamp>/iter_2/model_merged"

conda run -n spi python instruction_generation.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --model_path "$TRAINED_MODEL" \
    --model_type qwen-vl \
    --max_samples 30 \
    2>&1 | tee /tmp/gen_instructions.log
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mindcube` | 数据集名：`mmsibench` / `sat` / `mindcube` / `vsibench` |
| `--data_path` | 视数据集而定 | 测试数据 JSON 路径 |
| `--image_root` | 视数据集而定 | 图片根目录 |
| `--model_path` | `Qwen/Qwen2.5-VL-7B-Instruct` | Planner 模型路径（训练后的 merged model） |
| `--model_type` | `qwen-vl` | 模型类型：`qwen-vl`（Qwen2/2.5/3-VL 通用） |
| `--max_samples` | `-1` | 限制处理样本数；`-1` 为全部 |
| `--num_gpus` | `-1` | GPU 数量；自动多卡分片并行 |

### 输出

```
results/mmsibench/<model_name>/mmsibench_<timestamp>/
└── results.jsonl    # 每行一个样本，含 instructions 列表
```

---

## Step 3: 生成补充视频 (`generate_video_data.py`)

Wan2.1-VACE-14B 根据 Step 2 的 instructions，以原始 QA 图作为参考帧（R2V），生成 5 秒（80 帧，16fps）的补充视角视频。每个视频同时按每 10 帧取一个中间帧的策略下采样为 PNG，供后续推理模型使用。

### 命令

```bash
# Step 2 的输出目录名即为 planning_model 名（e.g. model_merged）
conda run -n spi python generate_video_data.py \
    --planning_model model_merged \
    --vace_ckpt checkpoints/Wan2.1-VACE-14B \
    --max_samples 30 \
    2>&1 | tee /tmp/video_gen.log

# 若 predicted_instructions/ 不在默认路径，可指定：
conda run -n spi python generate_video_data.py \
    --planning_model model_merged \
    --predicted_instructions_root results/mmsibench \
    --vace_ckpt checkpoints/Wan2.1-VACE-14B \
    --max_samples 30

# 处理所有 planning model
conda run -n spi python generate_video_data.py \
    --all_planning_models \
    --vace_ckpt checkpoints/Wan2.1-VACE-14B

# 强制重新生成已有视频
conda run -n spi python generate_video_data.py \
    --planning_model model_merged \
    --no_skip_existing
```

> **注意：** `generate_video_data.py` 在 `predicted_instructions/<planning_model>/results.jsonl` 查找 Step 2 的输出。Step 2 实际输出在 `results/mmsibench/<model_name>/mmsibench_<timestamp>/results.jsonl`，需确认路径匹配或用 `--predicted_instructions_root` 指定正确根目录。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--planning_model` | — | Step 2 输出对应的目录名（与 `--predicted_instructions_root` 拼接） |
| `--all_planning_models` | `False` | 处理 `predicted_instructions/` 下所有子目录 |
| `--vace_ckpt` | `checkpoints/Wan2.1-VACE-14B` | Wan2.1-VACE-14B 模型路径 |
| `--model_name` | `vace-14B` | VACE 变体：`vace-1.3B` / `vace-14B` |
| `--predicted_instructions_root` | `predicted_instructions` | Step 2 结果根目录 |
| `--output_root` | `generated_videos` | 生成视频输出根目录 |
| `--size` | `480p` | 视频分辨率（`480p` = 480×832） |
| `--frame_num` | `80` | 生成帧数（80 帧 = 5s @ 16fps） |
| `--sample_steps` | `50` | 扩散采样步数 |
| `--guide_scale` | `5.0` | 分类器自由引导 scale |
| `--num_gpus` | `-1` | GPU 数量 |
| `--max_samples` | `-1` | 限制处理样本数 |
| `--no_skip_existing` | `False` | 加此 flag 则强制重新生成已有视频 |

### 输出结构

```
generated_videos/
└── mmsibench/
    └── <planning_model>/
        └── Wan2.1-VACE-14B/
            ├── 0/                    # 样本 id=0
            │   ├── vid_0.mp4         # 完整视频（80帧，5秒，480p）
            │   ├── vid_0_frames/     # 下采样帧（每10帧取中间帧，共8帧）
            │   │   ├── frame_000.png
            │   │   ├── frame_001.png
            │   │   └── ...
            │   ├── vid_1.mp4
            │   ├── vid_1_frames/
            │   └── ...
            ├── 1/
            └── ...
```

---

## Step 4: 评估 (`evaluation.py`)

同时运行 **Baseline**（仅原图）和 **Augmented**（原图 + 生成图）两种模式并对比。

### 命令

```bash
# 本次实验使用的命令
conda run -n spi python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3.5-4B \
    --data_dir datasets/evaluation/MMSIBench \
    --gen_dir generated_videos/mmsibench/<planning_model>/Wan2.1-VACE-14B \
    --limit 30 \
    --output_dir results/eval_sat_dpo \
    2>&1 | tee /tmp/eval.log

# Baseline only（不传 gen_dir）
conda run -n spi python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3.5-4B \
    --data_dir datasets/evaluation/MMSIBench \
    --limit 30 \
    --output_dir results/eval_baseline
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | `qwen2.5-vl` | VQA 评估模型类型：`qwen2.5-vl` / `qwen3-vl` |
| `--model_path` | 必填 | VQA 评估模型路径（通常用 base model，不是 Planner） |
| `--data_dir` | `datasets/evaluation/MMSIBench` | MMSIBench 数据根目录 |
| `--gen_dir` | `None` | Step 3 生成视频帧根目录（`generated_videos/.../<planning_model>/Wan2.1-VACE-14B/`）；`None` 则只跑 Baseline |
| `--limit` | `None` | 限制评估样本数；`None` 为全部 1000 条 |
| `--batch_size` | `1` | 每次推理 batch 大小 |
| `--max_new_tokens` | `128` | 生成最大 token 数 |
| `--output_dir` | `results/mmsibench` | 结果输出目录 |

### 输出

```
results/eval_grpo/qwen3-vl/qwen3-vl_<timestamp>/
├── baseline_results.json       # 每题 baseline 预测与是否正确
├── augmented_results.json      # 每题 augmented 预测与是否正确
└── summary.json                # 准确率汇总 + 分类别分解
```

控制台同时打印：
```
RESULTS — BASELINE (original images only)
  Accuracy: XX.XX%

RESULTS — AUGMENTED (original + generated images)
  Accuracy: XX.XX%

SUMMARY COMPARISON
  Baseline ✓ → Augmented ✗  (gen images hurt)  :  N
  Baseline ✗ → Augmented ✓  (gen images helped) :  N
```

---

## 常见问题

### SIGSEGV / ImportError: liger-kernel

`iterative_dpo.py` 已固化 `use_liger_loss=False`，不再需要手动处理。

### generate_image_instructions.py 找不到图片

检查 `--image_root` 是否正确，以及数据集 JSON 中的 `image` 路径是相对于 `image_root` 的相对路径还是绝对路径。

### evaluation.py gen_dir 无生成图

`gen_dir` 应指向 `generated_videos/.../<planning_model>/Wan2.1-VACE-14B/`（直接包含按样本 id 命名的子目录），而非其父目录。
