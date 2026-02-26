# DPO Training for Spatial Planning Model

## Overview

通过 **Direct Preference Optimization (DPO)** 训练 Planning Model，使其学会生成更有效的 image-generation instructions，从而帮助下游 MLLM 更好地回答空间推理问题。

整个流程可用两种方式运行：

| 方案 | 文件 | 适用阶段 |
|------|------|----------|
| **单轮 DPO** | `generate_dpo_data.py` + `train_planner.py` | 快速验证思路 |
| **迭代 DPO（推荐）** | `iterative_dpo.py` | 正式训练，5 轮滚雪球 |

---

## 文件结构

```
spatial_planning/
├── generate_dpo_data.py              # 数据生成 pipeline（也被 iterative_dpo 调用）
├── train_planner.py                  # DPO 训练入口（也被 iterative_dpo 调用）
├── iterative_dpo.py                  # ★ 迭代 DPO 编排器（5 轮 snowball）
├── scripts/
│   ├── run_generate_dpo_data.sh      # 单轮数据生成启动脚本
│   ├── run_train_planner.sh          # 单轮训练启动脚本 (lora / full)
│   └── run_iterative_dpo.sh          # ★ 迭代 DPO 启动脚本 (pilot / full / resume)
├── results/
│   ├── dpo_data/                     # 单轮数据输出
│   │   └── mmsibench_<timestamp>/
│   └── iterative_dpo/               # ★ 迭代 DPO 输出
│       └── mmsibench_<timestamp>/
│           ├── config.json           # 运行配置
│           ├── shards.json           # 数据分片 ID 映射
│           ├── iter_0/               # 第 1 轮
│           │   ├── rollouts.jsonl
│           │   ├── generated_images/
│           │   ├── labeled.jsonl
│           │   ├── dpo_pairs.json    # 本轮偏好对
│           │   ├── dpo_accumulated.json  # 累积偏好对 (iter 0)
│           │   ├── model_lora/       # LoRA 适配器
│           │   ├── model_merged/     # 合并后模型 → 用于下一轮 rollout
│           │   └── summary.json
│           ├── iter_1/               # 第 2 轮 (数据累积 iter 0+1)
│           ├── ...
│           ├── iter_4/               # 第 5 轮 (数据累积 iter 0..4)
│           └── final_status.json
└── output/
    ├── planner_dpo_lora/             # 单轮 LoRA 训练输出
    └── planner_dpo_full/             # 单轮 Full 训练输出
```

---

## 方案 1: 数据生成 (`generate_dpo_data.py`)

### 核心思想

不需要人工标注，而是利用现有 pipeline 自动产出高质量偏好对：

```
(图片, 问题) → Planner 采样多条指令 → Generator 生成图片 → MLLM 打分 → 选出最好/最差
```

### 四步流程

#### Step 1: Rollout — 大规模采样

- 对每个 `(图片, 问题)` 样本，Planning Model 以高温度 (T=0.9) 随机生成 **K 组** 不同的 instruction（默认 K=8）
- 高温度确保指令多样性，涵盖不同视角、角度、距离等策略
- 支持多 GPU 并行

#### Step 2: Execution — 图像生成

- 将所有采样到的指令送入 Frozen Generator（Flux2Klein）
- 每条指令生成一张图片
- 同样多 GPU 并行

#### Step 3: Labeling — MLLM 评分

支持三种评分策略（`--scoring_method`）：

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| `confidence` (隐式) | 将原图 + 生成图 + 问题一起给 MLLM，测量回答时的 mean log-probability（置信度） | 快速、客观 |
| `explicit` (显式) | 直接问 MLLM "这张图对回答空间问题帮助有多大？" 打 1-10 分 | 更直观 |
| `both` | 两种方法加权平均 | 最稳健 |

**负样本处理**：
- Generator 生成失败 → score = -1.0（强负信号）
- 空指令（模型认为不需要额外图片）→ score = 0.0（中性）
- 指令语法通顺但生成图片与空间关系不符（幻觉）→ 由 MLLM 评分自然识别为低分

#### Step 4: 构建偏好对

- 从 K 组 rollout 中选出：
  - **Chosen** = 得分最高的 rollout 的完整输出（含 `<thinking>` + `<instructions>`）
  - **Rejected** = 得分最低的 rollout 的完整输出
- 过滤条件：`min_score_gap`（默认 0.5），分差太小的模糊样本不使用
- 输出格式兼容 Qwen-VL-Series-Finetune 的 DPO 数据格式：

```json
[
  {
    "image": ["path/to/scene1.jpg", "path/to/scene2.jpg"],
    "prompt": "<image><image>\nWhat is behind the sofa?",
    "chosen": "<thinking>...</thinking>\n<instructions><instruction>A view from behind...</instruction></instructions>",
    "rejected": "<thinking>...</thinking>\n<instructions><instruction>Darken the lighting...</instruction></instructions>"
  }
]
```

### 数据量建议

| 阶段 | 偏好对数量 | 效果 |
|------|-----------|------|
| 起步 | 500 – 1,000 | 基础的偏好学习 |
| 进阶 | 3,000 – 5,000 | 明显的空间规划逻辑提升 |

### 使用

```bash
# 完整 pipeline
bash scripts/run_generate_dpo_data.sh

# 自定义参数
python generate_dpo_data.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --judge_model_path  checkpoints/Qwen3-VL-4B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --num_rollouts 8 \
    --scoring_method confidence \
    --min_score_gap 0.5

# 断点续跑（跳过已完成的步骤）
python generate_dpo_data.py \
    --resume_from results/dpo_data/mmsibench_20260225_120000/ \
    --skip_rollout --skip_execution

# 快速测试（10 个样本）
python generate_dpo_data.py ... --max_samples 10
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_rollouts` | 8 | 每个问题采样的指令组数 |
| `--scoring_method` | confidence | 评分方式: confidence / explicit / both |
| `--min_score_gap` | 0.5 | 偏好对最小分差，低于此值跳过 |
| `--num_inference_steps` | 28 | Flux2Klein 去噪步数 |
| `--num_gpus` | -1 | GPU 数量，-1 为全部 |
| `--skip_rollout` | False | 跳过 rollout 步骤 |
| `--skip_execution` | False | 跳过图片生成步骤 |
| `--skip_labeling` | False | 跳过打分步骤 |

---

## 方案 2: DPO 训练 (`train_planner.py`)

### 核心思想

DPO 让模型学习 **相对的好坏**：
- ✅ "旋转 90° 俯视" → MLLM 看清物体背面（Chosen）
- ❌ "把光线调暗" → MLLM 看不清（Rejected）

模型逐渐学会：避开无效指令，倾向生成能带来"信息增量"的指令。

### 支持的模型

通过 `AutoConfig` 自动检测模型类型：

| 模型 | model_type |
|------|-----------|
| Qwen2-VL | `qwen2_vl` |
| Qwen2.5-VL | `qwen2_5_vl` |
| Qwen3-VL | `qwen3_vl` |
| Qwen3-VL-MoE | `qwen3_vl_moe` |

### 训练模式

#### LoRA 微调（推荐起步）

- 冻结 LLM + Vision Tower + Merger
- 仅训练 LoRA 适配器（默认 rank=64, alpha=16）
- 显存需求低，适合单卡 / 双卡
- DeepSpeed ZeRO-2

```bash
bash scripts/run_train_planner.sh lora
```

#### Full 微调

- 全参数训练，LLM / Vision / Merger 各用独立学习率
- 需要更多显存，推荐 4+ GPU
- DeepSpeed ZeRO-3 + CPU offload

```bash
bash scripts/run_train_planner.sh full
```

### 使用

```bash
# 自动找到最新的 dpo_train.json
bash scripts/run_train_planner.sh lora

# 手动指定数据
DPO_DATA=results/dpo_data/mmsibench_20260225/dpo_train.json \
  bash scripts/run_train_planner.sh lora

# 直接调用（完整控制）
deepspeed train_planner.py \
    --deepspeed ../Qwen-VL-Series-Finetune/scripts/zero2.json \
    --model_id checkpoints/Qwen3-VL-4B-Instruct \
    --data_path results/dpo_data/mmsibench_xxx/dpo_train.json \
    --image_folder "" \
    --output_dir output/planner_dpo_lora \
    --dpo_loss sigmoid \
    --beta 0.1 \
    --lora_enable True \
    --freeze_llm True \
    --freeze_vision_tower True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --bf16 True \
    --gradient_checkpointing True
```

### 主要训练参数

| 参数 | LoRA 默认 | Full 默认 | 说明 |
|------|-----------|-----------|------|
| `--beta` | 0.1 | 0.1 | DPO 的 KL 惩罚系数 |
| `--dpo_loss` | sigmoid | sigmoid | DPO loss 类型 |
| `--learning_rate` | 1e-5 | 5e-6 | LLM 学习率 |
| `--vision_lr` | — | 2e-6 | Vision tower 学习率 |
| `--merger_lr` | — | 5e-6 | Merger 学习率 |
| `--lora_rank` | 64 | — | LoRA rank |
| `--lora_alpha` | 16 | — | LoRA alpha |
| `--num_train_epochs` | 3 | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 2 | 1 | 单卡 batch size |
| `--gradient_accumulation_steps` | 8 | 16 | 梯度累积步数 |

### 训练后使用

LoRA 训练完成后，可以用 Qwen-VL-Series-Finetune 的 merge 脚本合并权重：

```bash
python ../Qwen-VL-Series-Finetune/src/merge_lora_weights.py \
    --model_id checkpoints/Qwen3-VL-4B-Instruct \
    --adapter_path output/planner_dpo_lora \
    --output_path checkpoints/Qwen3-VL-4B-Planner-DPO
```

然后在 `generate_image_instructions.py` 中使用训练好的模型：

```bash
python generate_image_instructions.py \
    --dataset mmsibench \
    --model_path checkpoints/Qwen3-VL-4B-Planner-DPO \
    ...
```

---

## 方案 3: 迭代 DPO — Snowball Training (`iterative_dpo.py`)

### 核心思想

单轮 DPO 用固定 (off-policy) 数据训练，模型自身策略分布没有参与数据生成。
**迭代 DPO** 解决这个问题：每一轮用 **当前最新模型** on-policy 采样，让训练信号不断跟紧策略的演化。

类比"滚雪球"：

```
  Iteration 0: base model → rollout shard 0 → pairs 0                 → DPO → model_0
  Iteration 1: model_0    → rollout shard 1 → pairs 0+1               → DPO → model_1
  Iteration 2: model_1    → rollout shard 2 → pairs 0+1+2             → DPO → model_2
  Iteration 3: model_2    → rollout shard 3 → pairs 0+1+2+3           → DPO → model_3
  Iteration 4: model_3    → rollout shard 4 → pairs 0+1+2+3+4         → DPO → model_4 ✓
```

### 三角色架构

| 角色 | 模型 | 状态 | 说明 |
|------|------|------|------|
| **Planning Model** | Qwen3-VL-4B-Instruct | LoRA 可训练 | 生成 `<thinking>` + `<instructions>` |
| **Executor** | Flux2Klein-4B | 冻结 | 根据指令生成新视角图片 |
| **Critic (Judge)** | Qwen3-VL-8B-Instruct | 冻结 | 评估 rollout 质量 |

### 关键设计

#### On-Policy 采样
- 每轮用 **当前模型** 采样 8 条 rollout（T=0.9）
- 随着迭代推进，模型策略分布变化 → 采样到的正/负例更贴合当前水平

#### 数据累积
- **第 N 轮训练数据 = 第 0~N 轮所有偏好对的并集**
- 防止遗忘早期学到的偏好信号
- 每轮都从 base model 初始化 LoRA，确保 reference model 一致

#### 数据分片
- 175K 数据集随机打乱后均匀切分为 5 份 (~35K/shard)
- 每轮只在新 shard 上做 rollout，避免重复采样
- Pilot 实验: 1700 → 5×340 per shard

#### 边界处理
- 全部 rollout 同分 → 丢弃该样本（无法构建有效偏好）
- 分差 < `min_score_gap` (0.3) → 保留为弱信号，增加覆盖
- 生成失败 → score = -1.0（强负样本）

### 使用

```bash
# ── Pilot 实验 (1700 样本, 5 轮 × ~340) ──
bash scripts/run_iterative_dpo.sh pilot

# ── Full scale (175K 样本, 5 轮 × ~35K) ──
bash scripts/run_iterative_dpo.sh full

# ── 断点续跑 (从第 3 轮继续) ──
bash scripts/run_iterative_dpo.sh resume results/iterative_dpo/mmsibench_xxx 3
```

或直接调用 Python：

```bash
python iterative_dpo.py \
    --dataset mmsibench \
    --data_path datasets/evaluation/MMSIBench/data/test_data_final.json \
    --image_root datasets/evaluation/MMSIBench \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --critic_model_path  checkpoints/Qwen3-VL-8B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --max_samples 1700 \
    --num_iterations 5 \
    --num_rollouts 8 \
    --min_score_gap 0.3 \
    --lora_rank 64 \
    --learning_rate 1e-5 \
    --beta 0.1
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_iterations` | 5 | 迭代轮数（= 数据分片数） |
| `--num_rollouts` | 8 | 每个问题的 on-policy 采样数 |
| `--min_score_gap` | 0.3 | 偏好对最小分差 |
| `--beta` | 0.1 | DPO KL 惩罚系数 |
| `--lora_rank` | 64 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--learning_rate` | 1e-5 | 学习率 |
| `--num_train_epochs` | 1 | 每轮 DPO 训练 epoch 数 |
| `--max_samples` | -1 | 最大样本数 (-1=全部, 1700=pilot) |
| `--resume_from` | — | 断点续跑: 已有 run 目录 |
| `--resume_iter` | 0 | 断点续跑: 从第几轮开始 (0-based) |

### 输出结构

每次运行生成 `results/iterative_dpo/<dataset>_<timestamp>/`，每轮子目录包含：

| 文件 | 说明 |
|------|------|
| `rollouts.jsonl` | 8× on-policy 采样结果 |
| `generated_images/` | Flux2Klein 生成图片 |
| `labeled.jsonl` | MLLM Critic 评分 |
| `dpo_pairs.json` | 本轮偏好对 |
| `dpo_accumulated.json` | **累积所有轮偏好对** (用于训练) |
| `model_lora/` | 本轮 LoRA 权重 |
| `model_merged/` | **合并后模型** → 下一轮 on-policy rollout |
| `summary.json` | 本轮统计信息 |

### 推荐实验路线

1. **Pilot** (1700 样本): 先跑通整个 5 轮循环，验证收益趋势
2. **调超参**: 观察每轮 pair 数量 / score gap 分布，微调 `min_score_gap`, `beta`
3. **Full** (175K 样本): 确认趋势后放 full scale
4. **评估**: 用最终 `model_merged` 跑 `generate_image_instructions.py` + `evaluation.py`

---

## 端到端流程总览

### 单轮 DPO

```
┌─────────────────────────────────────────────────────────────┐
│  generate_dpo_data.py                                       │
│                                                             │
│  Step 1: Rollout (Planner × K rollouts per question)        │
│       ↓                                                     │
│  Step 2: Execution (Flux2Klein → candidate images)          │
│       ↓                                                     │
│  Step 3: Labeling (MLLM judge scores each rollout)          │
│       ↓                                                     │
│  Step 4: Selection (best → Chosen, worst → Rejected)        │
│       ↓                                                     │
│  Output: dpo_train.json                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  train_planner.py                                           │
│                                                             │
│  DPO Training (Qwen VL + DeepSpeed + LoRA/Full)             │
│       ↓                                                     │
│  Output: fine-tuned Planning Model                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  generate_image_instructions.py                             │
│                                                             │
│  Inference with improved Planner → better instructions      │
│       ↓                                                     │
│  image_generation.py → evaluation.py                        │
└─────────────────────────────────────────────────────────────┘
```

### 迭代 DPO (iterative_dpo.py)

```
                    Dataset (175K / 1.7K pilot)
                              │
                    ┌─────────┴─────────┐
                    │ split_dataset(N=5) │
                    └─────────┬─────────┘
              ┌───────┬───────┼───────┬───────┐
           shard_0  shard_1 shard_2 shard_3 shard_4

  ╔══════════════════════════════════════════════════════════╗
  ║  for iteration i in 0..4:                                ║
  ║                                                          ║
  ║  current_model ───► Rollout (shard_i, 8×, T=0.9)        ║
  ║                          │                               ║
  ║              Flux2Klein ─┤─► generated images            ║
  ║                          │                               ║
  ║        Qwen3-VL-8B ─────┤─► scores                      ║
  ║                          │                               ║
  ║              build_preference_pairs()                    ║
  ║                          │                               ║
  ║    pairs_i ──► accumulate(pairs_0..i) ──► DPO training   ║
  ║                          │                               ║
  ║        LoRA merge ───► model_i (→ next iteration)        ║
  ╚══════════════════════════════════════════════════════════╝

                    Final model: iter_4/model_merged/
```

---

## 依赖

在现有 `SPR` conda 环境基础上，额外需要：

```
trl          # DPO Trainer
peft         # LoRA
deepspeed    # 分布式训练
diffusers    # Flux2Klein (数据生成阶段)
```

安装：
```bash
pip install trl peft deepspeed
```
