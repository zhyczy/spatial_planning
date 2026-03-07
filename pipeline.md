# Spatial Planning: 完整 Training & Evaluation Pipeline

## 概览

```
SAT 训练数据
     │
     ▼
【Step 1】iterative_dpo.py          ← 迭代 DPO 训练 Planner
     │  rollout → 生成图 → 打分 → 构建偏好对 → DPO 训练 → 合并 LoRA
     │  (重复 N 轮，每轮用累积偏好对)
     ▼
 trained planner (model_merged/)
     │
     ▼
【Step 2】generate_image_instructions.py   ← Planner 对测试集推断视角指令
     │  (MMSIBench test set)
     ▼
 results.jsonl  (每个问题→若干视角指令)
     │
     ▼
【Step 3】image_generation.py       ← Flux2Klein 生成辅助图
     │
     ▼
 generated_images/  (每个问题→若干生成图)
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
conda activate SPR        # Python 3.10, torch 2.11, transformers 4.57, trl 0.25

cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning
```

**模型检查点：**

| 角色 | 路径 |
|------|------|
| Planner (初始/微调目标) | `checkpoints/Qwen3-VL-4B-Instruct` |
| Critic (打分) | `checkpoints/Qwen3-VL-8B-Instruct` |
| Generator | `checkpoints/flux2-klein-4B` |
| Judge (VQA 评估) | `checkpoints/Qwen3-VL-4B-Instruct` |

---

## Step 1: 迭代 DPO 训练 (`iterative_dpo.py`)

### 原理

每轮迭代：
1. **Rollout** — 当前 Planner 以高温度对每个问题生成 K 组不同的 `<thinking>+<instructions>`（多卡并行）
2. **Execution** — Flux2Klein 将每组 instructions 渲染成图片（多卡并行）
3. **Labeling** — Critic 模型对每组生成图打分（问题回答置信度，多卡并行）
4. **Preference Pairs** — 选最高分 vs 最低分构成 (chosen, rejected)；累积历史所有轮偏好对
5. **DPO Training** — 在累积偏好对上做 LoRA / Full-model DPO 训练（torchrun 多卡 DDP）
6. **LoRA Merge** — 合并 LoRA 到 base model，作为下一轮 Planner

### 推荐方式：使用一键脚本

```bash
cd /egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning

# ── Pilot：SAT 2000 samples，5 iterations，LoRA，全四步 ──
bash scripts/run_dpo_pipeline.sh pilot lora all

# ── Full scale：SAT 全量 ~172K，5 iterations，LoRA ──
bash scripts/run_dpo_pipeline.sh full lora all

# ── 只跑训练（Step 1） ──
bash scripts/run_dpo_pipeline.sh pilot lora train

# ── 只跑评估（Steps 2-4，需已有训练好的模型） ──
bash scripts/run_dpo_pipeline.sh pilot lora eval

# ── Full-model DPO（解冻所有参数，需要更多显存） ──
bash scripts/run_dpo_pipeline.sh pilot full_model all
```

脚本参数：`$1 SCALE`（pilot|full）、`$2 TRAIN_MODE`（lora|full_model）、`$3 RUN_STEPS`（all|train|eval）

#### Pilot vs Full 配置对比

| 参数 | pilot | full |
|------|-------|------|
| `max_samples` | 2000 | -1（全量 ~172K） |
| `num_iterations` | 5 | 5 |
| `num_rollouts` | 8 | 8 |
| `num_train_epochs` | 2 | 2 |
| `learning_rate` | 1e-4 | 5e-5 |
| `gradient_accumulation_steps` | 4 | 8 |
| `eval_limit`（评估样本数） | 30 | 全部 |
| 输出目录 | `train_records/iterative_dpo_sat_pilot` | `train_records/iterative_dpo_sat_full` |

#### LoRA vs Full-model 训练模式

| 参数 | lora（默认） | full_model |
|------|-------------|------------|
| `--lora_enable` | `True` | `False` |
| 可训练参数 | 仅 LoRA 适配器 | 全部参数 |
| 多卡策略 | torchrun DDP（全部 GPU）| torchrun DDP（全部 GPU）|
| 显存需求 | 低（单卡 24GB 可运行） | 高（建议每卡 ≥ 60GB）|
| `min_score_gap` | 0.3 | 0.3 |

### 手动调用命令

```bash
# Pilot（手动等价命令）
conda run -n SPR python iterative_dpo.py \
    --dataset sat \
    --data_path datasets/evaluation/SAT/train_action_consequence.json \
    --image_root datasets/evaluation/SAT \
    --max_samples 2000 \
    --output_dir train_records/iterative_dpo_sat_pilot \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --critic_model_path checkpoints/Qwen3-VL-8B-Instruct \
    --num_iterations 5 \
    --num_rollouts 8 \
    --num_gpus -1 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --min_score_gap 0.3 \
    --lora_enable True \
    --lora_rank 64 \
    2>&1 | tee /tmp/sat_dpo_pilot.log

# Full scale（全量数据）
conda run -n SPR python iterative_dpo.py \
    --dataset sat \
    --data_path datasets/evaluation/SAT/train_action_consequence.json \
    --image_root datasets/evaluation/SAT \
    --max_samples -1 \
    --output_dir train_records/iterative_dpo_sat_full \
    --planner_model_path checkpoints/Qwen3-VL-4B-Instruct \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --critic_model_path checkpoints/Qwen3-VL-8B-Instruct \
    --num_iterations 5 \
    --num_rollouts 8 \
    --num_gpus -1 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --beta 0.1 \
    --min_score_gap 0.3 \
    --lora_enable True \
    --lora_rank 64 \
    2>&1 | tee /tmp/sat_dpo_full.log
```

### 断点续跑

```bash
conda run -n SPR python iterative_dpo.py \
    --resume_from train_records/iterative_dpo_sat_full/sat_<timestamp> \
    --resume_iter 2 \
    ... (其他参数与首次相同)
```

### 参数说明

#### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mmsibench` | 数据集名称：`sat` / `mmsibench` / `mindcube` / `vsibench` |
| `--data_path` | 必填 | 数据集 JSON/JSONL 文件路径 |
| `--image_root` | 必填 | 图片根目录 |
| `--max_samples` | `-1` | 限制训练样本数，`-1` 为全部；小规模验证建议 50–200 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--planner_model_path` | 必填 | Planner 初始模型路径（合并后模型也可继续用） |
| `--critic_model_path` | `None` | Critic 打分模型；`None` 则使用 Planner 本身打分 |
| `--flux_ckpt` | `checkpoints/flux2-klein-4B` | Flux2Klein 生成模型路径 |

#### 迭代参数

| 参数 | 默认值 | 推荐（正式） | 说明 |
|------|--------|------------|------|
| `--num_iterations` | `5` | `5` | 迭代轮数；更多轮数效果更稳定 |
| `--num_rollouts` | `8` | `8` | 每个样本采样几组 instructions；影响偏好对多样性 |
| `--scoring_method` | `confidence` | `confidence` | 打分方式：`confidence`（置信度）/ `explicit`（直接评分）/ `both` |
| `--min_score_gap` | `0.3` | `0.3` | 偏好对最小分差；`0.0` 保留所有对（用于数据稀少场景） |
| `--num_inference_steps` | `28` | `28` | Flux2Klein 去噪步数 |

#### 训练超参数

| 参数 | 默认值 | pilot 推荐 | full 推荐 | 说明 |
|------|--------|-----------|----------|------|
| `--lora_enable` | `True` | `True` | `True` | `True`=LoRA 模式；`False`=全参数 DPO |
| `--lora_rank` | `64` | `64` | `64` | LoRA 秩；越大表达能力越强但显存越多 |
| `--lora_alpha` | `16` | `16` | `16` | LoRA 缩放系数；通常设为 rank/4 |
| `--learning_rate` | `1e-5` | `1e-4` | `5e-5` | DPO 学习率；过大会遗忘；过小收敛慢 |
| `--beta` | `0.1` | `0.1` | `0.1` | DPO 正则强度；越大越保守（接近 reference model） |
| `--num_train_epochs` | `1` | `2` | `2` | 每轮 DPO 训练 epoch 数 |
| `--per_device_batch_size` | `2` | `2` | `2` | 每卡 batch size（训练内部强制 per_device=1 防 OOM）|
| `--gradient_accumulation_steps` | `8` | `4` | `8` | 梯度累积步数；等效 batch = per_device × accum × gpus |

#### 系统参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_gpus` | `-1` | 使用 GPU 数量；`-1` 为全部可见 GPU；Rollout/Execution/Labeling 多进程分片，DPO 训练用 torchrun DDP |
| `--output_dir` | `train_records/iterative_dpo` | 输出根目录 |
| `--seed` | `42` | 随机种子 |

### 输出结构

```
train_records/iterative_dpo_sat_pilot/
└── sat_<timestamp>/
    ├── config.json                   # 完整运行配置
    ├── iter_0/
    │   ├── rollouts.jsonl            # Planner 原始输出
    │   ├── generated_images/         # Flux2Klein 生成图
    │   ├── labeled.jsonl             # 带评分的 rollout
    │   ├── dpo_pairs.json            # 本轮偏好对
    │   ├── model_lora/               # LoRA 适配器
    │   └── model_merged/             # 合并模型（→ iter_1 的 Planner）
    ├── iter_1/
    │   └── ...
    └── iter_2/
        └── model_merged/             # ← 最终模型
```

---

## Step 2: 生成视角指令 (`generate_image_instructions.py`)

Planner 对测试集每个问题推断「需要哪些补充视角」，输出自然语言 instruction。

### 命令

```bash
TRAINED_MODEL="train_records/iterative_dpo_sat_pilot/sat_<timestamp>/iter_2/model_merged"

conda run -n SPR python generate_image_instructions.py \
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

## Step 3: 生成补充图像 (`image_generation.py`)

Flux2Klein 根据 Step 2 的 instructions 生成补充视角图片。

### 命令

```bash
# Step 2 的输出目录名即为 planning_model 名（e.g. model_merged）
conda run -n SPR python image_generation.py \
    --planning_model model_merged \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --max_samples 30 \
    2>&1 | tee /tmp/image_gen.log

# 或用 results/ 目录下的 timestamp 子目录名（自动查找 predicted_instructions/）
# 若 predicted_instructions/ 不在默认路径，可指定：
conda run -n SPR python image_generation.py \
    --planning_model model_merged \
    --predicted_instructions_root results/mmsibench \
    --flux_ckpt checkpoints/flux2-klein-4B \
    --max_samples 30
```

> **注意：** `image_generation.py` 在 `predicted_instructions/<planning_model>/results.jsonl` 查找 Step 2 的输出。Step 2 实际输出在 `results/mmsibench/<model_name>/mmsibench_<timestamp>/results.jsonl`，需确认路径匹配或用 `--predicted_instructions_root` 指定正确根目录。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--planning_model` | — | Step 2 输出对应的目录名（与 `--predicted_instructions_root` 拼接） |
| `--all_planning_models` | `False` | 处理 `predicted_instructions/` 下所有子目录 |
| `--flux_ckpt` | `checkpoints/flux2-klein-4B` | Flux2Klein 模型路径 |
| `--predicted_instructions_root` | `predicted_instructions` | Step 2 结果根目录 |
| `--output_root` | `generated_images` | 生成图输出根目录 |
| `--num_inference_steps` | `28` | 去噪步数；减少可加速但质量下降 |
| `--num_gpus` | `-1` | GPU 数量 |
| `--max_samples` | `-1` | 限制处理样本数 |
| `--no_skip_existing` | `False` | 加此 flag 则强制重新生成已有图片 |

### 输出结构

```
generated_images/
└── mmsibench/
    └── <planning_model>/
        └── flux2-klein-4B/
            ├── 0/          # 样本 id=0 的生成图
            │   ├── img_0.png
            │   └── img_1.png
            ├── 1/
            └── ...
```

---

## Step 4: 评估 (`evaluation.py`)

同时运行 **Baseline**（仅原图）和 **Augmented**（原图 + 生成图）两种模式并对比。

### 命令

```bash
# 本次实验使用的命令
conda run -n SPR python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
    --data_dir datasets/evaluation/MMSIBench \
    --gen_dir generated_images/mmsibench/mmsibench_<timestamp>/flux2-klein-4B \
    --limit 30 \
    --output_dir results/eval_sat_dpo \
    2>&1 | tee /tmp/eval.log

# Baseline only（不传 gen_dir）
conda run -n SPR python evaluation.py \
    --model_type qwen3-vl \
    --model_path checkpoints/Qwen3-VL-4B-Instruct \
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
| `--gen_dir` | `None` | Step 3 生成图根目录（`generated_images/.../flux2-klein-4B/`）；`None` 则只跑 Baseline |
| `--limit` | `None` | 限制评估样本数；`None` 为全部 1000 条 |
| `--batch_size` | `1` | 每次推理 batch 大小 |
| `--max_new_tokens` | `128` | 生成最大 token 数 |
| `--output_dir` | `results/mmsibench` | 结果输出目录 |

### 输出

```
results/eval_sat_dpo/qwen3-vl/qwen3-vl_<timestamp>/
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

## 本次实验结果（2026-03-06）

### 配置

| 项目 | 值 |
|------|-----|
| 训练数据 | SAT `train_action_consequence.json`，50 samples |
| 迭代轮数 | 3 |
| 每轮 rollouts | 4 |
| LoRA rank | 64，lr=1e-4，β=0.1 |
| 评估集 | MMSIBench，30 samples |
| 最终模型 | `train_results/iterative_dpo_sat_pilot/sat_20260306_153823/iter_2/model_merged` |

### 结果

| 方法 | 准确率 | Correct / Total |
|------|--------|-----------------|
| Baseline（原图） | **26.67%** | 8 / 30 |
| Augmented（+ DPO Planner 生成图） | **26.67%** | 8 / 30 |

**逐样本变化：**
- Gen images 帮助：2 题（Baseline ✗ → Augmented ✓）
- Gen images 损害：2 题（Baseline ✓ → Augmented ✗）
- 净效果：持平

### 分析

本次结果持平的主要原因：
1. **训练规模过小**：仅 50 SAT 样本，47 个累积偏好对，Planner 未能充分泛化
2. **领域迁移**：SAT action-consequence 数据与 MMSIBench 空间推理问题存在差异
3. **训练轮次短**：每轮仅 1 epoch，学习信号不足

### 下一步提升建议

| 改进点 | 建议值 |
|--------|--------|
| `--max_samples` | `-1`（SAT train set 共 172,384 条，用 full 模式全量训练）|
| `--num_iterations` | 5 |
| `--num_rollouts` | 8 |
| `--num_train_epochs` | 2 |
| `--min_score_gap` | 0.3（过滤模糊样本） |
| `--learning_rate` | 5e-5（full scale）/ 1e-4（pilot）|
| 数据集 | 尝试使用与 MMSIBench 更近似的评估域数据做 DPO |
| 运行方式 | 直接用 `bash scripts/run_dpo_pipeline.sh full lora all` |

---

## 常见问题

### SIGSEGV / ImportError: liger-kernel

`iterative_dpo.py` 已固化 `use_liger_loss=False`，不再需要手动处理。

### DPO 训练 OOM

`per_device_train_batch_size` 已强制为 1。如仍 OOM，减小 `--lora_rank`（如 32）或减少 `--num_gpus`（确保每卡显存 ≥ 40GB）。

### generate_image_instructions.py 找不到图片

检查 `--image_root` 是否正确，以及数据集 JSON 中的 `image` 路径是相对于 `image_root` 的相对路径还是绝对路径。

### evaluation.py gen_dir 无生成图

`gen_dir` 应指向 `generated_images/.../flux2-klein-4B/`（直接包含按样本 id 命名的子目录），而非其父目录。
