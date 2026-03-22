# train_correspondence — 相机位姿预测 LoRA 微调

## 概述

对 `SpaForConditionalGeneration`（Qwen3.5-VL）进行 LoRA 微调，预测场景中多帧图像之间的相对相机变换。

**架构：**
```
SpaCorrespondenceModel
  ├── SpaForConditionalGeneration  [骨干网络 + LoRA 适配器]
  │    ├── SpaVisionModel (ViT，冻结)
  │    └── SpaModel (LLM + 4D M-RoPE)
  └── PoseRegressionHead           [MLP: hidden_dim → 9D]
```

**输出：** 每个有序图像对 (i→j) 预测 9D 位姿向量，前 6 维为旋转（6D 连续表示），后 3 维为平移。

**坐标系约定：** 所有帧对齐到第一帧相机坐标系（第一帧光心为原点）。

---

## 前置数据准备

训练前需先完成以下两步数据预处理：

```bash
# 1. 生成 relative_transforms（相机位姿和相对变换矩阵）
python src/data_process/process_reconstruct.py --max -1

# 2. 生成 3D_pos（每像素 XYZ 坐标，供 4D M-RoPE 使用）
python src/data_process/process_pointcloud.py --max -1
```

输出目录：
- `datasets/train/SPAR_7M/spar/reconstruct/{entry_id}.npz`
- `datasets/train/SPAR_7M/spar/3D_pos/{entry_id}.npz`

---

## 目录结构要求

```
spatial_planning/
├── checkpoints/
│   └── Qwen3.5-4B/          # 预训练模型权重
├── datasets/train/SPAR_7M/spar/
│   ├── train_10k.json        # 训练数据索引
│   ├── reconstruct/          # process_reconstruct.py 输出
│   └── 3D_pos/               # process_pointcloud.py 输出
└── train_records/            # 训练输出（自动创建）
```

---

## 使用方式

### 方式一：直接用 Python 运行（单 GPU）

```bash
cd spatial_planning

python train_correspondence.py \
    --model_path checkpoints/Qwen3.5-4B \
    --output_dir checkpoints/spa_correspondence
```

### 方式二：用 Shell 脚本运行（多 GPU DDP，推荐）

```bash
cd spatial_planning

# 用法：bash scripts/train_correspondence.sh [num_gpus] [num_samples]

bash scripts/train_correspondence.sh            # 全部 GPU，完整数据集
bash scripts/train_correspondence.sh 2          # 2 GPU，完整数据集
bash scripts/train_correspondence.sh 2 100      # 2 GPU，100 条样本（调试用）
bash scripts/train_correspondence.sh 1 6        # 单 GPU，6 条样本（最小测试）
```

脚本内部使用 `torchrun --nproc_per_node $NPROC` 启动 DDP 训练，torchrun 路径固定为：
```
/egr/research-actionlab/caizhon2/miniconda3/envs/spc/bin/torchrun
```

---

## 命令行参数说明

### 路径参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model_path` | `checkpoints/Qwen3.5-4B` | 预训练模型权重目录 |
| `--json_path` | `datasets/.../train_10k.json` | 训练数据 JSON 文件路径 |
| `--output_dir` | `checkpoints/spa_correspondence` | 检查点输出目录 |
| `--pos3d_dir` | `datasets/.../3D_pos` | 每像素 XYZ `.npz` 目录（传空字符串可禁用 4D M-RoPE） |

### 训练超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | `3` | 训练轮数 |
| `--lr` | `2e-4` | 学习率 |
| `--lora_rank` | `16` | LoRA 秩 r |
| `--max_images` | `4` | 每个场景最多使用的图像数（控制显存用量） |
| `--grad_accum` | `8` | 梯度累积步数 |
| `--save_steps` | `200` | 每隔多少步保存一次检查点 |
| `--num_workers` | `4` | DataLoader 工作进程数 |
| `--max_samples` | `None`（全部） | 截断数据集到指定条数（调试用） |

### 模型结构参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--skip_layers` | `-8 -4 -1` | 从哪些 LLM 层取 hidden state 拼接后送入 PoseRegressionHead。例如 `--skip_layers -4 -1` 会将倒数第 4 层和最后一层拼接（input_dim 翻倍） |
| `--cycle_weight` | `0.1` | 旋转循环一致性损失权重（N≥3 帧时生效，设为 0 禁用） |
| `--train_vision` | `False`（flag） | 加上此 flag 则同时解冻 ViT 视觉编码器参与微调 |

### WandB 日志

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--wandb_project` | `""` | WandB 项目名，留空则不启用 WandB |
| `--wandb_entity` | `""` | WandB 用户名或团队名 |
| `--wandb_run_name` | `""` | WandB run 名称，留空则自动生成 |

---

## Shell 脚本默认超参数

`scripts/train_correspondence.sh` 中硬编码的默认值：

```
EPOCHS=3
LR=2e-4
LORA_RANK=16
MAX_IMAGES=4
GRAD_ACCUM=8
NUM_WORKERS=4
SKIP_LAYERS="-1"          # 注意：脚本默认只用最后一层（与 Python 默认 -8 -4 -1 不同）
CYCLE_WEIGHT=0.1
SAVE_STEPS=500            # 注意：脚本默认 500，Python 默认 200

WANDB_PROJECT="SPI"
WANDB_ENTITY="actmrv"
WANDB_RUN_NAME="lora_r${LORA_RANK}_ep${EPOCHS}_cycle${CYCLE_WEIGHT}"

OUTPUT_DIR="train_records/correspondence"
```

---

## 损失函数

```
loss = geodesic_rotation_loss(R_pred, R_gt)   # 旋转角度误差（弧度均值）
     + L1_loss(t_pred, t_gt)                  # 平移 L1 误差
     + cycle_weight * cycle_consistency_loss   # 三元组旋转循环一致性（可选）
```

---

## 输出文件

检查点保存在 `--output_dir`（或脚本中的 `train_records/correspondence/`）：

```
train_records/correspondence/
├── step_500/
│   ├── adapter_config.json       # LoRA 配置
│   ├── adapter_model.safetensors # LoRA 权重
│   └── pose_head.pt              # PoseRegressionHead 权重
├── step_1000/
│   └── ...
└── final/
    └── ...
```
