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


Both files pass syntax checks. Here's a summary of the changes:

Two ablation modes via --ablation {no_cam, vanilla}:

Pose Head	Position Embedding	Loss	Backbone
default	Yes	4D M-RoPE + image_xyz	pose (geodesic+L1+cycle)	SpaForConditionalGeneration
--plus	Yes	4D M-RoPE + image_xyz	pose + LM answer	SpaForConditionalGeneration
--ablation no_cam	No	4D M-RoPE + image_xyz	LM answer only	SpaForConditionalGeneration
--ablation vanilla	No	Original 3D M-RoPE	LM answer only	Qwen3_5ForConditionalGeneration
Key changes:

train_correspondence.py: Added AnswerOnlyModel class (LM loss only, no pose head), --ablation CLI flag, no_pose dataset flag to skip <pose> tokens and GT transforms, vanilla uses stock Qwen3_5ForConditionalGeneration
train_correspondence.sh: Added --ablation MODE flag parsing, e.g. bash scripts/train_correspondence.sh 2 100 --ablation no_cam

整体架构

CoordinatePlusModel
  ├── SpaForConditionalGeneration  [Qwen2-VL backbone + LoRA]
  ├── PoseRegressionHead           [<pose> token → 9D rot+trans]
  └── CoordinateRegressionHead     [<coord> tokens → xyz, MLP only]
数据准备 (Dataset.getitem)
Stage 1：Probe pass（获取 patch 数量）
构建一个不含 <coord> 的临时 prompt，只为了获取 image_grid_thw：


[IMG1][IMG2]...[IMG_N]  pose sentences...  question?
调用 processor 得到 thw_all: (N, 3)，其中每张图的 [T, H_vit, W_vit]。

Stage 2：构建最终 prompt
每张图 k 的 LLM patch 数 = (H_vit // 2) × (W_vit // 2)（spatial merge size = 2）：


n_tok_k = (thw_k[1] // 2) × (thw_k[2] // 2)
例如图 k 有 38 个 patch，则生成 38 个 <coord> token：


Image k 3D spatial coordinates: <coord> <coord> ... <coord>.
最终完整 text 结构：


<|im_start|>user
[IMG1][IMG2]...[IMG_N]
The camera pose of image 2 relative to image 1 is <pose>.
The camera pose of image 1 relative to image 2 is <pose>.
...
Image 1 3D spatial coordinates: <coord>×38.
Image 2 3D spatial coordinates: <coord>×29.
...
Question text?
<|im_end|>
<|im_start|>assistant
Answer text<|im_end|>
Token 序列分布
位置	内容	监督信号
[img_pad] ×M₁	Image 1 vision tokens（ViT patch embeddings）	无
[img_pad] ×M₂	Image 2 vision tokens	无
...	...	...
<pose> ×K	K = N×(N-1) 个姿态 token	→ PoseRegressionHead → geodesic+L1 loss
<coord> ×n₁	Image 1 的 n₁ 个坐标 token	→ CoordinateRegressionHead → L1 loss
<coord> ×n₂	Image 2 的 n₂ 个坐标 token	→ CoordinateRegressionHead → L1 loss
answer tokens	答案文字	→ LM cross-entropy loss
labels 只对 answer + <|im_end|> 部分有效，其余位置设为 -100。

前向推理 (CoordinatePlusModel.forward)

# 1. 单次 backbone forward（teacher forcing，完整序列输入）
outputs = spa_model(input_ids, pixel_values, image_grid_thw, ...)
hidden_coord = outputs.hidden_states[-1]  # (1, seq_len, 2560)

# 2. Pose loss（不变）
pose_positions = (input_ids[0] == pose_token_id).nonzero()
latents = hidden_pose[0, pose_positions]   # (K, D)
preds_9d = pose_head(latents)              # (K, 9)

# 3. LM loss（只对 answer tokens）
lm_loss = cross_entropy(shift_logits, labels, ignore_index=-100)

# 4. Coordinate loss（新架构核心）
coord_pos = (input_ids[0] == coord_token_id).nonzero()  # 所有 <coord> 位置
start = 0
for k in range(N):
    n_tok = (thw_k[1]//2) × (thw_k[2]//2)
    coord_h_k = hidden_coord[0, coord_pos[start:start+n_tok]]  # (n_tok, 2560)
    pred_k = coord_head(coord_h_k)           # (n_tok, 3)，MLP: 2560→256→3
    pred_k = pred_k.view(llm_h, llm_w, 3)   # reshape 为空间网格
    gt_k   = image_xyz[k]                    # (llm_h, llm_w, 3)，patch 级别 GT
    loss_k = L1_loss(pred_k, gt_k)
    start += n_tok
coord_loss = mean(per_img_losses)

# 5. 合并
total = pose_loss + answer_weight × lm_loss + coord_weight × coord_loss


N=2，H₁×W₁=38，H₂×W₂=34：


[0]      <|im_start|>
[1]      user
[2]      \n
─── 图1 视觉区段 ───────────────────────────────── 38 个 patch
[3]      <|vision_start|>
[4..41]  <|image_pad|> × 38
[42]     <|vision_end|>
─── 图2 视觉区段 ───────────────────────────────── 34 个 patch
[43]     <|vision_start|>
[44..77] <|image_pad|> × 34
[78]     <|vision_end|>
─── pose sentences ────────────────────────────
[79..88]  "The camera pose of image 2 relative to image 1 is"
[89]      <pose>          ← pose_pos[0]
[90]      "."
[91..100] "The camera pose of image 1 relative to image 2 is"
[101]     <pose>          ← pose_pos[1]
[102]     "."
─── coord sentences（密集展开，新版本核心）──────────
[103..107] "Image 1 3D spatial coordinates :"
[108..145]  <coord> × 38   ← coord_pos[0:38]  ★★★
[146]     "."
[147..151] "Image 2 3D spatial coordinates :"
[152..185]  <coord> × 34   ← coord_pos[38:72] ★★★
[186]     "."
─── question ──────────────────────────────────
[187..200] question tokens
[201]      <|im_end|>
[202]      \n
─── assistant ─────────────────────────────────
[203]      <|im_start|>
[204]      assistant
[205]      \n
[206..225] answer tokens   ← labels 不是 -100
[226]      <|im_end|>
[227]      \n