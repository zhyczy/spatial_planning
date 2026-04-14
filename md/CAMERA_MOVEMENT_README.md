# Camera Movement RL: 旋转矩阵预测模型

## 概述

`CameraMovementRL` 是一个四步前向传播的相机姿态估计模型，用于学习从MLLM隐层输出预测3×3旋转矩阵。

### 四步设计

#### Step 1️⃣ 冻结的MLLM特征提取 (Frozen Feature Extraction)
- 使用预训练的`SpaForConditionalGeneration` (Qwen3.5-VL + LoRA)
- 所有参数完全冻结，保护通用知识
- 输出：文本tokens $T = [t_1, t_2, ..., t_n]$

#### Step 2️⃣ 序列拼接与查询构建 (Sequence Concatenation)
- 引入可学习的相机令牌 $C_{init}$
- 拼接序列：$X = [C_{init}, t_1, t_2, ..., t_n]$
- $C_{init}$ 作为活跃的Query，去计算与所有tokens的注意力

#### Step 3️⃣ 浅层特征融合 (Shallow Transformer Encoder)
- 2层TransformerEncoderLayer (可配置)
- 自注意力机制融合所有特征
- 输出：$C_{out} = X'[0]$ (第一个位置的token)

#### Step 4️⃣ 旋转矩阵预测 (Rotation Matrix Prediction)
- MLP: $C_{out} \rightarrow$ 6D向量
- Gram-Schmidt正交化: 6D向量 $\rightarrow$ 3×3旋转矩阵 $R$
- 确保 $R \in SO(3)$（满足正交性和行列式为1的约束）

## 文件说明

| 文件 | 说明 |
|-----|------|
| `camera_movement_rl.py` | 核心模型实现 |
| `train_camera_movement.py` | 训练脚本框架 |
| `test_camera_movement.py` | 模型测试脚本 |

## 使用方法

### 1. 加载模型

```python
from camera_movement_rl import build_model_from_checkpoint

model, image_token_id = build_model_from_checkpoint(
    adapter_path="train_records/coordinate_no_cam_mindcube/step_550",
    hidden_dim=2560,
    d_encoder=512,
    encoder_nhead=8,
    encoder_depth=2,
    dtype=torch.bfloat16,
)
model = model.to(device)
```

**参数说明**：
- `adapter_path`: LoRA检查点路径（自动检测base model）
- `base_model_path`: Base model路径（可选，自动从adapter_config获取）
- `hidden_dim`: LLM隐层维度（Qwen3.5-4B为2560）
- `d_encoder`: Transformer Encoder维度
- `encoder_nhead`: 注意力头数
- `encoder_depth`: Encoder层数
- `dtype`: 数据类型（推荐bfloat16）

### 2. 前向传播

```python
outputs = model(
    input_ids=input_ids,              # (batch_size, seq_len)
    attention_mask=attention_mask,    # (batch_size, seq_len)
    pixel_values=pixel_values,        # (patches, C, H, W) or None
    image_grid_thw=image_grid_thw,    # (num_images, 3) or None
    gt_transforms=gt_transforms,      # (batch_size, 4, 4) or None
)
```

**输出**：
```python
{
    'preds_6d': torch.Tensor,        # (batch_size, 6) - 6D旋转向量
    'R_pred': torch.Tensor,          # (batch_size, 3, 3) - 旋转矩阵
    'rotation_loss': torch.Tensor,   # scalar or None
    'loss_dict': dict,               # 损失值详情
}
```

### 3. 运行测试

```bash
cd spatial_planning
python test_camera_movement.py
```

输出示例：
```
Loading base model from: checkpoints/Qwen3.5-4B
Loading LoRA adapter from: train_records/coordinate_no_cam_mindcube/step_550
<|image_pad|> token id = 151939
LoRA model loaded successfully
All MLLM parameters frozen
...
Orthogonality error (max |R^T R - I|): 0.000001
Determinants (should be ≈ 1): tensor([1.0000, 0.9999])
✓ All checks passed!
```

## 模型架构细节

### 可学习参数

仅冻结的MLLM之外的参数可训练：

```
├── cam_token: (1, 1, hidden_dim)           ~2.6M params
├── input_proj: hidden_dim → d_encoder      ~1.3M params
├── transformer_encoder (2层)               ~4.2M params
├── output_proj: d_encoder → hidden_dim     ~1.3M params
└── rotation_head: MLP (6D)                 ~2.1M params
                                             ────────────
                                             Total: ~11M
```

MLLM本身通常有数十亿参数，但完全冻结，只有~0.01%的参数可训练。

### 6D旋转表示的优势

相比直接输出9个数字：
- ✅ **满足约束**：Gram-Schmidt确保 $R^T R = I$ 和 $\det(R) = 1$
- ✅ **梯度友好**：正交化过程对反向传播友好
- ✅ **数值稳定**：避免参数化过度完备性导致的梗塞

### 损失函数

使用测地线损失（Geodesic Loss）计算SO(3)上的旋转距离：
$$\mathcal{L}_{rot} = \text{geodesic\_distance}(R_{pred}, R_{gt})$$

## 训练建议

### 1. 从冻结模型开始
完全冻结MLLM可以：
- 减少显存占用
- 加快训练速度
- 避免知识遗忘

### 2. 监控参数梯度
```python
model.print_trainable_parameters()
```

### 3. 学习率选择
- 推荐：`1e-4` 到 `5e-4`
- 与小学习率（1e-5）相比通常更稳定

### 4. 批量大小
- 目前实现支持 `batch_size=1`
- 未来可扩展支持更大批量

## 故障排除

### Q1: 如何验证旋转矩阵的正确性？

```python
R = outputs['R_pred']  # (batch_size, 3, 3)

# 验证正交性
RTR = torch.bmm(R.transpose(-2, -1), R)
print((RTR - torch.eye(3)).abs().max())  # 应接近0

# 验证行列式
dets = torch.det(R)
print(dets)  # 应接近1
```

### Q2: 显存不足如何处理？

- 减小 `d_encoder` （如512→256）
- 减少 `encoder_depth` （如2→1）
- 使用梯度累积

### Q3: 模型为什么不学习？

检查清单：
- [ ] MLLM参数是否正确冻结？
- [ ] 损失函数是否正确计算？
- [ ] 学习率是否太高或太低？
- [ ] 训练数据是否有效？

## 参考文献

### 6D旋转表示
> Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019

### Gram-Schmidt正交化
> 标准线性代数方法，用于确保数值稳定的正交基构造

## 相关文件

- 模型骨干：`src/models/spa_emb.py` (SpaForConditionalGeneration)
- 旋转转换：`src/models/correspondence_llm.py` (rot6d_to_rotmat)
- 损失函数：`src/loss/cam_loss.py` (geodesic_loss)
- 坐标模型：`src/models/coordinate_llm.py` (参考设计)
- 训练参考：`train_coordinate.py` (第一阶段训练)
