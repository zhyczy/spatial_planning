# CameraMovementRL 快速开始

## 📦 创建的文件

```
spatial_planning/
├── camera_movement_rl.py          # 核心模型实现 ⭐
├── train_camera_movement.py       # 训练脚本框架
├── test_camera_movement.py        # 测试和验证脚本
├── CAMERA_MOVEMENT_README.md      # 详细文档
└── QUICKSTART.md                  # 本文件
```

## 🚀 快速使用

### 1️⃣ 加载预训练冻结模型（30秒）

```python
import torch
from camera_movement_rl import build_model_from_checkpoint

# 加载模型
model, image_token_id = build_model_from_checkpoint(
    adapter_path="train_records/coordinate_no_cam_mindcube/step_550"
)
model = model.to("cuda:0")
model.eval()
```

### 2️⃣ 前向传播

```python
# 准备输入
input_ids = torch.randint(0, 151936, (batch_size, seq_len)).to("cuda:0")
attention_mask = torch.ones((batch_size, seq_len)).to("cuda:0")

# 推理
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

# 获取旋转矩阵
R_pred = outputs['R_pred']  # shape: (batch_size, 3, 3)
```

### 3️⃣ 验证旋转矩阵

```python
# 正交性检验
RTR = torch.bmm(R_pred.transpose(-2, -1), R_pred)
ortho_error = (RTR - torch.eye(3).unsqueeze(0)).abs().max()
print(f"Orthogonality error: {ortho_error:.6f}")  # 应接近0

# 行列式检验
dets = torch.det(R_pred)
print(f"Determinants: {dets}")  # 应接近1
```

## 📊 模型结构速览

```
CameraMovementRL
├── Input: (B, L, hidden_dim) from frozen MLLM
│
├── Step 1: MLLM Forward (Frozen ❄️)
│   └─ output: (B, L, hidden_dim)
│
├── Step 2: Learnable Cam Token
│   └─ concat [cam_token, mllm_output] → (B, L+1, hidden_dim)
│
├── Step 3: Shallow Transformer (2 layers)
│   ├─ Linear: hidden_dim → d_encoder (512)
│   ├─ TransformerEncoder (2 layers, 8 heads)
│   └─ Linear: d_encoder → hidden_dim
│
├── Step 4: Rotation Prediction
│   ├─ Extract cam_token_output = X[0]
│   ├─ MLP: (2560) → 6D
│   └─ Gram-Schmidt: 6D → (3, 3) rotation matrix
│
└─ Output: R_pred (B, 3, 3) ✓ SO(3) constrained
```

## 🔧 可配置参数

```python
model, image_token_id = build_model_from_checkpoint(
    adapter_path="...",          # LoRA checkpoint路径
    base_model_path=None,        # base model（自动检测）
    hidden_dim=2560,             # LLM hidden size
    d_encoder=512,               # Transformer encoder维度
    encoder_nhead=8,             # 注意力头数
    encoder_depth=2,             # Encoder层数
    dtype=torch.bfloat16,        # 数据类型
)
```

## ✅ 验证安装

运行测试脚本：
```bash
cd spatial_planning
python test_camera_movement.py
```

期望输出：
```
Loading base model from: checkpoints/Qwen3.5-4B
Loading LoRA adapter from: train_records/coordinate_no_cam_mindcube/step_550
<|image_pad|> token id = 151939
LoRA model loaded successfully
All MLLM parameters frozen
CameraMovementRL model created
trainable params: 11,500,000 || all params: 8,000,000,000 || trainable%: 0.14%

Running forward pass...
Forward pass completed successfully!
Output shapes:
  - preds_6d:      torch.Size([2, 6])
  - R_pred:        torch.Size([2, 3, 3])
  - rotation_loss: None

Orthogonality error (max |R^T R - I|): 0.000001
Determinants (should be ≈ 1): tensor([1.0000, 0.9999])
✓ All checks passed!
```

## 💡 关键设计决策

| 特性 | 原因 |
|-----|------|
| 冻结MLLM | 防止灾难性遗忘，减少显存占用 |
| 可学习Cam Token | 充当查询，从所有tokens中聚合几何信息 |
| 浅层Encoder | 参数高效，避免过度表示 |
| 6D旋转表示 | Gram-Schmidt自动保证SO(3)约束 |

## 📈 训练检查清单

- [ ] 模型可以加载
- [ ] Forward pass不报错
- [ ] 旋转矩阵验证通过（正交性+行列式）
- [ ] 损失函数计算正确
- [ ] 梯度流向可训练参数
- [ ] 可以保存和加载检查点

## 🐛 常见问题

**Q: MLLM参数没有被冻结？**
```python
# 检查
for name, param in model.named_parameters():
    if param.requires_grad and 'spa_model' in name:
        print(f"WARNING: {name} is trainable!")
```

**Q: 旋转矩阵不满足约束？**
```python
# 检查Gram-Schmidt是否正确执行
from src.models.correspondence_llm import rot6d_to_rotmat
r6d = torch.randn(10, 6)
R = rot6d_to_rotmat(r6d)
print((torch.bmm(R.transpose(-2, -1), R) - torch.eye(3)).abs().max())
```

**Q: 显存不足？**
```python
# 减少encoder维度
model = build_model_from_checkpoint(..., d_encoder=256, encoder_depth=1)
```

## 📚 相关文档

- 详细设计：[CAMERA_MOVEMENT_README.md](CAMERA_MOVEMENT_README.md)
- 模型源码：[camera_movement_rl.py](camera_movement_rl.py)
- 训练脚本：[train_camera_movement.py](train_camera_movement.py)
- 测试脚本：[test_camera_movement.py](test_camera_movement.py)

## 🎯 下一步

1. **集成数据加载**：实现 `train_camera_movement.py` 中的数据加载部分
2. **完整训练循环**：填补训练脚本中的占位符
3. **评估指标**：添加旋转误差、测地线距离等指标
4. **模型部署**：导出为ONNX或TorchScript

---

**快速支持**：查看 `CAMERA_MOVEMENT_README.md` 了解更多细节和最佳实践。
