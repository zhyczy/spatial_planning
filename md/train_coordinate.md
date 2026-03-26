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