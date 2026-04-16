两组按 "rotation_enc" in name 切分，基于 PyTorch named_parameters() 的命名空间：

维度	Group A: rotation_enc_params	Group B: other_params (LoRA + coord_head)
命名前缀	rotation_enc.*	spa_model.base_model.*.lora_A/B.*, coord_head.*
learning rate	args.rotation_enc_lr (默认 2e-4)	args.lr (默认 2e-4)
grad clip	args.rotation_enc_clip (默认 0.3)	args.lora_clip (默认 1.0)
初始化	从零训练（rot_head 零权重 + 恒等 bias）	LoRA 从 0 起；coord_head 默认 init
Group A — rotation_enc.* 具体参数（来自 rotation_rope_llm.py:263 CameraTokenRotationEncoder）

路径	形状来源
rotation_enc.cam_token	(1, d_model=1024)
rotation_enc.input_proj.weight / .bias	(1024, hidden_dim)
rotation_enc.layers.{i}.norm1.weight / .bias	(1024,)
rotation_enc.layers.{i}.norm2.weight / .bias	(1024,)
rotation_enc.layers.{i}.attn.q_proj.weight / .bias	(1024, 1024)
rotation_enc.layers.{i}.attn.k_proj.weight / .bias	(1024, 1024)
rotation_enc.layers.{i}.attn.v_proj.weight / .bias	(1024, 1024)
rotation_enc.layers.{i}.attn.o_proj.weight / .bias	(1024, 1024)
rotation_enc.layers.{i}.ffn.0.weight / .bias	(dim_ff=2048, 1024)
rotation_enc.layers.{i}.ffn.3.weight / .bias	(1024, 2048)
rotation_enc.rot_head.weight / .bias	(6, 1024) ← step 0 R=I
i ∈ [0, num_layers)，默认 num_layers=2。注意 rope_emb 只含 inv_freq buffer，不是参数，不进任何组。

Group B — other_params 的构成

子集	命名模板
LoRA 适配器	spa_model.base_model.model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.default.weight
LoRA 适配器	spa_model.base_model.model.language_model.layers.{i}.mlp.{gate,up,down}_proj.lora_{A,B}.default.weight
coord_head	coord_head.*（DepthPredictionTransformer 的所有参数）
互斥性保证

Group A 的所有参数名都以 rotation_enc. 开头。
LoRA 参数名前缀是 spa_model.base_model...，coord_head 前缀是 coord_head.。
两组的前缀没有重叠，且过滤条件严格用 "rotation_enc" in n：n in A ⇔ "rotation_enc" ∈ n，n in B ⇔ "rotation_enc" ∉ n，互为补集，数学上不可能有交集。


下面是 RotationRoPEModel 在 train_rotation.py 里的完整训练流（把数据、模型、梯度、优化器四层对齐画出来）。


════════════════════════════════════════════════════════════════════════════
  0.  BUILD PHASE  (build_model @ train_rotation.py:115)
════════════════════════════════════════════════════════════════════════════

  SpaForConditionalGeneration  (视觉 ViT + 文本 LLM)
     ├─ visual.*                   requires_grad = False   (freeze_vision)
     ├─ get_input_embeddings()     requires_grad = False   (非 LoRA target)
     ├─ language_model.layers.*    requires_grad = False   (非 LoRA target)
     └─ PEFT LoRA → {q,k,v,o,gate,up,down}_proj.lora_{A,B}.default
                                   requires_grad = True    ← Group B

  CameraTokenRotationEncoder  (rotation_enc)
     ├─ cam_token                  requires_grad = True
     ├─ input_proj                 requires_grad = True
     ├─ layers.{0..N}.*            requires_grad = True    ← Group A (全部)
     └─ rot_head  ← step 0: W=0, b=[1,0,0, 0,1,0]  →  R=I

  DepthPredictionTransformer  (coord_head)
     └─ *                          requires_grad = True    ← Group B


════════════════════════════════════════════════════════════════════════════
  1.  OPTIMIZER  (train_rotation.py:346-370)
════════════════════════════════════════════════════════════════════════════

  AdamW([
    { params: other_params        (LoRA + coord_head),  lr = args.lr          },
    { params: rotation_enc_params (rotation_enc.*),     lr = args.rotation_enc_lr },
  ], weight_decay = 0.01)

  CosineAnnealingLR → 两组 initial_lr 同步按 cosine 衰减


════════════════════════════════════════════════════════════════════════════
  2.  PER-STEP FORWARD  (model(**batch) @ train_rotation.py:463)
════════════════════════════════════════════════════════════════════════════

  DataLoader batch
    ├─ input_ids          (1, seq_len)
    ├─ attention_mask     (1, seq_len)
    ├─ pixel_values       (N_img, C, H, W)           bf16
    ├─ image_grid_thw     (N_img, 3)
    ├─ image_xyz          list[k]: (llm_H, llm_W, 3)          ← 世界坐标 (frame-0)
    ├─ image_xyz_hires    list[k]: (llm_H*up, llm_W*up, 3)
    └─ labels             (1, seq_len)

  Step 1 — merged inputs_embeds                    [rotation_rope_llm.py:434]
  ────────────────────────────────────────────────────────────────────────
    text_embed  = get_input_embeddings()(input_ids)          (frozen)
    img_feat    = get_image_features(pixel_values, ...)      (frozen, no_grad)
    inputs_embeds = text_embed.masked_scatter(img_mask, img_feat)

  Step 2 — rotation prediction                      [rotation_rope_llm.py:451]
  ────────────────────────────────────────────────────────────────────────
    token_txyz_int = _build_token_txyz_int(...)              (long, no grad)
    R = rotation_enc(inputs_embeds.detach(), token_txyz_int) (3, 3) float32
          └─ step 0: R = I          (rot_head 恒等初始化)

  Step 3 — rotate xyz  (float, gradient-preserving) [rotation_rope_llm.py:466]
  ────────────────────────────────────────────────────────────────────────
    rotated_xyz = [ R @ xyz  for xyz in image_xyz ]          (仍是 float, 带梯度)

  Step 4 — build FLOAT 5D position_ids              [rotation_rope_llm.py:471]
  ────────────────────────────────────────────────────────────────────────
    _build_float_position_ids(..., image_xyz = rotated_xyz, coord_scale)
      • text tokens : t=x=y=z = arange(pos)    float
      • image tokens: t = shared pos;
                      x,y,z = rotated_xyz * coord_scale
                      ★ 无 round / 无 .long()  — 梯度可穿过

  Step 5 — manual text-model pass                   [rotation_rope_llm.py:336]
  ────────────────────────────────────────────────────────────────────────
    cos, sin = DifferentiableMRoPE(inputs_embeds, rope_pos_float)
               ★ 自己实现、没有 @torch.no_grad
    for layer in text_model.layers:
        hidden = layer(hidden, position_embeddings=(cos, sin), ...)
    hidden2 = text_model.norm(hidden)
    logits2 = lm_head(hidden2)

  Step 6 — losses                                   [rotation_rope_llm.py:491]
  ────────────────────────────────────────────────────────────────────────
    lm_loss    = CE( shift_logits , shift_labels )
    coord_gt   = _apply_rotation_to_xyz(R.detach(), image_xyz_hires)
    coord_loss = L1( coord_head(visual hidden_k) , coord_gt_k )

    loss = answer_weight * lm_loss + coord_weight * coord_loss
    (rot_loss = None — 没有 gt_rotation 监督)


════════════════════════════════════════════════════════════════════════════
  3.  BACKWARD — 梯度归属                            [train_rotation.py:478]
════════════════════════════════════════════════════════════════════════════

  (loss / grad_accum).backward()

  ┌────────────────────┬─────────────────────────────────────────────────┐
  │   梯度源            │   流向                                           │
  ├────────────────────┼─────────────────────────────────────────────────┤
  │ lm_loss             │ → LoRA  (通过 q/k/v/o/gate/up/down LoRA A/B)     │
  │                     │ → rotation_enc                                  │
  │                     │     (hidden → cos/sin → DiffMRoPE →              │
  │                     │      rope_pos_float → rotated_xyz → R →          │
  │                     │      rot_head → layers → input_proj → cam_token)│
  │ coord_loss          │ → coord_head                                    │
  │                     │ → LoRA  (通过 hidden2)                          │
  │                     │ × rotation_enc  (coord_gt 用 R.detach(), 不回传) │
  └────────────────────┴─────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
  4.  OPTIMIZER STEP  (每 grad_accum 个 micro-batch 一次)
                                                     [train_rotation.py:485]
════════════════════════════════════════════════════════════════════════════

  if (step + 1) % grad_accum == 0:
      clip_grad_norm_(rotation_enc_params, max_norm = args.rotation_enc_clip)  # 0.3
      clip_grad_norm_(other_params,        max_norm = args.lora_clip)          # 1.0
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

  ── 同步 ─────────────────────────────────────────────────────────────────
  所有 all-reduce / ckpt / eval / wandb 按 global_step 节奏触发


════════════════════════════════════════════════════════════════════════════
  5.  TRAINING LOOP 行为特征 (step 0 → step N)
════════════════════════════════════════════════════════════════════════════

  step 0:
    • R = I   →   rotated_xyz = xyz           (等同于 baseline 无旋转)
    • lm_loss / coord_loss 正常计算
    • backward 后 rot_head.weight 首次获得非零梯度

  step 1..N:
    • rot_head.weight 被更新 →  r6d ≠ [1,0,0,0,1,0]
    • Gram-Schmidt 后 R 从 I 逐步偏离一个小角度
    • DifferentiableMRoPE 让高频 RoPE 维度的梯度被放大
         └─ 由 rotation_enc_clip=0.3 严格抑制
    • LoRA + coord_head 正常下降，用 lora_clip=1.0
    • 整个系统通过 lm_loss 端到端地学出一个"帮助 LLM 答题"的 R
关键点一句话总结：step 0 从恒等旋转起步，DifferentiableMRoPE 让 lm_loss 能把梯度送回 rotation_enc，两组独立 clip 让高频梯度不会冲垮训练。



rotation_enc
                         │
                         ▼
                      R, cam_feat
                 ┌───────┴───────────────────────┐
                 │                                │
                 ▼                                ▼
         rotated_xyz = R @ xyz           cam_feat（原始，带梯度）
                 │                                │
                 ▼                         ┌──────┴──── .detach() ────┐
          rope_pos_float                   │                          │
                 │                       coord_gt                coord_head
                 ▼                      (R.detach())              (cam 条件)
          DiffMRoPE cos,sin                  │                          │
                 │                           │                          │
                 ▼                           │                          │
         text_model.layers                   │                          │
                 │                           │                          │
                 ▼                           │                          │
             hidden2 ───────────────────────────────────┐               │
              │   │                                     ▼               │
              │   └─► lm_head → logits → lm_loss     coord_h_k          │
              │                              │          │               │
              │                              │          ▼               │
              │                              │       pred_k ◄───────────┘
              │                              │          │
              │                              │          ▼
              │                              │       coord_loss
              │                              │          │
              └──────────────────────────────┴──────────┘
                         （共享 hidden2）
.detach() 真正阻断的是什么
.detach() 位置	阻断的路径	没阻断的路径
coord_gt = R.detach() @ xyz	coord_loss → coord_gt → R（旋转 GT 去就 pred 的平凡解）	coord_loss → pred_k → hidden2 → cos/sin → R
cam_feat=cam_feat.detach()	coord_loss → cam_proj → cam_feat → rotation_enc（直连短路）	coord_loss → hidden2 → cos/sin → R → rotation_enc
修正后的结论
两个 loss 都通过 hidden2 → cos/sin → R → rotation_enc 这条共享通道回传给 encoder：

lm_loss → rotation_enc：唯一路径（hidden2 → DiffMRoPE → R）
coord_loss → rotation_enc：也通过 hidden2 → DiffMRoPE → R 回传；只是被 R.detach() / cam_feat.detach() 掐断了另两条"直连短路"
.detach() 的设计意图
R.detach() 在 coord_gt：防止 coord_loss 通过"旋转 GT"的平凡解（把 GT 转去迎合坏 pred）来最小化自己——保证 coord_loss 在每一步看到固定的几何目标。
cam_feat.detach() 在 coord_head 条件：防止 coord_head 把 cam_feat 当作自由可学的 bias 向量直接改写 rotation_enc，避免 coord_loss 绕过 LLM 抄近路。
两条 detach 都只是关掉"绕过 LLM 的近路"，coord_loss 对 rotation_enc 的监督依然会通过 LLM hidden state 这条长路径到达——只不过要"跨越整个 LLM"才能改 R。


rotation vs rotation_relative — 训练 / 测试对照
1. 模型结构差异（构建期）
组件	rotation	rotation_relative
CLI 标志	（无）	--relative
coord_head.cam_dim	0	rotation_enc.d_model（1024）
coord_head.cam_proj	不存在	nn.Linear(1024, d_model)
ckpt 识别	state 无 cam_proj.weight	state 含 cam_proj.weight
训练入口：train_rotation.py:200
Eval 识别：evaluation.py:401-469 _load_coord_head 通过 state["cam_proj.weight"].shape[1] 自动推断
2. 前向数据流差异（每 step 共通部分）
两者共用 Steps 1–5（inputs_embeds → rotation_enc → R, cam_feat → rotated_xyz → DifferentiableMRoPE / 推理时整数 RoPE → hidden2），差异只在 coord loss / coord MAE 分支：


          ┌────── rotation ──────┐           ┌─── rotation_relative ───┐
GT        │  R.detach() @ xyz     │           │  xyz（原坐标，不旋转）    │
          │  （rotated frame）      │           │  （original frame）       │
cam 条件  │  coord_head(hidden,   │           │  coord_head(hidden,      │
          │   cam_feat=None)     │           │   cam_feat.detach())    │
监督信号  │  让 head 在旋转帧预测   │           │  让 head 学"旋转帧特征    │
          │                      │           │   → 原帧坐标"的反变换      │
3. 代码锚点
训练 (rotation_rope_llm.py:822-830)：


if use_relative:
    coord_gt = coord_gt_src                           # 原坐标
    _cam_cond = cam_feat.detach()
else:
    coord_gt = _apply_rotation_to_xyz(R.detach(), coord_gt_src)  # 旋转后
    _cam_cond = None
pred_k = self.coord_head(coord_h_k, llm_h, llm_w, cam_feat=_cam_cond)
测试 (evaluation.py:1588-1593, 1617-1622)：


if is_relative:
    xyz_for_mae = [x.detach().cpu() for x in image_xyz]  # 原坐标做 MAE
    cam_feat_for_head = cam_feat
else:
    xyz_for_mae = [r.detach().cpu() for r in rotated]    # 旋转后坐标做 MAE
# ...
preds = _get_coord_predictions(..., cam_feat=cam_feat_for_head)
result["coord_mae"] = _compute_coord_mae(preds, xyz_for_mae)
4. 共同点（两者完全一致）
LM 路径：xyz_for_rope = rotated（RoPE 用旋转后坐标），lm_loss 回传到 LoRA + rotation_enc 完全相同
梯度：R.detach() / cam_feat.detach() 阻断 coord_loss 回流到 rotation_enc，仅 lm_loss 端到端学 R
step 0：R = I（rot_head 恒等初始化），两者初始等价于 baseline 无旋转



是的，cam_feat 被投影后作为序列第 0 个 token prepend 在最前面。

具体流程（coordinate_llm.py:108-120）

x = self.input_proj(hidden)                          # (h*w, d_model) —— vision tokens
x = x + self._sinusoidal_2d_pe(h, w, ...)            # 加 2D 正弦位置编码

# ① cam_feat 投影并 prepend 到序列最前
if cam_feat is not None and self.cam_proj is not None:
    cam_tok = self.cam_proj(cam_feat).unsqueeze(0)   # (1, d_model)
    x = torch.cat([cam_tok, x], dim=0)               # (1 + h*w, d_model)
    #              ↑ 位置 0

x = self.transformer(x.unsqueeze(0)).squeeze(0)      # Transformer 全序列自注意力

# ② transformer 后把位置 0 的 cam token 剥掉
if cam_feat is not None and self.cam_proj is not None:
    x = x[1:]                                        # (h*w, d_model)

x = self.output_proj(x)                              # 只用 h*w 个 vision token 出坐标
关键点
步骤	行为
prepend 位置	序列第 0 位（cam 在 [0]，vision tokens 在 [1 : 1+h*w]）
位置编码	不加 2D PE（只给 vision tokens 加 PE），cam 天然独立于网格
作用机制	通过 TransformerEncoder 的 self-attention 让每个 vision token 都能读到 cam 信息
输出截断	forward 尾部 x = x[1:] 剥掉 cam token，保证 PixelShuffle 的 (h, w) 形状不变
梯度	训练时 cam_feat.detach()（rotation_rope_llm.py:824），coord_loss 不回传到 rotation_enc
一句话：cam_feat → cam_proj → 作为序列 index 0 的 conditioning token 插在最前面，通过 transformer 全局注意力广播到所有 vision patch，最后再被剥掉不影响空间输出。