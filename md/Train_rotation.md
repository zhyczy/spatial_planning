两组按 "rotation_enc" in name 切分，基于 PyTorch named_parameters() 的命名空间：

维度	Group A: rotation_enc_params	Group B: other_params (LoRA + coord_head)
命名前缀	rotation_enc.*	spa_model.base_model.*.lora_A/B.*, coord_head.*
learning rate	args.rotation_enc_lr (默认 2e-4)	args.lr (默认 2e-4)
grad clip	args.rotation_enc_clip (默认 0.3)	args.lora_clip (默认 1.0)
初始化	从零训练（rot_head 零权重 + 恒等 bias）	LoRA 从 0 起；coord_head 默认 init
Group A — rotation_enc.* 具体参数（来自 rotation_llm.py:264 CameraTokenRotationEncoder）

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