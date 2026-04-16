train_alternate.py 数据流
1) 启动阶段

CLI args (parse_args)
   │
   ├── DDP init (LOCAL_RANK / nccl)  → local_rank, world_size, device
   ├── AutoProcessor + tokenizer      → image_token_id, spatial_merge_size
   └── build_model()
          ├── SpaForConditionalGeneration (bfloat16, sdpa, 4D M-RoPE section [2,X,X,X])
          ├── freeze ViT (visual.*)
          ├── get_peft_model(LoRA r=16, α=32, q/k/v/o/gate/up/down)
          ├── gradient_checkpointing_enable
          ├── CameraTokenRotationEncoder(d_model = rot_nhead × head_dim)
          └── DepthPredictionTransformer(cam_dim = d_model if relative else 0)
   → RotationRoPEModel → .to(device) → DDP(find_unused_parameters=True)
2) 数据集 / Loader

[mindcube|sat]_Train_Dataset_Rotation(json_path, results_dir, processor, …)
   → DistributedSampler → DataLoader(bs=1, collate_fn → batch[0])

batch dict keys: input_ids, attention_mask, pixel_values, image_grid_thw,
                 image_xyz, image_xyz_hires, labels
同时构造 Eval_Dataset_Coord ×2 (MindCube_tinybench / spinbench)。

3) 三组参数优化器

named_parameters  requires_grad=True
    ├── rotation_enc_params  → group "rotation_enc"  lr=args.rotation_enc_lr
    ├── coord_head_params    → group "coord_head"    lr=args.lr
    └── lora_params          → group "lora"          lr=args.lr
          (lora = 既非 rotation_enc 也非 coord_head 的可训参数)

AdamW (weight_decay=0.01)

per-phase cosine budget:
   phase_a_end              = steps_per_epoch // 2
   phase_a_total_optim_steps = epochs × phase_a_end        // grad_accum
   phase_b_total_optim_steps = epochs × (spe - phase_a_end) // grad_accum
4) 训练主循环（每个 step）

step < phase_a_end  → Phase A                step ≥ phase_a_end → Phase B
   requires_grad:                               requires_grad:
     rotation_enc = False                         rotation_enc = True
     lora         = True                          lora         = False
     coord_head   = True                          coord_head   = True
   use_rot_enc         = (epoch > 0)             use_rot_enc         = True
   detach_coord_hidden = False                   detach_coord_hidden = True
前向（model(...)，见 rotation_rope_llm.py:709）：


input_ids, attention_mask
pixel_values, image_grid_thw ────┐
image_xyz, image_xyz_hires       │
labels                           │
                                 ▼
(1) inputs_embeds = text_embed  + ViT scatter at <|image_pad|>
(2) if use_rot_enc:
       token_txyz_int = _build_token_txyz_int(...)
       R, cam_feat   = rotation_enc(inputs_embeds.detach(), token_txyz_int)
    else: R=None, cam_feat=None
(3) rotated_xyz = R @ image_xyz   (gradient-preserving, float)
(4) position_ids_float (5D, no discretization)  ← rotated_xyz
(5) hidden2 = manual text-model pass with differentiable M-RoPE
    logits2 = lm_head(hidden2)
        → lm_loss  = CE(shift_logits, shift_labels)
(6) coord_head:
       coord_h_k = hidden2[vis_pos_k]
       if detach_coord_hidden: coord_h_k = coord_h_k.detach()   ← Phase B
       if relative: cam_cond = cam_feat.detach(), gt = xyz_hires
       else:        cam_cond = None,             gt = R.detach() @ xyz_hires
       pred_k = coord_head(coord_h_k, llm_h, llm_w, cam_feat=cam_cond)
        → coord_loss = L1(pred, gt).mean()
(7) loss = answer_weight·lm_loss + coord_weight·coord_loss
梯度流：

Phase A: lm_loss → LoRA（rotation_enc 冻结 + R=I 或冻结输出）；coord_loss → LoRA + coord_head。
Phase B: lm_loss → rotation_enc（LoRA 冻结）；coord_loss → coord_head only（coord_h_k.detach() 切断回到 R 的通路）。
5) 反向 / 优化

(loss / grad_accum).backward()
running_loss += loss.item();  accumulate loss_dict

if (step+1) % grad_accum == 0:
    clip_grad_norm_(rotation_enc_params, rotation_enc_clip=0.3)
    clip_grad_norm_(lora + coord_head,   lora_clip=1.0)
    optimizer.step()

    # 每阶段独立 cosine
    if phase == "A":
        phase_a_optim_step += 1
        lr = lr_phase_a_base · 0.5·(1 + cos(π · t / phase_a_total))
        apply to groups {"lora","coord_head"}
    else:
        phase_b_optim_step += 1
        lr = lr_phase_b_base · 0.5·(1 + cos(π · t / phase_b_total))
        apply to groups {"rotation_enc","coord_head"}

    optimizer.zero_grad();  global_step += 1
    all_reduce(loss 向量) / world_size   →  rank0 log + wandb
6) 周期性 eval / checkpoint（rank 0）

每 save_steps:  _save_checkpoint()
    spa_model.save_pretrained + tokenizer
    torch.save(rotation_enc.state_dict, coord_head.state_dict)

每 eval_steps:  model.eval() + 关闭 gradient_checkpointing
    遍历 {mindcube, spinbench} Eval_Dataset_Coord
    model(..., use_rotation_enc=use_rot_enc, detach_coord_hidden=detach_coord_hidden)
    all_reduce(count, loss_sums) → rank0 log + wandb
    → model.train()
7) 收尾

最终 _save_checkpoint(..., suffix="final")
wandb.finish()
dist.destroy_process_group()
一图概览

            ┌───────────────── Phase A (step < N/2) ─────────────────┐
input ──►  ViT ──► embeds ──► (rot_enc frozen; R=I @epoch0, R_cur @ep≥1)
                            ──► rotated_xyz ──► float M-RoPE ──► LM (LoRA TRAIN)
                                                              ├► lm_loss ─► ∂LoRA
                                                              └► coord_head (TRAIN, 无 detach)
                                                                  └► coord_loss ─► ∂LoRA, ∂coord_head
            └────────────────────────────────────────────────────────┘
            ┌───────────────── Phase B (step ≥ N/2) ─────────────────┐
input ──►  ViT ──► embeds ──► rot_enc (TRAIN) ──► R, cam_feat
                            ──► rotated_xyz (grad→R) ──► float M-RoPE ──► LM (LoRA FROZEN)
                                                              ├► lm_loss ─► ∂rot_enc  (唯一到 R 的通路)
                                                              └► coord_head (TRAIN) with hidden.detach()
                                                                  └► coord_loss ─► ∂coord_head 仅
            └────────────────────────────────────────────────────────┘