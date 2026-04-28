# `train_atten` 训练 / 测试范式不一致 — 系统分析

> Cross-refs: [`spatial_attention_causal_leak.md`](spatial_attention_causal_leak.md) (前一轮 causal-mask leak postmortem),
> [`spatial_attention.md`](../model_design/spatial_attention.md) (atten 架构设计).

## 0.1 ⚠️ 真正的 root cause（2026-04-28 实测确认）

**先放结论，后面 §0–§8 是逐步逼近这个结论的过程。**

`load_spa_model` 的 wrap 顺序跟 `train_atten.py` 反了，导致**保存的 LoRA 路径跟加载时 PEFT 注入的路径不匹配**，**8 个 full-attn 层的 LoRA 权重在 deploy 时被静默丢弃**。

| | saved key（来自 ckpt safetensors） | load 时 PEFT 注入预期的 key |
|---|---|---|
| layer 3 self_attn | `...layers.3.self_attn.`**`attn.`**`q_proj.lora_A.weight` | `...layers.3.self_attn.q_proj.lora_A.`**`default.`**`weight` |

差两段：`.attn.` 有 vs 无 / `.default.` 无 vs 有 —— **全部 missing**。

**§5 leak 回归测试结果**（[`tests/test_atten_no_leak.py`](../../tests/test_atten_no_leak.py)）：

```
Position L_p - 1 = ...
GT first token: 'C' (id=34)
Gen prefill argmax: 'The' (id=760)
TF forward  argmax: 'The' (id=760)
max |tf - gen|: 0.000000e+00
cos sim:        1.000000
```

3/3 sample logits 逐 bit 相等。**没有 leak。**

那 96 % vs 0 % 怎么解释？因为 deploy 模型跟训练模型**根本不是同一个模型** —— 8 个 full-attn 层（layers 3, 7, 11, 15, 19, 23, 27, 31）的 LoRA 权重 64 个 tensor 全丢了，**deploy 时这 8 层是 stock Qwen3.5 权重**。

```
训练时 (LoRA 全在):  96% TF acc, model 在 letter 位置预测 letter
                              ↓ save_pretrained
checkpoint:  base_model.model.model.language_model.layers.3.self_attn.attn.q_proj.lora_A.weight
                                                              ^^^^^
                              ↓ PEFT.from_pretrained
deploy 模型 (8 层 LoRA 丢): full-attn 层回退到 base Qwen → 模型按 base reasoning 习惯输出 "The..."
                                                          → 我的 forward 测试 argmax 全部 'The'
                                                          → gen 1050/1050 输出 reasoning
                                                          → xyz=0 ablation Δ ≈ 0
                                                            （SpatialAttentionBias 装在这 8 层，
                                                             LoRA 没了，bias 也空转）
```

**修复**：调整 [`load_spa_model`](../../evaluation.py#L507-L522) 的顺序，wrap 必须在 PEFT load **之前**：

```python
# 修复后（已实装）
if atten:
    n_wrapped = patch_attention_layers_spatial(spa)
spa = PeftModel.from_pretrained(spa, str(ckpt_dir), is_trainable=False)
spa = spa.merge_and_unload()
```

**修复后实测验证**（[`tests/test_atten_no_leak.py`](../../tests/test_atten_no_leak.py) 同 3 个样本）：

```
修前:                           修后:
Sample 0: GT='C', Gen=TF='The' ❌  Sample 0: GT='C', Gen=TF='The' ❌（model 真不会答这题）
Sample 1: GT='A', Gen=TF='The' ❌  Sample 1: GT='A', Gen=TF='A'   ✓
Sample 2: GT='B', Gen=TF='The' ❌  Sample 2: GT='B', Gen=TF='B'   ✓
```

PEFT 的 `missing adapter keys` warning 也消失了。

**`.default.` 那段**：原来担心 saved key 没 `.default.`、load 注入有 `.default.` 也会 mismatch —— 实测跑下来 PEFT 内部能消化这个差异（可能因为 strict=False 加上 default 是 PEFT 唯一支持的 adapter name，PEFT 在 load 时做了路径归一化）。所以**只要 wrap 顺序对了，问题就解决了**。

### 影响范围

这个 bug 影响**所有用 `load_spa_model(atten=True)` 加载的 atten 模型**：

- [`evaluation.py`](../../evaluation.py) 主评估脚本
- [`validation.py`](../../validation.py) xyz=0 ablation 脚本
- 其他依赖 `load_spa_model` 的下游

**不影响**：
- `train_atten.py` 自己 —— 训练时 model in memory，LoRA 全在；
- `train_correspondence.py` / `train_coordinate.py` —— 它们的训练顺序里没有 patch_attention_layers，wrap 顺序问题不存在；
- `decouple` / `polar` / `rotation` / `vanilla` / `coord` 等其他 method —— 它们不调用 `patch_attention_layers_spatial`。

---

## 0. TL;DR — 关键事实

`train_atten.py` 训练时 eval 报 **~96 %** 的 MindCube tinybench accuracy。`evaluation.py` / `validation.py` 在同一份 ckpt 上 autoregressive 跑下来 **45.43 %**。

**实测的 smoking gun**（[ablation result](../../vis_results/atten_xyz0_step_1000/results_atten_normal.json) 1050 条 raw output）：

```
First non-whitespace character distribution: {'T': 1050}
```

**1050 / 1050 个样本**的第一个生成 token 都是 `T`（"The user wants to determine..."）。**没有一个**样本以 letter A/B/C/D 起头。

这跟训练 eval 的 96 % accuracy **数学上不可能同时成立** —— 在 greedy decoding 下：

> gen 的第一个生成 token = TF eval 第一个非 -100 位置的 `argmax(logits)`

这是 transformer 前向 + 因果掩码的**恒等式**，不依赖任何模型架构。两边不等只有一个解释：**TF eval 在泄漏**。

**关键反证（§6.0）**：训练数据是裸字母、监督只 3 token、regex 期待 `<answer>` tag —— 这些**潜在问题** train_coordinate 跟 train_atten 完全一样。但 train_coordinate 没爆这种 96 % vs 45 % 的剧裂。所以这些不是 atten 异常的根因。**唯一只属于 atten 的差异**是 [`spatial_attention_llm.py`](../../src/models/spatial_attention_llm.py) 在 attention path 上插入的三处手工 mask 操作（materialize / V↔V hole / wrapper），任一处 leak 都会污染**训练 loss 本身**（因为 LM loss 跟 TF eval 走同一条 forward），让模型 trivially 拟合，从未真正学 letter 预测。

下面把链条上 7 个环节摆出来，再用 train_coordinate 反证定位真正的根因。

---

## 1. 链条上 7 个环节

### 1.1 数据本身

[`MindCube_train.jsonl`](../../datasets/train/MindCube/MindCube_train.jsonl) 每行：

```json
{"question": "Based on these four images... A. ... B. ...", "gt_answer": "B"}
```

- `gt_answer` 是裸字母：`'A'/'B'/'C'/'D'`。
- 没有 `<answer>` tag，没有解释，没有 `</answer>`。

### 1.2 训练时 prompt 构造

[`train_dataset.py:231-235`](../../src/dataset/train_dataset.py#L231-L235):

```python
text_full = processor.apply_chat_template(
    [{"role": "user",      "content": [images, question]},
     {"role": "assistant", "content": _answer}],   # _answer = "B"
    tokenize=False, add_generation_prompt=False,
)
```

token 化结果：

```
<|im_start|>user\n[images][question]<|im_end|>\n<|im_start|>assistant\nB<|im_end|>\n
```

### 1.3 训练时监督信号

[`train_dataset.py:240-244`](../../src/dataset/train_dataset.py#L240-L244):

```python
suffix_ids = tokenizer(_answer + "<|im_end|>\n", ...)   # ≈ 3 个 token
labels = input_ids.clone()
labels[0, :-len(suffix_ids)] = -100
```

有效梯度只落在末尾 ~3 个 token：letter, `<|im_end|>`, `\n`。其余全 -100 ignore。

### 1.4 训练时 eval 度量（teacher-forcing first-token argmax）

[`train_atten.py:585-607`](../../train_atten.py#L585-L607):

```python
out = _spa(input_ids=t_ids, ..., labels=t_lbl)         # forward 一次
sl  = out.logits[..., :-1, :]; sb = t_lbl[..., 1:]
_m  = sb[0] != -100
_slm = sl[0, _m]; _sbm = sb[0, _m]
acc += _slm[0].argmax(-1) == _sbm[0]                    # 只看第一个非 -100 位置
```

设输入序列长度 L，labels 末尾 3 个非 -100，则：
- `_slm[0]` = `logits[L-4, :]`，模型在位置 `L-4`（assistant `\n`）预测下一 token；
- `_sbm[0]` = `labels[L-3]`，target = letter token id。

**目标**："给定到位置 L-4 的上下文，模型 argmax 是不是 letter？"

这个问题**理论上和 gen 第一个 token 等价**：在 gen prefill 里，prompt 长度 = L_prompt，`logits[L_prompt - 1, :]` 也是位置 `\n_after_assistant` 的预测。两者上下文一致，weights 一致，logits 一致，argmax 一致。

### 1.5 测试时 prompt 构造

[`evaluation.py:1218`](../../evaluation.py#L1218):

```python
messages = [{"role": "user", "content": [images, question]}]
prompt_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
```

token 化结果：

```
<|im_start|>user\n[images][question]<|im_end|>\n<|im_start|>assistant\n
```

**前缀部分跟训练完全一致**，到 `<|im_start|>assistant\n` 为止。

### 1.6 测试时生成

[`evaluation.py:1326-1331`](../../evaluation.py#L1326-L1331):

```python
generated_ids = model.generate(
    **inputs_dev,
    max_new_tokens=512,
    do_sample=False,                    # greedy
    pad_token_id=eos_token_id,
)
```

512 步贪心 autoregressive。

### 1.7 测试时解析

[`evaluation.py:1440`](../../evaluation.py#L1440), [`186-201`](../../evaluation.py#L186-L201):

```python
full_output = "<answer>" + output                                   # 强行前缀 tag
# extract_answer_letter regex 三层：
# 1) <answer>\s*([A-Za-z])\s*</answer>     ← 训练数据里从没出现过
# 2) (?:answer|option|choice)\s+(?:is\s+)?[:\s]*([A-Za-z])\b
# 3) 文本里最后一个 \b[A-D]\b               ← 兜底，实际生效那条
```

---

## 2. 实测数据：模型生成的第一个 token

跑 1050 条样本（[`results_atten_normal.json`](../../vis_results/atten_xyz0_step_1000/results_atten_normal.json)），扒出每条 raw output 的第一个非空白字符：

```
First non-whitespace character distribution: {'T': 1050}
```

样本前 80 字符（10 个 random 例子）：

```
[0] 'The user wants to determine the camera movement between two images.\n\n1.  **Analy'
[1] 'The user wants to determine the camera movement between two images.\n\n1.  **Analy'
[2] 'The user wants to identify the object behind the camera in image 1.\n1.  **Analyz'
[3] 'The user wants to determine the camera movement between two images.\n\n1.  **Analy'
[4] 'The user wants to know if moving forward and right from the position in image 2 '
[5] 'The user wants to determine the camera movement between two images.\n\n1.  **Analy'
[6] 'The user wants me to identify the object behind the camera position in image 3.\n'
[7] 'The user wants to know if moving forward and right will bring them closer to the'
[8] 'The user wants me to identify the object to the left of the office chair in imag'
[9] 'The user wants to determine the camera movement between two images.\n\n1.  **Analy'
```

**没有任何一条样本**以 letter 起头。Greedy 解码下，`argmax(logits[L_prompt-1])` 在 1050 条上**全部是 "The" 那个 token**。

---

## 3. 数学上的矛盾

把 §1.4 和 §2 摆一起：

| 量 | 值 |
|---|---|
| TF eval 的 `argmax(logits[L-4])` 命中 letter 的比例 | **96 %** |
| Gen 的 `argmax(logits[L_prompt-1])` 是 letter 的比例 | **0 %** |

这两个量在数学上是同一个量（同样的 weights，同样的上下文，同样的位置）。它们应该相等。

实测差了 96 个百分点。**只有一种可能：TF forward 的 logits 跟 gen forward 的 logits 不相等。**

这意味着 **TF forward 和 gen forward 走的不是同一条 forward path**。

---

## 4. 哪里 leak / mismatch

回顾 [`spatial_attention_causal_leak.md`](spatial_attention_causal_leak.md) 修过的那个 catastrophic bug：HF 的 `create_causal_mask` 在 SDPA 路径下会返回 `None`，旧代码 `else: new_mask = bias` 把 mask 替换成 zero tensor，SDPA 看到非 None mask + is_causal=False，**所有 token 互相能看到对方包括未来 token**。

那次修复在 [`spatial_attention_llm.py:379`](../../src/models/spatial_attention_llm.py#L379) 加了一段：

```python
if causal_mask is None and self._spatial_cache is not None and is_prefill:
    causal_mask = torch.triu(torch.full(...), diagonal=1)[None, None, :, :]
```

**只在 `_spatial_cache is not None` 时才补 mask**。

但是 — 训练时 _spatial_cache 是 set 的（atten 模式必走这条路），TF forward 也是 _spatial_cache set 的。两边都进了这个分支。所以**这个修复对 TF eval 和 gen 的影响应该是对称的**。

那 leak 在哪？两个候选：

### 候选 (a)：V↔V hole punch 在某些 batch shape 下漏掉文本位置

[`spatial_attention_llm.py:407-423`](../../src/models/spatial_attention_llm.py#L407-L423):

```python
is_vv_pair = (
    vision_mask.unsqueeze(1).unsqueeze(3)
    & vision_mask.unsqueeze(1).unsqueeze(2)
)
causal_mask = torch.where(
    is_vv_pair,
    torch.zeros((), dtype=causal_mask.dtype, device=causal_mask.device),
    causal_mask,
)
```

V↔V hole 把 vision 内的 lower-triangle 全打开。TF forward 的输入比 gen forward 的输入**多了 letter + im_end + \n 这 3 个 text token**，序列长度 L_TF = L_gen + 3。

如果 `vision_mask` 在 TF forward 里被错误地标到了文本区域（比如 padding 或 `<|im_end|>` 被误认成 vision），`torch.where` 会把那个文本位置的 -inf mask 抹平成 0，造成跨文本-文本 leak。**这可能是 TF eval 96 % 的源头**。

### 候选 (b)：M-RoPE position embedding 跨 prefill / gen 不一致

Qwen3.5-VL 的 4D M-RoPE 把 position 拆成 (text_seq, t, h, w) 四轴。`SpatialAttnVanillaTextModel.forward` 在 `position_ids is None` 时自己算：

```python
position_ids = cache_position.view(1, 1, -1).expand(4, B, -1)
```

但 TF forward 时 `cache_position` 长 L_TF，gen prefill 时 cache_position 长 L_gen。如果某条 RoPE 频率分配跟 cache_position 长度有耦合（不应该，但要验证），同一个绝对位置 L_prompt-1 在 TF 和 gen 里的 RoPE 投影会不同。

### 候选 (c)：dropout / bn / norm 在 train vs eval 行为差异

`_spa.eval()` 是否被正确调用过？如果 TF eval 在 train 模式下跑，dropout 噪声让 logits 看起来对，但 gen 在 eval 模式下 dropout off，logits 就不一样。

---

## 5. 决定性诊断（必跑）

写一个 30 行的最小回归脚本，固定一个 sample，**直接对比两条 forward 在同一个绝对位置的 logits**：

```python
# 1. 准备：完整序列（含 letter）和裸 prompt（不含 letter）
full_ids   = tokenizer(prompt + answer + "<|im_end|>\n")["input_ids"]   # length L
prompt_ids = tokenizer(prompt)["input_ids"]                             # length L_p
assert L_p == len(full_ids) - 3   # letter + im_end + \n

# 2. TF forward
out_tf = model(input_ids=full_ids, image_xyz=..., mm_token_type_ids=...)
tf_argmax = out_tf.logits[0, L_p - 1, :].argmax(-1)        # logits at "\n after assistant"

# 3. Gen prefill (same as gen step 1)
out_gen = model(input_ids=prompt_ids, image_xyz=..., mm_token_type_ids=...)
gen_argmax = out_gen.logits[0, L_p - 1, :].argmax(-1)      # same absolute position

# 4. 验证
print(f"TF argmax token: {tokenizer.decode([tf_argmax])!r}")
print(f"Gen argmax token: {tokenizer.decode([gen_argmax])!r}")
assert tf_argmax == gen_argmax, f"LEAK! TF={tf_argmax} GEN={gen_argmax}"
```

**预期**：两个 argmax 必须相等（同一个 weights，同一个 context，causal mask 必须屏蔽 L_p 之后所有 token）。

**如果不等**：定位到 leak。再分别 ablate：
- 拆掉 V↔V hole punch — 看是否消除差异 → 候选 (a)
- 把 prompt_ids 也 pad 到 L 长度（用 pad token 填充）— RoPE 长度对齐 → 候选 (b)
- `model.eval()` 显式调用 — 排除 candidate (c)

---

## 6. 四个系统级问题 — 重新定位

### 6.0 反证：train_coordinate 不炸说明 ①② 不是根因

下面四个因素**train_coordinate 和 train_atten 完全相同**：

| 因素 | 两边状态 |
|---|---|
| 训练 `gt_answer` 是裸字母 `'B'` | 同 |
| Label masking：末尾 3 个 token 真值，其余 -100 | 同（[`MindCube_Train_Dataset_Coord:391-395`](../../src/dataset/train_dataset.py#L391-L395) vs [`:240-244`](../../src/dataset/train_dataset.py#L240-L244)） |
| Chat template / prompt 构造 | 同 |
| `evaluation.py` 的 `extract_answer_letter` regex | 同 |

如果 ①（训练 label 不带 tag、解析期待 tag）和 ②（监督只 3 token）真是**根本原因**，那 train_coordinate **应该跟 train_atten 一样**：模型输出长篇 reasoning、deploy acc 比训练 metric 低 50pp。

但实测 train_coordinate 没这个症状。**这反证了 ①② 不是 atten 异常的原因** —— 它们是双方共有的**底层结构问题**，但不解释**为什么 atten 异常而 coord 不**。

### 6.1 真正只属于 train_atten 的东西：手工 mask 操作

[`spatial_attention_llm.py`](../../src/models/spatial_attention_llm.py) 在 attention path 上插入了三处 surgery（[`spa_emb.py`](../../src/models/spa_emb.py) 没有）：

| 入口 | 位置 | 操作 |
|---|---|---|
| **A** | [`spatial_attention_llm.py:382-388`](../../src/models/spatial_attention_llm.py#L382-L388) | `causal_mask is None and is_prefill` 时手工 materialize `torch.triu(...)[None, None, :, :]` |
| **B** | [`spatial_attention_llm.py:407-423`](../../src/models/spatial_attention_llm.py#L407-L423) | `torch.where(is_vv_pair, 0, causal_mask)` —— 把 V↔V cell 的 -inf 抹平 |
| **C** | [`spatial_attention_llm.py:158-179`](../../src/models/spatial_attention_llm.py#L158-L179) | `SpatialAttnWrapper.forward`：`new_mask = attention_mask + bias` 然后 `torch.where(is_masked, attention_mask, new_mask)` |

`spa_emb.SpaTextModel.forward` 完全不做这些手工活，stock HF causal_mask 直接传给 layer，SDPA 用 built-in 下三角 → 不可能 leak。

### 6.2 关键洞察：训练 loss 走的是同一条 forward 路径，所以 leak 不只污染 eval，**还污染训练**

[`train_atten.py`](../../train_atten.py#L598-L602) 的 LM loss 计算：

```python
out = _spa(input_ids=t_ids, ..., labels=t_lbl)
lm_loss = F.cross_entropy(out.logits[..., :-1, :].view(-1, V),
                          t_lbl[..., 1:].view(-1),
                          ignore_index=-100)
```

跟 §1.4 TF eval 用**同一个 forward** —— 一旦 §6.1 的 A/B/C 中任一个让 `logits[L-4]` 间接看到 letter token，**训练 loss 也假性收敛**：

> 模型根本不需要从 prompt 学怎么答题，loss 已经满了。

后果链：

```
训练 forward leak
  → train loss 漂亮、TF eval 报 96 %
  → 实际 logits[L-4] 在 prompt-only 上下文下完全没被训练
  → 部署时没有 leak、模型只剩 base Qwen 的 reasoning 习惯
  → 1050/1050 输出 "The user wants to ..."
```

train_coordinate 没有 §6.1 这套 surgery，训练 loss 是**诚实的**：要让模型在 letter 位置预测 letter，它必须真从 prompt + image_xyz 里学到信号。学到了，部署时也输出 letter。

### 6.3 因果图

```
        ┌─────────────────────────────────────┐
        │ 共同底层（双方都有，不解释差异）   │
        │  ① 训练 label = 裸字母              │
        │  ② 监督仅 3 token                   │
        │  ③ regex 期待 <answer> tag          │
        └─────────────────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │ atten 独有 surgery（差异源）         │
        │  A. materialize 三角阵              │
        │  B. V↔V hole `torch.where`          │
        │  C. wrapper 的 mask + bias 叠加     │
        └─────────────────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │ TF forward leak                     │
        │  ↘ 训练 loss 假性收敛              │
        │  ↘ TF eval 假性 96 %               │
        │  ↘ 部署 0 % letter-first           │
        └─────────────────────────────────────┘
```

底层 ①②③ 是**潜在风险**，但只有在 §6.1 surgery 引入 leak 之后才会**爆发**成 96 % vs 45 % 的剧裂。修了 leak，①②③ 退化为温和的"deploy acc 略低于训练 acc"，跟 train_coordinate 一个量级。

### 6.4 xyz 通路 — 同样被 leak 拖后腿

xyz=0 ablation Δ = −0.48 % 独立证明 `SpatialAttentionBias` 是 dead weight。这个结论本身**不依赖**有没有 leak（因为 normal/zero 同条件对比）。

但 leak 让 xyz 通路**永远不可能学到东西**：

- 训练 loss 因为 leak 已经满了，optimizer 没动力让 bias 模块去学 spatial → letter 的信号；
- bias 一直停在 zero-init 附近；
- 部署时（无 leak），bias 输出近 0，attention pattern 跟 stock Qwen3.5 完全一样 —— xyz 信号空转。

**所以 xyz 通路看起来"无效"也只是 leak 的副产物**。修了 leak，bias 才有机会真去学。如果修完之后 ablation Δ 还是接近 0，**那时**才是架构问题（监督链太长 / W₂ zero-init 没梯度信号）。

---

## 7. 修复优先级

```
1. §5 leak 回归测试（必须先跑）
       └─ 用 1 个 sample 直接对比 TF logits[L-4] vs gen logits[L_p-1]
       └─ 同时分别 ablate 入口 A / B / C 定位 leak 来源
                    ↓
2. 修 §6.1 的 surgery
       └─ 修复 leak 入口（最可能是 B：V↔V hole 配合错误的 vision_mask）
                    ↓
3. 重训 → 验证 TF eval 跟 deploy gen acc 对齐
       └─ TF acc 应该跟 train_coordinate 一个量级（没了 96 % 的虚高）
                    ↓
4. quick-gen eval（防御性，可选）
       └─ 在 train_atten.py eval 里加一个 model.generate(max_new_tokens=4)
       └─ 如果跟 TF acc 对齐，就证明 leak 已经修干净
                    ↓
5. xyz 通路再评估
       └─ 修了 leak 之后再跑 xyz=0 ablation
       └─ 如果 Δ 还接近 0，那才是真的 spatial 通路无效，
          不是 leak 副产物
                    ↓
6. （可选）解析对齐 / label format 调整
       └─ 等模型真能输出 letter 之后再考虑这些
```

**关键：先跑 §5 的诊断**。在没确认 leak 之前：

- 任何"模型不会答题"的修复都建立在错误的诊断假设上；
- xyz 通路看起来 dead 是 leak 的副产物，不是架构问题；
- ①②③ 的"修复"只会盖住症状，根本问题还在。

---

## 8. 结论

诊断收敛点：

- 训练 eval 报 96 % 但**实测每个样本第一个生成 token 都不是 letter**；
- 这两个事实在 greedy decoding 下数学上不能同时成立；
- **train_coordinate 用同样的训练数据 / label masking / regex 解析却没爆**，反证根因不在共同底层（裸字母、稀疏监督、regex tag 期待）；
- 唯一只属于 atten 的差异是 [`spatial_attention_llm.py`](../../src/models/spatial_attention_llm.py) 在 attention path 上插入的三处手工 mask 操作（materialize / V↔V hole / wrapper）；
- 这些 surgery 让 forward 在 `(L-4, L-3)` 之类的 cell 上 leak —— 而且 LM loss 走同一条 forward，意味着**训练本身在用 leak 拟合**，模型从来没真学 letter 预测；
- 部署时没 leak，模型只剩 base Qwen 的 reasoning 习惯 → 1050 / 1050 输出 "The user wants to..."；
- §5 的回归测试是定位入口 A/B/C 的唯一办法，必须先跑。

xyz=0 ablation 的 Δ = −0.48 %（spatial bias 是 dead weight）这个**对比结论本身独立 valid**，因为 normal/zero 同条件抵消了 leak 影响。但 spatial bias **为什么** dead 不能现在下定论 —— 它可能是被 leak 拖死的（loss 已经满了，optimizer 没动力训 bias），也可能是真架构问题。**修了 leak 才能区分**。
