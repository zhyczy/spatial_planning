# Input Template — Train / Eval Alignment

**Date:** 2026-04-30
**Scope:** Two input-template paths used in this codebase, with the matching
output extractors:
- **SFT path** — used by all 7 SFT'd methods (`vanilla`, `atten`, `coordinate`,
  `polar`, `decouple`, `rotation`, `rotation_rl`). Train + eval byte-identical.
- **Zero-shot baseline path** — used only by `--method baseline` (raw Qwen3.5,
  no LoRA). Different on purpose: the model has never seen `<answer>...</answer>`
  and must be prompted into the format.

Cross-references:
- Output extractor details + round-trip validation: [data_mllm.md](data_mllm.md)
- VST Q&A category breakdown: [../dataset/vst.md](../dataset/vst.md)
- Format helper (assistant-side wrapping): [../../src/dataset/answer_format.py](../../src/dataset/answer_format.py)

---

# 套 1: SFT 路径（7 个方法 — vanilla / atten / coordinate / polar / decouple / rotation / rotation_rl）

**Train + Eval 共享 byte-identical 前缀，只在 generate 起点位置分叉。**

## 1.1 Training（VST 为例 — [train_dataset_qwen35.py](../../src/dataset/train_dataset_qwen35.py)）

VST 训练数据按 8 个子集类型自动分流，所有 assistant content 都包成
`<answer>{raw}</answer>`。raw 内容随子集而变（letter 句、numeric+unit、
free-form caption…）。代码 `_vst_build_chat_with_labels`。

### 1.1.A VST 单轮（如 `mi_camera_motion`）

```python
qa_pairs = [(question, raw_gpt_answer)]   # 单 turn

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img_0},
        {"type": "image", "image": img_1},
        {"type": "text",  "text": question},
    ]},
    {"role": "assistant",
     "content": f"<answer>{raw_gpt_answer}</answer>"},
]
text_full = processor.apply_chat_template(
    messages,
    tokenize=False, add_generation_prompt=False, enable_thinking=False,
)
```

**实际渲染**（mi_camera_motion 真实样本，2 视角）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>The images are acquired continuously from a first-person perspective. How does the camera's location change through space? Options:
A. rightward, B. backward and rightward, C. left and forward, D. rightward and upward
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant
<think>

</think>

<answer>C. left and forward</answer><|im_end|>
```

**Label mask**：只 unmask 整段 `<answer>C. left and forward</answer><|im_end|>\n` 对应的 token 子序列；前面 system + user + 视觉 + question + `<think></think>` 全部 -100。Suffix 长度按 `<answer>` 内文字长度变（不是固定 8），dataset 通过 token 子序列搜索定位。

### 1.1.B VST 多轮（仅 `si_measurement`，每条 3 个 Q&A）

```python
qa_pairs = [
    (q1, "122 cm"),   # turn 1: about table
    (q2, "97 cm"),    # turn 2: about chair
    (q3, "213 cm"),   # turn 3: about bed
]

messages = [
    # turn 1: 图挂第一个 user turn
    {"role": "user", "content": [
        {"type": "image", "image": img_0},
        {"type": "text",  "text": q1},
    ]},
    {"role": "assistant", "content": "<answer>122 cm</answer>"},
    # turn 2: 仅文本
    {"role": "user", "content": [{"type": "text", "text": q2}]},
    {"role": "assistant", "content": "<answer>97 cm</answer>"},
    # turn 3: 仅文本
    {"role": "user", "content": [{"type": "text", "text": q3}]},
    {"role": "assistant", "content": "<answer>213 cm</answer>"},
]
```

**实际渲染**：

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is the dimension of greatest length of the table (blue point-1) in centimeters?
Please try to answer the question with short words or phrases if possible.<|im_end|>
<|im_start|>assistant
<think>

</think>

<answer>122 cm</answer><|im_end|>
<|im_start|>user
What is the value of the longest dimension of the chair (green point-2) in centimeters?<|im_end|>
<|im_start|>assistant
<think>

</think>

<answer>97 cm</answer><|im_end|>
<|im_start|>user
What is the length of the dimension that is maximum of the bed (red point-3) in centimeters?<|im_end|>
<|im_start|>assistant
<think>

</think>

<answer>213 cm</answer><|im_end|>
```

**Label mask**：3 段 `<answer>...</answer><|im_end|>\n` 各自被独立定位 + unmask；中间的 user turn 和 `<think></think>` 全部 -100。共 3 个监督块、3 倍 token 数的 loss。

### 1.1.C MindCube 训练（letter + 选项文本，同一管线）

MindCube `gt_answer` 只是单字母（`"C"`/`"B"`），但训练侧不直接包成 `<answer>C</answer>` —— 而是从 question 里抽出该字母对应的选项文本，包成 `<answer>C. Light purple sofa</answer>`。这给模型一个**含语义的 SUPERVISE 尾巴**，在缺乏推理监督的设定下能多给几个点准确率。

```python
from .answer_format import format_answer_with_text
formatted_answer = format_answer_with_text(_answer, _question)
# "C" + "... A. TV B. Wooden dining table C. Light purple sofa D. Brown curtains and windows"
# → "<answer>C. Light purple sofa</answer>"
```

抽取代码 [answer_format.py:format_answer_with_text](../../src/dataset/answer_format.py)：

1. **Precheck**：question 必须同时含 `A.\s` 和 `B.\s` inline 标记，否则直接 fallback 到 `<answer>{letter}</answer>`。这避免了 spinbench 这种"描述里偶有 `View C.`"的 question 被误识别成 MCQ。
2. **正则抽取**：`\b(letter)\.\s+(.+?)(?=\s+[A-Z]\.\s+|\s*$)` 非贪婪匹配，对 first/middle/last 选项以及 "2.5 cm" 这种含小数点的选项都正确。
3. **长度上限**：抽到的 option text 必须 ≤ 200 字符（兜底，runaway match 也不会污染 suffix）。

10/10 单元测试 + 真实数据集（MindCube/SAT 抽取，spinbench 全 fallback）round-trip 全部通过。

**Suffix 长度变化**：原本固定 8 token，现在按选项长度浮动（`<` `answer` `>C` `.` ` Light` ` purple` ` sofa` `</` `answer` `>` `<|im_end|>` `\n`）。但 **letter 仍在 masked subset 第 2 个位置**（`>C`），所以 `LETTER_OFFSET=2` 跟 train_atten / train_correspondence 的 letter-pos argmax 公式都不需要改。

**Eval 端复用同一 wrapping**（`Eval_Dataset_Coord` 调用同一个 `format_answer_with_text`），训练 / 训练时 eval / deploy generate 三路一致。

**Output extractor 兼容**：`extract_answer_letter` 用 leading-letter 正则抓 `<answer>` 内首字母，`<answer>C. Light purple sofa</answer>` → `C` 完整 round-trip（已验证）。

## 1.2 Eval — `prepare_batch_spa`（[evaluation.py](../../evaluation.py)）

```python
content = [
    *[{"type": "image", "image": p} for p in image_paths],
    {"type": "text", "text": question},
]
prompt_text = processor.apply_chat_template(
    [{"role": "user", "content": content}],         # 仅 user turn
    tokenize=False,
    add_generation_prompt=True,                      # ← 跟 train 唯一的不同
    enable_thinking=False,
)
```

**渲染**（任何 SFT 数据集都是这个形状）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>...{question}<|im_end|>
<|im_start|>assistant
<think>

</think>

                    ← model.generate() 从这里开始
```

## 1.3 对齐验证

```
Train prefix:  ...assistant\n<think>\n\n</think>\n\n
Eval prefix:   ...assistant\n<think>\n\n</think>\n\n
                                                   ↑ byte-identical
Train tail:    <answer>{...}</answer><|im_end|>\n              ← teacher-forced
Eval tail:     {model.generate}                                  ← 自由生成
```

## 1.4 Output 抽取

```python
output = processor.decode(generated_ids, skip_special_tokens=True)
fmt = item.get("format_type", "select")
if fmt == "fill":
    prediction = extract_answer_number(output)    # "<answer>2.6</answer>" → "2.6"
else:
    prediction = extract_answer_letter(output)    # "<answer>B</answer>" → "B"
```

VST-bleed 输出（如 `<answer>C. left and forward</answer>` 出现在 MCQ 评测）也能被 `extract_answer_letter` 正确取出 `C`（leading letter regex）。

## 1.5 Training-time periodic eval（train_atten.py 内置）

跟独立的 `evaluation.py` 不同：训练循环每 `--eval_steps` 步跑一次"validation"，**仍然走 train 数据 schema**（teacher-forced lm_loss + letter-pos argmax），不是真 generate。

### 数据源
仅 2 个 benchmark：
```python
for _ds_name, _ds_jsonl, _ds_results, _q_key, _a_key in [
    ("mindcube",  ".../MindCube_tinybench.jsonl", ".../MindCube/3d_results", "question", "gt_answer"),
    ("spinbench", ".../spinbench_data/test.jsonl",  ".../spinbench_data/3d_results", "problem", "answer"),
]:
    ds = Eval_Dataset_Coord(_ds_jsonl, _ds_results, processor, log,
                            question_key=_q_key, answer_key=_a_key, ...)
```

`Eval_Dataset_Coord` 跟 `MindCube_Train_Dataset_Coord` **共享 chat template + label mask 实现**（都包 `<answer>{letter}</answer>`，前面 -100，后 8 token unmask），保证训练时 eval 跟训练 forward 完全同 path。

### Eval 循环（[train_atten.py:548-639](../../train_atten.py#L548-L639)）

```python
LETTER_OFFSET = compute_letter_offset(processor.tokenizer)   # = 2

for tb in loader:
    out = _spa(input_ids=t_ids, attention_mask=t_mask, pixel_values=t_pv,
               image_grid_thw=t_thw, image_xyz=t_xyz,
               mm_token_type_ids=t_mm, return_dict=True)
    logits = out.logits
    sl = logits[..., :-1, :]    # shift_logits
    sb = t_lbl[..., 1:]          # shift_labels

    # ── lm_loss (跟 training-step 等价) ──────────────────────
    lm_loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sb.view(-1),
                              ignore_index=-100)

    # ── letter-position argmax（teacher-forced 等价 generate） ─
    first_ans  = (t_lbl[0] != -100).nonzero(as_tuple=False)
    ans_start  = first_ans[0, 0].item()
    letter_pos = ans_start + LETTER_OFFSET     # `>X` 这个 token
    pred       = sl[0, letter_pos - 1].argmax().item()
    target     = sb[0, letter_pos - 1].item()
    if pred == target: acc_sum += 1.0
```

→ logged: `eval/{ds}_lm_loss`, `eval/{ds}_acc`

**关键性质**：在贪心解码无 leak 前提下，TF letter-pos argmax ≡ deploy generate 的首个 letter token —— 训练时 eval acc 跟 evaluation.py 的 `overall_accuracy` 应当统计上一致。详细推导见 [data_mllm.md §3](data_mllm.md#3-why-tf-acc-training-time-≡-gen-acc-deploy)。

## 1.6 Deploy 时各数据集 input 构造（[`load_testing_dataset`](../../src/dataset/eval_dataset_qwen35.py)）

`evaluation.py` 拿 `--dataset` 参数后调用 `load_testing_dataset(data_dir, limit, dataset)`，从各 benchmark 原生格式归一成统一 dict：

```python
{
    "index": ...,           # 跟 3d_results/<index>/ 对齐
    "image": [paths],       # 绝对路径列表
    "question": "...",      # 含 choices 已拼成 "Q\nA. opt1\nB. opt2\n..." (MCQ)
    "answer": "...",        # 字母 / 数字 / 文本
    "category": "...",      # benchmark 内的子类标签
    "format_type": "...",   # select / fill / robospatial
}
```

**各数据集的 question 拼接策略**：

| Dataset | Source 字段 | Question 构造 | Answer | format_type |
|---|---|---|---|---|
| mindcube | `question`, `gt_answer` | 原文 | 单字母 | select |
| mmsibench | `question`, `answer` | 原文 | 单字母 | select |
| sat_real | `question`, `answer_choices`, `correct_answer` | `Q\nA. opt1\nB. opt2\n...` 拼接 | letter（answer_choices.index 转字母） | select |
| sparbench_multi_view / single_view | `question`, `answer` | 原文 | 单字母 | select |
| sparbench_mv | 同上 | 原文 | letter 或数字 | **来自源 JSON**: select / fill 混合 |
| viewspatial | `question`, `choices`, `answer` | `Q\n{choices}` | letter（answer 首字母） | select |
| omnispatial_pt | `question`, `options`, `answer` | `Q\nA. opt1\nB. opt2\n...` 拼接 | letter（options.index 转字母） | select |
| embspatial | `question`, `answer_options`, `answer` | `Q\nA. opt1\nB. opt2\n...` 拼接 | letter | select |
| spinbench | `problem`, `answer` | 原文 | 单字母 | select |
| robospatial | parquet `question`, `answer` | 原文 | free-form | robospatial |

**4 类拼接模式**：
1. **裸 question**（mindcube / mmsibench / sparbench_*  / spinbench / robospatial）：原始 question 直接用，benchmark 自带选项已嵌在文本里。
2. **A/B/C/D 自动拼接**（sat_real / omnispatial_pt / embspatial）：source 里 choices 是 list，loader 用字母编号拼成 markdown 形式。
3. **`question + "\n" + choices`**（viewspatial）：source 已自带格式好的 choices 字符串，直接拼接。
4. **base64 → 临时 .jpg**（sparbench 三个）：source 把图片 base64 编码进 JSON，loader 解码到 `tempfile.mkdtemp(...)/{id}_{i}.jpg`，再走标准 image path。

**Image cache 行为**：
- `embspatial`：源 JSON 含 base64 image，loader 第一次加载时解码到 `data_dir/images/{qid}.jpg`，之后命中缓存。
- `robospatial`：parquet 行存 image bytes + mask bytes，loader 解到 `data_dir/images/` 和 `data_dir/masks/`。
- 其他 7 个：图片就在文件系统里，直接绝对路径解析。

---

# 套 2: Zero-shot Baseline 路径（仅 `--method baseline`，vanilla Qwen3.5 无 LoRA）

**没有 train 配套**，只在 eval 阶段评原始未 SFT 模型。

**Direct-answer policy（无 reasoning）**：system 禁推理 + user 重申严格格式 + `enable_thinking=False` 预填空 think。三道闸门一起把模型逼到 direct-answer：首 token 即 `<answer>` 开头，generation 通常在 `</answer><|im_end|>` 处即停。

## 2.1 Eval — `prepare_batch_baseline`

```python
fmt = item.get("format_type")

# system prompt
if fmt == "robospatial":
    sys_prompt = ROBOSPATIAL_SYSTEM_PROMPT
elif fmt == "fill":
    sys_prompt = EVAL_SYSTEM_PROMPT_FILL
else:                              # select 或缺省
    sys_prompt = EVAL_SYSTEM_PROMPT

# user instruction
if fmt == "robospatial":
    text = question                                       # 不加 instruction
elif fmt == "fill":
    text = f"{question}\n{ANSWER_INSTRUCTION_FILL}"
else:
    text = f"{question}\n{ANSWER_INSTRUCTION}"

prompt = processor.apply_chat_template(
    [{"role": "system", "content": sys_prompt},
     {"role": "user",   "content": [*images, {"type": "text", "text": text}]}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

### 2.1.A `format_type == "select"`（9 个 MCQ 数据集 — letter 答案）

```
<|im_start|>system
You are a spatial reasoning expert. You must answer the question directly. Output your final answer strictly in the format <answer>X</answer> where X is the option letter.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>...{question}
Output your answer strictly as <answer>X</answer> where X is the option letter.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

→ 期望 output：`<answer>{letter}</answer>` 后立刻 `<|im_end|>` 停止。
→ extractor: `extract_answer_letter(output)` → `A`/`B`/`C`/`D`

### 2.1.B `format_type == "fill"`（仅 sparbench_mv 中 1354 条数值题）

```
<|im_start|>system
You are a spatial reasoning expert. You must answer the question directly. Output your final answer strictly in the format <answer>X</answer> where X is the numeric value.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>...{question}
Output your answer strictly as <answer>X</answer> where X is the numeric value.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

→ 期望 output：`<answer>{number}</answer>` 后立刻 `<|im_end|>` 停止。
→ extractor: `extract_answer_number(output)` → `2.6`

### 2.1.C `format_type == "robospatial"`（RoboSpatial — 当前 10 数据集 eval 集没启用，分支保留）

```
<|im_start|>system
You are a spatial reasoning expert helping with robot navigation tasks.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>...{question}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

→ 期望 output：free-form 文本。
→ extractor: `extract_answer_content(output)`

---

# 对照总览

| 维度 | 套 1 SFT 路径 | 套 2 Zero-shot Baseline |
|---|---|---|
| 适用 `--method` | vanilla / atten / coordinate / polar / decouple / rotation / rotation_rl | 仅 `baseline` |
| 训练数据来源 | MindCube + VST（8 子集，单轮 + 多轮） | 无（仅 eval） |
| 是否参与训练 | 训练 + eval 用同一 template | 仅 eval（无对应训练数据） |
| System prompt | `"You are a helpful assistant."`（Qwen 默认） | `EVAL_SYSTEM_PROMPT*` / `ROBOSPATIAL_SYSTEM_PROMPT`（含 "answer directly" 禁推理指令） |
| User instruction 后缀 | 无（仅 question） | `ANSWER_INSTRUCTION*`（select/fill）或无（robospatial） |
| `enable_thinking` | False | False |
| Assistant content 形态 | `<answer>{letter}</answer>` (MCQ) / `<answer>{free-form}</answer>` (VST) | N/A（无 train） |
| 期望 eval output | 模型从 SFT 数据学到的 `<answer>...</answer>` | `<answer>{letter\|number}</answer>` 后立停（无 reasoning） |
| Train prefix == Eval prefix | ✓ byte-identical | N/A（无 train 配套） |
| 模型从哪学 `<answer>` 格式 | SFT 数据里学到 | 靠 prompt instruction 显式逼迫 |
| Output extractor | `extract_answer_letter` / `_number` 按 `format_type` | 同上 |

**核心区别**：套 1 模型从 SFT 数据学到了 `<answer>...</answer>` 格式，eval 不用 prompt；套 2 模型完全没见过这格式，**只能靠 prompt + `<think></think>` 预填空块两道闸门一起把它逼到 direct-answer**。两套故意不一样，分别服务不同评测场景（SFT 模型对照 vs vanilla 基线对照）。
