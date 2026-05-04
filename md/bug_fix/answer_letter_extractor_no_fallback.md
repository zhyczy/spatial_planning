# `extract_answer_letter` 抽取器：raw-letter 输出被全部判错

> Cross-refs: [`atten_couple_xyz_validation.md`](../findings/atten_couple_xyz_validation.md) §2.1 / §2.2 — 这个 bug 让 SpinBench 单图 accuracy 从真实的 82.2% 被记成 0.00%。

## 0. TL;DR

`extract_answer_letter()` **只认 `<answer>X</answer>` 包装**，没标签就直接返回 `""`。但有些 benchmark（典型 SpinBench 单图）的 prompt **明确要求**模型 "Only answer with a single capital letter"，模型听话直接吐 `A` / `B`，结果 100% 被抽取器判成空字符串 → 全错。

| 抽取方式 | SpinBench 整体 | SpinBench 单图(926) |
|---|---:|---:|
| 当前（严格 `<answer>` only） | **33.88%** | **0.00%** |
| 加 raw-letter fallback | 66.5% | 82.2% |
| **被低估的样本量** | **+893** | **+761** |

## 1. 源码位置 — 同一份逻辑被复制了两次

| 文件 | 函数 | 行 |
|---|---|---|
| [`src/dataset/answer_format.py`](../../src/dataset/answer_format.py) | `extract_answer_letter` | L140-153 |
| [`evaluation.py`](../../evaluation.py) | `extract_answer_letter` | L229-256 |

两份实现等价（同样的两步：`extract_answer_content` 抓 `<answer>` → 失败返回 `""`），任意一份独立用都触发 bug。

### 1.1 `src/dataset/answer_format.py` 的实现

```python
# answer_format.py:128
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def extract_answer_content(text: str) -> str:
    """Return the raw inner content of the *last* <answer>...</answer> tag."""
    if not text or not isinstance(text, str):
        return ""
    matches = _ANSWER_TAG_RE.findall(text)
    return matches[-1].strip() if matches else ""        # ← 没标签 → ""

def extract_answer_letter(text: str) -> str:
    content = extract_answer_content(text)
    if not content:
        return ""                                          # ← 早返回，从不看原文
    m = re.match(r"\s*([A-Za-z])(?:\s|[.)]|$)", content)
    if m:
        return m.group(1).upper()
    return ""
```

`extract_answer_letter` 拿到 `content == ""` 就放弃，**不会去看 raw text**。模型输出 `"A"`、`"A. mug"`、`"The answer is A"` 全部抽成 `""`。

### 1.2 谁调用它

```bash
$ grep -rn extract_answer_letter src/ evaluation.py xyz_validation.py 2>/dev/null
src/dataset/answer_format.py:140:def extract_answer_letter(text: str) -> str:
evaluation.py:229:def extract_answer_letter(text: str) -> str:
evaluation.py:1399:    prediction = extract_answer_letter(full_output)   # ← deploy eval
xyz_validation.py:78:    extract_answer_letter as _extract_answer_letter
```

deploy-time 评测全部受影响：`evaluation.py`、`xyz_validation.py`、以及任何下游 import。**训练时**的 periodic eval 走 [`train_atten.py:656`](../../train_atten.py#L656) 的 teacher-forced argmax（直接 logit 域比对 letter token），**不走 extractor，所以训练 loss/acc 曲线不受影响**。

## 2. 数据层 — 真实失败样本

`atten_couple_mindcube/step_624_final` ckpt × SpinBench 全集 2739 sample。

### 2.1 单图样本：模型答对但被判错（典型 6 例）

prompt 末尾都是 `Only answer with a single capital letter from (A, B).` —— 模型完全遵从指令，输出 `A\n`，没有 `<answer>` 标签包装。

| # | id | question 摘要 | GT | model output | extractor 输出 | scored |
|---|---|---|---|---|---|---|
| 1 | f40e513d878d | "Which object is further away ... A. mug, B. mustard bottle ..." | `A` | `'A\n'` | `''` | ✗ |
| 2 | 8c82a934a3ea | "Which object is closer ... A. banana, B. pitcher base ..." | `A` | `'A\n'` | `''` | ✗ |
| 3 | 930313164ba1 | "Which object is further away ... A. large clamp, B. wood block ..." | `A` | `'A\n'` | `''` | ✗ |
| 4 | b028a90fe8ed | "Which object is further away ... A. large clamp, B. scissors ..." | `A` | `'A\n'` | `''` | ✗ |
| 5 | 8c447a8b4c07 | "Which object is closer ... A. pudding box, B. potted meat can ..." | `A` | `'A\n'` | `''` | ✗ |
| 6 | 1caf9982c8ae | "Which object is further away ... A. mug, B. mustard bottle ..." | `A` | `'A\n'` | `''` | ✗ |

每一条模型实际都是答对的，但 `extract_answer_letter` 返回 `""`，于是 `prediction != answer` → 判错。

### 2.2 多图样本也会偶发踩坑（不是单图独有）

| # | id | n_imgs | question 摘要 | GT | model output | extractor 输出 | scored |
|---|---|---|---|---|---|---|---|
| 1 | 992dea7dab79 | 3 | "Which of these three images (A,B,C) shows a different car ..." | `B` | `'B\n'` | `''` | ✗ |
| 2 | b274d12cb4fe | 3 | "Which of these three images (A,B,C) shows a different car ..." | `C` | `'C\n'` | `''` | ✗ |

模型在某些多图任务里也会偶尔忘记 `<answer>` 包装直接吐字母 —— bug **不只单图独有**。

### 2.3 对照：抽取器正常工作时长这样

| id | n_imgs | GT | model output | extractor 输出 | scored |
|---|---|---|---|---|---|
| fef2e7f9f6ae | 4 | `A` | `'<answer>A</answer>\n'` | `'A'` | ✓ |
| 315845e159b1 | 4 | `A` | `'<answer>A</answer>\n'` | `'A'` | ✓ |

只要模型套了 `<answer>` 标签，抽取器就能正常工作。

### 2.4 总量影响

```
total single-image rows                 : 926
extracted == '' (current code)          : 926  (100.0%)   ← 全部判空
raw output's first letter == GT         : 761  (82.2%)   ← 但 82.2% 实际答对
→ 761 samples are answered correctly but scored 0 by current extractor
```

### 2.5 根本原因 — 训练分布 vs 测试 prompt 不一致

为什么单图模型不套 `<answer>` 但多图就套？

1. **训练数据 MindCube 全是多图**（≥2 frame，4-frame 占 74%），训练 label 一律是 `<answer>X</answer>` 包装
2. 模型从训练里学到的不是"按 prompt 末尾的指令做"，而是"看到多图任务 → `<answer>` 包装"这个**条件反射**
3. spinbench 单图任务（深度比较）**不在 MindCube 训练分布里**，模型没形成 `<answer>` 反射，于是**字面遵从** prompt 末尾的 "Only answer with a single capital letter" → 直接吐 `A\n`
4. 多图 prompt 里也写了同样的指令，但模型已经被训练成"多图就 `<answer>` 包装"，**反而忽略了 prompt 字面意思**

→ extractor 只识 `<answer>` 一种格式，本质上是把"训练分布的输出习惯"硬编码进了评分逻辑。**任何超出训练分布的 prompt 模板都会让评分崩**。

## 3. 为什么单图会触发：prompt 模板差异

SpinBench 单图 prompt（`test.jsonl`）：

```
<image>
Which object is further away from the viewer in the image?
 A. mug, B.mustard bottle
Only answer with a single capital letter from (A, B).
```

最后一行命令模型只输出字母。模型遵循指令，吐 `A` 或 `B`，**没有 `<answer>` 包装**。

SpinBench 多图 prompt（4-frame perspective taking）：

```
You are shown four images of a car in order: [1] Front view, [2] View A, [3] View B, [4] View C.
Select the image that best shows the car from the back side.
Front view: <image>
A: <image>
B: <image>
C: <image>
Only answer with a single capital letter (A, B, or C).
The back side is defined from the viewer's perspective when looking at the front of the car.
```

提示也说 "single capital letter"，但**模型输出仍然是 `<answer>A</answer>`**。MindCube 训练数据 100% 是多图 + 全部用 `<answer>...</answer>` 包装，模型从训练分布学到的是"多图任务用包装"。单图不在训练分布里，模型直接听 prompt 字面 → raw letter。

这种"训练分布 vs 测试 prompt" 的不一致 **必然**会跨数据集出现，不只是 SpinBench：
- **SAT** —— 部分单图测试可能也有"只输出字母"的指令模板
- **EmbSpatial-Bench** —— 全部单图，类似深度比较任务
- **OmniSpatial-PT** —— 单图 perspective-taking
- **未来**任何不带训练时见过的 `<answer>` 习惯的 prompt 都会踩坑

## 4. 影响量化（couple ckpt × SpinBench）

按 image-count 分桶（lenient 抽取下的真实数字）：

| n_imgs | n | 严格抽取 | lenient 抽取 | 被低估 |
|---:|---:|---:|---:|---:|
| 1 | 926 | 0.0% | 82.2% | **+82.2 pp** |
| 2 | 509 | 36.3% | 52.1% | +15.8 pp |
| 3 | 341 | 73.3% | 85.9% | +12.6 pp |
| 4 | 745 | 56.8% | 58.0% | +1.2 pp |
| 5 | 218 | 32.1% | 32.6% | +0.5 pp |
| **all** | **2739** | **33.88%** | **66.5%** | **+32.6 pp** |

multi-image 也有少量低估（4-img +1.2 pp，5-img +0.5 pp）—— 模型偶尔在多图任务里也会忘记 `<answer>` 包装。**bug 影响所有 dataset，不只单图**。

## 5. 修复方案

只改 `extract_answer_letter`，不动 `extract_answer_content` 和 `extract_answer_number`（它们各自语义独立）。

```python
def extract_answer_letter(text: str) -> str:
    """Extract the multiple-choice letter from a `<answer>...</answer>` tag,
    falling back to the first leading letter in the raw text when no tag
    is present (some datasets — e.g. SpinBench single-image — instruct the
    model to emit a bare letter, and the model correctly obeys)."""
    if not text or not isinstance(text, str):
        return ""
    content = extract_answer_content(text)
    if content:
        m = re.match(r"\s*([A-Za-z])(?:\s|[.)]|$)", content)
        if m:
            return m.group(1).upper()
        return ""
    # Fallback: no <answer> tag — accept "A", "A.", "A)", "A: ...", "A " at
    # the start of the (stripped) raw output. Strict word-boundary check
    # prevents false positives on prose like "Among A and B …".
    m = re.match(r"([A-Za-z])(?:\s|[.):]|$)", text.strip())
    if m:
        return m.group(1).upper()
    return ""
```

要点：
1. **fallback 只在没 `<answer>` 标签时触发** —— 已经走包装的 dataset（mindcube tinybench、多图 SpinBench）**输出不变**
2. **严格匹配字符串开头** —— 不会从 prompt-leak 里误抽 "A. mug" 之类的 option text；模型必须真的把字母放在自己输出的最开头才认
3. **接受多种分隔符**（空格 / `.` / `)` / `:` / 字符串结束）—— 覆盖 `"A"`、`"A. blue chair"`、`"A) Yes"`、`"A: see fig"` 等常见格式

需要同步两份实现：[`src/dataset/answer_format.py:140`](../../src/dataset/answer_format.py#L140) 和 [`evaluation.py:229`](../../evaluation.py#L229)。

## 6. 风险与回归点

### 6.1 无风险
- **训练曲线** —— 不走这个 extractor，wandb 指标无影响
- **已用 `<answer>` 标签的 dataset** —— content 非空时走原路径，行为不变
- **数字抽取** —— `extract_answer_number` 已经有 `<answer>` 缺失时的 fallback（line 163：`nums = _NUMBER_RE.findall(text or "")`），逻辑不变

### 6.2 需要验证
- **历史评测数字会变** —— `vis_results/` 下的所有 dataset 严格 vs lenient 数字会有差异。建议：
  - 修后重跑一次受影响 dataset（spinbench / SAT / EmbSpatial）
  - 或在 results json 同时存 strict + lenient 两个 prediction 字段，让消费端选

### 6.3 误检概率（false positive）
fallback 用 `re.match`（必须从字符串开头），且要求第一个字母后接分隔符。可能误检的输出：
- `"A is closer than B"` → 抽 "A"（没 `.` 但有 `\s`） —— 这是**正确**的抽取（模型是在选 A）
- `"Apple is on top..."` → fail（"A" 后面是 "p"，非分隔符）—— **不会误抽**
- `"a) yes"` → 抽 "A"（小写自动 upper）—— 正确

可能 false positive 的边缘 case：
- `"All four objects are visible."` → 抽 "A"（"A" 后面是 "l"），**fail，不会触发**
- `"The answer is C."` → 抽 "T"（"T" 后面是 " " 空格）→ **错抽 T！**

→ **fallback regex 应该限制在 A-F**（多选选项最多到 F/G）：

```python
m = re.match(r"([A-Fa-f])(?:\s|[.):]|$)", text.strip())   # 限制 A-F
```

不过 "T " 这种开头实际罕见（模型按 prompt 出 letter 时不会先说 "The..."），保留 A-Z 也基本安全。**建议保守起见用 A-F**。

## 7. follow-ups

1. **应用修复**到两处 extractor，单元测试加 4 个 case：
   - `<answer>A</answer>` → "A"（不变）
   - `<answer>A. blue chair</answer>` → "A"（不变）
   - `A` → **"A"（新增）**
   - `A. mug` → **"A"（新增）**
   - `The answer is A` → "" 或 "A"（看是否要更激进的 fallback；保守 = ""）
2. **重跑 SpinBench / SAT / EmbSpatial xyz_validation**，更新 [`atten_couple_xyz_validation.md`](../findings/atten_couple_xyz_validation.md) 的"strict vs lenient"对比改成"修复前 vs 修复后"
3. **训练时 deploy-style eval**（如果未来加，例如 mindcube tinybench 在训练中跑全生成评测）也应该用同一个 extractor

## 8. 复现 / 诊断脚本

```python
# probe_extractor_bug.py — 在任何 vis_results/<run>/<dataset>/ 上跑一次
import json, glob, re, sys
sys.path.insert(0, ".")
from src.dataset.answer_format import extract_answer_letter

rows = []
for p in sorted(glob.glob("vis_results/<your_run>/<dataset>/atten_normal_cuda*.json")):
    rows.extend(json.load(open(p)))

n_total = len(rows)
n_extract_empty = sum(1 for r in rows if extract_answer_letter(r["output"]) == "")
n_raw_first_letter_correct = sum(
    1 for r in rows
    if (m := re.match(r"^([A-F])\b", r["output"].strip())) and m.group(1) == r["answer"]
)
print(f"total: {n_total}")
print(f"extract returns '': {n_extract_empty}  ({n_extract_empty/n_total*100:.1f}%)")
print(f"raw first letter == GT: {n_raw_first_letter_correct}  "
      f"({n_raw_first_letter_correct/n_total*100:.1f}%)")
print(f"  → {n_raw_first_letter_correct} samples are answered correctly "
      f"but scored 0 by current extractor")
```

跑一遍立刻能看出 dataset 是否被这个 bug 拖累。
