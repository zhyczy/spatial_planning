# `atten` step_900 — why SpinBench accuracy doesn't move during training

> Cross-refs:
> [`xyz_zero_ablation.md`](../findings/xyz_zero_ablation.md) (prior coord/decouple ablation),
> [`train_eval_paradigm_mismatch.md`](../bug_fix/train_eval_paradigm_mismatch.md) (atten TF-eval leak postmortem),
> [`spin_bench_data_note.md`](../spin_bench_data_note.md) (SpinBench data structure).

## 0. TL;DR

`train_atten.py` 在 MindCube 上把 TF eval acc 从 63% 训到 96%，但 SpinBench eval acc **全程卡在 60-65%**（lm_loss 倒是从 0.67 跌到 0.14，5× 下降）。

xyz_validation.py 跑 atten step_900 在 SpinBench 上的 normal vs zero ablation：

```
Δaccuracy (zero - normal) = -0.04 pp     (strict parser)
Δaccuracy (zero - normal) = +0.00 pp     (relaxed parser)
normal == zero predictions = 99.16 %     (out of 2739)
```

**xyz 通路在 SpinBench 上是 dead module**。SpinBench acc 不动的原因不是 atten 还没学会，而是四类失败模式：

| 失败模式 | 机制 | 影响估计 |
|---|---|---|
| ① `<image>X</image>` tag 替代 | 模型把答案包进 `<image>` 而非 `<answer>` tag | mental-rotation `?` family 损失 ~14pp，全局 +0.76pp 可救 |
| ② 真实推理瓶颈 | 心算旋转 / viewer-perspective 低于 random | 全局 ~10-15pp 损失，xyz 救不回来 |
| ③ 单图任务无跨视图 xyz 信号 | atten V↔V hole 在单图里就是 ViT patch 编码冗余 | 影响 33.8% (926/2739) 样本 |
| ④ pts3d 量级严重 OOD | cars / abo 样本 \|pts3d\|_max ~13× 训练分布 | 影响 32% (875/2739) 样本 |

---

## 1. 训练阶段的信号 — lm_loss 下降但 acc 不动

[`train_records/atten_mindcube/train.log`](../../train_records/atten_mindcube/train.log)，TF first-token argmax 指标：

| step | mindcube_lm_loss | mindcube_acc | spinbench_lm_loss | spinbench_acc |
|---:|---:|---:|---:|---:|
| 50  | 0.0854 | 0.6343 | 0.6739 | 0.6233 |
| 100 | 0.0596 | 0.7571 | 0.5019 | 0.6371 |
| 300 | 0.0453 | 0.8057 | 0.1616 | 0.6317 |
| 500 | 0.0406 | 0.8657 | 0.2718 | 0.6295 |
| 700 | 0.0190 | 0.9543 | 0.2314 | 0.6324 |
| 900 | 0.0239 | 0.9600 | 0.1427 | 0.6193 |
| 950 | 0.0219 | 0.9562 | 0.1655 | 0.6265 |

SpinBench `lm_loss` 5× 下降意味着模型**对正确答案的置信度在上升**。但 argmax 已经是这一批 62-65% 的样本 — 模型没法把不会的题目变成会的题目。

> 注：MindCube 的 `acc` 数字本身存在 leak 嫌疑（见 [`train_eval_paradigm_mismatch.md`](../bug_fix/train_eval_paradigm_mismatch.md)），但 SpinBench 上"acc 不动"这个事实跟 leak 无关 — 上限本来就被 1-4 类失败模式锁死。

---

## 2. xyz=0 ablation — bias module 是 dead

`scripts/xyz_validation.sh --method atten --ckpt train_records/atten_mindcube/step_900 --datasets spinbench`，6 GPU 并行，2739 样本，~25 分钟。

输出：[`vis_results/xyz_val_atten_step900/spinbench/`](../../vis_results/xyz_val_atten_step900/spinbench/)。

### 2.1 整体

| 解析方式 | normal acc | zero acc | Δ (zero − normal) | normal == zero pred |
|---|---:|---:|---:|---:|
| 严格 `<answer>X</answer>` | 60.17% | 60.13% | −0.04 pp | 98.98% |
| 宽松 (含 `<image>X</image>`) | 60.93% | 60.93% | +0.00 pp | 99.16% |

99% 预测一字不变 — bias module 输出对最终 logits 的影响小到完全不改 argmax。

### 2.2 按 family 拆

| dataset_type | n | norm_strict | norm_loose | zero_loose | Δ_loose | same_pred |
|---|---:|---:|---:|---:|---:|---:|
| `?` (mental rotation) | 120 | 0.0667 | 0.2083 | 0.2083 | +0.0000 | 97.50% |
| abo | 617 | 0.6434 | 0.6451 | 0.6434 | −0.0016 | 98.87% |
| cars | 258 | 0.5775 | 0.5891 | 0.5930 | +0.0039 | 97.29% |
| faces | 339 | 0.6460 | 0.6460 | 0.6460 | +0.0000 | 100.00% |
| infinigen | 1405 | 0.6228 | 0.6228 | 0.6228 | +0.0000 | 99.57% |
| **TOTAL** | **2739** | **0.6017** | **0.6093** | **0.6093** | **+0.0000** | **99.16%** |

faces 上 normal 和 zero 的预测**逐字节相同**（100% same_pred）— bias 在 faces 上完全无作用。

---

## 3. 四类失败模式

### 3.1 类 ① `<image>X</image>` tag 替代

输出 tag 类型按 family 统计（normal pass）：

| family | `<answer>X</answer>` | `<image>X</image>` | 其他 / 无 tag |
|---|---:|---:|---:|
| `?` (mental rotation) | 30 | **87 (72.5%)** | 3 |
| cars | 243 | 15 | 0 |
| abo | 613 | 4 | 0 |
| faces | 338 | 0 | 1 |
| infinigen | 1379 | 0 | 26 |

**机制**：mental-rotation 任务的 prompt 长这样：

```
<image>
The car turns 135 degrees counterclockwise. Which resulting view is correct?
A. <image>
B. <image>
C. <image>
D. <image>
Only answer with the capital letter from (A, B, C, D).
```

5 个 `<image>` 字面量，且选项是 `A. <image>` `B. <image>` 这种结构。模型把"选项"和"`<image>` tag"绑成了语法模板，于是输出 `<image>B</image>` 而不是 `<answer>B</answer>`。

严格 parser（只匹配 `<answer>X</answer>`）把这些当作空预测：

```
[🔴empty] gt='B'  pred=''  output='<image>B</image>\n'
[🔴empty] gt='B'  pred=''  output='<image>A</image>\n'
[🔴empty] gt='C'  pred=''  output='<image>B</image>\n'
```

宽松 parser 把 `<image>X</image>` 也接收 → mental-rotation `?` family acc 从 6.67% → 20.83%（+14pp）。但仍**低于 4-choice random baseline 25%** — format 修对也救不了真实推理能力。

### 3.2 类 ② 真实推理瓶颈

放宽 parser 后仍低于 random 的 task_type：

| task_type | n | acc (loose) | random baseline |
|---|---:|---:|---:|
| infinigen_mental_rotation | 116 | 6.9% | 25% |
| car_mental_rotation | 20 | 0.0% | 25% |
| object_mental_rotation | 78 | 17.95% | 25% |
| face_rotation_classification_viewer_perspective | 70 | 28.57% | 50% |
| face_rotation_classification_own_perspective | 78 | 67.95% | 50% |
| infinigen_rotation_selection_left_no_occlusion | 62 | 33.87% | 25-50% |

**心算旋转**（mental rotation）全线低于 random — 模型完全不会，xyz 救不回来。

**face rotation**: own-perspective 67.95%（合理）vs viewer-perspective 28.57%（低于 random 50%）— 模型系统性地混淆"自己视角的左右"vs"观察者视角的左右"。这是训练数据里没有强信号区分两套坐标系。

### 3.3 类 ③ 单图任务无跨视图 xyz 信号

SpinBench 单图任务分布：

| family | 总数 | 单图样本 | 单图比例 |
|---|---:|---:|---:|
| infinigen | 1405 | 926 | 65.9% |
| 其余 family | 1334 | 0 | 0% |
| **TOTAL** | **2739** | **926** | **33.8%** |

atten 的 V↔V hole punch（`spatial_attention_llm.py:407-423`）打开同一 prefill 内 vision token 之间的双向注意力。**单图情况下这只在一张图内部生效**，跟 ViT 自带的 patch grid positional encoding 高度冗余。bias module 学到的 inter-image relative geometry 在这里完全用不上。

### 3.4 类 ④ pts3d 量级 OOD

[`scripts/xyz_validation.sh`](../../scripts/xyz_validation.sh) 跑出来的同时，对 pts3d 做 sanity check：

| | `\|pts3d\|_max` | `\|cam_trans\|` | 大平移 (>5m) |
|---|---:|---:|---:|
| **MindCube 训练**（200 scene 抽样） | mean 5.17 (p99 8.06) | mean 2.86 (p99 3.84) | **0%** |
| spinbench infinigen | 3.44 | 0.42 | 0% |
| spinbench faces | 4.05 | 0.23 | 0% |
| spinbench `?` | 3.73 | 0.01 | 0% |
| **spinbench cars** | **109.72** | 4.65 | **33%** |
| **spinbench abo** | 11.45 | **7.62** | **60%** |

cars 和 abo 占 spinbench 32% (875/2739)。PI3 在「同物体多旋转视图」/「合成渲染」上跑飞，pts3d 量级是训练分布的 13×，相机平移在 abo 上 60% 都 >5m。

`SpatialAttentionBias` 是 2 层 MLP `Linear(4 → hidden) → GELU → Linear(hidden → num_heads)`，输入 per-pair edge feature `(n_x, n_y, n_z, d)`。MLP 在训练分布 d ∈ [0, 5]m 上学到的函数，外推到 d=110m 上是噪声。

---

## 4. Prompt format 完全 OOD

| | spinbench | MindCube 训练 |
|---|---:|---:|
| 问句里嵌 `<image>` 字面量 | 100.0% (2739/2739) | 0% (0/10000) |
| "Only answer with a single capital letter" | 95.6% (2619/2739) | 0% |
| 答案监督格式 | 期望裸字母 | `<answer>X</answer>` 包裹 |

模型从未在训练里见过这两种结构：
- **问句嵌 `<image>` 字面量** → 模型把它当语法模板（解释了 §3.1）
- **"Only answer with a letter"** vs 训练学到的 `<answer>X</answer>` 输出 → 内在矛盾，模型选了"包 tag"那条路

---

## 5. 修复优先级

按 ROI 排序：

| # | 改动 | 预期收益 | 成本 |
|---|---|---|---|
| 1 | xyz_validation.py parser 接 `<image>X</image>` | mental-rotation +14pp，全局 +0.76pp | 5 行代码 |
| 2 | 训练数据补充 spinbench-style prompt（`A. <image>` 选项 + 单图）几千条 | 类 ① ② 大概率显著改善 | 数据工程 |
| 3 | `SpatialAttentionBias` 输入做 quantile / log scaling | 缓解类 ④（cars/abo OOD），但只对多视图样本生效 | 单点改 model + 重训 |
| 4 | mental rotation 单独 head | 类 ② 心算旋转部分 | 高，atten 框架可能不适合 |

> **注意**：单跑 `parser` 修复（#1）会让 mental-rotation `?` family 从 6.67% → 20.83%，但仍低于 random — **不解决根本推理瓶颈**。先做 #1 再 #2 是合理的，因为没修 #1 之前 acc 数字会被 format 误差污染，看不清训练改 prompt 的真实贡献。

---

## 6. xyz 通路在 SpinBench 上是 dead — 这条结论 vs 之前 ablation 的关系

| ckpt | dataset | Δacc (zero − normal) | bias 状态 |
|---|---|---:|---|
| coord step_1000 | MindCube | −21.05 pp | 真在用 xyz |
| coord step_1000 | SpinBench | −0.66 pp | 死 |
| decouple step_1000 | MindCube | +0.10 pp | 死 |
| decouple step_1000 | SpinBench | +0.80 pp | 死 |
| **atten step_900** | **SpinBench** | **−0.04 pp** | **死** |
| atten step_1000 | MindCube | (不可信，evaluation.py 中途被改) | 待重测 |

跨方法在 SpinBench 上 xyz 通路全部是 dead — 这跟 SpinBench 数据本身的 §3.3、§3.4 性质强相关，**不是某个具体方法的设计 bug**。

atten 在 MindCube 上的 ablation 还是要重跑确认（之前那次 `atten_xyz0_step_1000/` 的 normal/zero 用了不同版本的 evaluation.py 解析器，数字不可比）。
