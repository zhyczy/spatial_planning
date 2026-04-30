# estimate_3D.py 新旧方案差异分析

**Date:** 2026-04-30
**Scope:** 对比磁盘现存 `3d_results/`（旧方案产物）与当前 [estimate_3D.py](../../src/data_process/estimate_3D.py) + [coord_esti.py](../../coord_esti.py) 的输出
**Why:** 在重新生成 mindcube/sat 训练集与 evaluation/ 全部数据集之前，先确认改动落地位置，评估对下游 dataloader 的影响。

---

## 背景

repo 当前没有任何 commit（`ollama_inf` 分支），无法用 `git diff` 比对旧版本。本文通过两条线索还原差异：

1. 磁盘上 `train/SAT/3d_results/`（36810 条，4 月 15 日生成）的形状分布。
2. 当前 [coord_esti.py:355-447](../../coord_esti.py#L355-L447) 中 `_estimate_single_view` 的实现 + [estimate_3D.py:38-47](../../src/data_process/estimate_3D.py#L38-L47) 的 docstring 声明。

---

## 实测的形状分布

抽样 `train/SAT/3d_results/` 共 5000 条目录：

| 视角数 | 占比（抽样） | pts3d 形状 |
|---|---|---|
| 1 (Depth Pro) | ~79% | **`(512, 512, 3)`** — 唯一形状 |
| 2 (MapAnything) | ~21% | **`(518, 518, 3)`** — 唯一形状 |

`evaluation/SAT/3d_results/` 抽样 500 条结论一致。

→ 旧方案下，**单视角和多视角的 pts3d 形状分布完全不重叠**。

MapAny 的目标 pool（[mapanything.utils.image.RESOLUTION_MAPPINGS](../../external/map-anything/mapanything/utils/image.py)[518]）：
```
1.000  (518, 518)
1.321  (518, 392)
1.542  (518, 336)
...
0.486  (252, 518)
```
共 10 个候选比例。SAT 渲染图本来就是方形，所以多视角全部落在 `(518, 518)` 上；其他数据集（MMSI、SPAR…）会出现非方形候选。

---

## 五处差异

### 1. 单视角加 Step A（最关键）

**旧:** 直接把原图喂给 Depth Pro，输出统一 512×512。
**新:** [coord_esti.py:382-388](../../coord_esti.py#L382-L388)

```python
target_w, target_h = find_closest_aspect_ratio(w_in / h_in, resolution_set=518)
pil_img = crop_resize_if_necessary(
    PIL.Image.fromarray(image_np).convert("RGB"),
    resolution=(target_w, target_h),
)[0]
```

先按 MapAny 的 10 个候选比例 crop-resize，再 Depth Pro。

**后果:**
- 1-view pts3d 形状从 `{(512,512,3)}` 扩散到 MapAny pool 的 10 种之一。
- **1-view 与 multi-view 共用同一个形状池**，下游 dataloader 不再需要 1-view / multi-view 特例分支。
- 落盘的 `image.png` 也是 crop-resize 后的版本，与 pts3d 严格对齐。
- 内参 K 的 `cx, cy` 走的是 crop 后的 `W/2, H/2`（[coord_esti.py:412](../../coord_esti.py#L412)）。

### 2. `qwen_align` 默认 False

|  | 旧 | 新 |
|---|---|---|
| 落盘 H,W | 对齐到 Qwen smart_resize（/28 倍数） | 模型原生形状（MapAny /14 倍数；Depth Pro Step A 后比例） |
| 对齐时机 | 落盘时 bake | dataloader 在 `__getitem__` 阶段做 |

[coord_esti.py:142-216](../../coord_esti.py#L142-L216) 的 `_resample_to_qwen_aligned` 现在只在 `--qwen_align` 时调用。

**后果:**
- 同一份 `3d_results/` 可以喂给不同 patch grid 的下游模型。
- 训练 ViT 改 patch_size 时不用全量重生成。
- 复现旧行为：`python estimate_3D.py --qwen_align ...`

### 3. 数据集注册表扩展

[estimate_3D.py:324-380](../../src/data_process/estimate_3D.py#L324-L380) 现在登记：

新增：`mindcube_train`、`vst_train`、`viewspatial`、`omnispatial_pt`、`embspatial`、`spinbench`、`sparbench_mv`。

**仍然缺失（待补）:**
- `sat_train` — 磁盘上已有 36810 条产物（来自 `train_36k.json`），但脚本注册表里没有这个 key，重跑要走 `--json_path` 临时模式。
- `RoboSpatial` — `evaluation/RoboSpatial/` 目录存在但未注册。

**已知 bug:** [estimate_3D.py:127](../../src/data_process/estimate_3D.py#L127) 的 `mindcube_train` 读 `train_10k.json`，但磁盘上文件是 `past_train_10k.json` 或 `MindCube_train.jsonl`。当前直接跑会 `FileNotFoundError`。

### 4. 多 GPU 分片

[estimate_3D.py:539-561](../../src/data_process/estimate_3D.py#L539-L561) 加了 `--shard i/n`。

- 通用数据集：iterator 输出后按 `i % n == shard_idx` 过滤（仍要 decode 一次）。
- `vst_train` 走 `shard_aware=True` 路径（[estimate_3D.py:254-258](../../src/data_process/estimate_3D.py#L254-L258)），分片在 parquet 行级判断，不属于本 shard 的根本不 decode，对 ~563k 量级的 VST 是必需优化。

### 5. 文件 layout 不变

`view_XXXX/{pts3d,depth,camera_pose,intrinsics,mask}.npy + image.png + cameras.json` 跟旧方案完全一样 → 下游加载代码不用改。

---

## 对重生成的影响

**形状层面**：旧 SAT 1-view 全是 `(512, 512, 3)`，新跑出来会变成 `(518, 518)` 或其他 MapAny 候选。任何下游缓存（深度直方图、归一化常数、shape-bucket 采样器）都会失效，需要重算。

**ID 层面**：
- `train/MindCube/3d_results/` 现存 10000 条，id 是 hash 风格（如 `among_3a582cdbad..._gen_6_1`），与现在的 `MindCube_train.jsonl`（10k, group 风格 `among_group381_q3_5_1`）**完全对不上** — 整个目录是孤儿，需要清空。
- `evaluation/MindCube/3d_results/` 同样 1050 条孤儿。
- `train/SAT/3d_results/` 36810 条 id 是 `database_idx`（整数），与新 `train_36k.json` 的 `database_idx` 一致 → id 体系本身没变，但形状会变。

**重生成前需要决定:**
1. 旧 hash-id 目录直接 `rm -rf` 还是保留？（建议清空，避免训练时混入孤儿）
2. SAT 走 `train_36k.json` 还是 `train.json`(172k) 还是再加上 `challenging_samples_2k.json`？
3. MindCube 训练用 `MindCube_train.jsonl` 还是 `past_train_10k.json`？两者都是 10k 条，但 id schema 可能不同。
4. RoboSpatial 是否纳入这次重生成？
