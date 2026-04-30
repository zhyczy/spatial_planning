# VST 训练数据 — Question 类别分析

**Source**: `datasets/train/VST/{si_*,mi_*}/*.parquet`
**Total**: 563,190 parquet rows → 551,013 unique IDs（重复 ID 已 dedup）
**Schema**: `id, images, conversations, type, data_source, meta_info`

VST 8 个子集分两大类：`mi_*`（multi-image, ≥2 视角）和 `si_*`（single-image, 1 视角），覆盖 12 个数据集 registry 中的 `vst_train` key。

---

## 多图任务 (mi_*, ≥2 视角)

### 1. `mi_camera_motion` — 相机自运动分类（49,398 条）

**任务**：第一人称连续帧，判断相机如何移动。MCQ。

```
Q: The images are acquired continuously from a first-person perspective.
   How does the camera's location change through space?
   A. rightward, B. backward and rightward, C. left and forward, D. rightward and upward
A: C. left and forward
```

### 2. `mi_correspondence` — 跨视角点匹配（119,648 条 → 106,353 unique）

**任务**：第一张图红圈点 → 第二张图（视角/光线变化）哪个标号点是同一物理位置。MCQ。

```
Q: A red circled point is present in the first image. The second image, taken
   after altering the camera or lighting, features several red-circled points
   with 'A, B, C, D' labels. Which is the corresponding point?
   A point-A / B point-B / C point-C / D point-D
A: D point-D
```

### 3. `mi_object_object_relation` — 跨视角物体方位推理（82,725 条）

**任务**：给定 A 在 B 的某方向，问 C 在 B 的哪个方向。8 个方位候选（N/S/E/W/NE/NW/SE/SW）。

```
Q: Suppose box (yellow point) is south of power strip (gold point). What is
   the direction of file cabinet (cyan point) from power strip (gold point)?
   A. west / B. southwest / C. southeast / D. north
A: B. southwest
```

### 4. `mi_scene_caption` — 多视角场景描述（11,000 条）

**任务**：开放式描述，逐视角讲场景内容差异。

```
Q: Frames are taken one after another, all reflecting the observer's outlook.
   Describe the features of these two frames.
A: The first image shows a room with a large white table at the center.
   Around the table, there are several wooden chairs... The second image
   presents a different perspective of the same room... [长段开放文本]
```

---

## 单图任务 (si_*, 1 视角)

### 5. `si_depth_comparison` — 相对深度排序（114,199 → 106,353 unique）

**任务**：图上标了几个橙色点，问哪个离相机最近。短答。

```
Q: In the image, orange point-1, point-2 and point-3 are visualized.
   Which point do you think is nearer to the camera?
A: point-2
```

### 6. `si_distance` — 度量距离比较（81,007 → 76,676 unique）

**任务**：判断三个物体里哪个离参考物更近，**回答带米制距离**。

```
Q: Please evaluate the distances and state which of the chair (brown box) or
   the pillow-B (yellow box) is closer to the pillow-A (pink box).
A: Distance[pillow-A, pillow-B]=0.7m
   Distance[pillow-A, chair]=5.8m
   So, the answer is pillow-B.
```

### 7. `si_measurement` — 物体尺寸度量（52,279 条，每条多轮）

**任务**：物体的最长边长度（cm）。一条样本 3 轮，问 3 个不同物体。

```
Q: What is the dimension of greatest length (length, width, or height) of the
   table (blue point-1) in centimeters?
A: 122 cm

Q: ... chair (green point-2) ...
A: 97 cm

Q: ... bed (red point-3) ...
A: 213 cm
```

### 8. `si_scene_caption` — 度量化单视角场景描述（52,934 条）

**任务**：开放式描述，**带具体米制深度和物体间距**。

```
Q: Please create a caption that describes how the elements in this scene are
   spatially related, focusing on the green point and brown point.
A: The scene shows a kitchen counter... stationery (brown point-2) at depth
   0.65m. Behind it... microwave (green point-1) at 1.68m. The stationery is
   1.06m away from the microwave. Further back... bottle (brown point-3) at
   2.68m. The bottle is 1.37m away from the microwave and 2.36m away from
   the stationery...
```

---

## 整体观察

| 答案形式 | 子集 |
|---|---|
| **MCQ 选择题** | mi_camera_motion, mi_correspondence, mi_object_object_relation |
| **短答（label / cm 数字）** | si_depth_comparison, si_measurement |
| **带度量的解释式答案** | si_distance（米制）, si_scene_caption（米制） |
| **开放式描述（无度量）** | mi_scene_caption |

| 能力维度 | 子集 |
|---|---|
| **跨视角对应** | mi_camera_motion, mi_correspondence, mi_object_object_relation |
| **空间度量（米/厘米）** | si_distance, si_measurement, si_scene_caption |
| **相对深度** | si_depth_comparison |
| **场景理解** | mi_scene_caption, si_scene_caption |

整体训练目标是 **空间度量 + 跨视角对应** 这两个核心能力。`si_*` 偏度量（提供精确数值），`mi_*` 偏方位/动作分类（MCQ 形式）。

---

## 数据规模

| 子集 | parquet 行数 | 唯一 ID | 视角数 |
|---|---|---|---|
| mi_camera_motion | 49,398 | 49,398 | ≥2 |
| mi_correspondence | 119,648 | 119,648 | ≥2 |
| mi_object_object_relation | 82,725 | 82,725 | ≥2 |
| mi_scene_caption | 11,000 | 11,000 | ≥2 |
| si_depth_comparison | 114,199 | 106,353 | 1 |
| si_distance | 81,007 | 76,676 | 1 |
| si_measurement | 52,279 | 52,279 | 1 |
| si_scene_caption | 52,934 | 52,934 | 1 |
| **总计** | **563,190** | **551,013** | — |

`si_depth_comparison` 和 `si_distance` 的源数据中存在重复 ID（同一张图被多条 SFT 样本引用），3D 重建脚本通过 `skip_existing=True` 自动 dedup，磁盘上保留唯一 ID。
