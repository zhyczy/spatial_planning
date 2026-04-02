All 14 types saved successfully. Here's a summary:


Preliminary Training
14 question types found (with counts out of 10k):

Type	Count	Description
camera_motion_infer	649	Find observer location of image 2 within image 1's coordinate space
depth_prediction_oc_mv	998	Distance from observer to a marked object
depth_prediction_oo_mv	959	Depth of one object given depth of another as reference
distance_infer_center_oc_mv	880	Euclidean distance from observer to object center
distance_infer_center_oo_mv	986	Euclidean distance between two object centers
distance_prediction_oc_mv	905	Distance prediction observer→object (free-form answer)
distance_prediction_oo_mv	929	Distance prediction object→object (free-form answer)
obj_spatial_relation_oc_mv	552	Spatial orientation of object relative to observer (left/right/above/below/front/behind)
obj_spatial_relation_oo_mv	750	Spatial relation between two objects
position_matching	851	Match a marked point to the same location across views
spatial_imagination_map_mv	264	Map-based spatial imagination                                            ❌
spatial_imagination_oc_mv	512	Imagine spatial layout from observer perspective
spatial_imagination_oo_mv	257	Imagine spatial layout between objects
view_change_infer	508	Infer how the scene changes when viewpoint moves


Full Training
SPAR-7M train/ 数据集 QA 类型总结
分类规则
单张图（1张）：路径来自 image_color/，仅1张
多张图·静态多视角（2-3张）：来自 image_color/，多个不同视角帧
视频帧（32张）：路径来自 video_color/frame{i}_XXXX.jpg


🟢 单张图（1张）— 单视角空间感知                                                                          ❌
类型	               任务描述	ScanNet	ScanNet++	Structured3D
depth_prediction_oc	物体到相机的深度	315K	239K	183K
depth_prediction_oo	两物体深度关系	220K	231K	179K
distance_infer_center_oc	物体离相机远近比较	182K	175K	88K
distance_infer_center_oo	两物体离相机远近比较	231K	245K	—
distance_prediction_oc	物体到相机距离（米）	308K	234K	180K
distance_prediction_oo	两物体间距离（米）	278K	237K	121K
obj_spatial_relation_oo	两物体空间关系（单图）	350K	244K	—
spatial_imagination_oc	相机移动后物体位置想象	211K	159K	61K
spatial_imagination_oo	两物体空间布局想象	203K	160K	61K
spatial_volume_infer	推断物体/房间体积	210K	156K	121K


🔵 多张图·静态多视角（2-3张，来自 image_color/）
2张图（camera pair）：

类型	               任务描述	ScanNet	ScanNet++	Structured3D
camera_motion_infer	第2帧相机在第1帧中的坐标+深度	275K	222K	80K
position_matching	第1帧bbox → 定位第2帧bbox	353K	239K	118K
view_change_infer	描述从帧1到帧2的相机运动	240K	5,517K	177K


3张图（multi-view）：

类型	               任务描述	ScanNet	ScanNet++	Structured3D
depth_prediction_oc_mv	3视角：物体到相机深度	359K	240K	173K
depth_prediction_oo_mv	3视角：两物体深度关系	345K	238K	167K
distance_infer_center_oc_mv	3视角：比较物体离相机远近	714K	221K	294K
distance_infer_center_oo_mv	3视角：两物体离相机远近	341K	252K	166K
distance_prediction_oc_mv	3视角：物体到相机距离	359K	239K	169K
distance_prediction_oo_mv	3视角：两物体间距离	359K	240K	172K
obj_spatial_relation_oc_mv	3视角：物体相对相机方向	239K	160K	56K
obj_spatial_relation_oo_mv	3视角：两物体空间关系	359K	245K	56K
spatial_imagination_oc_mv	3视角：相机移动后想象	211K	157K	49K
spatial_imagination_oo_mv	3视角：两物体布局想象	33K	100K	21K
spatial_imagination_map_mv	3视角：构建俯视BEV地图	113K	79K	39K                  ❌


🔴 视频帧（32帧，来自 video_color/frame{i}_XXXX.jpg）— 仅 ScanNet/ScanNet++          ❌
类型	               任务描述	ScanNet	ScanNet++
distance_infer_center_oo_video	视频中跨帧物体距离比较	326K	240K
distance_prediction_oo_video	视频中跨帧物体间距离	330K	240K
spatial_imagination_oc_video	视频中相机移动后物体位置	108K	79K
spatial_imagination_oc_video_hard	同上（难版）	73K	76K
spatial_imagination_oo_video	视频中两物体布局想象	107K	79K
spatial_imagination_oo_video_hard	同上（难版）	72K	76K
appearance_order	物体在视频中首次出现的顺序	93K	54K
obj_count	视频中物体计数	147K	152K
obj_frame_locate	在哪些帧中能看到某物体	25K	52K
room_size	视频估计房间面积	68K	46K


🔵 多张图·静态多视角（2-3张，来自 image_color/）
phase1:
2张图（camera pair）：

类型	               任务描述	ScanNet	ScanNet++	Structured3D
camera_motion_infer	第2帧相机在第1帧中的坐标+深度	275K	222K	80K                   ❌
    camera_motion_infer 的答案是什么
    这个任务问的不是 "cam2 在 cam1 坐标系下的 3D 位置"，而是：

    "cam2 的光心投影到 cam1 的图像上，落在哪个像素位置？深度是多少？"

    所以答案是 2D 像素坐标 (x_pixel, y_pixel) + 深度 Z（米）。

    计算流程：

    cam2 的 3D 位置（在 cam1 frame 下）= poses_ff[1][:3, 3] = (X, Y, Z) 单位：米
            ↓ 用相机内参 K 投影
    x_pixel = fx * X/Z + cx     ← 像素坐标（在某张图像的分辨率下）
    y_pixel = fy * Y/Z + cy
    answer  = (x_pixel, y_pixel), depth=Z
    为什么跟图像 size 有关
    关键在于投影得到的像素坐标依赖于图像分辨率。

    原始 ScanNet 图像是 1296×968，用原始 K 投影得到的 (x_pixel, y_pixel) 范围是 [0, 1296] × [0, 968]
    但 GT 生成时把图像 resize 到 1000×1000，K 也随之缩放了，投影后坐标范围变成 [0, 1000] × [0, 1000]
    两者描述的是同一个空间点在不同分辨率图像上的不同像素坐标，数值不一样。

    具体数值举例
    假设 ScanNet 原始 cx=648, cy=484（即 1296×968 图）：

    scale_x	scale_y
    原始图 → 1000×1000	1000/1296 ≈ 0.772	1000/968 ≈ 1.033
    一个点在原图是 (800, 500)，在 1000×1000 图上就是 (617, 517)。

    如果你直接把原始 K 算出来的像素坐标跟 GT（1000×1000 空间）比较，误差最大会有 296px（x 方向）—— 这就是初始准确率只有 44% 的原因。

    总结
    原始图像空间	GT 答案空间
    分辨率	1296×968 (ScanNet)	1000×1000
    x_pixel 范围	[0, 1296]	[0, 1000]
    换算关系	x_GT = x_orig × (1000/1296)	—
    3D 坐标 (X, Y, Z 米) 本身不受影响，只有投影后的像素坐标因 resize 而发生变化。
    gt 自动resize了，这种pattern不是我们想学的



position_matching	第1帧bbox → 定位第2帧bbox	353K	239K	118K                     
view_change_infer	描述从帧1到帧2的相机运动	240K	5,517K	177K          ✔


3张图（multi-view）：

类型	               任务描述	ScanNet	ScanNet++	Structured3D             ？
depth_prediction_oc_mv	3视角：物体到相机深度	359K	240K	173K          ✔
depth_prediction_oo_mv	3视角：两物体深度关系	345K	238K	167K          ✔
distance_infer_center_oc_mv	3视角：比较物体离相机远近	714K	221K	294K  ？
distance_infer_center_oo_mv	3视角：两物体离相机远近	341K	252K	166K      ？ visual point? question format
distance_prediction_oc_mv	3视角：物体到相机距离	359K	239K	169K      ?
distance_prediction_oo_mv	3视角：两物体间距离	359K	240K	172K          ?  annotation is not right, the object is mouse, but the red point is not on the object
obj_spatial_relation_oc_mv	3视角：物体相对相机方向	239K	160K	56K       ? Annotation may not be correct
obj_spatial_relation_oo_mv	3视角：两物体空间关系	359K	245K	56K
spatial_imagination_oc_mv	3视角：相机移动后想象	211K	157K	49K
spatial_imagination_oo_mv	3视角：两物体布局想象	33K	100K	21K         
abandon this dataset
