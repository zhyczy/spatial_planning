{"problem": "Look at the following four people:\nA. <image>\nB. <image>\nC. <image>\nD. <image>\nWhich image shows a different person?\nOnly answer with the capital letter from (A, B, C, D).", "answer": "C", "images": ["images/face_rotation_1e13a87d8a.jpg", "images/face_rotation_2579b8775b.jpg", "images/face_rotation_1b87d17b74.jpg", "images/face_rotation_ebae9f510d.jpg"], "metadata": {"task_type": "face_identity_quartet_interleaved", "dataset_type": "faces", "person_ids": ["037", "037", "040", "037"], "angles": [0, 45, 90, 90], "distractor_position": 2, "distractor_person_id": "040", "same_person_id": "037", "num_same_views": 3, "correct_answer_text": "Image C"}, "id": "0ab6e06c2826"}

question category: face_identity_quartet_interleaved


1. 规范视角选择 (Canonical View Selection) — 14.8%
car_canonical_view_selection_* / object_canonical_view_selection_* / face_canonical_view_selection_*
任务: 给定参考视角（通常是正面），选择目标视角（左/右/背面）的图像
答案: 选 A/B/C
例如: 给 4 张汽车图（正面 + 3 个选项），选哪个是背面？
2. 空间关系理解 (Spatial Relations) — 20.0%
infinigen_spatial_relation_grounding_left_right (286) / front_behind (198) / far_near (152)
infinigen_spatial_relationship_dynamic_* (156 total)
infinigen_spatial_relation_transformation_* (290)
任务: 判断物体间的空间位置关系（左右、前后、远近）
答案: 选 A/B（如"左还是右"）
例如: 一张图，判断"芥末瓶在魔方的左边还是右边？"
3. 物体旋转理解 (Rotation) — 22.2%
car_rotation_classification / object_rotation_classification_* / face_rotation_classification_*
infinigen_mental_rotation (120) / object_mental_rotation (78) / car_mental_rotation (20)
infinigen_rotation_selection_* (267 total，包括遮挡)
任务:
分类: 判断旋转方向（顺时针/逆时针）— 2 张图
心理旋转: 物体旋转后选择正确朝向 — 4-5 张图
答案: 选 A/B/C/D
例如: "物体顺时针旋转了 180°，哪张图显示了正确朝向？"
4. 物体身份识别 (Identity) — 12.4%
car_identity / object_identity_* / face_identity_*
包括单个 (single) 和四元组 (quartet) 版本，以及不同呈现顺序 (imagefirst/textfirst/interleaved)
任务:
单个: 识别物体是否是同一个（不同视角）
四元组: 给 4 张图，找出不同的那个（3 张相同，1 张不同）
答案: 选 A/B/C
例如: "这 3 张图中，2 张是同一物体不同角度，1 张不同，找出不同的。"
数据来源:
Infinigen (超 50%): 合成 3D 场景，图像质量一致
Car/Object/Face: 真实或标注数据
Cars: 汽车旋转/身份
Objects: 通用物体身份与旋转
Faces: 人脸旋转与身份
评估重点:
SpinBench 专注于 3D 空间理解与推理 — 需要：

识别物体在 3D 中的方向/位置
心理旋转（想象 3D 物体旋转后样子）
多视图一致性（同物体不同角度）