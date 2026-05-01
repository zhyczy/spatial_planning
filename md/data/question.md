各数据集 — 输入构造、Loss 计算、Accuracy 计算 完整对照
训练数据（loss 计算的来源）
数据集 / 子集	原始 GT 字段	Wrap 方式	是否填文本？	Loss 范围	涉及脚本
MindCube train	gt_answer = 单字母	format_answer_with_text(letter, question) 抽 inline 选项文本	✅ 填 <answer>C. Light purple sofa</answer>	F.cross_entropy 全 12 supervise token mean	train_correspondence.sh / train_coordinate.sh
VST mi_camera_motion	raw GPT = "C. left and forward"	<answer>{raw}</answer> 直接包	❌ 不调 helper（raw 自带）	F.cross_entropy 全 supervise token mean (~12 token)	train_atten.sh
VST mi_correspondence	raw GPT = "D point-D"	同上	❌	同上 (~10 token)	train_atten.sh
VST mi_object_object_relation	raw GPT = "B. southwest"	同上	❌	同上 (~10 token)	train_atten.sh
VST mi_scene_caption	raw GPT = 长 caption	同上	❌	全 caption tokens mean (可能 100+)	train_atten.sh
VST si_depth_comparison	raw GPT = "point-2"	同上	❌	~8 token mean	train_atten.sh
VST si_distance	raw GPT = 度量解释	同上	❌	多 token mean (~30)	train_atten.sh
VST si_measurement	3 个 raw GPT (multi-turn)	多轮，每轮 <answer>{cm}</answer>	❌	3 段独立 unmask，跨段 mean	train_atten.sh
VST si_scene_caption	raw GPT = 度量 caption	同上	❌	全 caption tokens mean	train_atten.sh
重点：只有 MindCube training 走 format_answer_with_text 做"填文本"。所有 VST 数据不填，因为 raw GPT response 自身就是最终 wrapped 内容。

训练时 periodic eval（仅 2 个数据集，每 EVAL_STEPS 跑一次）
通过 Eval_Dataset_Coord，3 个 train 脚本都用同一组：

数据集	原始 GT	Wrap 方式	是否填文本？	Loss / Acc
MindCube tinybench (1050)	gt_answer = 单字母	format_answer_with_text(letter, question)	✅ 100% 填	Loss = 12 token mean CE / Acc = letter-pos argmax
SpinBench test.jsonl (2739)	answer = 单字母	format_answer_with_text(letter, problem)	🟡 混合: 1164 (42.5%) 填 + 1575 (57.5%) fallback <answer>{letter}</answer>	Loss = 变长 token mean CE / Acc = letter-pos argmax
为什么 SpinBench 是混合？因为 SpinBench problem 字段格式异质：

部分用 A. {真文本} B. {真文本} → 抽出真文本，填入
部分用 A: <image> B: <image> 或 A. <image> B. <image> (image options) → precheck/placeholder 闸门拦下，fallback
不论填或 fallback：

Loss：F.cross_entropy(unmasked, ignore_index=-100) 跨所有 supervise token mean
Acc：只检查 LETTER_OFFSET=2 那一格 argmax 是否命中 GT letter token
Deploy eval（10 个数据集，evaluation.py，无 wrap）
Deploy 阶段不走 format_answer_with_text。模型 generate 完整文本，extractor 提取 letter/number/content：

数据集	样本数	format_type	GT 形态	Extractor	模型在 SPA 路径 emit 啥？
mindcube	1050	select	单字母	extract_answer_letter	SFT 后 emit <answer>X. {text}</answer>，extract 抓首字母
mmsibench	1000	select	单字母	同上	同上
sat_real	150	select	单字母	同上	同上
sparbench_multi_view	1462	select	单字母	同上	同上
sparbench_single_view	1038	select	单字母	同上	同上
sparbench_mv (select 部分)	1798	select	单字母	同上	同上
sparbench_mv (fill 部分)	1354	fill	数字 (e.g. 2.6)	extract_answer_number	emit <answer>{number}</answer>（baseline）/ 训练数据决定（SPA）
viewspatial	5712	select	单字母	extract_answer_letter	同上
omnispatial_pt	561	select	单字母	同上	同上
embspatial	3640	select	单字母	同上	同上
spinbench	2739	select	单字母	同上	同上
Deploy acc 公式：


prediction = extract_answer_letter(model_output)   # or _number for fill
correct = (prediction.lower() == gt.lower())       # gt 仍是单字母/数字
Loss / Accuracy 三档汇总（按"哪里"算）
阶段	Loss 范围	Acc 范围	"填文本"参与 loss？
训练 step（MindCube）	12 token mean CE（letter+option text+闭合）	N/A	✅ option text 进 loss
训练 step（VST）	raw GPT response token mean CE（变长）	N/A	❌ 但 raw 本身可能含半 free-form 内容
Periodic eval (train_*.py)（MindCube tinybench）	同训练 step：12 token mean CE	letter-pos argmax 1 格	✅ option text 进 loss / ❌ option text 不进 acc
Periodic eval（SpinBench, ~43% 填）	12 token / ~8 token mixed mean CE	同上	同上
Deploy eval（evaluation.py）所有 10 个	N/A（不算 loss）	extract_answer_letter / _number 比较 GT	N/A — 看模型输出，不看训练 wrapping
三档"acc 等价性"链条

训练时 Periodic eval acc        ─┐
                                  │  在贪心 + no-leak 前提下数学等价
                                  ├── 都检查"<answer> 后第一个 token 是不是 >X"
                                  │
Deploy eval acc                 ─┘
训练时 acc = TF logits[letter_pos-1] argmax == GT >X token id
Deploy acc = generate 后 extract_answer_letter 抓首字母 == GT 单字母
两者实际比较的都是同一件事："<answer> 标签后模型 emit 的第一个字符是不是 GT 字母"。

一句话总结
MindCube training + Eval_Dataset_Coord 是唯一会"填文本"的地方（通过 format_answer_with_text），10000 + 1050 + ~1164 spinbench 共 ~12k 条
VST 训练 raw GPT response 直接 wrap，从不调 helper
训练时 eval 的 loss 包括填的 option text，acc 只看 letter（与 evaluation.py deploy acc 数学等价）
Deploy eval 不 wrap，看模型 generate 后的输出，extractor 抓 letter / number 跟 GT 比