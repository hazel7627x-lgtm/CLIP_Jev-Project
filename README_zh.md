# CLIP–Visual Jev：SNLI-VE 三分类研究包

研究目标是提高基于 CLIP 的 SNLI-VE E/N/C 分类。Amazon 不作为主要目标。本包不使用 Qwen2.5，不引入自造 q/r、充分性头或所谓内部置信机制。

## 已有 100 条混淆矩阵

用户提供的数字，按行真实标签、按列预测标签，顺序都是 E/N/C：

|真实\预测|E|N|C|合计|
|---|---:|---:|---:|---:|
|E|31|9|4|44|
|N|3|20|3|26|
|C|0|3|27|30|

- Accuracy = 78/100 = **78.00%**；macro-F1 ≈ **77.61%**；balanced accuracy ≈ **79.13%**。
- E：precision 91.18%，recall 70.45%，F1 79.49%。
- N：precision 62.50%，recall 76.92%，F1 68.97%。
- C：precision 79.41%，recall 90.00%，F1 84.38%。
- E→N 为 9/44=20.45% 的真实 E，9/22=40.91% 的全部错误。N→E 为 3/26=11.54%。预测分布 34/32/34，真实分布 44/26/30。
- 这支持“检查 E→N”的优先级，不能证明错误全来自固定 E 偏置、模型能力或标签问题。缺少图像、假设、逐条输出，不能判断这 9 条的语义原因、logit margin 或 bias 可修复率。
- 若 100 条独立同分布，准确率 Wilson 95% 区间约 68.9%–85.0%；抽样方式和图像相关性未知，因此只作示意，不作模型优劣检验。

**不能复原同学的原样本。** 重新抽样、校准的结果不能解释为对原 100 条逐条纠错，也不能与原矩阵作配对显著性检验。

## 已核实的推理接口

[适配器模型卡](https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft)及下载的 adapter_config.json 均指向 `Qwen/Qwen3-VL-4B-Instruct`。本包固定模型、适配器、注释 SHA，以及官方代码 commit，详见 versions.json。配置文件记录 LoRA r=16、alpha=32、语言侧目标模块和 visual 排除规则；本包只使用已发布 adapter，不加载额外 decision heads。

直接调用[官方代码](https://github.com/guanxuyu-sv/Visual-Jev/tree/a392eaedfa41adae348f9767b55c0f54b3a13b57/code)：`VDM(..., with_heads=False)`、`prepare_group(qtype='claim')`、`run_independent`、`lm_option_logits`。使用同一模型和提示，通过 PEFT 的 `disable_adapter()` 获取无适配器基线，再启用适配器获取 Jev。两者都 eval，输入相同，分辨率上限 200704 pixels；模型加载失败会中止，不会退回其他模型。

官方 prompts.py 的选项顺序是 supported / contradicted / not determined，即 **E/C/N**；读出是 `Answer:` 最后位置的下一 token ` A`、` B`、` C` logits，官方会断言候选字符串各为单 token。保存前显式重排 `[0,2,1]`，因此所有输出数组都是 **E/N/C**。不是直接抽取词汇 E/N/C 的 logits，也不是自由生成后解析。

softmax 只在这三个候选分数上归一化，是条件于选项集合的分布，不是已校准的客观证据概率。这里保留官方 SNLI-VE claim 提示和空 shared_context，绝不输入 sentence1 文本 premise 或金标签。这里是新样本实验，不声称复现模型卡分数。

精确相等的最大分数按保存顺序 E/N/C 取首个类别，并在报告记录 tie 数量；N/C 完全平局时，这与官方 E/C/N 原顺序取首项存在差异。不要把零 margin 平局当作可靠类别偏好。

## 数据与独立性

[SNLI-VE 原仓库](https://github.com/necla-ml/SNLI-VE)规定官方划分图像不相交；[HF 注释镜像](https://huggingface.co/datasets/HuggingFaceM4/SNLI-VE)提供 JSONL，图片需另行获得 Flickr30K。download_annotations.py 只下载固定版本注释，不执行远程 dataset 脚本。

默认 seed=20261010：官方 train 选 6000 张图，dev 选 800 张图，test 选 1000 张图；每张图再按固定哈希顺序选一条假设，不按类别挑选，保留过滤后图像均匀抽样的自然类别分布（不等同于原始逐行数据分布）。所有候选先按 image ID 分组，跨官方集合检查重叠；同图相同假设去重，同标签时取字典序最小 pairID；若标签冲突，在抽样前排除整个同图同假设组。使用 seed 加 SHA256 排序，避免源文件顺序和不同随机库版本改变样本。dev 分成 val_fit 400 与 val_select 400。manifest 保存 pairID、图像 ID、hypothesis、标签及来源划分，元数据保存注释哈希和类别计数。

已实际生成的样本分布如下，顺序 E/N/C：train **2020/1970/2010**；val_fit **109/152/139**；val_select **138/132/130**；test **316/327/357**。全量注释 train/dev/test 分别排除 **150/5/4** 个冲突组。所有选定图像 ID 相互独立；尚无原图，像素级审计需在 Colab 补做。这个经过统一清理和每图抽一条的测试子集不是官方全量 benchmark，报告时注明区别。

图像到位后，audit-images 检查每个文件，并对 RGB 像素算 SHA256，拒绝换名/重编码的完全相同图像。这个检查不保证排除视觉近重复；如发现重复，必须在任何推理前制定排除规则并新建实验，不能按结果挑替代样本。

独立测试的含义是：不参与本研究的校准、训练、checkpoint 选择。发布适配器本来用过 SNLI-VE train，且作者用过 SNLI-VE test 做评估；官方 build_other.py 明确从该 test 构建评估数据。因此不能将本研究测试称为对作者模型开发完全未暴露，更不能排除 Qwen/CLIP 预训练污染。用于比较已发布模型及本地改进是合理的，但不足以宣称全新域泛化。

## 校准与 E→N 分析

设原始分数为 z=(zE,zN,zC)，使用 **softmax((z + [bE,0,0])/T)**。正温度不改变 argmax，只影响 NLL/Brier/ECE；bias 才改变分类。

预先固定：T 在 [0.25,4] 的 161 点对数网格，另含 T=1；bE 在 [-2,2] 间隔 0.05。val_fit 最小化 NLL 拟合 T；val_select 以 macro-F1 为主、NLL 为次选择 bE 和方案，允许负 bias 和不调整。输出 raw、temperature、E_bias、E_bias_temperature 四个消融。选择边界值会记录，不在看过测试后扩网格。

锁定 calibration.json 后才执行测试。测试输出 accuracy、macro-F1、balanced accuracy、各类 P/R/F1、混淆矩阵、NLL、Brier、10-bin ECE，以及按图像配对 bootstrap 95% 区间。代码拒绝覆写已有测试报告，但这不是防止人为重复调参的安全机制；研究者需遵守冻结协议。小验证集 ECE 波动较大。

验证集 E→N 清单保存 margin_N_minus_E 和使 E 获胜所需的最小 bias（相等是平局，实际需略大）。人工核查图像和假设，记录：对象/动作漏检、属性/数量/关系、不可见意图或背景知识、标注疑义、预处理/截断问题。这里的分类是待检验假设，不是已发现的错误原因。不得用测试错误决定规则或重标主要测试集；原标签指标保留，标签复核另列。

## 蒸馏方案与已实现范围

使用 `openai/clip-vit-base-patch32` 冻结编码器，以归一化 image/text 特征构成 `[v,t,|v-t|,v*t]`，训练 512 维隐藏层的三分类 MLP。CLIP 原始相似度本身不是 E/N/C 分类器。

三组严格共用样本、初始化种子、优化器与训练预算：CE 真标签基线；CE+原始 Jev 蒸馏；CE+验证集所选校准 Jev 蒸馏。损失为 `(1-alpha) CE + alpha*tau^2 KL(p_teacher_tau || p_student_tau)`，alpha=0.5、tau=2。校准温度 T 与蒸馏温度 tau 分开，软目标由校准 logits/tau 得到。只用 train 软目标，保留真实标签。30 epochs、AdamW lr=1e-3、weight_decay=1e-4、batch=128；按 val_select macro-F1/NLL 选 checkpoint，种子 20261010/11/12。

这一步蒸馏到“基于冻结 CLIP 的三分类系统”，**没有更新 CLIP 编码器本身**。它首先检验 teacher 软分布是否带来收益。后续若扩展到 CLIP 编码器微调，可解冻最后 1–2 层、使用更小学习率，并保持 CE 与 KD 同等训练能力。必须预先重新分配评估集或用新独立 holdout；不要看完当前测试后反复改善再称为一次性测试。若校准 KD 没有超过 CE，报告失败而不是预设 Jev 必须优于基础模型。

## Colab 操作

1. 将 `CLIP_Visual_Jev_SNLI_VE.ipynb` 上传到 Colab。它嵌入全部必要脚本，无需上传源码目录。选择支持 BF16 的 GPU（L4/A100）；显存占用尚未实测。
2. 安装单元固定官方记录的软件版本。它使用官方 CUDA 13.0 PyTorch 构建，要求 Colab 驱动兼容；若安装或 CUDA 检查失败，先换兼容运行环境。不要静默换版本后混合结果。安装完成后重启会话，再从路径设置单元继续。
3. 挂载 Drive，设置持久化 `WORK` 和已合法获取的 `flickr30k-images` 目录 `IMAGES`。图片不包含在交付物中；参照 SNLI-VE 原仓库下载说明。HF 需要认证时通过 Colab secrets/HF 登录配置，不在 notebook 写 token。
4. Notebook 写出代码，下载固定版本注释、生成清单、审计图片。若已有清单，则复用；不要为了得到好分数换种子。首次调通可另建 smoke 目录，使用少量图像，但不能把 smoke 当正式结果。
5. 先对 train/val_fit/val_select 生成 base 与 Jev logits；支持按 sample_id 续跑，恢复时检查配置一致性。推理逐条写入并 flush，意外断线造成最后一行损坏时备份后仅移除不完整尾行，不删除已完成记录。
6. 冻结校准；提取 CLIP train/val 特征，训练三种 student。每个阶段使用独立进程，释放教师 GPU 内存。
7. 最后生成 test logits 与 CLIP test 特征，输出两个测试报告。已完成结果不能覆写。下载或保留 results、predictions、manifests、features/clip_revision.json、students/selection.json 和环境记录。

命令行等价入口：`python study.py --help`、`python student.py --help`。若换成 FP16/T4，显式传 `--dtype float16`，单独命名实验目录；它不是模型卡 BF16 条件的等价复现。不要随意降低图像分辨率或使用量化来混称同一设置。

## 验证范围与交付状态

本地无 PyTorch/CUDA 模型运行环境，未运行 Qwen 权重、适配器推理或 CLIP 训练；没有生成任何真实模型性能提升结论。已经实际获取并核实官方推理代码、模型配置、版本 ID，下载三个官方划分的固定版本注释并生成 7800 条样本清单。CPU 单元检查覆盖矩阵计算、E/C/N→E/N/C 映射、温度不改变决策、bias 方向、抽样可复现与隔离、冲突标签剔除、缺失预测拒绝、校准及配对 bootstrap。GPU 路径仍须在 Colab 完成端到端验收。

来源：以上接口依据固定版本的官方代码与模型卡；[Qwen3-VL 模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)；[CLIP Transformers 文档](https://huggingface.co/docs/transformers/model_doc/clip)。依赖版本来自官方 code/requirements.txt；本包的抽样、校准、student 是本研究新增方法，不冒充官方算法。
