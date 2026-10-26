# 论文写作交付入口 · 2026-10-26

给Hazel：**先看主线，再取结果表。当前范围下不新增实验，按SRW长文组织；长文是篇幅选择，不代表录用保证。**

## 先读这两份

1. [论文主线与收尾范围](01_writing/论文主线与收尾范围.md)：正文问题、结果如何表述、附录安排、停止条件。
2. [Stage2统一评价结果解读](02_stage2_results/统一评价结果解读.md)：分类、校准、输入贡献、区间及错误转移。

**版本优先级：本README和第一份主线说明 > 历史报告中的待办建议。**历史文件按原样保留以便追溯，其中“补base KD”“继续标注”等早期建议不再是当前任务。完整项目总账已加版本提示。

## 写论文取哪些文件

| 用途 | 文件 |
|---|---|
| Stage2主表（全部方法、校准前后、均值与样本SD） | [recomputed_summary.csv](02_stage2_results/recomputed_summary.csv) |
| 逐种子与各类别F1 | [recomputed_per_seed.csv](02_stage2_results/recomputed_per_seed.csv) |
| 主要及辅助配对区间 | [paired_intervals.csv](02_stage2_results/paired_intervals.csv) |
| 错误转移分析 | [error_transitions_summary.csv](02_stage2_results/error_transitions_summary.csv)、[逐种子](02_stage2_results/error_transitions_per_seed.csv) |
| 逐条概率、类别及Hazel原始表 | [hazel_original/report](02_stage2_results/hazel_original/report) |
| Student后校准参数和拟合协议 | [student_temperature_LOCKED.json](03_student_postcalibration/student_temperature_LOCKED.json)、[protocol_LOCKED.json](03_student_postcalibration/protocol_LOCKED.json) |
| 评价方案与模型锁定清单 | [evaluation_protocol_LOCKED.json](04_protocol/evaluation_protocol_LOCKED.json)、[models_LOCKED.csv](04_protocol/models_LOCKED.csv) |
| 数据使用史和隔离审计 | [known_use_history.md](04_protocol/history/known_use_history.md)、[DATA_AUDIT.json](04_protocol/audit/DATA_AUDIT.json) |
| 第一阶段结果、旧消融和P0方法说明 | [P0](05_stage1_P0P1/P0) |
| Teacher同集能力、完整成本及计时口径 | [P1](05_stage1_P0P1/P1) |
| 完整项目历史与阶段区别 | [项目总账](06_project_history/CLIP_Jev项目总账与论文主线.md) |
| 完整原始包依赖及SHA256 | [SOURCE_ARCHIVES.json](SOURCE_ARCHIVES.json) |

### 数值口径

- Stage2 CSV中的F1、Accuracy、ECE和相应差值以0—1比例存储；写成百分数或百分点时乘100。NLL与Brier不乘100。
- `historical_test`为已打开的历史1000条；`new_dev_holdout`是沿用的文件名，论文称补充200条，不能据文件名声称新独立测试。
- `raw`和`post_temperature`是同一模型的概率版本，温度为正，类别和F1相同，不当作两次训练。
- 均值是三个种子指标的均值，SD是种子间样本标准差；不是把所有种子的预测拼起来计算一个指标。
- 配对差值为A−B；F1/Accuracy越高越好，NLL/ECE/Brier越低越好。区间固定三个训练种子、以图片为单位抽样；不覆盖全部训练随机性，也不能消除已查看评价集带来的选择偏差。辅助区间未作多重比较校正。
- `P1_KD_vs_CE`是Stage2统计文件中的主比较ID，与10月14日“P1 Teacher/成本工作”不是同一个命名概念。
- 第一阶段与Stage2学习率和校准参数不同，不能混为同设置、拼种子或跨阶段挑最好值。P1成本属于当时指定部署模型。

## 当前论文范围

聚焦同任务E/N/C分类蒸馏、Teacher/Student校准及能力—成本取舍。当前稿不纳入Winoground，也不主张组合推理迁移或跨任务泛化；已有外部探索继续保留在历史总账中，不改写或删除记录。分类区间跨0、Teacher校准F1下降、LS对照和数据使用史必须保留。

旧base、硬标签、图像消融作为第一阶段补充证据，因此目前不补Stage2匹配训练。已完成的评价、后校准和统计无需再跑。

## 核验和复现

[FINAL_AUDIT.json](02_stage2_results/FINAL_AUDIT.json)记录：21模型、72行指标、43200行概率已核验；63个模型—划分组合CPU重放类别全部一致；68项预定区间已复算。该核验不是新增训练。

本目录是**写作材料与数值证据交付**，不包含大模型权重、全部冻结特征、原始图片或完整训练历史。完整复现需保留[SOURCE_ARCHIVES.json](SOURCE_ARCHIVES.json)列出的原始ZIP。尤其保留Hazel统一评价原包、Stage2协议与输入包及P0/P1完整包。

目录内`04_protocol`是便于阅读的协议摘录（含锁定抽样），不是完整协议输入包：缺少大型特征文件，不能直接运行原包完整性检查并期待通过。复制来的`PACKAGE_SHA256.json`属于其**原始完整包**，不用于验证这个汇总目录。

本目录自身使用[HANDOFF_SHA256.json](HANDOFF_SHA256.json)校验：

```bash
python verify_handoff.py
```

若需要重新独立核验Stage2，请先解压完整“Stage2_统一评价_协议与输入_20261026.zip”，使用其`handoff`目录，并传入Hazel原始评价ZIP：

```bash
python 02_stage2_results/review.py --source-zip /path/to/Stage2_Hazel_UnifiedEvaluation_20261026-20261026T090809Z-1-001.zip --protocol-dir /path/to/full/handoff --out /path/to/new_audit
```

环境参考[execution_environment.json](02_stage2_results/execution_environment.json)；脚本依赖NumPy与PyTorch。这是可选复现说明，现有核验已完成，不是再派一次运行任务。

`SOURCE_FILES.json`记录复制前源文件校验值；项目总账添加了更新提示，所以它的交付文件校验值以`HANDOFF_SHA256.json`为准。本次仅整理已有结果，未重训、重新拟合或修改结果数值。
