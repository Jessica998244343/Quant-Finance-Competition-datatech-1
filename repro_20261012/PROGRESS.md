# V3全量复跑实时进度

本次由主Agent根据用户明确授权亲自实现和运行。原提交ZIP不修改。

## 当前已完成

- 核验原包与只读副本SHA：8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153。
- 重验336个公开数据parquet文件，与20261010来源清单一致。
- 核验Python3.11.17及numpy/pandas/pyarrow/catboost/torch/safetensors固定版本；使用已安装CPU训练环境。
- 原包解压到本目录`original/submission/`，原文件逐项核验。未安装依赖。
- 编写独立审查运行器`scripts/v3_repro_resume.py`：直接使用原包架构、变换、损失、导出和树训练函数，NN只增加逐epoch保存/恢复；记录原始数据身份和运行器hash。
- 恢复检查点包括模型/优化器/PyTorch和NumPy及独立Generator/Python随机状态、完整选轮历史、最佳epoch/IC、已完成epoch。写临时文件、fsync、原子替换，保留两代，载入前核验SHA和运行身份。
- TabM、TCN原训练与审查续跑小规模对照通过；参数、曲线、预处理完全一致。实测中断恢复、完成外层恢复、损坏最新检查点回退通过。见`resume_equivalence_selftest.json`；这些是审查测试，不是新的选型实验。
- 检查器补强：原ZIP/解压源码/参照模型真实绑定、参数逐元素容差与有限性、预处理全JSON门、完整树结构/特征/尺度门、真实InferenceModel输出契约和scorer一致性、成员文件hash、完整identity硬门、正确成分变化日期与同日membership过滤。

## 进行中

- 九成员正式复跑，逐epoch记录在`train.log`，完整成员列表在`progress.json`。

## 前置核验补记

- 完整面板475200行76列，从公开原始数据重建耗时320.9秒；序列重建耗时16.6秒，均已保存并绑定hash。
- 严格检查器原模型自对照通过，所有raw/融合差为0，五类负例全部被拒。报告`checker_self_reference.json`。
- 发现恢复后通道统计JSON键会由int变str，在最终组装时统一做JSON规范化，避免重启后不同成员记录键类型造成误报。训练函数/缓存构建函数逐AST比较未变。保留初始运行器和identity，缓存经hash核验后绑定当前运行器，记录`cache_identity_adoption.json`；当时正式训练尚未启动。

## 后续顺序

1. 前置自检通过后运行九成员，完成一成员即保存模型和阶段记录；NN逐epoch持久保存。
2. 对比原包和复跑的所有模型、预处理、真实输入raw/日秩/家族/最终score及本地IC差，仅用于复现。
3. 汇总验收结果、所有差异、运行命令/环境与恢复方法，核验原包SHA未变。

运行环境：`repro_20261010/train_env311/bin/python`，CatBoost线程6，NN线程2，OMP/MKL/OpenBLAS均2。
训练命令：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 repro_20261010/train_env311/bin/python -B -u scripts/v3_repro_resume.py`。
发生重启后，原命令可恢复，但必须先排除重复运行；脚本有文件锁并重新核验数据/源码身份。

本次没有沿用旧`repro_20261010/out`中单个C1为已完成进度：其旧跳过机制与过程绑定不足，耗时仅约3分钟，因此在新身份下从头重跑，更方便审查。
CatBoost保留原函数，不引入snapshot参数；中断最多重做当前成员，NN损失最多一个epoch。所有缓存和检查点在工作区持久目录，不使用`/tmp`保存训练进度。

完成状态以`progress.json`和最终严格检查报告为准。本文件不提前宣布最终可复现。

## 正式复跑阶段记录

- C1 seed20261005：完成，用时162.9秒，700棵树；独立预检查叶值、边界、特征顺序全部与原包一致，见`c1_first_member_precheck.json`。
- 原始重建面板与旧研究面板的全量比对：475200行，61个模型特征及future_return/label_end_date、键和NaN位置全部逐位一致，见`full_panel_alignment.json`。旧面板只用于重建后的比对，未参与本次训练输入构建。
- TabM seed20261005：内层32个epoch、outer20个epoch全部完成，用时1447.8秒；所有导出参数最大差0，完整内层曲线逐位一致，见`tabm_first_member_precheck.json`。
- TCN seed20261005：完整完成，用时1721.0秒，选中13轮；全部导出参数最大差0、完整内层曲线逐位一致，见`additional_member_precheck.json`。
- C1 seed20261017：完整完成，用时182.5秒，1250棵树；叶值及边界一致，同见`additional_member_precheck.json`。
- 会话恢复：旧进程已不存在，运行锁空闲，日志停在TabM seed20261017内层8轮。不能仅由此识别机器重启还是会话终止；已核验原数据/源码/模型身份，四个已完成成员凭hash完成记录跳过，从第8轮检查点恢复第二个TabM。记录均保留在同一`train.log`，未覆盖旧日志。
- TabM seed20261017：恢复后完成内层29轮/outer17轮，本次完成尝试耗时1049.2秒（不包括中断前未终结尝试的时间）；独立核验参数最大差0、完整内层曲线逐位一致，见`tabm_second_member_precheck.json`。当前5/9成员完成，正在训练第二个TCN。
- TCN seed20261017：完成内层20轮/outer10轮，用时946.9秒；独立核验参数最大差0、完整内层曲线一致，见`tcn_second_member_precheck.json`。已完成6/9，正在第三个C1。
- C1 seed20261029：209.1秒完成，1875棵树；叶值及边界全部一致，见`c1_third_member_precheck.json`。已完成7/9，正在第三个TabM。
- TabM seed20261029：内层20轮/outer8轮完成，导出参数最大差0，完整内层曲线一致，见`tabm_third_member_precheck.json`。已完成8/9，正在最后一个TCN。
- TCN seed20261029：内层25轮/outer15轮完成，用时1265.5秒；9/9正式训练完成，元数据组装成功，训练进程退出码0。正执行完整严格检查器，尚不提前声明放行。

## 最终完成

- 完整严格检查passed=true，退出码0，problems为空。九成员所有完整选轮曲线与原留存记录逐位一致。
- 发现固定探针与成员变更探针日期重复后去重，重跑完整验收再次通过；最终24个唯一日期，所有原子/家族/最终raw分数差0，日秩差0，五负例全部被拒。
- 六份NN NPZ字节一致；三个C1除GUID/训练结束时间外整个模型JSON完全一致；三份预处理JSON完全一致。
- 原ZIP SHA未改变，最终证据SHA已逐项核验。
- 完成报告`../handoff/REVIEW_V3_FULL_REPRO_COMPLETED_20261012.md`、说明`README_REPRODUCTION.md`、收据`FINAL_REPRO_RECEIPT.json`、严格检查`../artifacts/stage4_verified_ic/v3_train_repro.json`。
- 无需重新提交，本轮没有上传。训练进程已退出，缓存与检查点保留。
