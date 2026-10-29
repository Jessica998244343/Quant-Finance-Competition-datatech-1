# V3 进度日志（对照 `REVIEW_V2_COMPLETION_REV1_and_V3_RELEASE_20261007.md`）

规则：只对获审清单内的对象做一次性确认；任何冻结后的实现改动先撤销身份再重建；小任务完成即追加。

## A. 启动检查（先于任何 2022 计算）

| 项 | 完成时间 | 命令 | 结果 | 产物 |
|---|---|---|---|---|
| A1 解释器与控制身份 | 2026-10-07 09:2x | `uv pip install lightgbm==4.7.0`（原始日志） | 统一解释器 `.venv_v1train`：Python 3.11.17 / torch 2.14.1+cpu / catboost 1.2.10 / lightgbm 4.7.0 / numpy 2.4.6 / pandas 3.0.6 / 线程 2；12 个确认原子全部加载 | `deps/V3_env_install.log` |
| A1 截止边界 | 同上 | `python -m qf3.s6_v3_preflight` | deploy / TCN / L5 三处 `label_end<=20211231`；三处 `outer_metrics_computed=false` | `v2/v3_preflight.json` |
| A2 统一 validator | 同上 | 同上 | 正常样本与负例共用 `validate_day`；NaN 原子、缺股票、篡改参考被同一门拒绝 | 同上 `negative_cases` |
| A3 加载器等价 | 同上 | 同上 | 2019/2020/2021 各 2 个代表日（含 3 个真实成分变更生效日）重建三家族×三 seed 原子：**max abs diff 0.0**、池/秩一致 | 同上 `equivalence` |
| A4 确认清单 | 2026-10-07 09:37 | `python -m qf3.s6_v3_manifest` | 48 文件、路径唯一、verify 通过；hash `83550fbe6baf5d27…` / `0c239a7de02fd823…`；记录 supersedes 与采用规则 | `v2/v3_confirmation_manifest.json` |
| A4 身份更正 | 2026-10-07 09:37 | `--verify` 报 `s6_v3_models.py` hash mismatch | 第一次冻结（`8a9466d0…`）绑定的是**尚不能运行的加载器**；撤销该身份，重建后在新身份下重跑 A 与 B，结果逐位相同；作废的打开动作已登记 | `access_registry.json` |

## B. 一次性 2022 确认

| 项 | 完成时间 | 命令 | 结果 |
|---|---|---|---|
| B1 记录打开 | 2026-10-07 09:38:08 | `python -m qf3.s6_v3_confirm` | 打开时间/清单 hash/源码 hash 已写入；preflight 必须为真才允许打开（否则 SystemExit） |
| B2 统一评价 | 同上 | 同上 | 合法信号日 **240 天**（2022-01-04…2022-12-28）；同日期/池/标签掩码；逐日 Spearman→日均；四组合 RankIC：three_member 0.063700、V1 0.065424、C1 0.065981、**TREE_structure 0.070323**；配对块 10/20/40、2000 次、seed20261005 |
| B3 决策 | 同上 | 同上 | 三成员−V1 **−0.001725**（两半年均 ≈ −0.00172）、三成员−TREE **−0.006623** → 不支持；回退检查 V1−TREE **−0.004898**（半年 −0.002311/−0.007359）→ 不支持 |

## C. 最终拟合与包

| 项 | 状态 | 说明 |
|---|---|---|
| C1 最终拟合 | **不执行** | 规则只对"获支持对象"全量拟合；本轮无获支持对象 |
| C2 CPU 提交包 | **不产出** | 同上；保留既有已成功提交版本 |
| C3 IC/IR 与交付 | 完成（负结果） | 本轮无新包，故无"该包 IC/IR"可报告；IR 只对最终冻结对象跑，本轮不存在该对象 |

## 结论

2022 一次性时间外确认**未支持**三成员候选，也**未支持** V1 回退；按事前规则保留既有成功提交版本，
不据 2022 更换权重/seed/特征/选轮，不产出新包，不建议上传。详见 `handoff/DS_stage4_V3_REPORT.md`。

---

## 恢复点（Resume）— 2026-10-07 10:35

**当前状态**：审查 §6 的全部程序修正已完成；审查 §5 授权的"一次全量外部诊断包"进度为
**C1×3 与 TabM×3 已训练完成并落盘，TCN×3 未完成**（进程随会话结束被终止，工作区没有
`tcn_rev1_tcn_s*_FULL22` 目录）；**打包与验收未开始**。

**工作区与解释器**（新会话必须使用同一路径与解释器）：

- 工作区：`/home/jessica998244343/DATA/Quant-Finance3`（**不是 git 仓库**，一切以文件 + SHA-256 认身份）
- 统一解释器：`.venv_v1train/bin/python`（Python 3.11.17 / torch 2.14.1+cpu / catboost 1.2.10 /
  lightgbm 4.7.0 / safetensors / rtdl_num_embeddings / typeguard）；`.venv` 无 torch、`.venv_nn` 无 typeguard
- 运行约定：`PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`，**CPU 线程 ≤12**，单任务串行

**下一步（按序，勿跳步）**：

1. 续跑 TCN 全量（C1/TabM 已存在会被读回，不重训）：
   `PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv_v1train/bin/python -m qf3.s6_v3_full_fit --tcn-only`
   产物：`artifacts/stage4_verified_ic/v2/tcn_rev1_tcn_s*_FULL22/`
2. 组装全量三成员包（`inference.py` / `train.py` / `requirements.txt` / 9 个原子权重 / manifest）；
   训练截止 `label_end<=20221230`，融合 `{c1:0.375, tabm:0.375, tcn:0.25}`，冻结图不变。
3. 包验收 + 写报告：清洁环境按包内依赖安装、禁网、包外目录、平台入口；完整池/有限性、
   **raw 容差与秩分别报告**、冷启动/暖启动 P50·P95·max、峰值 RSS、≤1024MB；报告明确标注
   "**未通过 2022 本地采用门槛的一次全量外部诊断**"，历史分数保持原身份、不转挂，最多一份先看公榜反馈。

**必须保持的约束**：2022 已使用，**不得**再称为盲验证年；不得据 2022 改特征/seed/α/架构/训练窗口；
**执行 Agent 只能判"本地技术验收通过/待审查"，不得判"建议上传"**；上传只由用户完成。

**用户级要求（必须遵守，来自用户原话，不在审查文档里）**：

1. `submission/` 里封装好的包**必须是全量模型**（本阶段即 `label_end<=20221230` 的全量三成员），
   不得用年度/夹具包冒充。
2. **每次封装好包并交付时，必须同时向用户报告该包的 IC/IR**（IR 只报告、不参与选型；
   如实区分样本外证据与训练内烟雾回测）。
3. 严格按任务书与主 Agent 的指示执行；**有歧义先向用户确认，不要自作主张**；交付前先自审。

**身份速查**：

- 当前确认清单：`v3_confirmation_manifest.json`，canonical `211c953c025ef57b…`，
  文件 SHA-256 `8eb3245610f6b83c…`（48 文件）；历史链 `8a9466d0…`（作废）→ `83550fbe…`（2022 确认）
- 2022 确认结果：`v3_confirmation_2022.json`（240 日；三成员 0.063700 / V1 0.065424 /
  C1 0.065981 / TREE 0.070323；四组合逐日分数在 `v3_confirmation_scores.parquet`）
- 全量诊断产物：`full_2022/`（C1×3 + TabM×3，15 文件；TCN 待补）
- 已作废/保留的历史文件均带 `__superseded*` 后缀，结论只以无后缀或 `_v2`/`_v3` 的当前版本为准

---

## 完成记录（自"恢复点"继续）— 2026-10-07 11:1x

1. **TCN 全量续跑完成**（上一会话遗留进程在原 `--tcn-only` 下继续）：三 seed
   `tcn_rev1_tcn_s{20261005,20261017,20261029}_FULL22` 全部落盘（`train_cutoff=20221230`、
   `outer_metrics_computed=false`）；登记 `v2/v3_full_fit.json`。C1/TabM 全量产物此前已存在，未重训。
2. **全量三成员包已组装并 zip**：`releases/stage4_verified_ic/V3_full_pkg.zip`（34 文件，
   zip 11.81 MiB / 解压 16.23 MiB，SHA-256 `8cb919e4…`）。权重 `{c1 .375, tabm .375, tcn .25}`，
   9 个原子（C1×3 CatBoost、TabM×3 与 TCN×3 均为纯 NumPy 静态导出），`requirements.txt` **无 torch**。
   构建源码 `src/qf3/s6_v3_package.py`、训练入口 `src/qf3/s6_v3_train_src.py`。
3. **包验收完成**：干净 venv（`/tmp/v3clean`，无 torch）按交付依赖安装并从**工作区外**解压运行；
   平台入口同构加载；完整池/有限性、raw 容差与日秩**分别**报告且均通过（C1 差 0、TabM≤1.5e-8、
   TCN≤3.7e-8、融合差 0、秩一致）；标识/未来/置换不变性通过；暖 P95 0.263 s、峰值 RSS 771 MB、
   包体达标。证据 `artifacts/stage4_verified_ic/v3_acceptance.json`。
4. **IC/IR 已报（用户要求）**：该全量包为**训练内烟雾回测**（2021 IC 0.1849 / IR 13.27；
   2019–2020 IC 0.2198 / IR 17.87，非样本外）；样本外证据为截至2021模型的 2022 三成员
   IC 0.064519 / rankic_ir 0.418 / 本地 IR −4.544。报告 `handoff/DS_stage4_V3_FULLFIT_REPORT.md`。
5. **待办**：`train.py` 独立全量复跑（§14.1.3）在后台运行中，完成后补 `v3_train_repro.json`；
   上传只由用户完成，执行 Agent 判"本地技术验收通过 / 待审查"，不判"建议上传"。
