# 第四阶段 V3-全量诊断包报告：2016–2022 全量三成员、CPU 包与自审

日期：2026-10-07（Asia/Shanghai）。依据 `REVIEW_V3_RESULTS_and_full_fit_recommendation_20261007.md` §5 的
"一次全量外部诊断"授权、`DS_stage4_V3_PROGRESS.md` 的"恢复点"，以及用户级要求（全量包、交付报 IC/IR、
不自行上传、交付前自审）。**唯一选择目标是日均 RankIC；IR 仅报告。**

## 第一屏

| 维度 | 结论 |
|---|---|
| 这是什么模型 | 冻结原配方三成员：`0.375×C1(3 seed) + 0.375×TabM(3 seed) + 0.25×TCN(3 seed)`，`label_end_date<=20221230` **全量**拟合；仅合法训练截止扩展，特征/架构/目标/预处理/尾窗选轮/外层从头重拟合均不变 |
| 相对哪个参照改进 | **不作改进声明。** 这是审查 §5 授权的"**未通过 2022 本地采用门槛的一次全量外部诊断**"；历史成绩身份不变，新包分数不转挂任何历史 IC |
| 验证与包是否同一身份 | 是。包内 9 个原子文件与 `full_2022` 全量产物逐一绑定 hash；包内 NumPy 推理与研究工作区 PyTorch/CatBoost 在真实 2022 输入上等价（C1 差 0、TabM≤1.5e-8、TCN≤3.7e-8，日秩与融合完全相同） |
| 该包 IC/IR（用户要求） | **训练内烟雾回测**（非样本外）：2021 IC 0.1849 / IR 13.27；2019–2020 IC 0.2198 / IR 17.87。**样本外证据**（截至2021模型，2022）：IC 0.06452、rankic_ir 0.418、本地 IR −4.54 |
| 还有什么未做 | `train.py` 独立复跑（§14.1.3）**进行中**，本报告定稿时尚未完成；公榜反馈未取得（需用户手动上传） |
| 可否进入主 Agent 审查 | **本地技术自审：通过。判定"待审查"。** 执行 Agent 不判"建议上传"，不自行上传 |

## 1. 交付物与身份

| 项 | 值 |
|---|---|
| 包 | `releases/stage4_verified_ic/V3_full_pkg.zip` |
| SHA-256 | `8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153` |
| zip / 解压 | 12,389,102 B（11.81 MiB） / 17,016,667 B（16.23 MiB），34 文件 |
| 训练截止 | `label_end_date <= 20221230`（2022 已用于训练） |
| 融合图 | `atom raw -> per-seed daily centred rank -> seed mean -> family weight -> final rank` |
| 家族权重 | `c1 0.375 / tabm 0.375 / tcn 0.25` |
| 运行依赖 | `requirements.txt`：numpy 2.4.6 / pandas 3.0.6 / pyarrow 25.0.1 / catboost 1.2.10（**无 torch**） |
| 训练依赖 | `train_requirements.txt`：上述 + torch 2.14.1+cpu / safetensors 0.8.0 / rtdl_num_embeddings 0.0.12 |

包内文件：`inference.py`、`train.py`、`alpha_*` 自包含源码、`model/{9 个原子, meta.json, tabm_design.json,
tcn_design.json, tcn_channel_stats.json}`、`LICENSE.tabm`、`TABM_SOURCE.md`。
原子：C1×3 `catboost_c1_seed*.cbm`；TabM×3 `tabm_seed*.npz`（NumPy 静态导出）；TCN×3 `tcn_seed*.npz`（NumPy 静态导出）。

## 2. 全量拟合事实（仅训练，未做 2022 外层评价）

| 成员 | seed 20261005 | seed 20261017 | seed 20261029 |
|---|---|---|---|
| C1 选中树数（earliest near-opt） | 700 | 1250 | 1875 |
| C1 内层最优检查点 / 最优IC | 1275 / 0.076623 | 1300 / 0.076652 | 1975 / 0.079670 |
| TabM 选中 epoch / 内层IC | 20 / 0.077950 | 17 / 0.076221 | 8 / 0.076007 |
| TCN 选中 epoch / 内层IC | 13 / 0.076349 | 10 / 0.063318 | 15 / 0.074012 |

产物：`artifacts/stage4_verified_ic/v2/full_2022/`（C1/TabM）与 `v2/tcn_rev1_tcn_s*_FULL22/`（TCN）；
登记报告 `v2/v3_full_fit.json`（`outer_metrics_computed=false`，未预测 2022 外层）。
注：`v3_full_fit.json` 的 `models` 字段只含 TCN 记录，C1/TabM 记录在 `__curve`/`inner_curve`/`design` 文件里；
该报告的 `files` 段已覆盖全部 15 个全量文件的字节与 SHA-256。

## 3. 自审验收（可失败项逐条给出）

### 3.1 干净环境（按交付依赖安装，禁网运行）

- 新建 venv `/tmp/v3clean`，仅按 `requirements.txt` 安装：numpy/pandas/pyarrow/catboost 成功；`import torch` → `ModuleNotFoundError`（证明推理不依赖 torch）。
- 从 **工作区外**（`/tmp/v3sub` 解压目录）加载 `/tmp/v3sub/submission/inference.py` 并预测：
  20160105 / 20220613 / 20221228 均返回完整合法池、有限分数。证据：`artifacts/stage4_verified_ic/v3_acceptance.json`。

### 3.2 真实入口、完整池与有限性

平台入口 `InferenceModel.predict`（`submission/inference.py`）通过 `skd_backtest.SubmissionRunner` 同构加载；
抽查 2022（20220104/20220613/20221228）+ 抖动日（20160105 等）均：列名 `date/code/score`、日期一致、
池与当日 Barra 完全一致（无缺股/重复）、分数全有限。短历史（2016 首周，价格点<21）由序列构建器**确定性左填充
全 NaN 列**处理（标准化为 0、validity=0），不杜撰收益。

### 3.3 数值等价（raw 容差与秩**分别**报告）

初始容差 raw `atol=1e-5, rtol=1e-4`；秩判定独立进行。三个 2022 抽查日：

| 对象 | raw 最大绝对差 | raw 容差 | 日秩是否完全一致 | 最大秩位差 |
|---|---|---|---|---|
| C1×3 | 0.0 | 通过 | 是 | 0 |
| TabM×3 | ≤1.5e-8 | 通过 | 是 | 0 |
| TCN×3 | ≤3.7e-8 | 通过 | 是 | 0 |
| 三家族分数 | 0.0 | 通过 | 是 | 0 |
| **最终融合分** | **0.0** | 通过 | 是 | 0 |

raw 容差与秩两项**都通过**才算通过（不再用其一替代另一）。平台入口返回分与 `Scorer.score` 逐位一致。

### 3.4 场景与不变性

| 检查 | 结果 |
|---|---|
| 名称/is_st 标识扰动 | 最大绝对差 0.0（屏蔽有效） |
| t 之后未来行截断 | 输出逐位不变 |
| 表内行顺序打乱 | 按 code 合并后逐位一致 |
| 重复实例化 | 分数一致 |
| 短历史日 / 停牌股 / 特征缺失股 | 均得完整池有限分 |

### 3.5 时限、内存与包体

| 项 | 值 | 目标 |
|---|---|---|
| 冷启动（模型加载） | 0.072 s | — |
| 暖推理 P50 / P95 / max | 0.246 / 0.263 / 0.325 s | P95 ≤ 2 s |
| 峰值 RSS（含加载整套源表+模型，单进程） | 771 MB | ≤1024 MB |
| 峰值 RSS（验收进程含 torch 参照，仅参考） | 941 MB | — |
| zip / 解压 | 11.81 / 16.23 MiB | ≤900MiB（工程目标） |

P95 不大于同一组样本最大值；加载时间与首日推理分别计量，未用"已加载后的首日"冒充冷启动。

### 3.6 静态代码审计（自定义代码）

逐文件扫描：无 `getattr(`/Python `eval(`/`sys.modules`/`subprocess`/`importlib`/`exec`；`model.eval()` 为
torch 推理模式（仅训练侧），非内置 `eval`。推理侧磁盘读取仅限包内 `model/`（`json` + `np.load(allow_pickle=False)`）；
无包外数据读取、无评测期 fit、无跨调用状态。

## 4. IC / IR（用户要求：交付即报；IR 只报告、不参与选型）

### 4.1 该包（全量模型）的 IC/IR —— **训练内烟雾回测，非样本外**

通过平台入口 `submission_dir` 跑固定主费用契约（`data_zero_st` 视图）：

| 期间 | 引擎 IC (mean_rankic) | rankic_ir | 组合 IR | 年化净超额 | TE | 换手 | 成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 0.18485 | 1.346 | 13.27 | +55.1% | 0.0415 | 235.5 | 294,677 |
| 2019–2020 | 0.21976 | 1.423 | 17.87 | +69.5% | 0.0389 | 461.6 | 895,948 |

**必须按此读数：模型在 `label_end<=20221230` 上训练，上述年份已进入训练集，因此这组 IC/IR 只说明
"打包入口能跑、且在工作区内自洽"，不是样本外性能，不得据此声称改进或预测公榜。** 证据：
`artifacts/stage4_verified_ic/backtests/v3_full_pkg_{conf21,dev19_20}/metrics.json`。

### 4.2 该配方的**样本外**证据（截至2021模型 → 2022）

| 对象 | 面板日均 Spearman (240 日) | 引擎 IC | rankic_ir | 组合 IR | 年化净超额 |
|---|---:|---:|---:|---:|---:|
| **三成员 (c1 .375/tabm .375/tcn .25)** | **0.063700** | 0.064519 | 0.418 | −4.544 | −18.3% |
| V1 (c1 .5/tabm .5) | 0.065424 | 0.066228 | 0.435 | −4.625 | −17.6% |
| C1_fixed | 0.065981 | 0.066817 | 0.456 | −4.137 | −15.0% |
| TREE_structure (l5 .5/c1 .5) | 0.070323 | 0.071178 | 0.495 | −4.075 | −14.6% |

这是本批**唯一**样本外读数：三成员在 2022 仍为正 IC，但**低于** V1/C1/TREE，未获采用门槛支持
（详见 `DS_stage4_V3_REPORT.md`）。本地 IR 全为负（本地代理费率），与官方 IR 不可直接互推；IR 未用于任何选择。

## 5. 与审查 §5 授权的一致性

| 授权条款 | 落实 |
|---|---|
| 固定对象（仅原三成员、既有 seed、既有图） | 是；唯一变化为训练截止扩展到 2022 |
| 性质标注（未通过 2022 门槛的一次全量外部诊断，非新冠军） | 本报告第一屏与 §4.1 明确标注；历史分数身份不变 |
| 包先验收（全量源文件、CPU 依赖、真实入口、完整池/有限性/数值等价/时限内存/≤1024MB） | 均已完成（除 train.py 复跑进行中） |
| 最多一份先看反馈、不等同公榜调权 | 本会话只产出这一份；未做任何公榜试探 |

审查 §6 已修复项（A/B/C 阶段，见 `DS_stage4_V3_REPORT.md` 附录）：入口强制冻结校验、preflight 自洽校验、
validator 容差与秩分别判定、判定补 TREE 半年门、产物命名与 registry 状态更正、两次打开如实登记。本轮
registry 又补登了"全量拟合使用 2022 标签（仅训练、无外层评价）""烟雾回测""2022 组合回测"三条。

## 6. 未完成 / 限制（不掩盖）

1. **`train.py` 独立复跑（§14.1.3）本报告定稿时仍在运行**：从解压目录在工作区外执行 `train.py --data-dir Source/Data --out /tmp/v3sub/repro_model`，重建全量 9 成员并与包内权重逐原子比较。完成后追加 `artifacts/stage4_verified_ic/v3_train_repro.json`；在此之前"最终包可复现"一层标 **未完成**。
2. 2022 已是**已使用**年（既用于确认又用于全量训练），不得再称盲验证年；本包不能产生"2022 样本外"。
3. 公榜未取得：按用户约定**只由用户手动上传**，本次不提供"建议上传"结论。
4. `v3_confirmation_scores.parquet` 只含四组合分数，无 12 原子 raw（历史缺口，未变）。
5. 全量模型的内层选轮用了 2022 成熟训练标签（合法），非以外层成绩换超参；2022 训练标签的使用已如实登记。

## 7. 结论与下一步

- 交付物与技术验收：V3 全量三成员 CPU 包**本地技术验收通过、待主 Agent 独立审查**。执行 Agent
  **不判"建议上传"**。
- 待办：① `train.py` 复跑结果补入；② 若主 Agent 审查通过，由**用户**决定是否用此一份包取得一次
  公榜反馈（仅作外部诊断，不改写历史身份）。

## 交付物索引

| 内容 | 路径 |
|---|---|
| 本报告 | `handoff/DS_stage4_V3_FULLFIT_REPORT.md` |
| 全量包 | `releases/stage4_verified_ic/V3_full_pkg.zip`（SHA-256 见 §1） |
| 包构建目录 | `releases/stage4_verified_ic/build/V3_full_pkg/` |
| 验收结果 | `artifacts/stage4_verified_ic/v3_acceptance.json` |
| 烟雾回测 | `artifacts/stage4_verified_ic/backtests/v3_full_pkg_{conf21,dev19_20}/` |
| 全量拟合产物 | `artifacts/stage4_verified_ic/v2/full_2022/`、`v2/tcn_rev1_tcn_s*_FULL22/`、`v2/v3_full_fit.json` |
| 访问登记 | `artifacts/stage4_verified_ic/access_registry.json` |
| 构建/验收源码 | `src/qf3/s6_v3_package.py`、`s6_v3_train_src.py`、`s6_v3_accept.py`、`s6_v3_backtest.py` |
