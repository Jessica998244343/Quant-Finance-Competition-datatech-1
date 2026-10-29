# 第四阶段 V1 报告（含主 Agent 审查后的有限补正）

日期：2026-10-06。依据 `Prompt_for_DS_stage4_verified_IC_improvement.md`、
`REVIEW_stage4_V0_*`、`REVIEW_stage4_V1_and_V2_direction.md`。
**唯一选择目标是日均 RankIC；IR 仅报告。** 未计算任何 2022 性能，未上传。

## 第一屏

| 问题 | 回答 |
|---|---|
| 这是什么模型 | V1 锚点 = `0.5 × CatBoost C1（3 seed 日秩均值）+ 0.5 × TabM（修正训练，3 seed 日秩均值）`；冻结图：原子日秩 → seed 均值 → 家族加权 → 最终日秩 |
| 身份 | **2021 验证夹具**：训练标签截止 `20201231`，预测/验证年 `2021`，预处理拟合截止 `20201231`。**它不是 `label_end<=2021-12-31` 的部署模型，也不是全量提交模型**（按审查 §4.4 归档、不转挂身份） |
| 相对哪个固定参照 | DEV 合并 **0.084650**；相对 C1 三 seed **+0.003005**，相对实际 TREE 包结构控制约 **+0.0031**，相对 R_DEPLOY **+0.0068**；官方 IC 最好仍是 strong_baseline_v1（0.0657） |
| 验证与包是否同一身份 | **是**：现行 zip 在工作区外解压、干净环境（仅交付运行依赖、无 torch）跑 2021 全期 243 天，与锚点 max abs diff **0.0**；包内 `train.py` 独立复跑复现全部 6 成员（TabM `.npz` 逐字节一致） |
| 还有什么未做 | **真正的 `label_end<=2021-12-31` 部署模型/预处理（V2 冻结前完成）**、V2 主模型对照与特征/TCN、V3 2022 一次性确认与全量模型 |
| 可否进入主 Agent 提交审查 | 本 Agent 判定 **V1 有限补正完成（§4.1–4.6 已闭合）**，交主 Agent 审查；**不建议上传**，且按用户要求不再单独交 V1 |

## 0. 六项状态（分开，不合并）

| 项 | 判定 |
|---|---|
| 工程完成 | **通过**（含逐日复现、六成员年度入口、严格验收器、随包源码与依赖记录） |
| 比较完成 | 达成（粗权重网格、删除消融、ResMLP 边际、P0/P1、配对区间） |
| IC 增益 | DEV 合并相对 C1 **+0.003005**（区间下界 > 0）；相对 TREE/R_DEPLOY 为正但**不宣称稳定击败全部控制** |
| CPU 包验收 | **通过**（2021 全期零差异、无 torch、导入闭包 ⊆ 依赖、manifest 22 项 + 8 负例） |
| 主 Agent 审查 | 待 |
| 官方状态 | 无新提交；仍只有 strong_baseline_v1 / ALT_v1 / TREE_v1 |

## 1. 本次对审查意见的补正（§4.1–4.6）

### 4.1 六成员 NumPy 年度入口（新增）

新增 `src/qf3/s6_v1_annual_fixture.py`：用**现行包同一 scorer/构建路径**，装载各年度
三 seed 权重、经真实 `DataProvider` 逐日跑整年，保留每个原子 raw、家族均值、最终输出、
池、日秩与每日 IC；期望日期取自独立面板日历。

| 窗口 | 独立日历天数 | 池/唯一/有限 | 原子 raw vs 已存预测 | 融合 vs 锚点 | 日秩 | rankIC |
|---|---:|---|---:|---:|---|---:|
| DEV19 | 244 | 全部通过 | C1 0.0；TabM ≤ 5.96e-08 | **0.0** | 全部一致 | **0.082577** |
| DEV20 | 243 | 全部通过 | C1 0.0；TabM ≤ 5.96e-08 | **0.0** | 全部一致 | **0.086732** |
| CONF21 | 243 | 全部通过 | （由现行包级验收覆盖） | **0.0** | 全部一致 | **0.064410** |

TabM 原子 raw 差异为静态 NumPy 前向 vs 保存的 PyTorch 前向的浮点残差，采用预登记
`TABM_ATOL=1e-5`（V0 实测 ≤1.19e-07）；C1 要求严格 0。证据 `v1_annual_fixture_{DEV19,DEV20}.json`。

### 4.2 逐日复现（修正原跨年取秩错误）

`s6_v1_repro_check.py` 原先把整年 72,900 行一次交给只接受单日的 scorer，导致 IC 口径错误。
现改为**逐日完整池调用后再拼接**，并新增契约守卫与检查：

| 检查 | 结果 |
|---|---|
| 六个原子 raw（逐 seed）vs 包内 | 全部 **0.0** |
| 家族原子 / 融合输出 / 日秩 | 全部 **0.0** / 一致 |
| 与冻结锚点 | keys 完全匹配，max abs diff **0.0** |
| rankIC（复现 vs 锚点） | **0.064410371**（两边一致，与审查给的 0.064410371 相同） |
| 混合日期输入 | **被拒绝**（scorer 单日契约） |
| `tabm_design.json` / config | 哈希一致 / 一致 |

证据 `v1_repro_check.json`（`passed=true`）。

### 4.3 manifest 硬校验

`s6_v1_manifest_check.py` 重写为 22 项检查：成员名集合、哈希键集合**恰好等于**全部成员、
原子记录绑定、专家与 seed 身份、权重键集合、融合图（含"无额外家族 rank"）、训练截止 /
验证年 / 预处理截止、特征 61 列顺序、`tabm_config` 架构、design 哈希。
**产包流程 `s6_v1_package.build()` 直接调用该校验，失败抛 `SystemExit`（非零退出）。**

负例 8/8 全部被检出：缺成员、空哈希表、额外家族 rank、未知权重键、非法权重、
错误训练截止、重复 seed 哈希、特征顺序错。证据 `v1_manifest_check.json`（`all_pass=true`）。

包内静态加载器同步换成研究版严格实现（全键集、dtype、有限性、整数 k、正维度、逐数组 shape）。

### 4.4 训练截止身份

meta 现显式分列 `train_label_cutoff=20201231`、`validation_year=2021`、`preprocessing_fit_cutoff=20201231`，
不再用含混的单一 `cutoff`。现行包与权重**冻结归档为 2021 年度夹具**，不覆盖、不转挂新身份：

| 归档 | SHA-256 | 说明 |
|---|---|---|
| `V1_pkg_REV0.zip` | `8f30e8ce…20ee1` | 修复 `train.py` 之前 |
| `V1_pkg_2021fixture.zip` | `fac8f67c…7b15a` | 修复 `train.py` 之后（冻结夹具身份） |
| `V1_pkg.zip`（现行） | `ea82d562…e3dca` | 补正代码/身份后；**权重文件与上两版逐字节相同** |

真正 `label_end<=20211231` 的部署模型与预处理，按审查 §4.4 安排在 **V2 候选冻结前**生成
（训练入口须支持"只训练保存、不评价外层"，避免提前打开 2022）。

### 4.5 随包源码与依赖

- 删除通用 `tabm_official.py`（含 `getattr(nn, activation)` 动态分派），改为**显式固定架构 + `nn.ReLU`**
  （`alpha_train_model.py`），保留 `LICENSE.tabm` 与新增 `TABM_SOURCE.md` 来源说明（Yandex TabM v0.0.3，Apache-2.0，commit `28e47ae3…`）。
- 等价性证明（`v1_tabm_equiv.json`）：参数布局、初始化、eval/train 前向（含 dropout）、
  损失、梯度、3 步 AdamW **全部 0.0 差异**。
- 用显式架构重训单折：产出 `.npz` 与包内**逐字节一致**（sha `4ef2b818…`），design 一致；
  在 `/tmp/v1venv` 与全新环境 `.venv_v1train` 两次均通过（`v1_tabm_retrain_check.json`）。
- 训练依赖**安装来源与顺序已记录并实测**（`deps/V1_train_install.log` 为原始 stdout/stderr）：
  `torch==2.14.1+cpu` 必须先以 `--extra-index-url https://download.pytorch.org/whl/cpu` 安装，
  否则 `rtdl_num_embeddings` 会从 PyPI 拉入带 CUDA 的 torch（本轮实测复现该现象）。
- `deps/V1_run_install.log` 已标注为**人工整理摘要**，不是原始日志。

### 4.6 比较完整性

**粗权重网格（仅 DEV 选权重）**：`artifacts/stage4_verified_ic/v1_anchor_compare.json`

| TabM 权重 | 0.0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| DEV 合并 IC | 0.081645 | 0.083758 | **0.084650** | 0.083935 | 0.081665 |

DEV 最优即 0.5，相对默认 0.5 的增量为 **0.0** → 按"差<0.0003 取简单版本"保留
`V1_FIXED_ANCHOR(0.5/0.5)`，**不建立** `V1_SELECTED_ANCHOR`；2021 只作冻结复核（0.064410）。

**删除消融（冻结权重）**，DEV19 ΔIC：`drop_c1_20261005 −0.002272`、`drop_c1_20261017 +0.001244`、
`drop_c1_20261029 −0.000153`、`drop_tabm_20261005 −0.000312`、`drop_tabm_20261017 +0.000103`、
`drop_tabm_20261029 −0.000221`。

**配对移动块自助**（固定 seed 20261005，2000 次，block 10/20/40）：

| 比较 | ΔIC | block20 区间 |
|---|---:|---|
| DEV 合并 vs C1 三 seed | +0.003005 | [0.000351, 0.005494] |
| DEV 合并 vs TabM 三 seed | +0.002985 | [0.000525, 0.005709] |
| 2021 vs C1 三 seed | +0.000506 | [−0.002007, 0.003011] |

结论：DEV 对单体的优势区间下界为正（block10 下界约 +0.0003，接近零）；**2021 区间跨零**，
属已用开发数据上的条件性证据，不宣称稳定领先。

**ResMLP N0 控制（正确重拟合，单 seed，两份 DEV）**：full artifacts 已保存。

| 窗口 | 最优 epoch | 单体 IC | 相对锚点最佳边际 ΔIC |
|---|---:|---:|---|
| DEV19 | 28 | 0.062785 | −0.000025（α=0.05） |
| DEV20 | 13 | 0.077975 | +0.000246（α=0.10） |

两项均 < 0.0003（近似持平），按审查规则**不扩 seed/2021**。证据 `v1_resmlp_marginal.json`、
`v1_resmlp/`。`v1_anchor_analysis.json` 的 `p0_epoch` 已改为从各折 atom manifest 读取（15/13/12）。

**冷启动拆分**（修正混称）：加载 0.078 s、首次预测 0.118 s、冷启动合计 **0.196 s**；
暖启动 P50 0.104 / P95 0.113 / max 0.129 s；峰值 RSS **388 MiB**。

## 2. 修复 `train.py` 复现（上轮硬缺口，已闭合）

原因：写死 `C1_TREES=875`，且权重按全体 finite 行归一，而产包协议是**逐 seed 内层 IC 选轮**
（875/650/1225）+ **fold-local 权重归一**（`l2_leaf_reg=15>0` 使尺度不可忽略）。
训练输入已先核验：重建面板与 `panel_platform.parquet` **逐列逐值一致**（475,200×76，差异 0 列）。

从实际 zip 在工作区外执行 `train.py --data-dir Source/Data`：CatBoost 树数 875/650/1225、
内层最优 checkpoint 1050/825/1300、TabM 选轮 12/11/43 全部命中；CatBoost 去 CatBoost 自带
随机 `model_guid` 与 `train_finish_time` 后模型内容一致、2021 全期预测 0.0；TabM `.npz` 逐字节一致。

## 3. 干净环境推理验收（现行 zip 字节）

仅按交付 `requirements.txt` 安装的独立环境（无 torch），从 zip 解压目录外驱动真实入口：
2021 全期 243 天，每日 300 只全池、无重复、有限、三列、日期正确；vs 锚点 max abs diff **0.0**、
日秩全一致；包自身第三方 import `{catboost, scipy}` 均在依赖传递闭包内（越界项为空）；
真未来扰动（追加 2022 真实 500 行、close×3）输出不变；重复调用 / 跨调用状态 / 股票行置换均一致。
证据 `v1_run_acceptance_CONF21.json`。

## 4. V1 范围外（未开始）

V2（§10–13）：CatBoost C2/C3 与目标对照、TabM dropout/相关损失单变量、F1–F4 特征组、
TCN（含无时间支路控制）、有限融合与删除消融预算、真正 cutoff-2021 控制与最多两候选的实物冻结清单。
V3（§9）：2022 一次性确认与全量提交模型。**2022 性能至今未打开。**

## 5. 产物索引

| 内容 | 路径 |
|---|---|
| 本报告 | `handoff/DS_stage4_V1_REPORT.md` |
| 包与归档 | `releases/stage4_verified_ic/V1_pkg.zip`、`V1_pkg_2021fixture.zip`、`V1_pkg_REV0.zip` |
| 包身份/权重不变性 | `artifacts/stage4_verified_ic/v1_pkg_identity.json` |
| train.py 逐日复现 | `v1_repro_check.json` |
| 六成员年度入口 | `v1_annual_fixture_DEV19.json`、`v1_annual_fixture_DEV20.json` |
| manifest 硬校验 | `v1_manifest_check.json` |
| 干净环境验收 | `v1_run_acceptance_CONF21.json` |
| TabM 等价与重训 | `v1_tabm_equiv.json`、`v1_tabm_retrain_check.json` |
| 权重/删除/区间 | `v1_anchor_compare.json`、`v1_anchor_analysis.json` |
| ResMLP 控制 | `v1_resmlp_marginal.json`、`v1_resmlp/` |
| 依赖与环境 | `deps/V1_*`（含原始安装日志 `V1_train_install.log`） |
| 访问登记 | `access_registry.json` |
| 源码 | `src/qf3/s6_{nn_v1,batch,v1_fixture,v1_anchor,v1_package,v1_repro_check,v1_run_acceptance,v1_manifest_check,v1_annual_fixture,v1_tabm_equiv,v1_resmlp,v1_anchor_compare}.py`、`s4_tabm_wrapper_src.py`、`s6_v1_train{,_tabm}_src.py` |

**判定：V1 有限补正完成，交主 Agent 审查；不建议上传现行包（它是 2021 验证夹具，不是部署/全量模型）。**
