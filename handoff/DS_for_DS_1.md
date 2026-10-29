# 交接文档：Quant-Finance3 执行 Agent 交接（DS_for_DS_1）

> **状态更新（2026-10-06，接手 Agent 追加指针）**：本文 §5 记录的硬缺口
> "`train.py` 不能复现包内成员" **已修复并机器验证通过**；V1 现已达到本地技术验收。
> 本文正文作为交接时点的历史记录保留，不再代表当前状态；当前状态以
> `handoff/DS_stage4_V1_REPORT.md` 为准（证据：`artifacts/stage4_verified_ic/`
> 下 `v1_repro_check.json`、`v1_run_acceptance_CONF21.json`、`v1_manifest_check.json`、
> `v1_package_identity.json`）。

日期：2026-10-06（Asia/Shanghai）。作者：前一任执行 Agent。接收方：同级执行 Agent。
**本文只交接，不含任何新训练或新实现**（用户本轮明确要求）。

## 0. 第一屏：接手后立刻要知道的事

| 问题 | 回答 |
|---|---|
| 项目是什么 | 沪深300指数增强 + 截面选股竞赛：每日为当日沪深300成分股输出 `date, code, score`，主办方用统一组合优化 + 回测评分 |
| 现在处在哪个阶段 | 第四阶段（任务书 `handoff/Prompt_for_DS_stage4_verified_IC_improvement.md`，403 行）的 **V1 中途** |
| V0 状态 | **已完成并由主 Agent 关闭**（见 `REVIEW_stage4_V0_REV1_and_V1_entry.md`） |
| V1 状态 | **未完成、不能交付**。工程主体（9 折 TabM、年度夹具对照、V1 包、干净环境推理验收）已完成；唯一硬缺口是 **包内 `train.py` 不能复现包内 CatBoost 成员** |
| 我这一轮做了什么 | 按任务书写了本文档；未训练、未实现其他内容 |
| 接手后的第一件事 | 修 `releases/stage4_verified_ic/build/V1_pkg/train.py` 的 CatBoost 复现路径：改为**逐 seed IC 选轮树数**（875/650/1225），并对齐训练样本/权重/线程；随后完成 TabM 六成员逐参数对照；两者都通过后才可把 V1 标为完成并交主 Agent 审查 |
| 允许自行上传吗 | **不允许**。任何包先交主 Agent 独立审查；上传只由用户完成 |

阅读优先级：**用户最新指令 → 第四阶段任务书 → 本文 → 更旧的任务书/报告**。
`handoff/DS_SUBMISSION_STATUS_REPORT.md` 已标注被取代，只作历史记录；
其中“提交包不得依赖 torch”“失败原因已定位”**不是有效规则**，不得据此继续执行。

---

## 1. 项目整体目标与评测口径

### 1.1 任务

- 用主办方提供的 2016–2022 官方数据训练模型，对每日沪深300成分股打截面分数。
- 标签：H1 = `O[t+2]/O[t+1] - 1`（t 收盘信号，t+1 开盘交易）。**用户明确：真实预测标签的价基是复权 open（`MarketData.open`）**，不是 close。
- 主办方用统一组合优化 + 回测（B2）在隐藏期评测。官方总分 = 0.70 × 标准化 MeanRankIC + 0.30 × 标准化组合 IR。
- 本地没有评测机回测，只有本地回测（`vendor/skd-backtest`，commit `ec5d35872fb3dec58d385039c8c18787460c18a9`，v1.0.1）。

### 1.2 提交规范（来源：`Source/Submission_Rules.pdf`）

- 提交包必须自包含：`inference.py`、`train.py`、`requirements.txt`、模型权重及全部已拟合参数。
- Python 3.11、CPU 运行、评测期禁网、推理不重新 fit、依赖按提交的 `requirements.txt` 锁版本预装。
- 输出恰好三列 `date, code, score`；`date` 为 `YYYY-MM-DD` 字符串；覆盖当日**完整合法池**（停牌也给分）、无重复、无 NaN/Inf。
- 禁止动态执行、反射规避审查、修改 `sys.modules`、子进程、包外数据读取、跨调用文件状态。
- 包体上限 1024 MB（用户确认）；项目内部保守目标：zip 与解压均 ≤ 900 MiB（自身工程目标，非官方限制）。
- 允许 Neural Networks；评测机无 GPU，但**没有材料支持“禁止 torch”**。

### 1.3 用户/项目约定（非官方规则，但必须遵守）

- 名称、`is_st` 不得进入模型特征。`is_st` 禁用是用户转述的主办方追加规则；也不得用名称反推 ST。
- 所有模型/特征/超参/融合/候选选型**只看日均 RankIC**；IR 只计算、只报告，**不参与任何决策**（第四阶段任务书 §0）。
- 2021 是**已使用过的研究数据**，只能作稳健性复核，且每次访问要登记；不得称为独立确认集。
- 2022 在本轮冻结方案前**不做性能评估**，只在 V3 做一次性时间外确认（用只训练到 2021 的模型）。
- 用户手动上传；主 Agent 不调度其他 Agent；执行 Agent 只能判“本地技术验收通过/待审查”，不能判“建议上传”。

### 1.4 工作区与数据

- 工作区：`/home/jessica998244343/DATA/Quant-Finance3`。**不是 git 仓库**（`git status` 报 not a repository），没有版本控制兜底，一切以文件 + SHA-256 为身份依据。
- 数据：`Source/Data/` 下 `MarketData`、`MarketDataRawOpen`、`Factor33_winsor`、`Barra_factor`、`HS300_index`、`HS300_weight`、`HS300_industry`，各覆盖 2016–2022。
- 环境：`.venv`（树模型）、`.venv_nn`（torch CPU 研究环境）。
- 平台：`vendor/skd-backtest`；主费用与组合约束沿用 `configs/beginning/evaluation_contract.json` 与 `src/qf3/constants.py`（二者需保持一致，主 Agent 已修过 TOML 语法/相对路径/R_DEPLOY 目录，**不得回退这些修复**）。

---

## 2. 全局阶段与官方反馈现状

### 2.1 官方成绩（来源：用户 2026-10-06 评测截图，非官方 API）

| 提交 | 官方 RankIC | 官方 IR | 标准化 RankIC | 标准化 IR | 总分 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| strong_baseline_v1 | **0.0657** | 0.5524 | 0.7756 | 0.2339 | 0.6131 | 已成功提交 |
| ALT_v1 | 0.0642 | 0.7070 | 0.7578 | 0.2541 | 0.6067 | 已成功提交 |
| TREE_v1 | 0.0650 | **1.3057** | 0.7676 | 0.3321 | **0.6370** | 已成功提交 |
| FINAL_v1（两次同名上传） | — | — | — | — | 失败，exit code 1 | 无日志、根因未定位 |

历史观察（2026-10-05）：strong 总分 0.6125、标准化 IC 0.7747。两次观察都保留，**不篡改、不擅自解释标准化机制差异**。

**关键结论：官方 IC 最好的仍是 strong_baseline_v1（0.0657）；TREE_v1 是综合分改善，不是官方 IC 改善。** 本地 IC 提升不能被说成官方提升。

### 2.2 本地归档哈希（同名文件 ≠ 已核验上传字节）

| 包 | 路径 | SHA-256 |
|---|---|---|
| strong_baseline_v1 | `releases/strong_baseline_v1.zip` | `b34712933d952da40ff903804096e0f104ad501b598d643b335a7b31dc0f87c5` |
| ALT_v1 | `releases/stage3_fusion/ALT_v1.zip` | `5e794e86fdaefd06472545687ce083e60024724b7b49f08d165a996bb32d9b59` |
| TREE_v1 | `releases/stage3_fusion/TREE_v1.zip` | `11d07b8c1033a61b4f008d36a0fd98976747c8ad3794f32167046efe9c1e53e9` |
| FINAL_v1（现存的 CPU 索引修订版） | `releases/stage3_fusion/FINAL_v1.zip` | `dabb26c7c45ca50ae497680e8637602d201d68c688accffb0c82947ef9967999` |
| FINAL_v1 初版（**文件已被覆盖，只剩哈希记录**） | — | `64bfb8593df71dfd3a12febc934f6d0c47a07de4bfc8ab88acd3d7927020eb30` |
| baseline_v1 = B0 回退 | `releases/baseline_v1.zip` | `565c7907c7519d3f801237dcaee98a862c7aed2527a0ec27e1703b0ac8fcb171` |

用户截图未提供上传哈希，**不能凭文件名声称后台运行字节与某个本地文件一致**。

### 2.3 FINAL_v1 两次失败的证据边界（重要，别重复犯错）

- 观测：两次 `evaluation failed with exit code 1`，**无任何日志、无阶段信息**。
- 假设 A：初版交付普通 `torch==2.14.1`（PyPI Linux wheel 554.6 MB，解析出 29 个含 CUDA 的包）。
- 假设 B：改用 `--extra-index-url` + `torch==2.14.1+cpu`（本地解析 10 包、零 CUDA）后**仍然失败**。
- 反例：无法观测评测机安装阶段，假设 B 的失败**不能证明**是索引问题。“评测机不认 `--extra-index-url`”只是**未证实假设**。
- 处置：只修可复现的本地缺陷；不据假设反复换包试上传（用户手动上传，不消耗我方的验证额度）。

### 2.4 全局规划（6 阶段，摘自 `PROJECT_PLAN.md`）

1. 规则确认 + 本地环境 + 基线 + 强基线（已完成，产出 strong_baseline_v1 / ALT_v1 / TREE_v1 / FINAL_v1）。
2. 单模型与信息增量研究（已完成大部分，阶段 2/3 文档在 `handoff/`）。
3. 模型互补与稳健性、融合与组合表现（TREE_v1 / FINAL_v1 出自此阶段）。
4. 候选交付与官方反馈循环（**当前阶段**）。
5. 最终选型与发布冻结。
6. 私榜、人工审查与收尾。

阶段 2–4 可循环；最终冻结才意味着停止选型。

---

## 3. 当前阶段目标：第四阶段 V0–V3

任务书：`handoff/Prompt_for_DS_stage4_verified_IC_improvement.md`（阅读优先级最高，除用户最新指令）。
**唯一模型选择目标 = 日均 RankIC；IR 仅计算和报告。** PIT、工程正确性、可复现、CPU 可运行、包体限制仍是硬约束。

| 交付 | 必须完成 | 状态 |
|---|---|---|
| **V0：证据修正与 CPU 部署可行性** | 更新状态、冻结哈希、列影响范围、单一 manifest/融合定义、最小 TabM 静态 CPU 导出与依赖复现证据；不称恢复了旧 IC | **已完成，主 Agent 已关闭** |
| **V1：可信修正版基线** | 合法重训 NN、年度包入口全链路对齐、冻结修正版树/NN 组合、可审查 CPU 包、§8 训练程序比较 | **进行中，本地验收未通过（见 §5）** |
| **V2：有限 IC 研究与候选冻结** | §10–12 必做/触发分支、小范围时间外比较、冻结最多 2 个新候选与备选规则、2022 确认计划 hash | 未开始 |
| **V3：2022 一次性确认、全量模型和交付** | 一次性时间外确认、固定算法全量拟合、完整包复现与 CPU 验收、最终报告 | 未开始 |

每个交付都必须**分开**列出六项状态：工程完成 / 比较完成 / IC 增益 / CPU 包验收 / 主 Agent 审查 / 官方状态，**不得合并成一个“通过”**。

### 3.1 冻结融合图（反复出错的点，必须背下来）

```
原子模型原始分 s
  → 原子日秩 r_t(s) = (average_rank(s) - (n_t+1)/2) / n_t   （只在 t 日完整合法池计算）
  → 同配方多 seed：q_g = mean_seed r_t(s_g, seed)            （不再对 q_g 做一次家族排名！）
  → 融合：u = Σ_g w_g · q_g                                    （w ≥ 0，Σw = 1）
  → 提交输出 r_t(u)
```

- TabM 内部 k=8 是**参数共享集成**，先在原子内部对 8 个成员均值，作为一个原子模型，再参与日秩；k=8 **不是** 8 个独立 seed。
- 最终 rank 不改变 IC，但会影响下游组合，IR 必须用实际最终输出重算。
- 历史统计：多出一次家族 rank 会真实改变 IC（不能声称“不影响 IC”）。

### 3.2 第四阶段必做清单（V2/V3，别遗漏）

- **TCN 必做**（§12）：W20、静态支路保留、时间支路 32 通道 / 两个残差块 / 因果卷积 dilation 1-2、AdamW、max80 epoch/patience10、必须有**无时间支路的同预算控制**；只在有增量时才扩 W60。
- **特征组 F1–F4**（§11）：F1 近期反转与较早趋势分离（`C[t-1]/C[t-5]-1`、`C[t-5]/C[t-20]-1`，注意**不能**写成 `C[t]/C[t-1]-1`）；F2 价量条件关系（最多 4 列）；F3 隔夜/日内/路径持续性（最多 5 列）；F4 行业相对状态（接口不可得就登记跳过）。默认新增列 ≤ 18。
- **CatBoost C2/C3**（§10.1）：depth6/l2=15 与 depth8/l2=30。
- **TabM 单变量对照**（§10.2）：dropout .1→.2；MSE + 0.1×(1-Pearson) 日内相关损失。各最多一项，不扫 k/宽度/层数/lr 组合。
- **融合与删除消融**（§13.2）：对父组合测 α ∈ {0, 0.05, 0.10, 0.25, 0.50}；最终联合融合 ≤ 40 个**不同完整权重向量**；禁止负权重/分年权重/自适应门控/stacker。
- **IC 证据标准**（§13.3）：相同日期/合法池/标签掩码，按日等权 Spearman（不是 pooled 相关）；逐日 ΔIC 的配对移动块自助（block20，10/40 敏感性，2000 次，固定 seed）；DEV ΔIC ≥ 0.0005 为保留目标，差 < 0.0003 视为近似持平并选简单版本（**IR 不参与平局裁决**）。
- **搜索预算**（§13.1）：V2 新增最多 18 个配方，预分配 Cat 4 / TabM 3 / 特征组 5 / 特征转 TabM 2 / TCN 3 / 余 1；先单 seed 两 DEV，再补 seed。
- 未触发、未完成的项要**明确写“未完成/跳过及理由”**，不能把任务缩成“够用”。

### 3.3 V3 的 2022 一次性确认边界（§9）

- 打开前必须先落盘只读冻结清单（最多 2 个新候选 + 优先顺序、精确特征/seed/权重/程序、截至 2021 的模型 hash、评价日/label 边界、成功失败解释、不回调承诺），保存清单 hash 并写入报告。
- 2022 评价必须用**只训练到 2021** 的模型；不能用已经全量训练到 2022 的 zip 制造“2022 样本外 IC”。
- 预设解释门槛：主候选 ΔIC > 0 且两个半年 ΔIC 均不低于 −0.001 → “时间确认支持”；否则“不支持/不稳定”。支持与否都不自动等于可上传。
- 确认后再按**打开前已冻结的算法**用 2016–2022 成熟标签做最终全量拟合；记录确认模型与最终模型各自的训练截止。

---

## 4. 我的工作（截至交接时点）

### 4.1 已完成：V0 独立交付（已被主 Agent 关闭）

- 状态纠正：撤回“提交包不得依赖 torch”“失败根因已定位”“树模型上限 ~0.0816”“融合图只影响 IR”等越界表述。
- TabM 静态 CPU 导出可行性：`tabm_static_fp32.npz`（源 `tabm.safetensors` 哈希 `0a81bcae00cc210162b4b69d768a42fa6bdcaf894d4c29a05c059e6dfce3814a`，NPZ 哈希 `e580b2fce4ee78b0f4554215befa3069560214d01839f3d771b446c11641e796`），九场景逐层等价，成员均值最大绝对误差 1.19e-07。
- 按主 Agent 的 REV1 要求补正：
  - 验收器“缺测也能通过”漏洞已修：声明必需九场景、拒空集/缺项/替换、先做 shape 校验、覆盖/形状/有限性/秩纳入总判定、负向样例全被拒（`v0_rev1_acceptance.json`，取代旧判定）。
  - 溯源哈希扩展到验收脚本/预处理实现/模型配置/参考前向/输入面板；加载器加 k≥1、d_in≥1、d_block≥1 严格正整数校验，负向 5/5 拒绝（`v0_loader_negative.json`）。
  - 报告生成器与报告的错误断言已同步修正（“两次同名上传均失败；本地仅保留 CPU 修订版”）。

### 4.2 已完成：V1 主体工程（9 折 + 年度对照 + 包）

**TabM 正确重训 9/9 折**：`artifacts/stage4_verified_ic/v1_tabm/tabm_pc_{seed}__{window}/`。
每折流程：内层独立选 epoch → 外层**重新初始化模型/优化器/RNG** + 外层重新 fit 预处理 + 从头训练选中 epoch → 立即落盘权重/预处理/内层曲线/manifest/预测（缺项即失败）。
**修正后的分数普遍低于旧 warm-start 版，这是如实结果，不要追着旧数调参。**

| 窗口 | seed 20261005 | seed 20261017 | seed 20261029 | 三种子日秩均值 |
|---|---|---|---|---|
| DEV19 | 0.075438 (ep15) | 0.067774 (ep34) | 0.073027 (ep9) | 0.076984 |
| DEV20 | 0.081463 (ep13) | 0.086376 (ep19) | 0.084389 (ep24) | 0.086365 |
| CONF21 | 0.052270 (ep12) | 0.061058 (ep11) | 0.062107 (ep43) | 0.061184 |

（对照：旧 warm-start DEV19 为 0.081463；旧 FINAL 研究分数 0.084721 **不可转挂**到本锚点。）

**年度入口全链路对照三窗口全通过**（`v1_fixture_{DEV19,DEV20,CONF21}.json`、`v1_fixtures/`）：
用真实包入口（平台 `spec_from_file_location` 私有包加载）+ 真实 provider 三表（`DataProvider.as_of` 前向遍历）；
全年 max abs diff **0.0**、日秩全一致、真未来扰动（+500 行、close×3）当日输出不变。
此步骤抓到并修掉两个**真实缺陷**：夹具曾用年度并集窗口（差 0.87）、scorer 曾只给 C1 取秩却把 TabM 原始分加权（差 0.41）。

**V1 锚点**（`0.5 × C1 三种子 + 0.5 × TabM 修正三种子`，冻结图）：

| 窗口 | TabM 三种子 | C1 三种子 | 锚点 |
|---|---|---|---|
| DEV19 | 0.076984 | 0.082152 | **0.082577** |
| DEV20 | 0.086365 | 0.081136 | **0.086732** |
| CONF21 | 0.061184 | 0.063904 | **0.064410** |

DEV 合并约 **0.08465**，高于 TREE_v1 结构控制（DEV 0.081559，+0.0031）与 R_DEPLOY（DEV 0.077805，+0.0068）。

**控制与选轮**：C1 三 seed 三个窗口全部存在且哈希互不相同（非复制）；P0 vs P1（近优 ≤0.0005 取最早）选中 epoch 完全相同（15/13/12），IC 差 0.0 → **保留 P0**；ResMLP N0 正确重拟合控制 DEV19 0.062785 / DEV20 0.077975。

### 4.3 已完成：V1 CPU 包（可运行，但训练不可复现）

`releases/stage4_verified_ic/V1_pkg.zip`（解压目录 `releases/stage4_verified_ic/build/V1_pkg/`，顶层前缀 `submission/`）：

| 项 | 值 |
|---|---|
| zip / 解压 | 8,787,672 B / 12,249,089 B / 27 文件 |
| SHA-256 | `8f30e8ce0e5a99c2575096fb2880686018fb928a435634c4c1472f1e48920ee1` |
| 成员 | CatBoost C1 × 3（`cutoff 2021` 年度模型，`label_end_date <= 2020-12-31`）+ TabM × 3（NumPy 静态导出） |
| 权重 | `{c1: 0.5, tabm: 0.5}`，图 `atom rank -> seed mean -> family weight -> final rank` |
| 运行依赖 | `numpy==2.4.6`、`pandas==3.0.6`、`pyarrow==25.0.1`、`catboost==1.2.10`（**无 torch**） |
| 训练依赖 | 同上 + `torch==2.14.1+cpu`、`safetensors==0.8.0`、`rtdl_num_embeddings==0.0.12` |

实测三个 `.cbm` 的树数分别为 **875 / 650 / 1225**（loss RMSE，seed 20261005/20261017/20261029），与包内逐 seed IC 选轮一致。

已通过验收：干净 venv（仅运行依赖）安装 + 导入 + 推理通过且**未导入 torch**；包输出 vs 锚点 max abs diff **0.0**；代表日 300 只全池、无重复、有限、约 0.12 s/日；manifest 机器校验 10/10；manifest 负向样例（缺成员/重复 seed 哈希/非法权重）全部检出；V0 REV1 验收器 `passed=True`。

---

## 5. 未完成任务（接手后的具体待办）

### 5.1 阻塞 V1 的硬缺口：`train.py` 不能复现包内成员

在干净环境（补装 `rtdl_num_embeddings` 后）`train.py` 可运行并产出 6 个模型，但**不能复现包内成员**：

| 成员 | 复现差异 | 诊断 |
|---|---|---|
| CatBoost seed20261005 | max\|Δpred\| **2.4e-2**（树数 875/875） | 树数一致但预测不同 → 训练样本或权重/线程协议仍有差异 |
| CatBoost seed20261017 | **2.7e-2**（树数 650/875） | **树数规则错**：包内是逐 seed IC 选轮（650），`train.py` 用了固定 875 |
| CatBoost seed20261029 | **4.4e-2**（树数 1225/875） | 同上 |
| TabM ×3 | 未完成逐参数对照 | NPZ 短键名与 safetensors 模块键名需映射后比较 |

根源位置：`releases/stage4_verified_ic/build/V1_pkg/train.py` 里 `C1_TREES = 875` 写死；包内实际身份是逐 seed 选轮（875/650/1225）。
另外已修：`train_requirements.txt` 原漏 `rtdl_num_embeddings`，现已补并重建包。

修复顺序（任务书 §14.1 第 3 层“最终包可复现”）：

1. 训练样本/权重/线程与产包时的实际身份对齐（含内层选轮所使用的合法尾窗口径）。
2. 改 `train.py` 为逐 seed IC 选轮，而非写死 875。
3. 目标：CPU 树模型**位一致**，或预先声明并解释的紧容差；NN 用预登记的数值容差。
4. 完成 TabM 六成员逐参数/逐预测对照。
5. 重跑 `train.py`，把结果写入 `handoff/DS_stage4_V1_REPORT.md`；未通过前不称 V1 完成、不给上传建议。

### 5.2 V1 剩余收尾

1. V1 年度夹具集合整理（三窗口夹具已在 `artifacts/stage4_verified_ic/v1_fixtures/`）。
2. 运行/训练依赖复现记录归档到 `artifacts/stage4_verified_ic/deps/`（目前该目录只有第三阶段的 `TREE_v1_*`、`FINAL_v1_requirements.txt` 等旧文件，**V1 的运行/训练日志尚未整理进去**）。
3. `DS_stage4_V1_REPORT.md` 已有草稿，第六项状态里“CPU 包验收”写的是“部分”，修好 `train.py` 后需要更新。

### 5.3 V2 未开始（见 §3.2 必做清单）

### 5.4 V3 未开始（见 §3.3）

### 5.5 环境/工具层面的注意

- `artifacts/stage4_verified_ic/v1_anchor_analysis.json` 里 `tabm_seed_means` 为空对象（种子均值另有 parquet 产物），若要引用该 JSON 需先补齐或改引用源，别当成缺数据。
- 训练脚本入口：`src/qf3/s6_nn_v1.py`、`s6_v1_batch.py`、`s6_v1_anchor.py`、`s6_v1_fixture.py`、`s6_v1_package.py`，以及随包源码模板 `s6_v1_train_src.py` / `s6_v1_train_tabm_src.py`。

---

## 6. 行为约束（继承清单，必须逐条遵守）

### 6.1 选型与证据

1. **只看 IC 选型**：特征、超参、轮数、架构、权重、候选晋级一律不看 IR；IR 只计算和报告（用户与任务书双重明确）。
2. 不把本地 IC 提升说成官方提升；不把假设写成禁令；不写“唯一可能”“全部通过”这类超出证据的表述。
3. 每个断言必须标注证据类型：官方规则 → PDF 页码；官方成绩 → 用户反馈；本地实测 → artifact 路径；推断 → 写“分析/假设/尚待验证”。
4. 报告第一屏必须回答：这是什么模型；相对哪个固定参照改进；验证与包是否同一身份；还有什么未做；可否进入主 Agent 提交审查。必须附机器结果路径，不能只给截图或最高分。

### 6.2 PIT 与工程正确性（禁止项）

- 禁止跨日排名；禁止用未来标签筛样本；禁止在验证集上拟合预处理统计；禁止用测试年份选 epoch/树数。
- 禁止用年度并集替代 provider 每日可见性；历史市场信息依赖最多 120 个交易日。
- 禁止 `inner join` 隐藏缺股、禁止 `np.where(isfinite, score, 0)` 或 `drop_duplicates` 静默掩盖实现错误；短历史缺失应在特征/模型规则层确定性处理。
- 禁止动态执行、`sys.modules`、子进程、包外数据读取、跨调用状态；包内只保留所用模块 + LICENSE + 来源 commit。
- 验收器不得“缺测也能通过”；JSON 的 `passed` 必须由检查汇总生成，禁止手写“全部通过”掩盖 false/N/A/未运行项。
- 真正做未来不变性要**加入 t 之后的未来行并大幅扰动**；对已截到 t 的数据再截一次不算检验。

### 6.3 交付与上传流程

- 执行 Agent 只能判“本地技术验收通过/待审查”；**不能判“建议上传”**。
- 任何新包必须先交主 Agent 独立审查；**上传只由用户完成**。
- 每位交付都分开列六项状态（工程完成 / 比较完成 / IC 增益 / CPU 包验收 / 主 Agent 审查 / 官方状态）。
- 未完成就标未完成，交清晰阻塞理由，不伪造完成、不缩小任务。

### 6.4 报告与沟通习惯（用户明确要求过）

- 报告用中文。
- 严格按任务书规定的交付节点交付；**不要额外报告节点**；一旦决定交付，先自审任务书要求的交付前全部内容是否完成，再交付。
- 不要自由发挥、不要自作主张扩大范围（用户多次强调“严格按照任务书实施”）。普通实现细节可自主决定，涉及标签、验证协议、禁用字段、回测口径的改动必须明确登记。
- 用户对效率敏感（工作曾被反复打回）：不要扩大成全面审计；A/V0 类交付保持精简，只回答“差异多大、是否需要重训、主要问题是什么、下一步优先哪两条路线”。

### 6.5 资源降载（用户硬约束）

- **CPU 线程数不得超过 12**（用户硬约束，即使允许开 subagent 也一样）。
- 内存不要跑满：机器约 15 GiB；研究期峰值目标 ≤ 8 GiB，提交期 ≤ 4 GiB；单个重任务串行。
- 既定降载配置：树 6 线程；`BLAS/OMP/MKL = 2`；NN CPU 2 线程。
- 每折约 3.5 分钟，**一折一命令**；不要用 `nohup` 跑长批处理（用户看不到输出会误判卡死，曾因此误以为任务卡住）。
- 推理工程目标（非官方上限）：日全池暖推理 P95 ≤ 2 秒。

---

## 7. 代码仓库外无法获知的上下文

这些是读代码看不出来、但接手必须知道的信息：

1. **官方成绩来自用户截图**，不是 API；截图没有上传哈希，所以“本地文件 = 后台运行字节”永远只能是有条件的推断。
2. **FINAL_v1 两次失败无日志**；初版 zip 文件已被覆盖、只剩哈希，任何“根因已定位”的说法都是回溯性假设。
3. **“提交包不得依赖 torch”已被撤回**。第四阶段明确允许合法的 PyTorch CPU 路线；但 FINAL 的两次失败确实是本项目推进神经网络交付的主要顾虑来源，选用 torch 路线要准备真实制品与干净环境证据。
4. **TREE_v1 已成功上传一次，之后本地包被修改过**（用户问过是否需要重新提交）。本地 `TREE_v1.zip` 的哈希与官方成绩不一定对应同一字节；不要把本地修改后的包当成“已提交的那一版”。
5. **strong_baseline_v1 存在已知合规问题**：`train.py` 用 `sys.modules` 注册 + 动态加载（审查已点名）；它虽然官方成绩最好，但不能作为新包的封装模板。
6. **旧 0.084721 不可转挂**：那是旧 warm-start 训练流程的 FINAL 研究分数；修正协议后 TabM 分数下降是真实结果（任务书 §7.3 明确要求如实交付）。
7. **DEV 合并、2021、TREE 结构控制是三个不同口径**：R_WORK（旧语义 DEV 0.077268 / 2021 0.061007）、R_DEPLOY（真实输入桥接 DEV 0.077805 / 2021 0.062708）、TREE_ORIGINAL_LOCAL（DEV 0.081559）。比较时不要混口径。
8. **融合图的历史差异是真实事故**：研究口径曾对种子均值再 rank，包内直接融合种子秩均值；FINAL 包还缺最终 rank。旧图保留独立身份，**不静默“修复”旧成绩**。
9. **R_WORK/MAIN/TREE/ALT 各自保留身份**，不重新挂官方成绩；`R_DEPLOY` 引用 A REV1 目录并核对哈希。
10. **评测机行为未知**：Python 3.11、按 `requirements.txt` 预装、评测期禁网，但预装是否有外网、是否有 `--extra-index-url` 支持都**未证实**。
11. **用户手动控制上传节奏**；本会话没有确认每日提交次数，不要沿用 `others/SHARE.md` 里的数字。
12. **`others/SHARE.md` 只是参考**：其旧项目路径、成绩、规则转述和环境快照未经本项目核验，不能当官方事实。
13. **资源环境真实值**（规划时实测）：CPU 为 Ryzen 7 8845H（8 核 16 线程），内存约 15 GiB（可用约 13 GiB）；`nvidia-smi` 曾被操作系统阻止，GPU 可用性未确认。
14. 用户曾因电脑意外重启怀疑 CPU 跑太满，之后要求“降载重跑”——线程/内存约束是**用户真实关切**，不是形式条款。

---

## 8. 关键文件索引

| 类别 | 路径 |
|---|---|
| 第四阶段任务书（最高优先级） | `handoff/Prompt_for_DS_stage4_verified_IC_improvement.md` |
| 主 Agent 审查 | `handoff/REVIEW_stage4_V0_and_V1_direction.md`、`handoff/REVIEW_stage4_V0_REV1_and_V1_entry.md` |
| 阶段报告 | `handoff/DS_stage4_V0_REPORT.md`、`DS_stage4_V1_REPORT.md`、`DS_stage4_V1_PROGRESS.md` |
| 历史（作废入口，仅历史记录） | `handoff/DS_SUBMISSION_STATUS_REPORT.md`、`handoff/DS_submission_failure_diagnosis.md`、`handoff/DS_two_versions_report.md` |
| 早前阶段 | `handoff/Prompt_for_DS_beginning.md`、`Prompt_for_DS_stage2_IC_IR.md`、`Prompt_for_DS_stage3_IC_first_models.md`、`Prompt_for_DS_stage3_rounds_to_fusion.md`、`DS_stage3_fusion_D1/D2/D3_REPORT.md` |
| 规划 | `handoff/PROJECT_PLAN.md`、`handoff/PHASE1_TASK.md` |
| V1 包 | `releases/stage4_verified_ic/V1_pkg.zip`、`releases/stage4_verified_ic/build/V1_pkg/` |
| V1 TabM 年度折 | `artifacts/stage4_verified_ic/v1_tabm/tabm_pc_{seed}__{window}/` |
| 锚点/种子均值 | `v1_anchor__*.parquet`、`v1_tabm_seedmean__*.parquet`、`v1_c1_seedmean__*.parquet`、`v1_anchor_analysis.json` |
| 年度入口对照 | `v1_fixture_{DEV19,DEV20,CONF21}.json`、`artifacts/stage4_verified_ic/v1_fixtures/` |
| 控制 | `v1_resmlp.json`、`v1_resmlp__*.parquet`、`v1_manifest_check.json` |
| V0 验收 | `v0_rev1_acceptance.json`、`v0_loader_negative.json`、`v0_acceptance.json`、`tabm_static_fp32.npz` |
| 源码 | `src/qf3/s5_{tabm_export,v0_acceptance,v0_report}.py`、`s6_{nn_v1,batch,v1_fixture,v1_anchor,v1_package}.py`、`s6_v1_train{,_tabm}_src.py` |
| 数据/规则 | `Source/Competition_Rules.pdf`、`Source/Problem.pdf`、`Source/Submission_Rules.pdf`、`Source/Data/README.pdf` |

---

## 9. 交接自查

- [x] 项目整体目标、评分口径、提交规范、用户约定已写明（§1）。
- [x] 全局阶段位置与官方反馈现状（含时间戳与来源性质）已写明（§2）。
- [x] 当前阶段 V0–V3 目标、冻结融合图、必做清单、2022 边界已写明（§3）。
- [x] 我的工作（V0、V1 主体、V1 包）与所有关键数字/哈希/路径已写明（§4）。
- [x] 未完成任务给出可直接执行的下一步，定位到具体文件与具体数值差异（§5）。
- [x] 行为约束（选型、PIT、交付流程、报告习惯、资源降载）逐条列出（§6）。
- [x] 代码仓库外无法获知的上下文（失败无日志、撤回的结论、上传身份、口径混杂、环境实测值、用户关切）已写明（§7）。
- [x] 本轮仅写本文档，未训练、未改实现、未上传。

**结论：V1 本地技术验收未通过（`train.py` 复现失败），不可交付、不建议上传；接手后第一优先是修 `train.py` 的 CatBoost 逐 seed 选轮与训练协议对齐，其次是 TabM 六成员逐参数对照。**
