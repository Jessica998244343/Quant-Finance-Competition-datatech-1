# 第四阶段 V2 收尾修订报告（COMPLETION REV1）

日期：2026-10-07。依据 `REVIEW_stage4_V2_COMPLETION_20261007.md`（板块 A/B/C）。
**唯一选择目标是日均 RankIC；IR 仅报告。未计算任何 2022 性能，未封装提交包，未建议上传。**

## 第一屏

| 维度 | 结论 |
|---|---|
| 研究结果 | 三成员 `C1=0.375 / TabM=0.375 / TCN=0.25` 为**待确认候选（带起点敏感性警告）**；`V1_FIXED_ANCHOR` 为回退；二成员（删 TabM）**不晋级** |
| 工程复现 | 推理端改为**只读真实 provider 截断三表 + t + 保存权重/预处理**；三窗口代表日重建三 seed 原子与三成员融合，最大绝对误差 **0.0**；负例与未来不变性全部按预期 |
| 比较完成度 | A/B/C 三板块执行完毕：删除消融纠正、半年比较持久入口、机器台账、在线验收、冻结身份与四类拒绝负例 |
| 冻结完整性 | `v2_completion_frozen_manifest_v2.json`：33 个文件、路径唯一、含三家族权重/全部预处理/有序特征与设计/通道/训练选轮程序/依赖锁/结构化采用规则；缺必需项 build 失败 |
| 是否具备 V3 条件 | **待主 Agent 核对本报告**；本 Agent 判定收尾完成，不自行开放 V3 |
| 是否可上传 | **不建议上传任何现有包** |

## 1. 逐条回应审查问题

| 审查条目 | 处理 | 证据 |
|---|---|---|
| §3 删除消融权重写错（两个 0.5/0.5） | 改为"先删除成员再归一化"并加 assert；12/12 数值与审查表一致；错误的 0.5/0.5 产物**保留为错误记录**，不覆盖 | `v2_completion_deletion_v2.json`（旧 `v2_completion_deletion.json` 保留） |
| §2 B4 自动淘汰条件被主 Agent 撤销 | 保留原命中记录，新增字段说明"自动淘汰撤销、敏感性警告保留"，**不把旧检验写成通过** | `DS_stage4_V2_COMPLETION_REV1_REPORT.md` §2、进度日志 |
| §4.1 输入重建仍依赖缓存、且未把覆盖/池/有限/秩纳入总判定 | 新增 `s6_v2_input_acceptance.py`：推理端仅用真实 provider 截断三表 + t + 保存权重/预处理；显式断言期望日数=执行日数；总 passed 为覆盖∧池∧有限∧数值/秩∧融合秩∧不变性的合取 | `v2_completion_input_acceptance.json`、`v2_completion_input_negatives.json` |
| §4.2 冻结校验不核对 manifest hash，且允许 missing:true 豁免；遗漏 constants/cv/训练程序/依赖锁 | 重写冻结与校验：核对 canonical hash 与清单文件 SHA-256；missing 项直接判失败；补齐 11 个代码文件与 2 个依赖锁；路径去重 | `v2_completion_frozen_manifest_v2.json`、`v2_completion_freeze_fixtures_v2.json` |
| §4.3 `family_q` 无 expected_keys 时仍缩池 | 改为**无条件逐 seed 与首个 seed 键集对比**（正反序都拒绝），业务调用方继续传独立期望池 | `s6_fusion.py`、`v2_rev1_fusion_contract.json` |
| §4.4 台账自相矛盾、旧 TCN 汇总入口坏、无持久半年入口、训练目录可覆盖 | 台账四层机器生成并 assert；旧 TCN 跨窗口汇总废弃并指向新入口；新增 `s6_v2_halfyear_compare.py`；新增 `s6_io.new_run_dir` 默认拒绝覆盖 | `v2_completion_ledger_v2.json`、`s6_v2_completion.py`、`s6_v2_halfyear_compare.py`、`s6_io.py` |
| §4.4 `final-stats` 只输出摘要、无逐日产物 | `--command final-stats` 同时保存逐日 IC/ΔIC 与有效标签掩码规则 `v2_completion_final_stats_daily.parquet`（730 行）；B1 入口保存 `v2_completion_halfyear_daily.parquet` | 两个 parquet |
| §4.4 临时脚本产物无持久来源 | 三个 H2 的单 seed/三 seed 比较全部由 `s6_v2_halfyear_compare.py` 复算；原先由临时脚本生成的 `v2_completion_halfyear_start.json` 改名 `*__superseded_ad_hoc.json` 保留 | 同上 |

## 2. 候选结论更正（保留原判定 + 修订说明）

- **原记录（保留）**：B4 事前规则下，2020H2 的"同起点候选−父组合"比年度起点下降 −0.002119470（>0.001），2021H2 单 seed 也触发；据此上一轮把 TCN 标为"半年起点无大幅反向未通过"。
- **主 Agent 修订（新字段）**：该"增量下降>0.001 即自动淘汰"的判据被主 Agent**撤销**（见审查 §1/§2），因为年度→半年时**三成员自身 IC 反而上升**（2020H2 +0.001039045）且仍强于同起点 V1；正确解释是**TCN 的边际优势对拟合起点敏感**，不是模型失效。
- **当前判定**：三成员保留为**待确认候选、附敏感性警告**；**不声称它通过了原先未变的预注册门槛**，也不据此把历史表改写为通过。
- **次要结论**：二成员（删 TabM，C1 0.6/TCN 0.4）在三个 H2 起点均弱于三成员（−0.00139 / −0.00179 / −0.00173），**不晋级**；也不再试 0.55/0.45 等新向量。

## 3. 纠正后的删除消融（年度，三 seed，相对三成员）

| 删除方案 | 向量 | DEV19 | DEV20 | DEV 合并 | 2021 |
|---|---|---:|---:|---:|---:|
| 删 C1 | tabm 0.6 / tcn 0.4 | −0.004352000 | +0.000145566 | −0.002107835 | −0.002365773 |
| 删 TabM | c1 0.6 / tcn 0.4 | +0.001070823 | −0.000709426 | **+0.000182527** | +0.000413661 |
| 删 TCN | c1 0.5 / tabm 0.5（V1） | −0.000394856 | −0.003107879 | −0.001748582 | −0.000051006 |

删 TabM 的 DEV 合并改进 **+0.000183 < 0.0003**（近似持平），且两 DEV 方向分化 → 不能宣称"每年都更好"；
是否保留 TabM 由其在起点敏感性下的稳定作用与主 Agent 判断决定，本 Agent 保留三成员、不晋级二成员。
上一轮"删 TabM DEV +0.000594"的表格来自错误权重，**已撤回**。

**同一规定向量在 H2（复用已训练 H2 模型）**

| H2 窗口 | seed 数 | 二成员 IC | 二成员−V1 | 二成员−三成员 |
|---|---:|---:|---:|---:|
| 2019H2 | 1 | 0.067533486 | −0.000779552 | −0.001392271 |
| 2020H2 | 3 | 0.098166441 | −0.000619542 | −0.001788250 |
| 2021H2 | 3 | 0.053982913 | −0.001816025 | −0.001725780 |

证据 `v2_completion_halfyear_compare.json`（含逐日 IC/ΔIC 与共同日期，`v2_completion_halfyear_daily.parquet`）。

## 4. 在线推理验收（B1–B3）

**推理端只允许真实输入**（`s6_v2_input_acceptance.py`）：从 `DataProvider.as_of(t)` 取截断三表 →
`featurelib.build_day_features` 生成静态特征 → 用 provider 行情窗口重建 **21 价格点/20 步**序列 →
加载保存的 C1/TabM/TCN 权重与各自预处理 → 计算三 seed 原子、家族日秩均值与三成员融合。
**不读取 `panel_platform` 或预测缓存**；参考侧才读缓存用于比较。

| 窗口 | 代表日（起始日 + 真实成分变更邻近日） | 池/有限 | 原子 max abs diff | 融合 max abs diff | 日秩/融合秩 |
|---|---|---|---:|---:|---|
| DEV19 | 2019-01-02、2019-06-14 | 通过 | **0.0** | **0.0** | 一致 |
| DEV20 | 2020-01-02、2020-06-12 | 通过 | **0.0** | **0.0** | 一致 |
| CONF21 | 2021-01-04、2021-06-11 | 通过 | **0.0** | **0.0** | 一致 |

**必拒/必不变**：缺样本日、缺股票、NaN 原子、篡改参考 → 全部被拒；追加 500 行 t 之后真实行情 →
t 输出不变；总 `passed` 为覆盖、池、有限性、数值/秩、融合秩与不变性检查的**合取**。
证据 `v2_completion_input_acceptance.json`、`v2_completion_input_negatives.json`。

## 5. 冻结身份与清单自校验（C1/C2）

- `v2_completion_frozen_manifest_v2.json`：**33 个文件、路径唯一**；覆盖三家族×各 seed 的权重、
  TCN `preprocessing.json`、TabM `design_seed*.json`、有序 61 特征/141 维设计/8 通道定义、
  训练与选轮程序（含 `constants.py`、`cv.py`、特征生成与融合源码）、依赖锁
  （`requirements.lock`、`V1_train_requirements.txt`）以及**结构化采用规则**。
- manifest_hash `cb3010ee90d1579a…`；清单文件 SHA-256 `2686577f0b738e94…`
  （构建后曾编辑两个被绑定的源文件，`--verify` 正确报出 hash mismatch，重建后已重新校验通过；
  这正是该校验器应有的行为）。
- 负例（隔离副本）：干净副本通过；**改文件、篡改 `training_config`（hash 不变）、删必需文件、
  用 `missing:true` 伪装缺失** 四类全部被拒。证据 `v2_completion_freeze_fixtures_v2.json`。

## 6. 台账与预算（机器生成）

`v2_completion_ledger_v2.json`（四层分开、计数自明细计算并 assert）：
配方 **13**（选择用 11、其中新配方 7）、运行 **58**、完整向量 **43 行 / 38 个去重身份（≤40）**、
最终选择向量 5、2021 访问 14 条、2022 访问 3 条（**均无性能**）。
成员初筛向量与最终选择向量分开计数；错误消融与失效尝试保留可追溯；`tabm_recent504` 标注产物缺失。

## 7. 未完成 / 下一步

- 未计算任何 2022 性能；旧清单与旧报告仅作历史记录。
- **唯一受支持汇总入口**：`python -m qf3.s6_v2_completion --command family --windows DEV19,DEV20[,CONF21]`；
  旧 `s6_v2_tcn.py` 跨窗口汇总已废弃并给出指引。
- V3（待主 Agent 核对后释放）：按清单内结构化采用规则执行 2022 一次性确认 → 冻结算法全量拟合 →
  提交包复现与 CPU/源码验收 → 固定主费用 IR（2019–2020 连续账户 + 2021）→ 交付时报告该包 IC/IR。

**判定：V2 收尾修订完成（A/B/C），交主 Agent 审查；不建议上传。**
