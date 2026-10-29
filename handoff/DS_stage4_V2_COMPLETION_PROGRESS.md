# V2 收尾进度日志（对照 `REVIEW_stage4_V2_COMPLETION_20261007.md`）

规则：新运行独立 ID、**默认拒绝覆盖**（`--allow-existing` 才能复用）；被取代的文件只改名/加
superseded 标记，不覆盖旧结论；每完成一项立即追加。**2022 未开放。**

## A. 修正研究证据与台账

| 项 | 状态 | 完成时间 | 命令 | 主要输入 → 产物 | 结果 |
|---|---|---|---|---|---|
| A1 固定删除算法 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_completion --command deletion --windows DEV19,DEV20,CONF21 --alpha 0.25` | `v1_c1/tabm_seedmean`、`tcn_rev1_tcn_s*_{window}` → `v2_completion_deletion_v2.json` | 删C1 {tabm .6,tcn .4}、删TabM {c1 .6,tcn .4}、删TCN {c1 .5,tabm .5}；12/12 数值与审查表一致（DEV19 −0.004352/+0.001070823/−0.000394856；DEV20 +0.000145566/−0.000709426/−0.003107879；DEV合并 −0.002107835/+0.000182527/−0.001748582；2021 −0.002365773/+0.000413661/−0.000051006）。旧 `v2_completion_deletion.json`（0.5/0.5 错误消融）保留 |
| A2 半年比较入口 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_halfyear_compare` | 年度+H2 的 C1/TabM/TCN 预测（含 SHA-256）→ `v2_completion_halfyear_compare.json`、`v2_completion_halfyear_daily.parquet` | 二成员−V1：2019H2 −0.000779552、2020H2 −0.000619542、**2021H2 −0.001816025**；二成员−三成员 −0.001392271/−0.001788250/−0.001725780；start_sensitivity −0.000130941/−0.00211947/+0.000715921（全部与审查一致）；保存逐日 IC/ΔIC、共同日期与掩码规则 |
| A3 候选结论更正 | 完成 | 2026-10-07 | — | `DS_stage4_V2_COMPLETION_REV1_REPORT.md` §2 | 保留原 B4 命中记录；新增字段记录"主 Agent 撤销自动淘汰、敏感性警告保留"；三成员=待确认候选，V1=回退，二成员不晋级（H2 实际损失）。旧结论标 superseded |
| A4 机器台账 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_ledger` | 实际产物目录扫描 + 向量枚举 → `v2_completion_ledger_v2.json` | 四层分开、计数从明细计算并 assert：配方 13（选择用 11、新 7）、运行 58、向量 43 行 / **38 个去重身份（≤40）**、最终选择向量 5、2021 访问 14、2022 访问 3（均无性能） |

## B. 实际输入与会失败的验收

| 项 | 状态 | 完成时间 | 命令 | 产物 | 结果 |
|---|---|---|---|---|---|
| B1/B2 在线输入验收 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_input_acceptance --windows DEV19,DEV20,CONF21` | `v2_completion_input_acceptance.json` | 推理端**只读真实 provider 截断三表 + t + 保存权重/预处理**（不读 panel/预测缓存）；序列用 21 价格点→20 步；每日重建三 seed 原子与三成员融合；DEV19/DEV20/CONF21 各 2 个代表日（起始日 + 真实成分变更邻近日）**max abs diff 0.0**、池/有限/日秩/融合秩全通过 |
| B3 必拒/必不变 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_input_acceptance --negatives` | `v2_completion_input_negatives.json` | 缺日、缺股、NaN 原子、篡改参考**全部被拒**；追加 500 行 t 之后真实行情**不改变 t 输出**；总 passed = 覆盖∧池∧有限∧数值/秩∧不变性 的合取 |
| B4 公共函数与入口 | 完成 | 2026-10-07 | `python -m qf3.s6_fusion_negative` | `v2_rev1_fusion_contract.json` + 新增缩池负例 | `family_q` 无 `expected_keys` 时也逐 seed 对比首个 seed 键集（正反序都拒绝）；旧 `s6_v2_tcn.py` 跨窗口汇总已废弃并指向 `s6_v2_completion`；TCN/部署/特征新运行默认拒绝覆盖 |

## C. 冻结与集中交付

| 项 | 状态 | 完成时间 | 命令 | 产物 | 结果 |
|---|---|---|---|---|---|
| C1 完整身份 | 完成 | 2026-10-07 | `python -m qf3.s6_v2_freeze --build` | `v2_completion_frozen_manifest_v2.json`（manifest_hash `cb3010ee90d1579a…`，文件 SHA-256 `2686577f0b738e94…`） | 33 个文件、路径唯一；覆盖三家族×各 seed 的权重、TCN preprocessing、TabM design、有序 61 特征/141 维设计/8 通道、训练与选轮程序、日期窗口、依赖锁；含结构化采用规则；缺必需项 build 直接失败 |
| C2 校验清单 | 完成 | 2026-10-07 | `--fixtures` / `--verify` | `v2_completion_freeze_fixtures_v2.json` | 干净副本通过；**改文件 / 篡改 training_config（hash 不变）/ 删必需文件 / missing:true 伪装** 四类全部被拒；`verify` 核对 canonical hash 与文件 SHA-256 |
| C3 事前 2022 规则 | 完成 | 2026-10-07 | — | 同上清单 `adoption_rule` | 顺序：三成员 → V1；控制：固定 C1、TREE 结构；主比较 = 候选−V1（全年>0 且两半年 ≥−0.001，并相对 TREE 同条件），否则检查 V1−TREE；区间 10/20/40、按年抽样；IR 不裁决；敏感性警告保留 |
| C4 集中交付 | 完成 | 2026-10-07 | — | `DS_stage4_V2_COMPLETION_REV1_REPORT.md` | 逐项链接 A/B/C 证据与新结论 |

## 身份

- 冻结清单：`v2_completion_frozen_manifest_v2.json`（manifest_hash `cb3010ee90d1579a…`，文件 SHA-256 `2686577f0b738e94…`，33 文件）
- 增量产物：`v2_completion_deletion_v2.json`、`v2_completion_halfyear_compare.json`、`v2_completion_halfyear_daily.parquet`、`v2_completion_ledger_v2.json`、`v2_completion_input_acceptance.json`、`v2_completion_input_negatives.json`、`v2_completion_freeze_fixtures_v2.json`

## 遗留与例外

- `v2_completion_deletion.json`（0.5/0.5 错误消融）保留为**错误记录**，结论以 `_v2` 为准。
- `v2_completion_frozen_manifest/ledger/freeze_fixtures.json` 三个 v1 文件已改名为 `*__superseded_v1.json`
  保留；当前身份以 `*_v2.json` 为准。
- `v2_completion_halfyear_start.json`（临时脚本产物）已改名 `*__superseded_ad_hoc.json`，
  当前由 `s6_v2_halfyear_compare.py` 复算。
- `tabm_recent504_20261005__*` 无模型/预处理/曲线：**产物缺失**，不用于部署或晋级。
- 二成员（删 TabM）在各 H2 起点均弱于三成员，**不晋级**。
