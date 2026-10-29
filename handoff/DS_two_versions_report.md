# 两版提交对照报告：strong_baseline_v1 与 FINAL_v1

日期：2026-10-05（Asia/Shanghai）。**IR 只记录、不参与选择。** 未上传。

## 0. 路径确认

**是**，FINAL_v1 的包就是 `releases/stage3_fusion/FINAL_v1.zip`。

| 版本 | 包路径 | zip 字节 | 解压字节 | 文件数 | SHA-256 |
|---|---|---|---|---|---|
| strong_baseline_v1（已提交，官方有成绩） | `releases/strong_baseline_v1.zip` | 2,337,622 | 4,420,153 | 16 | `b34712933d952da40ff903804096e0f104ad501b598d643b335a7b31dc0f87c5` |
| **FINAL_v1（本轮主候选）** | **`releases/stage3_fusion/FINAL_v1.zip`** | **2,866,631** | **3,983,976** | **20** | **`64bfb8593df71dfd3a12febc934f6d0c47a07de4bfc8ab88acd3d7927020eb30`** |
| （回退）baseline_v1 = B0 | `releases/baseline_v1.zip` | 435,055 | 1,034,124 | 15 | `565c7907c7519d3f801237dcaee98a862c7aed2527a0ec27e1703b0ac8fcb171` |

两版顶层目录都是 `submission/`，符合官方模板。

## 1. 身份与配方

| 项 | strong_baseline_v1 | FINAL_v1 |
|---|---|---|
| 成员 | LightGBM P5（F0+G1，61 列）+ CatBoost d8/l2=15 | CatBoost C1（IC 选轮，d8/l2=15）+ TabM（官方实现，k=8） |
| 融合 | 0.75 × LGBM 日秩 + 0.25 × CatBoost 日秩 | 0.5 × CatBoost 日秩 + 0.5 × TabM 日秩 |
| 训练输入语义 | 年度并集（旧口径） | 平台真实语义（按每日 Barra 成分过滤） |
| 权重归一化 | 全局（含未来样本，**已知问题 CORR-2**） | 折内（CORR-2 已修） |
| CatBoost 选轮 | RMSE 早停 | 内层逐日 H1 Spearman 选树数（CORR-1 已修） |
| 全量训练轮数规则 | 中位数（418 / 310） | 中位数（875 棵树 / 13 epochs） |
| 打包方式 | `train.py` 用 `sys.modules` 注册 + 动态加载（**审查已点名的合规项**） | 普通静态导入、safetensors 权重、无 sys.modules/eval/exec |
| 推理依赖 | numpy/pandas/pyarrow/scipy/sklearn/lightgbm/catboost | 同左 + **torch(CPU)** + safetensors + rtdl_num_embeddings |

## 2. 官方成绩（仅 strong_baseline_v1 拥有）

| 版本 | RankIC | 组合 IR | 标准化 RankIC | 标准化 IR | 总分 |
|---|---|---|---|---|---|
| **strong_baseline_v1** | **0.0657** | **0.5524** | **0.7747** | **0.2339** | **0.6125** |
| FINAL_v1 | 无（未提交） | 无 | 无 | 无 | 无 |

FINAL_v1 **没有**官方成绩；本地数字不能与官方 0.0657 直接比较算收益。

## 3. 本地可比指标（同一本地代理契约、同一引擎、同一日期集合）

### 3.1 预测端日均面板 RankIC

| 版本 | 2019 | 2020 | DEV 合并 | 2021 | 说明 |
|---|---|---|---|---|---|
| strong_baseline_v1（legacy 口径） | 0.078757 | 0.077577 | 0.078169 | 0.060836 | 旧输入语义 |
| **FINAL_v1** | **0.084628** | **0.084815** | **0.084721** | **0.064761** | 平台真实语义 |
| 参照 R_WORK（同语义） | 0.079542 | 0.074984 | 0.077268 | 0.061007 | 便于公平比较 |
| 参照 R_DEPLOY（v1 配方+真实输入） | 0.078458 | 0.077151 | 0.077805 | 0.062708 | 桥接列 |

**FINAL_v1 相对 strong_baseline_v1 本地口径**：2019 +0.005871、2020 +0.007238、
DEV 合并 **+0.006552**、2021 **+0.003925**。

**但这列不是干净对照**：两版的训练输入语义、权重归一化、CatBoost 选轮都不同。
公平对照应用同语义参照：

| 对比 | DEV ΔIC | 2021 ΔIC |
|---|---|---|
| FINAL_v1 vs R_WORK | **+0.007453** | **+0.003754** |
| FINAL_v1 vs R_DEPLOY | **+0.006916** | **+0.002054** |

### 3.2 组合回测（主费用；**IR 仅记录，不参与判断**）

| 版本 | 期间 | 引擎 IC | 年化净超额 | TE | IR | 换手 |
|---|---|---|---|---|---|---|
| strong_baseline_v1 | 2019-01-02 至 2020-12-31 | 0.078838 | −12.04% | 0.034664 | −3.473 | 448.7 |
| FINAL_v1 | 同上 | **0.085303** | **−11.91%** | 0.034714 | **−3.430** | 448.8 |
| strong_baseline_v1 | 2021 | 0.061050 | −14.80% | 0.043583 | −3.397 | 231.5 |
| FINAL_v1 | 2021 | **0.064488** | −17.50% | 0.040658 | −4.305 | 234.3 |

按用户规则，**IR 不影响任何结论**；此处只作记录。注意 2021 的净超额与 IR 在 FINAL_v1 上更差，
若将来官方综合分（70% IC + 30% IR）复核时这可能是权衡点。

## 4. 合规状态对照

| 检查 | strong_baseline_v1 | FINAL_v1 |
|---|---|---|
| 名称 / is_st 入模 | 否 | 否（61 列实查） |
| 联网 | 无 | 无（验收中 socket 全部拒绝） |
| GPU 依赖 | 无 | 无（`cuda.is_available()=False`） |
| 推理期 fit / 跨调用状态 | 无 | 无 |
| 120 日上限 | 遵守 | 遵守（只用当日特征） |
| 外部数据 | 无 | 无（仅 vendored 代码） |
| `sys.modules` / 动态执行 | **有（已知问题）** | **无** |
| pickle 可执行载荷 | 使用 joblib/原生格式 | TabM 用 **safetensors**；CatBoost 原生 `.cbm` |
| `train.py` 独立复跑 | 已做过（逐位一致） | 已做（复现模型与随包模型差 **0.0**） |
| 平台入口加载 | 已验收 | 已验收（7 信号日、2,100 行） |
| CPU 耗时 / 峰值内存 | P95 ≈0.099 s/日、1.60 GiB | P95 **0.143 s/日**、**1.27 GiB** |

## 5. 结论与建议

1. **本地证据支持 FINAL_v1 明显优于 strong_baseline_v1**：同语义公平对照下 DEV +0.0069 ~ +0.0075，
   2021 +0.0021 ~ +0.0038，三年全正。
2. **但收益**（本地代理 IR）**没有同步改善**：DEV 略好（−3.43 对 −3.47），2021 反而更差
   （−4.31 对 −3.40）。官方综合分含 30% IR，这是唯一需要权衡的地方。
3. **合规上 FINAL_v1 更干净**：修掉了 strong_baseline_v1 的动态加载问题，权重改为 safetensors。
4. 风险：FINAL_v1 引入 torch(CPU) 依赖；若评测环境装不上，应改用纯树的 `ALT_v1`（DEV 0.080688，
   2021 0.064230，同样过 IC 门槛，无 torch）。
5. 官方成绩仍只属于 `strong_baseline_v1.zip`。**是否用 FINAL_v1 替换提交由用户决定，本 Agent 不上传。**

## 6. 数据来源

`artifacts/stage3_fusion/two_version_summary.json`、`package_info.json`、`d3_backtests.json`、
`d3_fusion_scan.json`、`d3_acceptance.json`；`artifacts/stage3_ic_first/s1_finalize.json`；
`artifacts/beginning/backtests/bt_SSTAR_*`；官方成绩来自用户转述。
