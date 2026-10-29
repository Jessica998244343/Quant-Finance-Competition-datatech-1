"""Write the B0 freeze record, feature catalogue and pre-CONF21 decision log."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import analysis, cv, featurelib, trainers
from .constants import F0_COLUMNS, FEATURE_COLUMNS, OUTER_WINDOWS

OUT = Path("artifacts/beginning")


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame(featurelib.feature_catalog())


def write_splits(panel: pd.DataFrame) -> dict:
    rows = {}
    for window in OUTER_WINDOWS:
        split = cv.fold_split(panel, window)
        rows[window] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in split.items() if not hasattr(v, "tolist")}
        rows[window]["n_outer"] = split["n_outer"]
        rows[window]["n_internal"] = split["n_internal"]
        rows[window]["n_es"] = split["n_es"]
        rows[window]["n_predict"] = split["n_predict"]
    (OUT / "splits.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows


def write_b0_freeze(log: pd.DataFrame, panel: pd.DataFrame | None = None) -> str:
    b0 = log[(log["experiment_id"] == "B0")].set_index("window")
    if panel is None:
        panel = pd.read_parquet("artifacts/beginning/cache/panel.parquet")
    text = [
        "# B0 冻结记录",
        "",
        "B0 是本地强基线的比较参照，冻结后不再覆盖。",
        "",
        "## 身份",
        "",
        "- 模型：LightGBM 4.7.0 `gbdt` / `regression`（L2），目标为每日截面平均并列秩 "
        "`z=(rank-(n+1)/2)/n`。",
        "- 特征集 F0（42 列）：Factor33_winsor 去掉 `名称`/`is_st` 后的 32 列，"
        "加 10 列 Barra 风格（原值减 0.5）。连续列用当日合法池截面秩中心化，"
        "`is_suspend`/`is_new` 保留 0/1 原值。",
        "- 样本权重：日期内股票等权 `1/n_t`，再整体归一到均值 1；不使用时间衰减。",
        "- 线程 8，`deterministic=True`，`force_row_wise=True`，主种子 20261005。",
        "",
        "## 参数",
        "",
        "```json",
        json.dumps({k: v for k, v in trainers.B0_PARAMS.items()}, indent=2),
        "```",
        "",
        "## 训练协议",
        "",
        "按年向前 expanding：外层训练只看 `label_end_date <= 外层截止日` 的样本；"
        "外层训练末端 126 个交易日为早停区间，内部拟合样本的 `label_end_date` 严格早于"
        "早停区间起始日；早停指标为每日平均 Spearman（",
        "`higher_is_better=True`，默认 metric 关闭）；找到轮数后用全部外层合格样本"
        "按固定轮数重新拟合，再预测外层年份。",
        "",
        "## 开发折结果（冻结时）",
        "",
        "| 窗口 | 外层截止 | 早停起始 | 外层样本 | 内部拟合 | 早停样本 | 最佳轮数 | 早停日 Spearman | "
        "Mean RankIC | RankICIR | 正 IC 比例 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for window in ("DEV19", "DEV20"):
        row = b0.loc[window]
        summary = row["summary"]
        split = cv.fold_split(panel, window)
        text.append(
            f"| {window} | {int(split['train_end'])} | {int(split['es_start'])} "
            f"| {int(row['n_outer'])} | {int(row['n_internal'])} | {int(row['n_es'])} "
            f"| {int(row['best_iteration'])} | {float(row['es_score']):.4f} "
            f"| {float(summary['mean_rankic']):.5f} | {float(summary['rankic_ir']):.3f} "
            f"| {float(summary['positive_ratio']):.3f} |")
    text += [
        "",
        "两个开发折的早停轮数（266/273）都远低于 3000 上限，因此不触发扩到 5000 的条件。",
        "",
        "## 特征列名（F0，42 列）",
        "",
        "```",
        ", ".join(F0_COLUMNS),
        "```",
        "",
        "## 预测与回测",
        "",
        "- 预测：`artifacts/beginning/predictions/B0__DEV19.parquet`、`B0__DEV20.parquet`。",
        "- 回测：见 `artifacts/beginning/backtests/` 下 B0 对应运行目录。",
        "- CONF21 与全量回退包在冻结后补齐。",
    ]
    path = OUT / "B0_FREEZE.md"
    path.write_text("\n".join(text))
    return str(path)


def write_decisions(panel: pd.DataFrame) -> str:
    table = analysis.summary_table()
    lines = [
        "# 打开 CONF21 之前登记的主候选 / 备选决定",
        "",
        "本文件在读取任何 2021 结果之前写入。CONF21 只用于核验与否决，不用于循环调参。",
        "",
        "## 决定",
        "",
        "1. **B0（冻结参照）**：LightGBM / F0 / 截面秩标签，参数见 `B0_FREEZE.md`。",
        "2. **备选 S-backup**：LightGBM P5 容量配置，特征集 F0+G1（61 列），主种子 20261005。",
        "3. **主候选 S-star**：0.75 × (P5 的每日中心化秩) + 0.25 × (CatBoost d8 l2=15 的每日中心化秩)，"
        "两者都用 F0+G1、同一训练协议与样本、主种子 20261005。",
        "",
        "融合权重只在 {0.25, 0.5, 0.75} 中选择；开发折上 0.25 在两个窗口都优于纯 P5，故冻结为 0.25。",
        "",
        "## 开发折证据（DEV19+DEV20 合并，按日等权）",
        "",
        "| 版本 | Mean RankIC | RankICIR | 正 IC 比例 |",
        "|---|---|---|---|",
    ]
    for name in ("B0", "P5_s20261005", "SC_d8_l2_15"):
        row = table.loc[table["experiment_id"] == name]
        if len(row):
            r = row.iloc[0]
            lines.append(f"| {name} | {r['mean_rankic']:.5f} | {r['rankic_ir']:.3f} | "
                         f"{r['positive_ratio']:.3f} |")
    lines += [
        "| 主候选混合 0.25 | 0.07817 | 0.563 | 0.721 |",
        "",
        "## 相对 B0 的判定（预登记门槛）",
        "",
        f"- 主候选 ΔIC = {0.078169 - 0.047832:.5f}，门槛 = max(0.003, 0.10×|IC_B0|) = "
        f"{max(0.003, 0.10*0.047832):.5f} → 通过。",
        "- 2019 与 2020 两个开发年 ΔIC 均为正。",
        "- 三种子配对差异：主候选 3 个种子对 B0 均为正（见 `experiments.jsonl`）。",
        "",
        "## 明确不做的事",
        "",
        "- 不因为 CONF21 的表现循环调整特征、容量、目标或融合权重。",
        "- 不用 2022 的模型表现做选型；2022 只用于最终全量训练（合法标签）与 RES22 诊断。",
    ]
    path = OUT / "decisions_pre_conf21.md"
    path.write_text("\n".join(lines))
    return str(path)


def main() -> None:
    panel = pd.read_parquet("artifacts/beginning/cache/panel.parquet")
    log = analysis.load_log()
    catalog = feature_catalog()
    catalog.to_csv(OUT / "feature_catalog.csv", index=False)
    write_splits(panel)
    print(write_b0_freeze(log, panel))
    print(write_decisions(panel))


if __name__ == "__main__":
    main()
