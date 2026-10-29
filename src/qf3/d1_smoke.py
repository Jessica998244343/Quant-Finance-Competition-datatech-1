"""D1 functional smoke test: fixed signal, short replay, independent checks."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_runner, dataio, labels as labels_mod, metrics

DATA_VIEW = "artifacts/beginning/data_zero_st"
OUT = Path("artifacts/beginning/d1/smoke")


def independent_labels(data_dir: str, calendar: list[int]) -> pd.DataFrame:
    tables = dataio.load_all(data_dir)
    label_table = labels_mod.build_labels(tables, calendar)
    return label_table


def check_constraints(result_dir: Path, data_dir: str) -> dict:
    """Verify the Barra LP constraints actually bind on the produced targets."""
    target = pd.read_csv(result_dir / "target_weights.csv")
    equity = pd.read_csv(result_dir / "equity_curve.csv")
    positions = pd.read_csv(result_dir / "positions.csv")
    barra = dataio.load_range(data_dir, "Barra_factor", 2019, 2019)
    industry = pd.read_csv(Path(data_dir) / "HS300_industry" / "2019" /
                           "hs300_industry_2019.csv", encoding="gbk")
    industry.columns = ["date", "code", "name", "industry_name", "industry_code"]

    single_max = float(target["target_weight"].max())
    active_max = float((target["target_weight"] - target["benchmark_weight"]).abs().max())
    daily_turnover_max = float(equity["turnover"].max()) if len(equity) else 0.0

    ind_max = 0.0
    for day, block in target.groupby("signal_date"):
        day_int = int(str(day).replace("-", ""))
        ind_day = industry.loc[industry["date"] == day_int, ["code", "industry_code"]]
        merged = block.merge(ind_day, on="code", how="left")
        active = merged["target_weight"] - merged["benchmark_weight"]
        by_ind = active.groupby(merged["industry_code"]).sum().abs()
        if len(by_ind):
            ind_max = max(ind_max, float(by_ind.max()))

    style_max = 0.0
    for day, block in target.groupby("signal_date"):
        day_int = int(str(day).replace("-", ""))
        styles = barra.loc[barra["日期"] == day_int].set_index("代码")
        styles = styles.drop(columns=[c for c in ("日期", "名称") if c in styles.columns])
        styles = styles.apply(pd.to_numeric, errors="coerce")
        merged = block.set_index("code").join(styles, how="left")
        active = (merged["target_weight"] - merged["benchmark_weight"])
        exposures = (merged[styles.columns].mul(active, axis=0)).sum()
        if len(exposures):
            style_max = max(style_max, float(exposures.abs().max()))

    target_codes = target.groupby("signal_date")["code"].nunique().min()
    actual_positions = positions.groupby("date")["code"].nunique()
    return {
        "max_single_name_weight": single_max,
        "max_active_weight": active_max,
        "max_daily_turnover": daily_turnover_max,
        "max_industry_exposure": ind_max,
        "max_style_exposure": style_max,
        "min_target_codes_per_signal": int(target_codes),
        "actual_positions_min": int(actual_positions.min()),
        "actual_positions_max": int(actual_positions.max()),
        "target_vs_actual_differs": bool(actual_positions.max() != target_codes),
    }


def check(result_dir: Path, metrics_out: dict, tables: dict, data_dir: str) -> dict:
    predictions = tables["predictions"]
    rankic_engine = tables["rankic"]
    equity = tables["equity_curve"]
    target = tables["target_weights"]
    trades = tables["trades"]
    orders = tables["orders"]

    calendar = sorted(pd.unique(dataio.load_range(data_dir, "MarketData", 2016, 2022,
                                                  columns=["日期"])["日期"]).tolist())
    labels = independent_labels(data_dir, calendar)
    labels = labels.assign(
        date=pd.to_datetime(labels["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    merged = predictions.merge(labels.loc[:, ["date", "code", "future_return"]],
                               on=["date", "code"], how="left",
                               suffixes=("", "_ref"), validate="one_to_one")
    merged["date_int"] = pd.to_datetime(merged["date"]).dt.strftime("%Y%m%d").astype(int)
    both = merged.dropna(subset=["future_return", "future_return_ref"])
    label_match = bool(np.allclose(both["future_return"], both["future_return_ref"],
                                   atol=1e-10, rtol=1e-10))
    label_missing_agree = bool((merged["future_return"].isna()
                                == merged["future_return_ref"].isna()).all())

    rankic_ref = metrics.daily_rankic(merged, "score", "future_return")
    engine_valid = rankic_engine.dropna(subset=["rankic"]).set_index("date")["rankic"]
    ref_valid = rankic_ref.dropna(subset=["rankic"]).set_index("date")["rankic"]
    common = engine_valid.index.intersection(ref_valid.index)
    ic_match = bool(np.allclose(engine_valid.loc[common], ref_valid.loc[common],
                                atol=1e-9, rtol=1e-9))

    summary = metrics.summarize_rankic(rankic_ref)

    # independent net-value / excess-return / IR recomputation
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    n = len(eq)
    nav = eq["portfolio_nav"].to_numpy(dtype=float)
    bnav = eq["benchmark_nav"].to_numpy(dtype=float)
    if n > 0:
        ann_p = float(nav[-1] ** (252.0 / n) - 1.0)
        ann_b = float(bnav[-1] ** (252.0 / n) - 1.0)
        active = eq["active_return"].to_numpy(dtype=float)
        te = float(np.std(active, ddof=1) * math.sqrt(252))
        ir = (ann_p - ann_b) / te if te > 0 else None
    else:
        ann_p = ann_b = te = ir = None

    cost_total = float(pd.to_numeric(trades["total_cost"], errors="coerce").sum())
    target_ok = bool(target["target_weight"].notna().all())
    barra_days = dataio.load_range(data_dir, "Barra_factor", 2019, 2019,
                                   columns=["日期", "代码"])
    counts = barra_days.groupby("日期").size()
    coverage_ok = all(int(counts.get(int(d), -1)) == int(c)
                      for d, c in merged.groupby("date_int")["code"].size().items())
    duplicate_ok = not bool(merged.duplicated(["date", "code"]).any())
    finite_ok = bool(np.isfinite(merged["score"].to_numpy(dtype=float)).all())

    return {
        "rows": int(len(merged)),
        "label_values_match": label_match,
        "label_missing_mask_match": label_missing_agree,
        "rankic_matches_engine": ic_match,
        "engine_metrics": metrics_out,
        "independent_rankic": summary,
        "independent_annualized_portfolio": ann_p,
        "independent_annualized_benchmark": ann_b,
        "independent_tracking_error": te,
        "independent_ir": ir,
        "trades": int(len(trades)),
        "orders": int(len(orders)),
        "reject_reasons": orders["reject_reason"].value_counts(dropna=False).to_dict(),
        "transaction_cost_sum": cost_total,
        "target_weight_complete": target_ok,
        "universe_coverage_ok": coverage_ok,
        "no_duplicate_scores": duplicate_ok,
        "all_scores_finite": finite_ok,
        "n_signal_days": int(merged["date"].nunique()),
        "target_weights_per_day": int(target.groupby("signal_date").size().max()),
        "constraints": check_constraints(result_dir, data_dir),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inference = backtest_runner.barra_column_inference("动量")
    run_dir = OUT / "run_barra_momentum"
    engine = backtest_runner.make_engine(
        data_dir=DATA_VIEW,
        start_date="2019-01-02",
        end_date="2019-02-15",
        inference=inference,
        output_dir=run_dir,
        async_inference=False,
    )
    metrics_out = engine.run()
    report = check(run_dir, metrics_out, engine.tables, DATA_VIEW)
    report["contract"] = "documented in configs/beginning/evaluation_contract.json"
    report["signal"] = "Barra_factor.动量 percentile (engineering signal, not a baseline)"
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
