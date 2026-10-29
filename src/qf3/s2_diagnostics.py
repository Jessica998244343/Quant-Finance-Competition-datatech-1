"""A-stage diagnostics: turnover, cost, optimiser use, gross/net and signal conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("artifacts/stage2_ic_ir")
DIAG = OUT / "diagnostics"
BT_V1 = Path("artifacts/beginning/backtests")
BT_S2 = OUT / "backtests"
PRED_V1 = Path("artifacts/beginning/predictions")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def rank_stability(window: str) -> dict:
    members = {
        "lgbm_p5": PRED_V1 / f"P5_s20261005__{window}.parquet",
        "catboost": PRED_V1 / f"SC_d8_l2_15__{window}.parquet",
        "blend_v1": PRED_V1 / f"SSTAR__{window}.parquet",
    }
    out = {}
    for name, path in members.items():
        frame = pd.read_parquet(path).sort_values(["code", "date"])
        pivoted = frame.pivot(index="date", columns="code", values="score")
        ranks = pivoted.rank(axis=1, method="average")
        common = pivoted.notna() & pivoted.shift(1).notna()
        corr = []
        rank_move = []
        top10, top20 = [], []
        previous = ranks.shift(1)
        for day in ranks.index[1:]:
            mask = common.loc[day]
            if mask.sum() < 10:
                continue
            current_rank = ranks.loc[day, mask]
            previous_rank = previous.loc[day, mask]
            corr.append(float(current_rank.corr(previous_rank)))
            rank_move.append(float((current_rank - previous_rank).abs().mean()))
            k10 = max(1, int(round(0.10 * mask.sum())))
            k20 = max(1, int(round(0.20 * mask.sum())))
            c_codes = set(current_rank.nsmallest(k10).index)
            p_codes = set(previous_rank.nsmallest(k10).index)
            top10.append(len(c_codes & p_codes) / k10)
            c20 = set(current_rank.nsmallest(k20).index)
            p20 = set(previous_rank.nsmallest(k20).index)
            top20.append(len(c20 & p20) / k20)
        out[name] = {
            "mean_adjacent_rank_corr": float(np.mean(corr)),
            "mean_abs_rank_change": float(np.mean(rank_move)),
            "mean_top10_overlap": float(np.mean(top10)),
            "mean_top20_overlap": float(np.mean(top20)),
            "n_days": len(corr),
        }
    return out


def cost_breakdown(run_dir: Path, label: str) -> dict:
    trades = _read(run_dir / "trades.csv")
    equity = _read(run_dir / "equity_curve.csv")
    orders = _read(run_dir / "orders.csv")
    value = pd.to_numeric(trades["trade_value"], errors="coerce")
    commission = pd.to_numeric(trades["commission"], errors="coerce")
    return {
        "label": label,
        "n_trades": int(len(trades)),
        "commission_total": float(commission.sum()),
        "commission_is_minimum_share": float((commission.round(6) == 5.0).mean()),
        "stamp_tax_total": float(pd.to_numeric(trades["stamp_tax"], errors="coerce").sum()),
        "other_cost_total": float(pd.to_numeric(trades["other_cost"], errors="coerce").sum()),
        "total_cost": float(pd.to_numeric(trades["total_cost"], errors="coerce").sum()),
        "small_trade_value_lt_10k_share": float((value < 10_000).mean()),
        "median_trade_value": float(value.median()),
        "turnover_sum": float(pd.to_numeric(equity["turnover"], errors="coerce").sum()),
        "turnover_mean_per_day": float(pd.to_numeric(equity["turnover"], errors="coerce").mean()),
        "cost_over_initial_capital": float(
            pd.to_numeric(equity["transaction_cost"], errors="coerce").sum() / 1_000_000),
        "cost_pct_of_nav_mean": float((pd.to_numeric(equity["transaction_cost"], errors="coerce")
                                       / (pd.to_numeric(equity["portfolio_value"],
                                                        errors="coerce") / 1.0)).mean()),
        "rejected_orders": int((orders["status"] == "REJECTED").sum()),
        "rejection_reasons": orders.loc[orders["status"] == "REJECTED",
                                        "reject_reason"].value_counts().to_dict(),
    }


def optimizer_use(run_dir: Path, label: str) -> dict:
    tw = _read(run_dir / "target_weights.csv")
    active = pd.to_numeric(tw["target_weight"], errors="coerce") - pd.to_numeric(
        tw["benchmark_weight"], errors="coerce")
    weight = pd.to_numeric(tw["target_weight"], errors="coerce")
    at_limit = (active.abs() > 0.00999)
    return {
        "label": label,
        "active_share_mean": float(active.abs().groupby(tw["signal_date"]).sum().mean()),
        "active_weight_max": float(active.abs().max()),
        "share_active_at_limit": float(at_limit.mean()),
        "target_weight_max": float(weight.max()),
        "n_rows": int(len(tw)),
    }


def head_tail_conversion(run_dir: Path, label: str, buckets: int = 5) -> dict:
    tw = _read(run_dir / "target_weights.csv")
    pred = _read(run_dir / "predictions.csv")
    pred = pred.rename(columns={"date": "signal_date"})
    merged = tw.merge(pred[["signal_date", "code", "future_return"]],
                      on=["signal_date", "code"], how="left", validate="many_to_one")
    merged = merged.dropna(subset=["future_return"])
    merged["bucket"] = merged.groupby("signal_date")["target_weight"].transform(
        lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    table = merged.groupby("bucket").agg(
        mean_future_return=("future_return", "mean"),
        mean_target_weight=("target_weight", "mean"),
        mean_benchmark_weight=("benchmark_weight", "mean"),
        n=("future_return", "size")).reset_index()
    table["active_weight"] = table["mean_target_weight"] - table["mean_benchmark_weight"]
    # realised contribution per bucket = mean(active * future_return) within day, averaged
    merged["contrib"] = ((merged["target_weight"] - merged["benchmark_weight"])
                         * merged["future_return"])
    contrib = merged.groupby("bucket")["contrib"].mean()
    table["mean_active_contribution"] = table["bucket"].map(contrib)
    return {"label": label, "buckets": table.to_dict("records"),
            "top_minus_bottom_future_return": float(
                table["mean_future_return"].iloc[-1] - table["mean_future_return"].iloc[0])}


def gross_vs_net(main_dir: Path, zero_dir: Path, label: str) -> dict:
    main = _read(main_dir / "equity_curve.csv")
    zero = _read(zero_dir / "equity_curve.csv")
    n = min(len(main), len(zero))
    nav_main = pd.to_numeric(main["portfolio_nav"], errors="coerce").to_numpy()[:n]
    nav_zero = pd.to_numeric(zero["portfolio_nav"], errors="coerce").to_numpy()[:n]
    cost = pd.to_numeric(main["transaction_cost"], errors="coerce").to_numpy()[:n]
    value = pd.to_numeric(main["portfolio_value"], errors="coerce").to_numpy()[:n]
    return {
        "label": label,
        "n_days": int(n),
        "net_cagr": float(nav_main[-1] ** (252.0 / n) - 1.0),
        "gross_cagr": float(nav_zero[-1] ** (252.0 / n) - 1.0),
        "cagr_gap": float((nav_zero[-1] ** (252.0 / n)) - (nav_main[-1] ** (252.0 / n))),
        "cost_sum": float(cost.sum()),
        "cost_sum_over_mean_nav": float(cost.sum() / np.nanmean(value)),
        "mean_daily_cost_bp": float(np.nanmean(cost / value) * 1e4),
        "note": ("CAGR gap mixes cash/execution path effects with explicit fees; "
                 "it is not a static fee attribution"),
    }


def ir_decomposition(main_dir: Path, zero_dir: Path, label: str) -> dict:
    def one(path: Path) -> dict:
        eq = _read(path / "equity_curve.csv")
        n = len(eq)
        nav = pd.to_numeric(eq["portfolio_nav"], errors="coerce").to_numpy()
        bnav = pd.to_numeric(eq["benchmark_nav"], errors="coerce").to_numpy()
        active = pd.to_numeric(eq["active_return"], errors="coerce").to_numpy()
        ann_p = float(nav[-1] ** (252.0 / n) - 1.0)
        ann_b = float(bnav[-1] ** (252.0 / n) - 1.0)
        te = float(np.nanstd(active, ddof=1) * np.sqrt(252))
        downside = active[active < 0]
        eq = eq.assign(_q=pd.to_datetime(eq["date"]).dt.quarter,
                       _y=pd.to_datetime(eq["date"]).dt.year)
        quarterly = eq.groupby(["_y", "_q"])["active_return"].sum()
        return {
            "ann_excess": ann_p - ann_b, "tracking_error": te,
            "ir": (ann_p - ann_b) / te if te else None,
            "downside_te": float(np.nanstd(downside, ddof=1) * np.sqrt(252))
            if len(downside) > 1 else None,
            "mean_daily_active_bp": float(np.nanmean(active) * 1e4),
            "quarterly_active_sum_min": float(quarterly.min()),
            "quarterly_active_sum_max": float(quarterly.max()),
            "quarters_negative": int((quarterly < 0).sum()),
            "quarters_total": int(len(quarterly)),
        }
    return {"label": label, "main_cost": one(main_dir), "zero_cost": one(zero_dir)}


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    report = {}
    report["rank_stability"] = {w: rank_stability(w) for w in ("DEV19", "DEV20", "CONF21")}
    report["cost_breakdown"] = {
        "REF-v1 legacy (2019-2020)": cost_breakdown(BT_V1 / "bt_SSTAR_dev19_20", "REF-v1"),
        "REF-v1 real input (2019-2020)": cost_breakdown(BT_S2 / "bt_realinput_dev19_20",
                                                        "real-input"),
        "pure P5 (2019-2020)": cost_breakdown(BT_V1 / "bt_P5_dev19_20", "P5"),
    }
    report["optimizer_use"] = {
        "REF-v1 legacy (2019-2020)": optimizer_use(BT_V1 / "bt_SSTAR_dev19_20", "REF-v1"),
        "pure P5 (2019-2020)": optimizer_use(BT_V1 / "bt_P5_dev19_20", "P5"),
        "B0 (2019-2020)": optimizer_use(BT_V1 / "bt_B0_dev19_20", "B0"),
    }
    report["head_tail"] = {
        "REF-v1 legacy (2019-2020)": head_tail_conversion(BT_V1 / "bt_SSTAR_dev19_20",
                                                          "REF-v1"),
        "pure P5 (2019-2020)": head_tail_conversion(BT_V1 / "bt_P5_dev19_20", "P5"),
    }
    report["gross_vs_net"] = {
        "REF-v1": gross_vs_net(BT_V1 / "bt_SSTAR_dev19_20",
                               BT_V1 / "bt_SSTAR_dev19_20_zerocost", "REF-v1"),
        "B0": gross_vs_net(BT_V1 / "bt_B0_dev19_20",
                           BT_V1 / "bt_B0_dev19_20_zerocost", "B0"),
    }
    report["ir_decomposition"] = {
        "REF-v1": ir_decomposition(BT_V1 / "bt_SSTAR_dev19_20",
                                   BT_V1 / "bt_SSTAR_dev19_20_zerocost", "REF-v1"),
        "B0": ir_decomposition(BT_V1 / "bt_B0_dev19_20",
                               BT_V1 / "bt_B0_dev19_20_zerocost", "B0"),
        "REF-v1 2021": {"label": "REF-v1-2021",
                        "main_cost": ir_decomposition(BT_V1 / "bt_SSTAR_conf21",
                                                      BT_V1 / "bt_SSTAR_conf21",
                                                      "REF-v1-2021")["main_cost"],
                        "zero_cost": None,
                        "note": "no zero-cost replay was run for 2021"},
    }
    (DIAG / "diagnostics.json").write_text(json.dumps(report, indent=2, default=str))
    for key, value in report["rank_stability"].items():
        print("rank stability", key)
        for member, stats in value.items():
            print(f"  {member}: corr={stats['mean_adjacent_rank_corr']:.4f} "
                  f"rank_move={stats['mean_abs_rank_change']:.2f} "
                  f"top10_overlap={stats['mean_top10_overlap']:.3f}")
    print(json.dumps(report["cost_breakdown"], indent=2))
    print(json.dumps(report["optimizer_use"], indent=2))
    print(json.dumps(report["head_tail"], indent=2))
    print(json.dumps(report["gross_vs_net"], indent=2))
    print(json.dumps(report["ir_decomposition"], indent=2))


if __name__ == "__main__":
    main()
