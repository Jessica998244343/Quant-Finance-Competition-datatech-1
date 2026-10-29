"""REV1 corrected diagnostics: stability, optimiser use, cost incl. slippage, conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path("artifacts/stage2_ic_ir/a_revision1")
DIAG = REV / "diagnostics"
PRED = REV / "predictions"
BT = REV / "backtests"
V1_BT = Path("artifacts/beginning/backtests")
V1_PRED = Path("artifacts/beginning/predictions")


def score_stability(window: str) -> dict:
    """Adjacent-day stability of each member; head = high score."""
    members = {
        "lgbm_p5": V1_PRED / f"P5_s20261005__{window}.parquet",
        "catboost": V1_PRED / f"SC_d8_l2_15__{window}.parquet",
        "blend_v1": PRED / f"legacy_oof_blend__{window}.parquet",
        "blend_realinput_fixed": PRED / f"origmodel_realinput__{window}.parquet",
    }
    out = {}
    for name, path in members.items():
        frame = pd.read_parquet(path)
        pivoted = frame.pivot(index="date", columns="code", values="score").sort_index()
        ranks = pivoted.rank(axis=1, method="average")          # 1 = lowest score
        n = pivoted.shape[1]
        common = pivoted.notna() & pivoted.shift(1).notna()
        previous = ranks.shift(1)
        corr, move, head10, head20, tail10, tail20 = [], [], [], [], [], []
        for day in ranks.index[1:]:
            mask = common.loc[day]
            if mask.sum() < 10:
                continue
            cur = ranks.loc[day, mask]
            prev = previous.loc[day, mask]
            corr.append(float(cur.corr(prev)))
            move.append(float((cur - prev).abs().mean()))
            k10 = max(1, int(round(0.10 * mask.sum())))
            k20 = max(1, int(round(0.20 * mask.sum())))
            head10.append(len(set(cur.nlargest(k10).index) & set(prev.nlargest(k10).index)) / k10)
            head20.append(len(set(cur.nlargest(k20).index) & set(prev.nlargest(k20).index)) / k20)
            tail10.append(len(set(cur.nsmallest(k10).index) & set(prev.nsmallest(k10).index)) / k10)
            tail20.append(len(set(cur.nsmallest(k20).index) & set(prev.nsmallest(k20).index)) / k20)
        out[name] = {
            "mean_adjacent_rank_corr": float(np.mean(corr)),
            "mean_abs_rank_change": float(np.mean(move)),
            "mean_head10_overlap_high_score": float(np.mean(head10)),
            "mean_head20_overlap_high_score": float(np.mean(head20)),
            "mean_tail10_overlap_low_score": float(np.mean(tail10)),
            "mean_tail20_overlap_low_score": float(np.mean(tail20)),
            "n_days": len(corr), "n_stocks_in_pool": int(n),
        }
    return out


def optimizer_use(run_dir: Path, label: str) -> dict:
    tw = pd.read_csv(run_dir / "target_weights.csv")
    active = (pd.to_numeric(tw["target_weight"], errors="coerce")
              - pd.to_numeric(tw["benchmark_weight"], errors="coerce"))
    per_day = active.abs().groupby(tw["signal_date"]).sum()
    at_limit = active.abs() > 0.00999
    row_share = at_limit.groupby(tw["signal_date"]).mean()
    return {
        "label": label,
        "active_weight_L1_mean": float(per_day.mean()),
        "active_share_mean_half_L1": float(per_day.mean() / 2.0),
        "active_weight_max": float(active.abs().max()),
        "row_share_at_limit_mean": float(row_share.mean()),
        "row_share_at_limit_min": float(row_share.min()),
        "row_share_at_limit_max": float(row_share.max()),
    }


def head_tail_by_score(run_dir: Path, label: str, buckets: int = 5) -> dict:
    """Score quintiles inside each day's signal universe, H=1 O2O diagnostics."""
    pred = pd.read_csv(run_dir / "predictions.csv")
    tw = pd.read_csv(run_dir / "target_weights.csv")
    tw = tw.rename(columns={"signal_date": "date"})
    merged = pred.merge(tw[["date", "code", "benchmark_weight", "target_weight"]],
                        on=["date", "code"], how="left", validate="one_to_one")
    merged = merged.dropna(subset=["future_return"])
    merged = merged.loc[merged["target_weight"].notna()]
    merged["bucket"] = merged.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    merged["active"] = merged["target_weight"] - merged["benchmark_weight"]
    merged["contrib"] = merged["active"] * merged["future_return"]
    daily = merged.groupby(["date", "bucket"]).agg(
        contrib_sum=("contrib", "sum"), n=("contrib", "size")).reset_index()
    summary = merged.groupby("bucket").agg(
        mean_future_return=("future_return", "mean"),
        mean_active_weight=("active", "mean"),
        mean_target_weight=("target_weight", "mean"),
        n_stock_days=("contrib", "size")).reset_index()
    contribution = daily.groupby("bucket")["contrib_sum"].mean().rename(
        "mean_daily_contribution")
    summary = summary.merge(contribution, on="bucket")
    summary["bucket"] = summary["bucket"].map(
        {0: "lowest score", 1: "q2", 2: "q3", 3: "q4", 4: "highest score"})
    return {"label": label, "buckets": summary.to_dict("records"),
            "note": ("H=1 O2O target-weight diagnostic; contributions are summed inside "
                     "each day before averaging, and are not a close-to-close NAV "
                     "attribution")}


def cost_with_slippage(run_dir: Path, label: str) -> dict:
    trades = pd.read_csv(run_dir / "trades.csv")
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    value = pd.to_numeric(trades["trade_value"], errors="coerce")
    position = pd.to_numeric(trades["position_value"], errors="coerce")
    commission = pd.to_numeric(trades["commission"], errors="coerce")
    nav = pd.to_numeric(equity["portfolio_value"], errors="coerce")
    slippage = (value - position).abs()
    explicit = pd.to_numeric(trades["total_cost"], errors="coerce")
    at_min = commission.round(6) == 5.0
    small = value < 10_000
    return {
        "label": label,
        "n_trades": int(len(trades)),
        "explicit_cost_sum": float(explicit.sum()),
        "slippage_sum": float(slippage.sum()),
        "total_friction_sum": float(explicit.sum() + slippage.sum()),
        "mean_daily_explicit_bp": float((explicit.groupby(
            trades["date"]).sum() / nav.groupby(equity["date"]).first()).mean() * 1e4),
        "mean_daily_slippage_bp": float((slippage.groupby(
            trades["date"]).sum() / nav.groupby(equity["date"]).first()).mean() * 1e4),
        "trades_at_min_commission_share": float(at_min.mean()),
        "commission_amount_from_min_share": float(
            commission[at_min].sum() / commission.sum()),
        "trades_below_10k_value_share": float(small.mean()),
        "traded_value_below_10k_share": float(value[small].sum() / value.sum()),
        "median_trade_value": float(value.median()),
    }


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    report = {
        "score_stability": {w: score_stability(w) for w in ("DEV19", "DEV20", "CONF21")},
        "optimizer_use": {
            "rev1 real input (2019-2020)": optimizer_use(
                BT / "rev1_bt_realinput_dev19_20", "real-input"),
            "v1 legacy REF (2019-2020)": optimizer_use(
                V1_BT / "bt_SSTAR_dev19_20", "REF-v1"),
        },
        "head_tail_by_score": {
            "rev1 real input (2019-2020)": head_tail_by_score(
                BT / "rev1_bt_realinput_dev19_20", "real-input"),
            "v1 legacy REF (2019-2020)": head_tail_by_score(
                V1_BT / "bt_SSTAR_dev19_20", "REF-v1"),
        },
        "cost_with_slippage": {
            "rev1 real input (2019-2020)": cost_with_slippage(
                BT / "rev1_bt_realinput_dev19_20", "real-input"),
            "v1 legacy REF (2019-2020)": cost_with_slippage(
                V1_BT / "bt_SSTAR_dev19_20", "REF-v1"),
        },
    }
    (DIAG / "diagnostics_rev1.json").write_text(json.dumps(report, indent=2, default=str))
    for window, value in report["score_stability"].items():
        print(window)
        for member, stats in value.items():
            print(f"  {member}: corr={stats['mean_adjacent_rank_corr']:.4f} "
                  f"head10={stats['mean_head10_overlap_high_score']:.3f} "
                  f"tail10={stats['mean_tail10_overlap_low_score']:.3f}")
    print(json.dumps(report["optimizer_use"], indent=2))
    print(json.dumps(report["cost_with_slippage"], indent=2))
    for key, value in report["head_tail_by_score"].items():
        print(key)
        for bucket in value["buckets"]:
            print(f"   {bucket['bucket']}: n={bucket['n_stock_days']} "
                  f"fwd={bucket['mean_future_return']:.5f} "
                  f"active={bucket['mean_active_weight']:.5f} "
                  f"daily_contrib={bucket['mean_daily_contribution']:.6f}")


if __name__ == "__main__":
    main()
