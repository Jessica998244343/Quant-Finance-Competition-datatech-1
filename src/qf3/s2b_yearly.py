"""Per-calendar-year decomposition of the stage-B runs (sub-periods of each account)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import s2b_common


def yearly(run_dir: Path) -> dict:
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    equity["year"] = pd.to_datetime(equity["date"]).dt.year
    out = {}
    for year, block in equity.groupby("year"):
        n = len(block)
        nav = pd.to_numeric(block["portfolio_nav"], errors="coerce").to_numpy()
        bnav = pd.to_numeric(block["benchmark_nav"], errors="coerce").to_numpy()
        active = pd.to_numeric(block["active_return"], errors="coerce").to_numpy()
        if n < 2 or nav[0] <= 0 or bnav[0] <= 0:
            continue
        ann_p = (nav[-1] / nav[0]) ** (252.0 / n) - 1.0
        ann_b = (bnav[-1] / bnav[0]) ** (252.0 / n) - 1.0
        te = float(np.nanstd(active, ddof=1) * np.sqrt(252))
        out[int(year)] = {
            "days": int(n),
            "annualized_excess_return": float(ann_p - ann_b),
            "tracking_error": te,
            "information_ratio": float((ann_p - ann_b) / te) if te else None,
            "mean_daily_active_bp": float(np.nanmean(active) * 1e4),
        }
    return out


def main() -> None:
    runs = {
        "reference_cat0.5": s2b_common.BT / "b1_bt_cat0.5_dev19_20",
        "seed_average": s2b_common.BT / "b2_seedavg",
        "E1_a0.1": s2b_common.BT / "b2_E1_a0.1",
        "E1_a0.2": s2b_common.BT / "b2_E1_a0.2",
        "E2_a0.1": s2b_common.BT / "b2_E2_a0.1",
    }
    conf = {
        "reference_cat0.5_2021": s2b_common.BT / "b1_bt_cat0.5_conf21",
        "seed_average_2021": s2b_common.BT / "b2_seedavg_conf21",
        "E1_a0.1_2021": s2b_common.BT / "b2_E1_a0.1_conf21",
    }
    out = {}
    for name, path in {**runs, **conf}.items():
        if path.exists():
            out[name] = yearly(path)
    s2b_common.write_json(s2b_common.B / "b_yearly.json", out)
    for name, years in out.items():
        print(name)
        for year, stats in years.items():
            print(f"   {year}: excess={stats['annualized_excess_return']:+.4f} "
                  f"TE={stats['tracking_error']:.4f} IR={stats['information_ratio']:+.3f} "
                  f"days={stats['days']}")


if __name__ == "__main__":
    main()
