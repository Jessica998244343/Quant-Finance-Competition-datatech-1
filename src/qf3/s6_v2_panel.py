"""§11: build the extended research panel and verify the new columns.

The legal universe is reproduced exactly as in ``panel.build_panel(membership_filter=True)``
(market/factor history inner-joined with the same-day Barra membership); the
resulting ``(date, code)`` key set must equal ``panel_platform.parquet`` exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio, featurelib
from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE
from .panel import _date_bounds
from .s6_v2_features import MIN_VALID, add_interactions, day_features

V2 = Path("artifacts/stage4_verified_ic/v2")
BASE = Path("artifacts/stage2_ic_ir/panel_platform.parquet")


def verify_arithmetic() -> dict:
    """Hand-checkable short sequences for every endpoint/offset."""
    days = list(range(20200101, 20200101 + 21))
    market = pd.DataFrame({
        "日期": days, "代码": "SH000001",
        "open": [float(i + 1) for i in range(21)],
        "close": [float(i + 1) for i in range(21)],
        "amount": [100.0] * 21})
    factor = pd.DataFrame({"日期": days, "代码": "SH000001", "turn": [2.0] * 21})
    out = day_features(market, factor, ["SH000001"], days[-1]).iloc[0]
    expected_1_5 = 20.0 / 16.0 - 1.0
    expected_5_20 = 16.0 / 1.0 - 1.0
    expected_mean_5 = float(np.log(21.0 / 16.0) / 5.0)
    checks = {
        "f1_ret_1_5": abs(out["f1_ret_1_5"] - expected_1_5) < 1e-12,
        "f1_ret_5_20": abs(out["f1_ret_5_20"] - expected_5_20) < 1e-12,
        "f2_amt_abn_zero": abs(out["f2_amt_abn"]) < 1e-12,
        "f2_turn_abn_zero": abs(out["f2_turn_abn"]) < 1e-12,
        "f3_mean_5": abs(out["f3_on_intra_mean_5"] - expected_mean_5) < 1e-12,
        "f3_path_eff_one": abs(out["f3_path_eff_20"] - 1.0) < 1e-9,
        "f3_downside_zero": abs(out["f3_downside_share_20"]) < 1e-12,
    }
    falling = pd.DataFrame({
        "日期": days, "代码": "SH000001",
        "open": [float(21 - i) for i in range(21)],
        "close": [float(21 - i) for i in range(21)],
        "amount": [100.0] * 21})
    down = day_features(falling, factor, ["SH000001"], days[-1]).iloc[0]
    checks["falling_path_eff_one"] = abs(down["f3_path_eff_20"] - 1.0) < 1e-9
    checks["falling_downside_one"] = abs(down["f3_downside_share_20"] - 1.0) < 1e-9
    sparse = market.copy()
    sparse.loc[: MIN_VALID - 1, "amount"] = np.nan
    short = day_features(sparse, factor, ["SH000001"], days[-1]).iloc[0]
    checks["short_history_amount_nan"] = bool(np.isnan(short["f2_amt_abn"]))
    early = day_features(market, factor, ["SH000001"], days[5]).iloc[0]
    checks["insufficient_price_history_nan"] = bool(np.isnan(early["f1_ret_5_20"]))
    checks = {key: bool(value) for key, value in checks.items()}
    checks["passed"] = all(checks.values())
    return checks


def build(data_dir: str = "Source/Data") -> dict:
    tables = dataio.load_all(data_dir)
    membership = tables[BARRA_TABLE].loc[:, ["日期", "代码"]]
    filtered = dict(tables)
    for name in (MARKET_TABLE, FACTOR_TABLE):
        filtered[name] = tables[name].merge(membership, on=["日期", "代码"],
                                            how="inner", sort=False)
    calendar = dataio.calendar(filtered)
    bounds = {name: _date_bounds(table) for name, table in filtered.items()}
    lookback = 120
    first_index = min(lookback - 1, max(0, len(calendar) - 1))
    signal_days = calendar[first_index:]
    market, factor, barra = (filtered[MARKET_TABLE], filtered[FACTOR_TABLE],
                             filtered[BARRA_TABLE])
    frames = []
    for count, day in enumerate(signal_days, start=1):
        pos = calendar.index(day)
        window_start = calendar[max(0, pos - lookback + 1)]
        m0, m1 = bounds[MARKET_TABLE][window_start][0], bounds[MARKET_TABLE][day][1]
        f0, f1 = bounds[FACTOR_TABLE][window_start][0], bounds[FACTOR_TABLE][day][1]
        barra_day = barra.loc[barra["日期"] == day]
        codes = barra_day["代码"].drop_duplicates().tolist()
        frame = day_features(market.iloc[m0:m1], factor.iloc[f0:f1], codes, day)
        frame.insert(0, "date", day)
        frames.append(frame)
        if count % 300 == 0:
            print(f"  v2 panel {count}/{len(signal_days)} days", flush=True)
    new = pd.concat(frames, ignore_index=True)
    new = new.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    base = pd.read_parquet(BASE).sort_values(["date", "code"]).reset_index(drop=True)
    keys_match = new[["date", "code"]].equals(base[["date", "code"]])
    merged = base.merge(new, on=["date", "code"], how="left",
                        validate="one_to_one", sort=False)
    merged = add_interactions(merged)
    V2.mkdir(parents=True, exist_ok=True)
    out = V2 / "panel_v2.parquet"
    merged.to_parquet(out, index=False)
    coverage = {column: float(merged[column].notna().mean())
                for column in day_features.__globals__["NEW_COLUMNS"]}
    report = {"panel": str(out), "base_rows": int(len(base)), "new_rows": int(len(new)),
              "keys_match_panel_platform": bool(keys_match), "coverage": coverage,
              "min_valid_history": MIN_VALID}
    report["passed"] = bool(keys_match)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_panel_report.json"))
    args = parser.parse_args()
    V2.mkdir(parents=True, exist_ok=True)
    report = {"arithmetic": verify_arithmetic(), "panel": build()}
    report["passed"] = bool(report["arithmetic"]["passed"] and report["panel"]["passed"])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
