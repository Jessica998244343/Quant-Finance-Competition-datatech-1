"""A-stage input parity: platform research semantics vs the legacy offline panel."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio, featurelib
from .constants import BARRA_TABLE, FACTOR_TABLE, FEATURE_COLUMNS, MARKET_TABLE

OUT = Path("artifacts/stage2_ic_ir")


def calendar_of(tables: dict) -> list[int]:
    return sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())


def legacy_window(tables: dict, day: int, lookback: int = 120) -> dict:
    """Current offline semantics: date slice over the annual union."""
    calendar = calendar_of(tables)
    pos = calendar.index(day)
    first = calendar[max(0, pos - lookback + 1)]
    return {name: table.loc[(table["日期"] >= first) & (table["日期"] <= day)]
            for name, table in tables.items()}


def platform_window(tables: dict, day: int, lookback: int = 120) -> dict:
    """Platform research semantics: inner join with that day's Barra membership."""
    window = legacy_window(tables, day, lookback)
    membership = tables[BARRA_TABLE].loc[:, ["日期", "代码"]]
    out = {}
    for name in (FACTOR_TABLE, MARKET_TABLE):
        out[name] = window[name].merge(membership, on=["日期", "代码"], how="inner",
                                       sort=False)
    out[BARRA_TABLE] = window[BARRA_TABLE]
    return out


def _feature_matrix(window: dict, day: int) -> pd.DataFrame:
    frame = featurelib.build_day_features(
        window[MARKET_TABLE], window[FACTOR_TABLE], window[BARRA_TABLE], day)
    return frame.set_index("code")


def affected_codes(tables: dict, day: int, lookback: int = 120) -> dict:
    """Stocks in the day-t universe whose membership rows inside the window are incomplete."""
    calendar = calendar_of(tables)
    pos = calendar.index(day)
    first = calendar[max(0, pos - lookback + 1)]
    barra = tables[BARRA_TABLE]
    universe = set(barra.loc[barra["日期"] == day, "代码"])
    window_rows = barra.loc[(barra["日期"] >= first) & (barra["日期"] <= day)]
    present = window_rows.groupby("代码").size()
    n_days = int(window_rows["日期"].nunique())
    incomplete = {code for code in universe if int(present.get(code, 0)) < n_days}
    return {
        "day": day,
        "window_days": n_days,
        "universe": len(universe),
        "incomplete_membership_codes": sorted(incomplete),
        "n_incomplete": len(incomplete),
        "fraction_incomplete": len(incomplete) / max(1, len(universe)),
    }


def compare_day(tables: dict, day: int, lookback: int = 120) -> dict:
    legacy = legacy_window(tables, day, lookback)
    platform = platform_window(tables, day, lookback)
    legacy_feat = _feature_matrix(legacy, day)
    platform_feat = _feature_matrix(platform, day)

    universe = platform_feat.index
    legacy_feat = legacy_feat.reindex(universe)
    diff = np.abs(legacy_feat[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
                  - platform_feat[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    finite = np.isfinite(diff)
    per_code_max = np.nanmax(np.where(finite, diff, np.nan), axis=1)
    nan_mask_diff = (np.isnan(legacy_feat[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
                     != np.isnan(platform_feat[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))
    codes_changed = universe[(np.nan_to_num(per_code_max, nan=0.0) > 0)
                             | nan_mask_diff.any(axis=1)]

    info = affected_codes(tables, day, lookback)
    return {
        "day": day,
        "n_universe": int(len(universe)),
        "n_codes_feature_changed": int(len(codes_changed)),
        "changed_codes": list(codes_changed),
        "changed_is_subset_of_incomplete": bool(set(codes_changed)
                                               <= set(info["incomplete_membership_codes"])),
        "max_abs_feature_diff": float(np.nanmax(diff[finite])) if finite.any() else 0.0,
        "n_nan_mask_changed_cells": int(nan_mask_diff.sum()),
        "legacy_rows": int(len(legacy[MARKET_TABLE])),
        "platform_rows": int(len(platform[MARKET_TABLE])),
        "legacy_rows_dropped_fraction": float(
            1.0 - len(platform[MARKET_TABLE]) / max(1, len(legacy[MARKET_TABLE]))),
        "incomplete_membership": info,
    }


def _constituent_change_dates(tables: dict, year: int) -> list[int]:
    barra = tables[BARRA_TABLE]
    days = sorted(pd.unique(barra.loc[barra["日期"] // 10000 == year, "日期"]).tolist())
    sets = {day: set(barra.loc[barra["日期"] == day, "代码"]) for day in days}
    changes = []
    for previous, current in zip(days, days[1:]):
        if sets[previous] != sets[current]:
            changes.append(current)
    return changes


def select_dates(tables: dict, year: int = 2019) -> dict:
    calendar = calendar_of(tables)
    days = [d for d in calendar if d // 10000 == year]
    changes = _constituent_change_dates(tables, year)
    change_adjacent = []
    for day in changes:
        idx = calendar.index(day)
        change_adjacent.extend(calendar[max(0, idx - 1):idx + 2])
    normal = [d for d in days if d not in change_adjacent]
    return {
        "constituent_change_days": changes,
        "change_adjacent": sorted(set(change_adjacent)),
        "normal_sample": normal[:: max(1, len(normal) // 6)][:6],
        "early_short_history": [d for d in calendar if d < 20160701][:3],
    }


def run(days: list[int], data_dir: str = "Source/Data", lookback: int = 120) -> dict:
    tables = dataio.load_all(data_dir)
    records = []
    started = time.time()
    for day in days:
        records.append(compare_day(tables, day, lookback))
    summary = {
        "data_dir": data_dir,
        "lookback": lookback,
        "n_days": len(records),
        "seconds": round(time.time() - started, 1),
        "legacy_rows_dropped_fraction_min": min(r["legacy_rows_dropped_fraction"]
                                                for r in records),
        "legacy_rows_dropped_fraction_max": max(r["legacy_rows_dropped_fraction"]
                                                for r in records),
        "max_abs_feature_diff": max(r["max_abs_feature_diff"] for r in records),
        "n_days_with_changed_features": sum(1 for r in records
                                            if r["n_codes_feature_changed"] > 0),
        "max_codes_feature_changed": max(r["n_codes_feature_changed"] for r in records),
        "changed_always_subset_of_incomplete": all(r["changed_is_subset_of_incomplete"]
                                                   for r in records),
        "records": records,
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = dataio.load_all("Source/Data")
    selection = select_dates(tables, 2019)
    days = sorted(set(selection["change_adjacent"] + selection["normal_sample"]))
    summary = run(days)
    (OUT / "input_parity.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "date_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    for record in summary["records"]:
        print(f"  {record['day']}: changed {record['n_codes_feature_changed']}/"
              f"{record['n_universe']} rows_dropped {record['legacy_rows_dropped_fraction']:.1%} "
              f"incomplete_membership {record['incomplete_membership']['n_incomplete']}")


if __name__ == "__main__":
    main()
