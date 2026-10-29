"""REV1: input verification through the locked platform's real research entry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skd_backtest.config import BacktestConfig
from skd_backtest.data_provider import DataProvider

from . import featurelib
from .constants import (BARRA_TABLE, F0_COLUMNS, FACTOR_TABLE, G1_COLUMNS,
                        MARKET_TABLE)

REV = Path("artifacts/stage2_ic_ir/a_revision1")
MODEL_FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
ALL_FEATURES = list(featurelib.FEATURE_COLUMNS)


def _iso(day: int) -> str:
    text = str(day)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def provider_as_of(day: int, start_date: str, lookback: int = 120,
                   data_dir: str = "Source/Data") -> dict:
    config = BacktestConfig(
        data_dir=Path(data_dir), start_date=start_date, end_date=_iso(day),
        initial_cash=1_000_000.0, rebalance_interval=1, holding_period=1,
        lookback=lookback, price_mode="adjusted_return", trading_days_per_year=252,
        risk_free_rate=0.0, output_dir=None, read_batch_months=3,
        prefetch=False, async_inference=False)
    provider = DataProvider(config)
    try:
        provider.prepare()
        return provider.as_of(_iso(day))
    finally:
        provider.close()


def legacy_window(tables: dict, day: int, lookback: int = 120) -> dict:
    calendar = sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
    pos = calendar.index(day)
    first = calendar[max(0, pos - lookback + 1)]
    return {name: table.loc[(table["日期"] >= first) & (table["日期"] <= day)]
            for name, table in tables.items()}


def _features(window: dict, day: int) -> pd.DataFrame:
    frame = featurelib.build_day_features(
        window[MARKET_TABLE], window[FACTOR_TABLE], window[BARRA_TABLE], day)
    return frame.set_index("code")


def compare_features(left: pd.DataFrame, right: pd.DataFrame,
                     columns: list[str]) -> dict:
    common = left.index.intersection(right.index)
    a = left.reindex(common)[columns].to_numpy(dtype=float)
    b = right.reindex(common)[columns].to_numpy(dtype=float)
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    nan_left = np.isnan(a)
    nan_right = np.isnan(b)
    nan_mask_diff = nan_left != nan_right
    out = {
        "n_common_stocks": int(len(common)),
        "n_columns": len(columns),
        "cells": int(diff.size),
        "cells_exact_diff_gt0": int(np.nansum(diff > 0)),
        "cells_gt_1e-12": int(np.nansum(diff > 1e-12)),
        "cells_gt_1e-9": int(np.nansum(diff > 1e-9)),
        "cells_gt_1e-6": int(np.nansum(diff > 1e-6)),
        "max_abs_finite_diff": float(np.nanmax(diff[finite])) if finite.any() else 0.0,
        "cells_nan_mask_diff": int(nan_mask_diff.sum()),
        "columns_with_material_diff": sorted(
            [columns[j] for j in range(len(columns))
             if np.nansum(diff[:, j] > 1e-9) > 0 or nan_mask_diff[:, j].any()]),
    }
    per_column = []
    for j, name in enumerate(columns):
        material = (diff[:, j] > 1e-9) | nan_mask_diff[:, j]
        per_column.append({"feature": name,
                           "n_stocks_affected": int(np.nansum(material)),
                           "max_abs_diff": float(np.nanmax(diff[:, j]))
                           if np.isfinite(diff[:, j]).any() else 0.0,
                           "nan_mask_diff_cells": int(nan_mask_diff[:, j].sum())})
    out["per_column"] = per_column
    return out


def run(dates: list[dict], data_dir: str = "Source/Data") -> dict:
    from . import dataio
    full = dataio.load_all(data_dir)
    records = []
    for item in dates:
        day = item["day"]
        start = item["start_date"]
        provider = provider_as_of(day, start)
        legacy = legacy_window(full, day)
        provider_feat = _features(provider, day)
        legacy_feat = _features(legacy, day)
        record = {
            "day": day,
            "label": item["label"],
            "provider_start_date": start,
            "provider_window_days": int(provider[MARKET_TABLE]["日期"].nunique()),
            "provider_rows": {k: int(len(v)) for k, v in provider.items()},
            "legacy_rows": int(len(legacy[MARKET_TABLE])),
            "model_features": compare_features(legacy_feat, provider_feat, MODEL_FEATURES),
            "all_features": compare_features(legacy_feat, provider_feat, ALL_FEATURES),
        }
        # provider vs my own platform-semantics reconstruction
        membership = full[BARRA_TABLE].loc[:, ["日期", "代码"]]
        rebuilt = {name: legacy[name].merge(membership, on=["日期", "代码"],
                                            how="inner", sort=False)
                   for name in (MARKET_TABLE, FACTOR_TABLE)}
        rebuilt[BARRA_TABLE] = legacy[BARRA_TABLE]
        key_provider = set(map(tuple, provider[MARKET_TABLE][["日期", "代码"]]
                               .to_numpy().tolist()))
        key_rebuilt = set(map(tuple, rebuilt[MARKET_TABLE][["日期", "代码"]]
                              .to_numpy().tolist()))
        record["provider_matches_reconstruction_keys"] = key_provider == key_rebuilt
        record["provider_vs_reconstruction_rows"] = {
            "provider": int(len(provider[MARKET_TABLE])),
            "rebuilt": int(len(rebuilt[MARKET_TABLE]))}
        records.append(record)
    return {"data_dir": data_dir, "records": records}


def main() -> None:
    REV.mkdir(parents=True, exist_ok=True)
    dates = [
        {"day": 20190603, "start_date": "2019-01-02", "label": "normal day"},
        {"day": 20190617, "start_date": "2019-01-02",
         "label": "constituent-change adjacent"},
        {"day": 20160630, "start_date": "2016-01-04", "label": "first panel day (full window)"},
        {"day": 20160105, "start_date": "2016-01-04", "label": "short history (2 days)"},
    ]
    result = run(dates)
    (REV / "rev1_input_parity.json").write_text(json.dumps(result, indent=2, default=str))
    for record in result["records"]:
        mf = record["model_features"]
        print(f"{record['day']} [{record['label']}] window_days={record['provider_window_days']} "
              f"provider_rows={record['provider_rows'][MARKET_TABLE]} "
              f"legacy_rows={record['legacy_rows']}")
        print(f"   61 model features: max_diff={mf['max_abs_finite_diff']:.3e} "
              f">1e-9 cells={mf['cells_gt_1e-9']} nan_mask_cells={mf['cells_nan_mask_diff']} "
              f"material_columns={mf['columns_with_material_diff']}")
        print(f"   provider==reconstruction keys: "
              f"{record['provider_matches_reconstruction_keys']}")


if __name__ == "__main__":
    main()
