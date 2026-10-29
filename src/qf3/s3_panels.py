"""S1 feature panels: L1 (ranked G1) and the G_sep / G_path additions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio, featurelib
from .constants import BARRA_TABLE, FACTOR_TABLE, G1_COLUMNS, MARKET_TABLE

S3 = Path("artifacts/stage3_ic_first")
BASE_PANEL = Path("artifacts/stage2_ic_ir/panel_platform.parquet")

G_SEP = ["sep_1_5", "sep_5_20"]
G_PATH = ["path_downvar_share", "path_efficiency", "path_drawdown"]


def _tradable(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr == 0.0, False)


def _shift_bool(arr: np.ndarray, steps: int) -> np.ndarray:
    """Shift a boolean mask, filling the leading border with False."""
    out = np.zeros(arr.shape, dtype=bool)
    if steps < arr.shape[1]:
        out[:, steps:] = arr[:, :-steps]
    return out


def path_features(market: pd.DataFrame, day: int, codes: list[str]) -> pd.DataFrame:
    """G_sep and G_path on the same 120-day window, same validity rules."""
    window = market.loc[market["日期"] <= day]
    dates = sorted(pd.unique(window["日期"]))
    close = featurelib._pivot(window, "close", codes, dates)
    susp = featurelib._pivot(window, "is_suspend", codes, dates)
    tradable = _tradable(susp)
    price_ok = np.isfinite(close) & (close > 0)

    prev1 = featurelib._shift(close, 1)
    prev5 = featurelib._shift(close, 5)
    prev20 = featurelib._shift(close, 20)
    t1 = _shift_bool(tradable, 1)
    t5 = _shift_bool(tradable, 5)
    t20 = _shift_bool(tradable, 20)

    with np.errstate(invalid="ignore", divide="ignore"):
        sep_1_5 = close / prev1 - 1.0
        sep_5_20 = prev5 / prev20 - 1.0
    sep_1_5[~(tradable & t1 & price_ok & (prev1 > 0))] = np.nan
    sep_5_20[~(t5 & t20 & np.isfinite(prev5) & (prev5 > 0)
               & np.isfinite(prev20) & (prev20 > 0))] = np.nan

    ret1 = np.full(close.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret1 = close / prev1 - 1.0
    ret1[~(tradable & t1 & price_ok & (prev1 > 0))] = np.nan

    out = pd.DataFrame(index=codes)
    last = -1
    out["sep_1_5"] = sep_1_5[:, last]
    out["sep_5_20"] = sep_5_20[:, last]

    # last 20 daily returns (t-19..t)
    recent = ret1[:, max(0, ret1.shape[1] - 20):]
    valid = np.isfinite(recent)
    count = valid.sum(axis=1)
    filled = np.where(valid, recent, 0.0)
    squares = np.sum(filled ** 2, axis=1)
    down = np.sum(np.where(filled < 0, filled ** 2, 0.0), axis=1)
    abs_sum = np.sum(np.abs(filled), axis=1)
    net = np.sum(filled, axis=1)
    eps = 1e-12
    share = down / (squares + eps)
    efficiency = np.abs(net) / (abs_sum + eps)
    enough = count >= 16
    share = np.where(enough, share, np.nan)
    efficiency = np.where(enough, efficiency, np.nan)
    out["path_downvar_share"] = share
    out["path_efficiency"] = efficiency

    recent_close = close[:, max(0, close.shape[1] - 20):]
    ok = np.isfinite(recent_close) & (recent_close > 0)
    enough_close = ok.sum(axis=1) >= 16
    masked = np.where(ok, recent_close, np.nan)
    with np.errstate(invalid="ignore"):
        highest = np.nanmax(masked, axis=1)
    drawdown = close[:, last] / highest - 1.0
    drawdown = np.where(enough_close & np.isfinite(highest) & (highest > 0)
                        & np.isfinite(close[:, last]), drawdown, np.nan)
    out["path_drawdown"] = drawdown
    return out


def build(data_dir: str = "Source/Data") -> pd.DataFrame:
    panel = pd.read_parquet(BASE_PANEL)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)

    # --- L1: rank the 19 G1 columns inside each day's pool ---------------
    for column in G1_COLUMNS:
        ranked = panel.groupby("date")[column].transform(
            lambda s: featurelib.cs_rank_center(s) if s.notna().any() else s)
        panel[f"L1_{column}"] = ranked

    # --- G_sep / G_path ---------------------------------------------------
    market = dataio.load_all(data_dir)[MARKET_TABLE]
    frames = []
    days = sorted(panel["date"].unique().tolist())
    started = time.time()
    for index, day in enumerate(days, start=1):
        block = panel.loc[panel["date"] == day, ["code"]]
        codes = block["code"].tolist()
        day_int = int(day)
        window = market.loc[market["日期"] <= day_int]
        first = sorted(pd.unique(window["日期"]))
        first = first[max(0, len(first) - 120)]
        window = market.loc[(market["日期"] >= first) & (market["日期"] <= day_int)]
        features = path_features(window, day_int, codes).reset_index()
        features.columns = ["code", *G_SEP, *G_PATH]
        features["date"] = day_int
        frames.append(features)
        if index % 400 == 0:
            print(f"  path features {index}/{len(days)}", flush=True)
    extras = pd.concat(frames, ignore_index=True)
    panel = panel.merge(extras, on=["date", "code"], how="left",
                        validate="one_to_one")
    print("built in", round(time.time() - started, 1), "s", panel.shape)
    return panel


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    panel = build()
    panel.to_parquet(S3 / "panel_ext.parquet", index=False)
    meta = {
        "base": str(BASE_PANEL),
        "rows": int(len(panel)),
        "l1_columns": [f"L1_{c}" for c in G1_COLUMNS],
        "g_sep": G_SEP, "g_path": G_PATH,
        "note": "L1 = daily centred rank of G1 with no max-abs rescaling; "
                "G_path uses the last 20 daily returns (>=16 valid) and 20 visible closes",
    }
    (S3 / "panel_ext_meta.json").write_text(json.dumps(meta, indent=2))
    missing = {c: float(panel[c].isna().mean()) for c in [*G_SEP, *G_PATH]}
    print(json.dumps(missing, indent=2))


if __name__ == "__main__":
    main()
