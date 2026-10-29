"""Day-level cross-sectional feature construction.

The same function is used by the offline panel builder and by the submission
``predict`` path, so training and inference cannot drift apart.

All market-history features obey the official 120 trading-day look-back limit.
Nothing here reads the banned ``名称`` / ``is_st`` columns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import (
    BARRA_FEATURE_NAMES,
    BARRA_STYLE_COLUMNS,
    F0_FACTOR_COLUMNS,
    FEATURE_COLUMNS,
    STATUS_COLUMNS,
)

_EPS = 1e-12


# --------------------------------------------------------------------------
# 2-D rolling helpers (rows = stocks, columns = trading days)
# --------------------------------------------------------------------------
def _fill_nan(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    return np.where(finite, arr, 0.0), finite


def _window_sums(arr: np.ndarray, k: int):
    filled, finite = _fill_nan(arr)
    counts = finite.astype(np.float64)
    cs = np.cumsum(filled, axis=1)
    cs2 = np.cumsum(filled * filled, axis=1)
    cc = np.cumsum(counts, axis=1)
    zeros = np.zeros((arr.shape[0], 1))
    cs = np.concatenate([zeros, cs], axis=1)
    cs2 = np.concatenate([zeros, cs2], axis=1)
    cc = np.concatenate([zeros, cc], axis=1)
    if k == 0:
        raise ValueError("window length must be positive")
    total = cs[:, k:] - cs[:, :-k]
    total2 = cs2[:, k:] - cs2[:, :-k]
    cnt = cc[:, k:] - cc[:, :-k]
    out = np.full(arr.shape, np.nan)
    out[:, k - 1:] = total
    out2 = np.full(arr.shape, np.nan)
    out2[:, k - 1:] = total2
    cnt_out = np.zeros(arr.shape)
    cnt_out[:, k - 1:] = cnt
    return out, out2, cnt_out


def _rolling_mean(arr: np.ndarray, k: int, min_count: int) -> np.ndarray:
    total, _, cnt = _window_sums(arr, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / cnt
    mean[cnt < min_count] = np.nan
    return mean


def _rolling_std(arr: np.ndarray, k: int, min_count: int) -> np.ndarray:
    total, total2, cnt = _window_sums(arr, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / cnt
        var = (total2 - total * total / cnt) / (cnt - 1.0)
    var = np.where(var > 0, var, 0.0)
    std = np.sqrt(var)
    std[cnt < min_count] = np.nan
    return std


def _shift(arr: np.ndarray, steps: int = 1) -> np.ndarray:
    out = np.full(arr.shape, np.nan)
    if steps < arr.shape[1]:
        out[:, steps:] = arr[:, :-steps]
    return out


def _min_count(k: int, ratio: float = 0.8, floor: int = 2) -> int:
    return max(floor, int(math.ceil(k * ratio)))


# --------------------------------------------------------------------------
# cross-sectional transforms
# --------------------------------------------------------------------------
def cs_rank_center(values: pd.Series) -> pd.Series:
    """Daily cross-sectional centred rank in (-0.5, 0.5); constants map to 0."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    v = v.replace([np.inf, -np.inf], np.nan)
    finite = v.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    n = int(finite.sum())
    if n == 0:
        return out
    ranks = v[finite].rank(method="average").to_numpy()
    if n > 1:
        scaled = (ranks - ranks.mean()) / n
        span = np.abs(scaled).max()
        if span > 0:
            scaled = scaled / (2.0 * span)
    else:
        scaled = np.zeros(1)
    out.loc[finite] = scaled
    return out


def _pivot(df: pd.DataFrame, column: str, codes, dates) -> np.ndarray:
    table = df.loc[:, ["代码", "日期", column]].dropna(subset=["代码", "日期"])
    if table.duplicated(["代码", "日期"]).any():
        table = table.drop_duplicates(["代码", "日期"], keep="last")
    wide = table.pivot(index="代码", columns="日期", values=column)
    wide = wide.reindex(index=codes, columns=dates)
    return wide.to_numpy(dtype=float)


def feature_catalog() -> list[dict]:
    """Human-readable definition of every model feature."""
    rows = []
    for name in F0_FACTOR_COLUMNS:
        rows.append({"feature": name, "group": "F0",
                     "definition": f"Factor33_winsor.{name} (current day)",
                     "window": 0, "transform": "raw" if name in STATUS_COLUMNS else "cs_rank"})
    for name, source in zip(BARRA_FEATURE_NAMES, BARRA_STYLE_COLUMNS):
        rows.append({"feature": name, "group": "F0",
                     "definition": f"Barra_factor.{source} (current day) - 0.5",
                     "window": 0, "transform": "raw_minus_half"})
    g1 = {
        "ret_1": "C_t/C_(t-1)-1", "ret_2": "C_t/C_(t-2)-1", "ret_5": "C_t/C_(t-5)-1",
        "ret_10": "C_t/C_(t-10)-1", "ret_20": "C_t/C_(t-20)-1",
        "ret_60": "C_t/C_(t-60)-1", "ret_119": "C_t/C_(t-119)-1",
        "rv_5": "std(r, 5)", "rv_20": "std(r, 20)", "rv_60": "std(r, 60)",
        "relma_5": "C_t/mean(C[t-4..t])-1", "relma_20": "C_t/mean(C[t-19..t])-1",
        "relma_60": "C_t/mean(C[t-59..t])-1", "hl_range": "H_t/L_t-1",
        "close_pos": "(2C_t-H_t-L_t)/(H_t-L_t)",
        "overnight": "O_t/C_(t-1)-1", "intraday": "C_t/O_t-1",
        "overnight_mean_5": "mean(overnight, 5)", "intraday_mean_5": "mean(intraday, 5)",
    }
    for name, definition in g1.items():
        rows.append({"feature": name, "group": "G1", "definition": definition,
                     "window": 120, "transform": "cs_rank"})
    g2 = {
        "amt_abn": "A_t/mean(A[t-20..t-1])-1", "vol_abn": "V_t/mean(V[t-20..t-1])-1",
        "turn_abn": "turn_t/mean(turn[t-20..t-1])-1", "std_shock": "r_t/std(r[t-20..t-1])",
        "dir_x_amt": "cs_rank(r_t)*cs_rank(amt_abn)",
        "on_intra_diff": "overnight_intraday",
        "on_intra_diff_z": "(overnight-intraday)/std(r[t-20..t-1])",
    }
    for name, definition in g2.items():
        rows.append({"feature": name, "group": "G2", "definition": definition,
                     "window": 22, "transform": "cs_rank"})
    g3 = {
        "g3_roe_minus_pb": "rank(roe_avg_ttm)-rank(pb)",
        "g3_roa_minus_debt": "rank(roa_avg_ttm)-rank(debt_to_asset_lf)",
        "g3_gross_minus_net": "rank(gross_profit_rate_ttm)-rank(net_profit_rate_ttm)",
        "g3_rev5_x_roe": "rank(reversal_5)*rank(roe_avg_ttm)",
    }
    for name, definition in g3.items():
        rows.append({"feature": name, "group": "G3", "definition": definition,
                     "window": 0, "transform": "cs_rank"})
    return rows


def build_day_features(
    market: pd.DataFrame,
    factor: pd.DataFrame,
    barra: pd.DataFrame,
    day_int: int,
) -> pd.DataFrame:
    """Feature matrix for the legal universe on ``day_int``.

    ``market`` / ``factor`` are the research-window long tables (at most the
    last ``lookback`` trading days, ending at ``day_int``); ``barra`` defines
    the legal universe for that day.
    """
    barra_day = barra.loc[barra["日期"] == day_int]
    codes = barra_day["代码"].drop_duplicates().tolist()
    out = pd.DataFrame({"date": day_int, "code": codes})
    if not codes:
        for name in FEATURE_COLUMNS:
            out[name] = np.nan
        return out

    market = market.loc[market["日期"] <= day_int]
    factor = factor.loc[factor["日期"] <= day_int]
    dates = sorted(pd.unique(market["日期"]))
    if not dates:
        for name in FEATURE_COLUMNS:
            out[name] = np.nan
        return out

    close = _pivot(market, "close", codes, dates)
    open_ = _pivot(market, "open", codes, dates)
    high = _pivot(market, "high", codes, dates)
    low = _pivot(market, "low", codes, dates)
    amount = _pivot(market, "amount", codes, dates)
    volume = _pivot(market, "volume", codes, dates)
    susp = _pivot(market, "is_suspend", codes, dates)
    turn_hist = _pivot(factor, "turn", codes, dates)

    tradable = np.where(np.isfinite(susp), susp == 0.0, False)
    price_ok = np.isfinite(close) & (close > 0)

    raw = {}
    prev_close = _shift(close)
    prev_susp = _shift(tradable.astype(float))
    prev_ok = np.where(np.isfinite(prev_susp), prev_susp == 1.0, False)

    with np.errstate(invalid="ignore", divide="ignore"):
        ret1 = close / prev_close - 1.0
    ret1[~(tradable & prev_ok & price_ok & np.isfinite(prev_close) & (prev_close > 0))] = np.nan
    raw["_ret"] = ret1

    for k in (1, 2, 5, 10, 20, 60, 119):
        prev = _shift(close, k)
        stos = _shift(tradable.astype(float), k)
        stos_ok = np.where(np.isfinite(stos), stos == 1.0, False)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = close / prev - 1.0
        value[~(tradable & stos_ok & price_ok & np.isfinite(prev) & (prev > 0))] = np.nan
        raw[f"ret_{k}"] = value

    for k in (5, 20, 60):
        raw[f"rv_{k}"] = _rolling_std(ret1, k, _min_count(k))
        mean_close = _rolling_mean(close, k, _min_count(k))
        with np.errstate(invalid="ignore", divide="ignore"):
            relma = close / mean_close - 1.0
        relma[~(tradable & price_ok & (mean_close > 0))] = np.nan
        raw[f"relma_{k}"] = relma

    with np.errstate(invalid="ignore", divide="ignore"):
        hl = high / low - 1.0
        hl[(~(np.isfinite(high) & np.isfinite(low) & (low > 0) & (high >= low))) | ~tradable] = np.nan
        raw["hl_range"] = hl
        denom = high - low
        pos = (2.0 * close - high - low) / denom
        pos[~((denom > 0) & tradable & price_ok)] = np.nan
        raw["close_pos"] = pos
        overnight = open_ / prev_close - 1.0
        overnight[~(tradable & prev_ok & np.isfinite(open_) & (open_ > 0)
                    & np.isfinite(prev_close) & (prev_close > 0))] = np.nan
        raw["overnight"] = overnight
        intraday = close / open_ - 1.0
        intraday[~(tradable & price_ok & np.isfinite(open_) & (open_ > 0))] = np.nan
        raw["intraday"] = intraday
    raw["overnight_mean_5"] = _rolling_mean(overnight, 5, _min_count(5, 0.8))
    raw["intraday_mean_5"] = _rolling_mean(intraday, 5, _min_count(5, 0.8))

    amt_prev = _shift(amount)
    vol_prev = _shift(volume)
    turn_prev = _shift(turn_hist)
    amt_base = _rolling_mean(amt_prev, 20, _min_count(20, 0.8, 2))
    vol_base = _rolling_mean(vol_prev, 20, _min_count(20, 0.8, 2))
    turn_base = _rolling_mean(turn_prev, 20, _min_count(20, 0.8, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        amt_abn = amount / amt_base - 1.0
        vol_abn = volume / vol_base - 1.0
        turn_abn = turn_hist / turn_base - 1.0
    trade_ok = tradable & np.isfinite(amount) & (amount >= 0)
    amt_abn[~(trade_ok & (amt_base > 0))] = np.nan
    vol_abn[~(trade_ok & (vol_base > 0))] = np.nan
    turn_abn[~(tradable & np.isfinite(turn_hist) & (turn_base > 0))] = np.nan
    raw["amt_abn"] = amt_abn
    raw["vol_abn"] = vol_abn
    raw["turn_abn"] = turn_abn

    ret_prev_std = _rolling_std(_shift(ret1), 20, _min_count(20, 0.8))
    with np.errstate(invalid="ignore", divide="ignore"):
        std_shock = ret1 / ret_prev_std
    std_shock[~(np.isfinite(ret_prev_std) & (ret_prev_std > 0))] = np.nan
    raw["std_shock"] = std_shock
    diff = overnight - intraday
    raw["on_intra_diff"] = diff
    with np.errstate(invalid="ignore", divide="ignore"):
        diff_z = diff / ret_prev_std
    diff_z[~(np.isfinite(ret_prev_std) & (ret_prev_std > 0))] = np.nan
    raw["on_intra_diff_z"] = diff_z

    last = -1  # current day is the final column of the aligned window
    for name in ("ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "ret_60", "ret_119",
                 "rv_5", "rv_20", "rv_60", "relma_5", "relma_20", "relma_60",
                 "hl_range", "close_pos", "overnight", "intraday",
                 "overnight_mean_5", "intraday_mean_5", "amt_abn", "vol_abn",
                 "turn_abn", "std_shock", "on_intra_diff", "on_intra_diff_z"):
        out[name] = raw[name][:, last]

    # --- F0 current-day snapshot -----------------------------------------
    factor_day = factor.loc[factor["日期"] == day_int, ["代码", *F0_FACTOR_COLUMNS]]
    factor_day = factor_day.drop_duplicates("代码").set_index("代码").reindex(codes)
    barra_day_indexed = barra_day.drop_duplicates("代码").set_index("代码").reindex(codes)

    for name in F0_FACTOR_COLUMNS:
        series = pd.to_numeric(factor_day[name], errors="coerce")
        if name in STATUS_COLUMNS:
            out[name] = series.to_numpy(dtype=float)
        else:
            out[name] = cs_rank_center(series).to_numpy(dtype=float)
    for name, source in zip(BARRA_FEATURE_NAMES, BARRA_STYLE_COLUMNS):
        style = pd.to_numeric(barra_day_indexed[source], errors="coerce")
        out[name] = (style - 0.5).to_numpy(dtype=float)

    # --- G2 / G3 interaction terms (need cross-sectional ranks) ----------
    rank_ret = cs_rank_center(pd.Series(raw["ret_1"][:, last], index=out.index))
    rank_amt = cs_rank_center(pd.Series(amt_abn[:, last], index=out.index))
    out["dir_x_amt"] = (rank_ret * rank_amt).to_numpy(dtype=float)

    f33_rank = {
        name: cs_rank_center(pd.to_numeric(factor_day[name], errors="coerce"))
        for name in ("roe_avg_ttm", "roa_avg_ttm", "gross_profit_rate_ttm",
                     "net_profit_rate_ttm", "debt_to_asset_lf", "pb", "reversal_5")
    }
    out["g3_roe_minus_pb"] = (f33_rank["roe_avg_ttm"] - f33_rank["pb"]).to_numpy()
    out["g3_roa_minus_debt"] = (f33_rank["roa_avg_ttm"] - f33_rank["debt_to_asset_lf"]).to_numpy()
    out["g3_gross_minus_net"] = (
        f33_rank["gross_profit_rate_ttm"] - f33_rank["net_profit_rate_ttm"]
    ).to_numpy()
    out["g3_rev5_x_roe"] = (f33_rank["reversal_5"] * f33_rank["roe_avg_ttm"]).to_numpy()

    for name in FEATURE_COLUMNS:
        if name not in out.columns:
            out[name] = np.nan
    return out.loc[:, ["date", "code", *FEATURE_COLUMNS]]
