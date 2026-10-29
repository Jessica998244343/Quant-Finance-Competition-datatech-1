"""§11: F1-F3 incremental features, defined on legal endpoints and recomputable.

Every column is a deterministic function of the provider-visible window:

  F1  f1_ret_1_5  = C[t-1]/C[t-5] - 1        (never C[t]/C[t-1]-1)
      f1_ret_5_20 = C[t-5]/C[t-20] - 1
  F2  f2_amt_abn  = log1p(amount_t)  - median(log1p(amount[t-20:t-1]))
      f2_turn_abn = log1p(turn_t)    - median(log1p(turn[t-20:t-1]))
      (+ two cross-sectional rank interactions built in the panel step)
  F3  f3_on_intra_mean_5 / _20 = mean over the window of (log(O_j/C[j-1]) - log(C_j/O_j))
      f3_path_eff_20   = |sum(r)| / (sum|r| + eps), r = 20 one-day log returns
      f3_downside_share_20 = sum(min(r,0)^2) / (sum(r^2) + eps)

Minimum valid history is 16 observations in the 20-day windows; anything shorter
stays NaN (never forward-filled, never zero-filled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

F1_COLUMNS = ("f1_ret_1_5", "f1_ret_5_20")
F2_COLUMNS = ("f2_amt_abn", "f2_turn_abn")
F3_COLUMNS = ("f3_on_intra_mean_5", "f3_on_intra_mean_20",
              "f3_path_eff_20", "f3_downside_share_20")
F2_INTERACTIONS = ("f2_rank_ret1_x_amt", "f2_rank_ret5_x_amt")
NEW_COLUMNS = F1_COLUMNS + F2_COLUMNS + F3_COLUMNS + F2_INTERACTIONS
GROUPS = {"F1": F1_COLUMNS, "F2": F2_COLUMNS + F2_INTERACTIONS, "F3": F3_COLUMNS}
EPS = 1e-12
MIN_VALID = 16


def _pivot(table: pd.DataFrame, column: str, codes, dates) -> np.ndarray:
    sub = table.loc[:, ["代码", "日期", column]].dropna(subset=["代码", "日期"])
    sub = sub.drop_duplicates(["代码", "日期"], keep="first")
    wide = sub.pivot(index="代码", columns="日期", values=column)
    return wide.reindex(index=codes, columns=dates).to_numpy(dtype=float)


def day_features(market: pd.DataFrame, factor: pd.DataFrame, codes, day: int,
                 lookback: int = 120) -> pd.DataFrame:
    """New columns for the legal universe on ``day`` (caller supplies the window)."""
    codes = list(codes)
    out = pd.DataFrame({"code": codes})
    for column in NEW_COLUMNS:
        out[column] = np.nan
    if not codes:
        return out
    market = market.loc[market["日期"] <= day]
    factor = factor.loc[factor["日期"] <= day]
    dates = sorted(pd.unique(market["日期"]).tolist())
    if day not in dates:
        return out
    pos = dates.index(day)
    close = _pivot(market, "close", codes, dates)
    open_ = _pivot(market, "open", codes, dates)
    amount = _pivot(market, "amount", codes, dates)
    turn = _pivot(factor, "turn", codes, dates)
    n = len(codes)

    def at(array: np.ndarray, offset: int) -> np.ndarray:
        index = pos - offset
        if index < 0:
            return np.full(n, np.nan)
        return array[:, index]

    # ---- F1: split the recent reversal from the earlier trend ------------
    c1, c5, c20 = at(close, 1), at(close, 5), at(close, 20)
    out["f1_ret_1_5"] = np.where((c5 > 0) & (c1 > 0), c1 / c5 - 1.0, np.nan)
    out["f1_ret_5_20"] = np.where((c20 > 0) & (c5 > 0), c5 / c20 - 1.0, np.nan)

    # ---- F2: 20-day median-relative anomalies (history excludes t) ------
    first = max(0, pos - 20)
    history = slice(first, pos)                      # t-20 .. t-1
    for name, array, raw in (("f2_amt_abn", amount, at(amount, 0)),
                             ("f2_turn_abn", turn, at(turn, 0))):
        window = np.log1p(np.where(array[:, history] > 0, array[:, history], np.nan))
        valid = np.isfinite(window).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(window, axis=1)
        current = np.log1p(np.where(raw > 0, raw, np.nan))
        value = current - median
        value[valid < MIN_VALID] = np.nan
        out[name] = value

    # ---- F3: overnight/intraday persistence and path statistics ---------
    overnight = np.full_like(close, np.nan)
    intraday = np.full_like(close, np.nan)
    log_ret = np.full_like(close, np.nan)
    if len(dates) > 1:
        prev_close = close[:, :-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            overnight[:, 1:] = np.log(np.where((prev_close > 0) & (open_[:, 1:] > 0),
                                               open_[:, 1:] / prev_close, np.nan))
            intraday[:, 1:] = np.log(np.where(open_[:, 1:] > 0,
                                              close[:, 1:] / open_[:, 1:], np.nan))
            log_ret[:, 1:] = np.log(np.where((prev_close > 0) & (close[:, 1:] > 0),
                                             close[:, 1:] / prev_close, np.nan))
    diff = overnight - intraday
    for label, days in (("f3_on_intra_mean_5", 5), ("f3_on_intra_mean_20", 20)):
        window = diff[:, max(0, pos - days + 1):pos + 1]
        valid = np.isfinite(window).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = np.nanmean(np.where(np.isfinite(window), window, np.nan), axis=1)
        value[valid < min(MIN_VALID, days)] = np.nan
        out[label] = value
    returns = log_ret[:, max(0, pos - 19):pos + 1]           # 20 one-day returns
    valid = np.isfinite(returns).sum(axis=1)
    total = np.nansum(returns, axis=1)
    absolute = np.nansum(np.abs(returns), axis=1)
    downside = np.nansum(np.minimum(returns, 0.0) ** 2, axis=1)
    squares = np.nansum(returns ** 2, axis=1)
    efficiency = np.abs(total) / (absolute + EPS)
    share = downside / (squares + EPS)
    short = valid < MIN_VALID
    efficiency[short] = np.nan
    share[short] = np.nan
    out["f3_path_eff_20"] = efficiency
    out["f3_downside_share_20"] = share
    return out


def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """F2 interactions: centred cross-sectional rank products on the visible pool."""
    out = frame.copy()
    for rank_col, name in (("ret_1", "f2_rank_ret1_x_amt"),
                           ("ret_5", "f2_rank_ret5_x_amt")):
        pieces = []
        for _, block in out.groupby("date", sort=False):
            def centred(series):
                ranks = series.rank(method="average")
                return (ranks - (series.notna().sum() + 1) / 2.0) / series.notna().sum()
            pieces.append((centred(block[rank_col]) * centred(block["f2_amt_abn"])).to_numpy())
        out[name] = np.concatenate(pieces) if pieces else np.nan
    return out
