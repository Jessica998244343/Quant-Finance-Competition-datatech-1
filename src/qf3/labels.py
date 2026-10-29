"""Open-to-open labels replicating the skd-backtest label contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MARKET_TABLE


def _wide(table: pd.DataFrame, column: str, codes, dates) -> np.ndarray:
    sub = table.loc[:, ["代码", "日期", column]].dropna(subset=["代码", "日期"])
    wide = sub.pivot(index="代码", columns="日期", values=column).reindex(index=codes, columns=dates)
    return wide.to_numpy(dtype=float)


def build_labels(tables: dict[str, pd.DataFrame], calendar: list[int],
                 holding_period: int = 1) -> pd.DataFrame:
    """Label for every (signal day, code) with a complete, tradable O2O pair.

    Matches ``skd_backtest.label_provider`` with ``label_price_basis=adjusted_open``:
    the row must exist at both t+1 (entry) and t+2 (exit), be explicitly
    non-suspended, and carry a finite positive ``open``.
    """
    market = tables[MARKET_TABLE]
    codes = sorted(pd.unique(market["代码"]).tolist())
    open_wide = _wide(market, "open", codes, calendar)
    susp_wide = _wide(market, "is_suspend", codes, calendar)

    tradable = np.isfinite(susp_wide) & (susp_wide == 0)
    price_ok = np.isfinite(open_wide) & (open_wide > 0)
    entry_ok = tradable & price_ok

    n = len(calendar)
    step = holding_period + 1
    rows = []
    for i in range(n - step):
        entry = i + 1
        exit_ = i + step
        ok = entry_ok[:, entry] & entry_ok[:, exit_]
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = open_wide[:, exit_] / open_wide[:, entry] - 1.0
        ok &= np.isfinite(ret)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        rows.append(pd.DataFrame({
            "date": calendar[i],
            "code": [codes[j] for j in idx],
            "entry_date": calendar[entry],
            "exit_date": calendar[exit_],
            "future_return": ret[idx],
        }))
    if not rows:
        return pd.DataFrame(columns=["date", "code", "entry_date", "exit_date", "future_return"])
    labels = pd.concat(rows, ignore_index=True)
    return labels.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def attach_label_end_date(labels: pd.DataFrame, calendar: list[int],
                          holding_period: int = 1) -> pd.DataFrame:
    """Add ``label_end_date`` (the calendar day the label is realised on)."""
    out = labels.copy()
    positions = {day: idx for idx, day in enumerate(calendar)}
    ends = []
    for date in out["date"].to_numpy():
        pos = positions[int(date)] + holding_period + 1
        ends.append(calendar[pos] if pos < len(calendar) else None)
    out["label_end_date"] = ends
    return out
