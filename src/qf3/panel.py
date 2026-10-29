"""Build the reusable feature panel and label table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dataio, featurelib
from . import labels as labels_mod
from .constants import BARRA_TABLE, FACTOR_TABLE, FEATURE_COLUMNS, MARKET_TABLE


def _date_bounds(table: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Row-position bounds per calendar date for a table sorted by 日期."""
    dates = table["日期"].to_numpy()
    bounds: dict[int, tuple[int, int]] = {}
    unique, starts = np.unique(dates, return_index=True)
    ends = np.append(starts[1:], len(dates))
    for day, start, end in zip(unique.tolist(), starts.tolist(), ends.tolist()):
        bounds[int(day)] = (int(start), int(end))
    return bounds


def build_panel(data_dir, start_year: int = 2016, end_year: int = 2022,
                lookback: int = 120, warmup_days: int = None,
                progress_every: int = 200,
                membership_filter: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, labels).

    ``warmup_days`` skips the first N calendar days (default ``lookback - 1``)
    so that every scored day has a full 120-day window, matching evaluation.

    ``membership_filter=True`` replicates the locked platform's research
    semantics: market/factor history is inner-joined with the same-day Barra
    membership.  The default (False) preserves the legacy annual-union cache.
    """
    tables = dataio.load_all(data_dir, start_year, end_year)
    # Labels must always be built from the unfiltered MarketData: the platform's
    # label provider reads t+1 / t+2 endpoints directly and does not apply the
    # same-day membership filter to them.
    label_source = tables
    if membership_filter:
        tables = dict(tables)
        membership = tables[BARRA_TABLE].loc[:, ["日期", "代码"]]
        for name in (MARKET_TABLE, FACTOR_TABLE):
            tables[name] = tables[name].merge(membership, on=["日期", "代码"],
                                              how="inner", sort=False)
    calendar = dataio.calendar(tables)
    bounds = {name: _date_bounds(table) for name, table in tables.items()}

    if warmup_days is None:
        warmup_days = lookback - 1
    first_index = min(warmup_days, max(0, len(calendar) - 1))
    signal_days = calendar[first_index:]

    market = tables[MARKET_TABLE]
    factor = tables[FACTOR_TABLE]
    barra = tables[BARRA_TABLE]

    frames = []
    for count, day in enumerate(signal_days, start=1):
        pos = calendar.index(day)
        window_start = calendar[max(0, pos - lookback + 1)]
        m0 = bounds[MARKET_TABLE][window_start][0]
        m1 = bounds[MARKET_TABLE][day][1]
        f0 = bounds[FACTOR_TABLE][window_start][0]
        f1 = bounds[FACTOR_TABLE][day][1]
        b0 = bounds[BARRA_TABLE][window_start][0]
        b1 = bounds[BARRA_TABLE][day][1]
        frame = featurelib.build_day_features(
            market.iloc[m0:m1], factor.iloc[f0:f1], barra.iloc[b0:b1], day,
        )
        frames.append(frame)
        if progress_every and count % progress_every == 0:
            print(f"  panel {count}/{len(signal_days)} days", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)

    label_table = labels_mod.build_labels(label_source, calendar)
    label_table = labels_mod.attach_label_end_date(label_table, calendar)
    return panel, label_table


def merge_labels(panel: pd.DataFrame, label_table: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(
        label_table.loc[:, ["date", "code", "future_return", "label_end_date"]],
        on=["date", "code"], how="left", validate="one_to_one", sort=False,
    )
    return merged
