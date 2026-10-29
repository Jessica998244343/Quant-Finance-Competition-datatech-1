"""Time-ordered folds, cross-sectional rank targets and the daily-Spearman metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EARLY_STOP_DAYS, OUTER_WINDOWS


def to_int_date(values) -> np.ndarray:
    """Numeric date array; missing values stay NaN (never silently dropped)."""
    return pd.to_numeric(values, errors="coerce").astype("float64").to_numpy()


def make_target(frame: pd.DataFrame, label_col: str = "future_return"):
    """Daily average-tied rank target ``z=(rank-(n+1)/2)/n`` plus equal weights."""
    label = pd.to_numeric(frame[label_col], errors="coerce")
    finite = label.notna()
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(finite).groupby(frame["date"]).transform("count")
    z = (ranks - (counts + 1.0) / 2.0) / counts
    weights = 1.0 / counts
    weight_mean = weights[finite].mean()
    weights = weights / weight_mean
    return z.to_numpy(dtype=float), weights.to_numpy(dtype=float), finite.to_numpy()


def daily_spearman(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    frame = frame.loc[np.isfinite(frame["y"]) & np.isfinite(frame["p"])]
    if frame.empty:
        return float("nan")
    grouped = frame.groupby("date")
    def _one(sub):
        if len(sub) < 2:
            return np.nan
        r1 = sub["p"].rank(method="average")
        r2 = sub["y"].rank(method="average")
        if r1.nunique() == 1 or r2.nunique() == 1:
            return np.nan
        return float(np.corrcoef(r1, r2)[0, 1])
    values = grouped.apply(_one, include_groups=False)
    return float(values.dropna().mean())


def fold_split(frame: pd.DataFrame, window: str) -> dict:
    """Outer train / predict / early-stopping split for one forward-year window."""
    spec = OUTER_WINDOWS[window]
    train_end = int(spec["train_end"].replace("-", ""))
    year = int(spec["predict_year"])
    dates = to_int_date(frame["date"])
    label_end = to_int_date(frame["label_end_date"])

    eligible = label_end <= train_end
    train_dates = np.sort(pd.unique(dates[eligible]))
    if len(train_dates) == 0:
        raise ValueError(f"no training samples before {train_end}")
    es_days = min(EARLY_STOP_DAYS, max(1, len(train_dates) // 4))
    es_start = int(train_dates[-es_days])
    outer_train = eligible
    internal = eligible & (label_end < es_start)
    early_stop = eligible & (dates >= es_start)
    predict = (dates >= year * 10000 + 101) & (dates <= year * 10000 + 1231)
    return {
        "window": window,
        "train_end": train_end,
        "es_start": es_start,
        "outer_train": np.asarray(outer_train),
        "internal": np.asarray(internal),
        "early_stop": np.asarray(early_stop),
        "predict": np.asarray(predict),
        "n_outer": int(outer_train.sum()),
        "n_internal": int(internal.sum()),
        "n_es": int(early_stop.sum()),
        "n_predict": int(predict.sum()),
        "es_days": int(es_days),
    }
