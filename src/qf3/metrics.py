"""Rank-IC metric helpers (identical definition to the platform evaluator)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def daily_rankic(df: pd.DataFrame, score_col: str = "score",
                 label_col: str = "future_return") -> pd.DataFrame:
    """Per-day Spearman RankIC on finite score/label pairs."""
    rows = []
    for date, day in df.groupby("date", sort=True):
        score = pd.to_numeric(day[score_col], errors="coerce")
        label = pd.to_numeric(day[label_col], errors="coerce")
        finite = score.map(math.isfinite) & label.map(math.isfinite)
        s = score.loc[finite]
        y = label.loc[finite]
        n = len(s)
        rankic = None
        if n >= 2:
            if s.rank(method="average").nunique() == 1:
                rankic = 0.0
            elif y.rank(method="average").nunique() > 1:
                rankic = s.rank(method="average").corr(y.rank(method="average"))
        rows.append((date, rankic, n))
    out = pd.DataFrame(rows, columns=["date", "rankic", "n_stocks"])
    out["rankic"] = pd.to_numeric(out["rankic"], errors="coerce")
    return out


def summarize_rankic(rankic: pd.DataFrame) -> dict:
    valid = rankic["rankic"].dropna()
    if valid.empty:
        return {"mean_rankic": None, "rankic_std": None, "rankic_ir": None,
                "positive_ratio": None, "n_days": 0, "n_pairs": 0}
    std = float(valid.std(ddof=1)) if len(valid) >= 2 else None
    ir = float(valid.mean() / std) if std not in (None, 0) else None
    return {
        "mean_rankic": float(valid.mean()),
        "rankic_std": std,
        "rankic_ir": ir,
        "positive_ratio": float((valid > 0).mean()),
        "n_days": int(len(valid)),
        "n_pairs": int(pd.to_numeric(rankic["n_stocks"], errors="coerce").fillna(0).sum()),
    }


def paired_ic_difference(rankic_a: pd.DataFrame, rankic_b: pd.DataFrame,
                         block: int = 20, n_boot: int = 2000,
                         seed: int = 20261005) -> dict:
    """Paired time-block bootstrap of mean IC difference (a - b).

    Both inputs are per-day RankIC frames; they are aligned on the intersection
    of valid days.
    """
    left = rankic_a.set_index("date")["rankic"].rename("a")
    right = rankic_b.set_index("date")["rankic"].rename("b")
    joined = pd.concat([left, right], axis=1).dropna()
    if joined.empty:
        return {"n": 0}
    diff = (joined["a"] - joined["b"]).to_numpy()
    n = len(diff)
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    starts = np.arange(n_blocks) * block
    means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        idx = (starts[picks][:, None] + np.arange(block)[None, :]).ravel()
        idx = idx[idx < n]
        means[i] = diff[idx].mean()
    return {
        "n": int(n),
        "mean_diff": obs,
        "ci90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "std_err": float(means.std(ddof=1)),
    }


def yearly_summary(rankic: pd.DataFrame) -> pd.DataFrame:
    frame = rankic.dropna(subset=["rankic"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["year", "mean_rankic", "n_days"])
    frame["year"] = frame["date"].astype(str).str[:4].astype(int)
    out = frame.groupby("year").agg(
        mean_rankic=("rankic", "mean"),
        rankic_std=("rankic", lambda s: s.std(ddof=1)),
        n_days=("rankic", "size"),
    ).reset_index()
    return out
