"""S0: one date-grouped rank implementation shared by every fusion layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

S3 = Path("artifacts/stage3_ic_first")


def rank_within_day(values: pd.Series) -> pd.Series:
    """Average-tied centred rank inside one signal day: (rank-(n+1)/2)/n."""
    array = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(array.to_numpy(dtype=float)).all():
        raise ValueError("rank input contains NaN/Inf inside the daily pool")
    n = len(array)
    if n == 0:
        raise ValueError("empty daily pool")
    return (array.rank(method="average") - (n + 1) / 2.0) / n


def fuse(frame: pd.DataFrame, columns: list[str], weights: list[float],
         *, universe: pd.Series | None = None, label: str = "fusion") -> pd.DataFrame:
    """Date-grouped weighted fusion with full key/pool validation.

    ``universe`` is an optional per-date expected code set (a DataFrame with
    ``date``/``code`` columns); when given, the member pool is compared by
    outer join rather than by self-check on already joined rows.
    """
    if len(columns) != len(weights):
        raise ValueError("columns/weights length mismatch")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    data = frame.loc[:, ["date", "code", *columns]].copy()
    data["date"] = pd.to_numeric(data["date"]).astype("int64")
    if data.duplicated(["date", "code"]).any():
        raise ValueError(f"{label}: duplicate (date, code) keys")
    values = data.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: member scores contain NaN/Inf")

    if universe is not None:
        expected = universe.loc[:, ["date", "code"]].copy()
        expected["date"] = pd.to_numeric(expected["date"]).astype("int64")
        merged = data.merge(expected, on=["date", "code"], how="outer",
                            indicator=True)
        missing = merged.loc[merged["_merge"] == "left_only"]
        absent = merged.loc[merged["_merge"] == "right_only"]
        if len(missing) or len(absent):
            raise ValueError(
                f"{label}: pool mismatch (members only {len(missing)}, "
                f"universe only {len(absent)})")

    pieces = []
    for date, block in data.groupby("date", sort=True):
        acc = np.zeros(len(block), dtype=float)
        for column, weight in zip(columns, weights):
            acc += (weight / total) * rank_within_day(
                block[column].reset_index(drop=True)).to_numpy()
        pieces.append(pd.DataFrame({"date": date,
                                    "code": block["code"].to_numpy(),
                                    "score": acc}))
    return pd.concat(pieces, ignore_index=True)


def seed_average(frames: list[pd.DataFrame], columns: list[str],
                 weights: list[float], *, universe: pd.Series | None = None) -> pd.DataFrame:
    """Per-seed, per-day fused ranks, then averaged across seeds (no outer ranking)."""
    ranked = []
    for index, frame in enumerate(frames):
        fused = fuse(frame, columns, weights, universe=universe,
                     label=f"seed{index}")
        ranked.append(fused.rename(columns={"score": f"score_{index}"}))
    combined = ranked[0]
    for frame in ranked[1:]:
        combined = combined.merge(frame, on=["date", "code"], how="outer",
                                  validate="one_to_one", indicator=True)
        if (combined["_merge"] != "both").any():
            raise ValueError("seed pools differ")
        combined = combined.drop(columns="_merge")
    score_columns = [f"score_{i}" for i in range(len(ranked))]
    combined["score"] = combined[score_columns].mean(axis=1)
    return combined.loc[:, ["date", "code", "score"]]


def computation_graph(members: list[str], weights: list[float], *,
                      layers: dict | None = None, path: Path | None = None) -> dict:
    """Explicit, auditable description of every rank/fusion layer."""
    graph = {
        "rank_operator": "(average_rank(x) - (n+1)/2)/n computed inside each "
                         "signal day's pool",
        "members": members,
        "weights": weights,
        "layers": layers or {},
        "invariants": [
            "no cross-date ranking at any layer",
            "member pool must equal the day's legal universe by outer join",
            "NaN/Inf or missing members raise",
            "other dates cannot change a fixed date's score",
            "permuting stock order does not change per-code values",
        ],
    }
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(graph, indent=2))
    return graph
