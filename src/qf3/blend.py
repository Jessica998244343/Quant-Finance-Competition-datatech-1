"""Daily centred-rank blending of two prediction sets, evaluated on dev folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

PRED_DIR = Path("artifacts/beginning/predictions")


def centered_rank(frame: pd.DataFrame, column: str = "score") -> pd.Series:
    def _one(day: pd.Series) -> pd.Series:
        ranks = day.rank(method="average")
        return (ranks - (len(day) + 1) / 2.0) / len(day)
    return frame.groupby("date")[column].transform(_one)


def blend_frames(left: pd.DataFrame, right: pd.DataFrame, weight: float) -> pd.DataFrame:
    a = left.copy()
    b = right.copy()
    a["rank"] = centered_rank(a)
    b["rank"] = centered_rank(b)
    merged = a[["date", "code", "rank"]].merge(
        b[["date", "code", "rank"]], on=["date", "code"], suffixes=("_a", "_b"),
        validate="one_to_one")
    merged["score"] = (1.0 - weight) * merged["rank_a"] + weight * merged["rank_b"]
    return merged[["date", "code", "score"]]


def evaluate_blend(left_id: str, right_id: str, weight: float,
                   window: str, labels: pd.DataFrame) -> dict:
    left = pd.read_parquet(PRED_DIR / f"{left_id}__{window}.parquet")
    right = pd.read_parquet(PRED_DIR / f"{right_id}__{window}.parquet")
    blended = blend_frames(left, right, weight)
    merged = blended.merge(labels[["date", "code", "future_return"]],
                           on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    summary["window"] = window
    summary["weight"] = weight
    return summary, rankic, blended


def search(left_id: str, right_id: str, windows=("DEV19", "DEV20"),
           weights=(0.25, 0.5, 0.75)):
    labels = pd.read_parquet("artifacts/beginning/cache/labels.parquet")
    rows = []
    frames = {}
    for weight in weights:
        rankics = []
        for window in windows:
            summary, rankic, blended = evaluate_blend(left_id, right_id, weight, window, labels)
            rankics.append(rankic)
            frames.setdefault(weight, {})[window] = blended
        pooled = metrics.summarize_rankic(pd.concat(rankics, ignore_index=True))
        rows.append({"left": left_id, "right": right_id, "weight": weight,
                     **{k: v for k, v in pooled.items() if k != "n_pairs"}})
    return pd.DataFrame(rows), frames


def main() -> None:
    table, _ = search("P5_s20261005", "SC_d8_l2_15")
    for other in ["SC_d6_l2_5", "SC_d8_l2_5"]:
        extra, _ = search("P5_s20261005", other)
        table = pd.concat([table, extra], ignore_index=True)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
