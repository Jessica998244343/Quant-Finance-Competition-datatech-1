"""REV1: correct per-signal-day centred-rank blending plus its verification harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path("artifacts/stage2_ic_ir/a_revision1")
PRED = REV / "predictions"


def centered_rank_within(values: pd.Series) -> pd.Series:
    """Average-tied centred rank inside one signal day's full universe."""
    ranks = values.rank(method="average")
    return (ranks - (len(values) + 1) / 2.0) / len(values)


def blend_members(members: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Blend member scores inside a single (date, code) aligned block."""
    total = sum(weights.values())
    out = pd.Series(0.0, index=members[list(members)[0]].index)
    for name, scores in members.items():
        out = out + (weights[name] / total) * centered_rank_within(scores.astype(float))
    return out


def blend_frame(frame: pd.DataFrame, columns: list[str], weights: list[float]) -> pd.DataFrame:
    """Per-date blending of a long frame; validates keys and the daily pool."""
    data = frame.loc[:, ["date", "code", *columns]].copy()
    if data.duplicated(["date", "code"]).any():
        raise ValueError("member frame has duplicate (date, code) keys")
    sizes = data.groupby("date").size()
    expected = frame.groupby("date")["code"].nunique()
    if not sizes.equals(expected):
        raise ValueError("inconsistent daily universe")
    # member values must cover the same daily pool
    if data.loc[:, columns].isna().any().any():
        raise ValueError("member frame contains missing scores inside the pool")

    pieces = []
    for date, block in data.groupby("date", sort=True):
        scores = {column: block[column].reset_index(drop=True) for column in columns}
        weight_map = dict(zip(columns, weights))
        blended = blend_members(scores, weight_map)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": blended.to_numpy()}))
    return pd.concat(pieces, ignore_index=True)


def blend_whole_frame_wrong(frame: pd.DataFrame, columns: list[str],
                            weights: list[float]) -> pd.DataFrame:
    """The stage-2 A bug, kept only to quantify its effect."""
    total = sum(weights)
    out = np.zeros(len(frame))
    for column, weight in zip(columns, weights):
        series = pd.Series(frame[column].to_numpy(dtype=float))
        ranks = series.rank(method="average")
        centered = ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()
        out += (weight / total) * centered
    return frame.loc[:, ["date", "code"]].assign(score=out)


def rank_disagreement(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    merged = left.rename(columns={"score": "a"}).merge(
        right.rename(columns={"score": "b"}), on=["date", "code"], how="inner",
        validate="one_to_one")
    diff = []
    for _, block in merged.groupby("date", sort=True):
        ra = block["a"].rank(method="average")
        rb = block["b"].rank(method="average")
        diff.append((ra != rb).mean())
    return {"rows": int(len(merged)),
            "share_rank_changed": float(np.mean(diff)),
            "max_abs_diff": float((merged["a"] - merged["b"]).abs().max())}


def perturbation_check(frame: pd.DataFrame, columns: list[str], weights: list[float],
                       probe_day) -> dict:
    """Other dates must not change a fixed day's scores."""
    reference = blend_frame(frame, columns, weights)
    reference_day = reference.loc[reference["date"] == probe_day].reset_index(drop=True)
    results = {}

    perturbed = frame.loc[frame["date"] != probe_day].copy()
    for column in columns:
        perturbed[column] = perturbed[column] * -3.0 + 7.0
    combined = pd.concat([frame.loc[frame["date"] == probe_day], perturbed],
                         ignore_index=True)
    for label, data in (("other_days_rescaled", combined),
                        ("probe_day_alone",
                         frame.loc[frame["date"] == probe_day].reset_index(drop=True))):
        got = blend_frame(data, columns, weights)
        got = got.loc[got["date"] == probe_day].reset_index(drop=True)
        results[label] = {
            "max_abs_diff": float((got["score"].to_numpy()
                                   - reference_day["score"].to_numpy()).max())
            if len(got) == len(reference_day) else None,
            "identical": bool(len(got) == len(reference_day)
                              and np.array_equal(got["score"].to_numpy(),
                                                 reference_day["score"].to_numpy())),
        }
    return results


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
