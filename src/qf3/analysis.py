"""Aggregate experiment logs and compare candidates to the frozen B0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

LOG_PATH = Path("artifacts/beginning/experiments.jsonl")
PRED_DIR = Path("artifacts/beginning/predictions")


def load_log(path=LOG_PATH) -> pd.DataFrame:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pooled(experiment_id: str, windows=("DEV19", "DEV20")) -> dict:
    frames = []
    for window in windows:
        path = PRED_DIR / f"{experiment_id}__{window}__rankic.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["window"] = window
        frames.append(frame)
    if not frames:
        return {}
    rankic = pd.concat(frames, ignore_index=True)[["date", "rankic", "n_stocks"]]
    summary = metrics.summarize_rankic(rankic)
    summary["per_window"] = {}
    for window in windows:
        part = rankic[rankic["date"].astype(str).str[:4] ==
                      {"DEV19": "2019", "DEV20": "2020", "CONF21": "2021"}.get(window, "")[:4]]
        if not part.empty:
            summary["per_window"][window] = metrics.summarize_rankic(part)["mean_rankic"]
    return summary


def compare(experiment_id: str, reference: str = "B0",
            windows=("DEV19", "DEV20")) -> dict:
    a = pooled(experiment_id, windows)
    b = pooled(reference, windows)
    if not a or not b:
        return {}
    delta = a["mean_rankic"] - b["mean_rankic"]
    rel = delta / abs(b["mean_rankic"]) if b["mean_rankic"] else None
    frames = []
    for window in windows:
        pa = PRED_DIR / f"{experiment_id}__{window}__rankic.parquet"
        pb = PRED_DIR / f"{reference}__{window}__rankic.parquet"
        if pa.exists() and pb.exists():
            frames.append((pd.read_parquet(pa), pd.read_parquet(pb)))
    boot = {}
    if frames:
        left = pd.concat([f[0] for f in frames], ignore_index=True)
        right = pd.concat([f[1] for f in frames], ignore_index=True)
        boot = metrics.paired_ic_difference(left, right)
    return {
        "candidate_ic": a["mean_rankic"],
        "reference_ic": b["mean_rankic"],
        "delta_ic": delta,
        "relative": rel,
        "per_window": a.get("per_window"),
        "bootstrap": boot,
        "threshold": max(0.003, 0.10 * abs(b["mean_rankic"])),
        "passes_threshold": bool(delta >= max(0.003, 0.10 * abs(b["mean_rankic"])))
        and a["mean_rankic"] > 0,
    }


def summary_table(windows=("DEV19", "DEV20")) -> pd.DataFrame:
    log = load_log()
    ids = sorted(set(zip(log["experiment_id"], log["tag"])))
    rows = []
    for experiment_id, tag in ids:
        stat = pooled(experiment_id, windows)
        if not stat:
            continue
        rows.append({
            "experiment_id": experiment_id,
            "tag": tag,
            "features": int(log.loc[log["experiment_id"] == experiment_id, "n_features"].iloc[0]),
            "mean_rankic": stat["mean_rankic"],
            "rankic_ir": stat["rankic_ir"],
            "positive_ratio": stat["positive_ratio"],
            "n_days": stat["n_days"],
        })
    return pd.DataFrame(rows).sort_values("mean_rankic", ascending=False).reset_index(drop=True)
