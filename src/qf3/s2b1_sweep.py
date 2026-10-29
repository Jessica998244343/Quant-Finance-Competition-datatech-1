"""B1: CatBoost weight sweep on same-semantics member predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import metrics, s2b_common

WEIGHTS = [0.0, 0.125, 0.25, 0.5, 1.0]


def run_source(source: str, weights: list[float], windows: tuple[str, ...],
               train_end_map: dict) -> dict:
    panel_ref = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    records = {}
    for cat_weight in weights:
        frames = []
        per_window = {}
        for window in windows:
            blended = s2b_common.blend_weight(source, window, cat_weight)
            frames.append(blended)
            summary, _ = s2b_common.ic_summary(blended, panel_ref)
            per_window[window] = summary
        pooled = pd.concat(frames, ignore_index=True)
        dev = pd.concat([f for f, w in zip(frames, windows)
                         if w in ("DEV19", "DEV20")], ignore_index=True)
        dev_summary, dev_rankic = s2b_common.ic_summary(dev, panel_ref)
        tag = f"{source}_cat{cat_weight:g}"
        pooled.to_parquet(s2b_common.PRED / f"{tag}__pooled.parquet", index=False)
        for window, frame in zip(windows, frames):
            frame.to_parquet(s2b_common.PRED / f"{tag}__{window}.parquet", index=False)
        dev_rankic.to_parquet(s2b_common.PRED / f"{tag}__dev_rankic.parquet", index=False)
        record = {
            "experiment_id": f"B1_{tag}",
            "source": source,
            "cat_weight": cat_weight,
            "per_window": {w: {k: v for k, v in s.items()} for w, s in per_window.items()},
            "dev_mean_rankic": dev_summary["mean_rankic"],
            "dev_rankic_ir": dev_summary["rankic_ir"],
            "dev_n_days": dev_summary["n_days"],
        }
        records[tag] = record
        s2b_common.log({"stage": "B1", **record})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ref_corrected")
    parser.add_argument("--weights", default="")
    args = parser.parse_args()
    weights = [float(x) for x in args.weights.split(",")] if args.weights else WEIGHTS
    s2b_common.PRED.mkdir(parents=True, exist_ok=True)
    records = run_source(args.source, weights, ("DEV19", "DEV20", "CONF21"), {})
    s2b_common.write_json(s2b_common.B / f"b1_{args.source}.json", records)
    table = pd.DataFrame([{
        "cat_weight": v["cat_weight"],
        "DEV19": v["per_window"]["DEV19"]["mean_rankic"],
        "DEV20": v["per_window"]["DEV20"]["mean_rankic"],
        "CONF21": v["per_window"]["CONF21"]["mean_rankic"],
        "dev_pooled": v["dev_mean_rankic"],
        "dev_rankic_ir": v["dev_rankic_ir"],
    } for v in records.values()])
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
