"""D1: first tree fusion — seed-mean L3 trunk plus the retained L5 expert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, s3_ranks

FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first/predictions")
ALPHAS = [0.0, 0.05, 0.10, 0.25, 0.50]
SEEDS = (20261005, 20261017, 20261029)
WINDOWS = ("DEV19", "DEV20", "CONF21")


def seed_mean_rank(prefix: str, window: str) -> pd.DataFrame:
    """Expert output q = mean over seeds of the daily rank of that seed's score."""
    merged = None
    for index, seed in enumerate(SEEDS):
        frame = pd.read_parquet(S3 / f"{prefix}_{seed}__{window}.parquet")
        frame = frame.rename(columns={"score": f"s{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(SEEDS)):
            value += s3_ranks.rank_within_day(
                block[f"s{index}"].reset_index(drop=True)).to_numpy() / len(SEEDS)
        pieces.append(pd.DataFrame({"date": date,
                                    "code": block["code"].to_numpy(),
                                    "score": value}))
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame, panel: pd.DataFrame) -> float:
    merged = frame.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def main() -> None:
    FUS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    report = {"alpha_grid": []}
    experts = {}
    for window in WINDOWS:
        experts[window] = seed_mean_rank("L5_L0", window)
        experts[window].to_parquet(FUS / f"EXPERT_L5_seedmean__{window}.parquet",
                                   index=False)
    trunk = {w: pd.read_parquet(S3 / f"MAIN_SEED_ENSEMBLE__{w}.parquet")
             for w in WINDOWS}
    for alpha in ALPHAS:
        row = {"alpha": alpha}
        rows = []
        for window in WINDOWS:
            both = trunk[window].rename(columns={"score": "trunk"}).merge(
                experts[window].rename(columns={"score": "expert"}),
                on=["date", "code"], validate="one_to_one")
            fused = s3_ranks.fuse(both, ["trunk", "expert"], [1 - alpha, alpha],
                                  label=f"alpha{alpha}")
            row[window] = evaluate(fused, panel)
            rows.append(fused)
            fused.to_parquet(FUS / f"TREEFUSION_a{alpha:g}__{window}.parquet",
                             index=False)
        row["dev_pooled"] = evaluate(
            pd.concat([r for r, w in zip(rows, WINDOWS) if w != "CONF21"],
                      ignore_index=True), panel)
        row["all_pooled"] = evaluate(pd.concat(rows, ignore_index=True), panel)
        report["alpha_grid"].append(row)
    base = report["alpha_grid"][0]
    for row in report["alpha_grid"]:
        row["delta_dev_vs_alpha0"] = row["dev_pooled"] - base["dev_pooled"]
        row["delta_2021_vs_alpha0"] = row["CONF21"] - base["CONF21"]
    report["trunk_identity"] = "MAIN_SEED_ENSEMBLE_v1 (L3, 3 seeds, daily rank mean)"
    report["expert_identity"] = "L5 63/800/40, 3 seeds, daily rank mean"
    (FUS / "d1_tree_fusion.json").write_text(json.dumps(report, indent=2, default=str))
    print(pd.DataFrame(report["alpha_grid"]).to_string(index=False))


if __name__ == "__main__":
    main()
