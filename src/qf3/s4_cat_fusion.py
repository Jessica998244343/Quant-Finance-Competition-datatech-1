"""D1: CatBoost C1 expert (3 seeds) fused with the frozen LGB trunk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, s3_ranks

FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first/predictions")
CAT = FUS / "predictions_cat"
ALPHAS = [0.0, 0.05, 0.10, 0.25, 0.50]
SEEDS = (20261005, 20261017, 20261029)
WINDOWS = ("DEV19", "DEV20", "CONF21")


def seed_mean_rank(prefix_files: list[Path], window: str) -> pd.DataFrame:
    merged = None
    for index, path in enumerate(prefix_files):
        frame = pd.read_parquet(path).rename(columns={"score": f"s{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(prefix_files)):
            value += s3_ranks.rank_within_day(
                block[f"s{index}"].reset_index(drop=True)).to_numpy() / len(prefix_files)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": value}))
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame, panel: pd.DataFrame) -> float:
    merged = frame.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def main() -> None:
    FUS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    report = {"single_seed": {}, "alpha_grid": []}
    experts = {}
    for window in WINDOWS:
        paths = [CAT / f"C1_{seed}__{window}.parquet" for seed in SEEDS]
        experts[window] = seed_mean_rank(paths, window)
        experts[window].to_parquet(FUS / f"EXPERT_C1_seedmean__{window}.parquet",
                                   index=False)
        report["single_seed"][window] = {
            str(seed): evaluate(pd.read_parquet(path), panel)
            for seed, path in zip(SEEDS, paths)}
    trunk = {w: pd.read_parquet(S3 / f"MAIN_SEED_ENSEMBLE__{w}.parquet")
             for w in WINDOWS}
    for alpha in ALPHAS:
        row = {"alpha": alpha}
        frames = []
        for window in WINDOWS:
            both = trunk[window].rename(columns={"score": "trunk"}).merge(
                experts[window].rename(columns={"score": "expert"}),
                on=["date", "code"], validate="one_to_one")
            fused = s3_ranks.fuse(both, ["trunk", "expert"], [1 - alpha, alpha],
                                  label=f"cat-alpha{alpha}")
            row[window] = evaluate(fused, panel)
            frames.append(fused)
            fused.to_parquet(FUS / f"CATFUSION_a{alpha:g}__{window}.parquet", index=False)
        row["dev_pooled"] = evaluate(
            pd.concat([f for f, w in zip(frames, WINDOWS) if w != "CONF21"],
                      ignore_index=True), panel)
        row["all_pooled"] = evaluate(pd.concat(frames, ignore_index=True), panel)
        report["alpha_grid"].append(row)
    base = report["alpha_grid"][0]
    for row in report["alpha_grid"]:
        row["delta_dev_vs_alpha0"] = row["dev_pooled"] - base["dev_pooled"]
        row["delta_2021_vs_alpha0"] = row["CONF21"] - base["CONF21"]
    report["expert_identity"] = "CatBoost C1 (d8/l2=15, IC-selected trees), 3 seeds, daily rank mean"
    report["trunk_identity"] = "MAIN_SEED_ENSEMBLE_v1 (L3, 3 seeds)"
    (FUS / "d1_cat_fusion.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["single_seed"], indent=2))
    print(pd.DataFrame(report["alpha_grid"]).to_string(index=False))


if __name__ == "__main__":
    main()
