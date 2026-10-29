"""REV1: build REF-corrected (platform semantics + fold-local weights + CORR-1)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, metrics, s2r_blend, s2r_train, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

REV = Path("artifacts/stage2_ic_ir/a_revision1")
MODELS = REV / "models_ref_corrected"
PRED = REV / "predictions"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
WINDOWS = ["DEV19", "DEV20", "CONF21"]
WEIGHTS = [0.75, 0.25]


def evaluate(preds: pd.DataFrame, panel: pd.DataFrame, label: str) -> dict:
    merged = preds.merge(panel.loc[:, ["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    summary["label"] = label
    return summary


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    report = {}
    for window in WINDOWS:
        split = cv.fold_split(panel, window)
        print(f"== {window}: LGBM", flush=True)
        started = time.time()
        lgbm = s2r_train.train_lgbm_fold_local(
            panel, FEATURES, split, trainers.CAPACITY_GRID["P5"],
            seed=20261005, threads=8)
        print(f"   rounds={lgbm['rounds']} ({time.time()-started:.0f}s)", flush=True)
        print(f"== {window}: CatBoost", flush=True)
        started = time.time()
        cat = s2r_train.train_catboost_fold_local(
            panel, FEATURES, split, depth=8, l2_leaf_reg=15.0,
            seed=20261005, threads=8)
        print(f"   tree_count={cat['tree_count']} best_index={cat['best_index']} "
              f"({time.time()-started:.0f}s)", flush=True)

        lgbm["predictions"].to_parquet(PRED / f"ref_corrected_lgbm__{window}.parquet",
                                       index=False)
        cat["predictions"].to_parquet(PRED / f"ref_corrected_catboost__{window}.parquet",
                                      index=False)
        lgbm["booster"].save_model(str(MODELS / f"ref_corrected_lgbm_{window}.txt"))
        cat["model"].save_model(str(MODELS / f"ref_corrected_catboost_{window}.cbm"),
                                format="cbm")
        members = lgbm["predictions"].merge(
            cat["predictions"], on=["date", "code"], suffixes=("_lgbm", "_cat"),
            validate="one_to_one").rename(columns={"score_lgbm": "lgbm_p5",
                                                   "score_cat": "catboost"})
        blended = s2r_blend.blend_frame(members, ["lgbm_p5", "catboost"], WEIGHTS)
        blended.to_parquet(PRED / f"ref_corrected__{window}.parquet", index=False)
        report[window] = {
            "lgbm_rounds": lgbm["rounds"],
            "catboost_tree_count": cat["tree_count"],
            "catboost_best_index": cat["best_index"],
            "weight_mean_internal_lgbm": lgbm["weight_mean_internal"],
            "weight_mean_outer_lgbm": lgbm["weight_mean_outer"],
            "lgbm": evaluate(lgbm["predictions"], panel, "lgbm"),
            "catboost": evaluate(cat["predictions"], panel, "catboost"),
            "blend": evaluate(blended, panel, "blend"),
        }
    (REV / "ref_corrected.json").write_text(json.dumps(report, indent=2, default=str))
    for window, value in report.items():
        print(window, "lgbm", round(value["lgbm"]["mean_rankic"], 5),
              "cat", round(value["catboost"]["mean_rankic"], 5),
              "blend", round(value["blend"]["mean_rankic"], 5),
              "rounds", value["lgbm_rounds"], value["catboost_tree_count"])


if __name__ == "__main__":
    main()
