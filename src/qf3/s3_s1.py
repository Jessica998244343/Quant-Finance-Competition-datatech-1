"""S1: LightGBM main-model research (representation, capacity, path features)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, metrics, s2r_train, s3_panels
from .constants import F0_COLUMNS, G1_COLUMNS

S3 = Path("artifacts/stage3_ic_first")
PRED = S3 / "predictions"
MODELS = S3 / "models"
LOG = S3 / "experiments.jsonl"

BASE = list(F0_COLUMNS) + list(G1_COLUMNS)
L1 = list(F0_COLUMNS) + [f"L1_{c}" for c in G1_COLUMNS]

CAPACITY = {
    "L2": {"num_leaves": 31, "min_data_in_leaf": 500, "lambda_l2": 20.0},
    "L3": {"num_leaves": 63, "min_data_in_leaf": 300, "lambda_l2": 20.0},
    "L4": {"num_leaves": 63, "min_data_in_leaf": 800, "lambda_l2": 40.0},
    "L5": {"num_leaves": 127, "min_data_in_leaf": 800, "lambda_l2": 40.0},
}


def feature_sets(representation: str) -> dict[str, list[str]]:
    base = BASE if representation == "L0" else L1
    return {
        "base": base,
        "G_sep": base + s3_panels.G_SEP,
        "G_path": base + s3_panels.G_PATH,
        "joint": base + s3_panels.G_SEP + s3_panels.G_PATH,
    }


def log(record: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def evaluate(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    merged = preds.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    return summary, rankic


def train_recipe(panel: pd.DataFrame, name: str, features: list[str],
                 params: dict, windows: list[str], seed: int = 20261005) -> dict:
    records = {}
    for window in windows:
        split = cv.fold_split(panel, window)
        started = time.time()
        result = s2r_train.train_lgbm_fold_local(panel, features, split, params,
                                                 seed=seed, threads=8)
        seconds = time.time() - started
        preds = result["predictions"]
        tag = f"{name}_{seed}__{window}"
        preds.to_parquet(PRED / f"{tag}.parquet", index=False)
        (MODELS).mkdir(parents=True, exist_ok=True)
        result["booster"].save_model(str(MODELS / f"{tag}.txt"))
        curve = result.get("history", {})
        values = curve.get("valid_0", {}).get("daily_spearman", [])
        pd.DataFrame({"iteration": np.arange(1, len(values) + 1),
                      "daily_spearman": values}).to_parquet(
            MODELS / f"{tag}__curve.parquet", index=False)
        summary, rankic = evaluate(preds, panel)
        rankic.to_parquet(PRED / f"{tag}__rankic.parquet", index=False)
        record = {
            "stage": "S1", "recipe": name, "seed": seed, "window": window,
            "n_features": len(features), "rounds": result["rounds"],
            "seconds": round(seconds, 1), "mean_rankic": summary["mean_rankic"],
            "rankic_ir": summary["rankic_ir"], "n_days": summary["n_days"],
            "es_score": float(values[-1]) if values else None,
            "best_curve_value": float(np.nanmax(values)) if values else None,
            "best_curve_iteration": int(np.nanargmax(values) + 1) if values else None,
            "curve_points": len(values),
            "weight_mean_internal": result["weight_mean_internal"],
            "weight_mean_outer": result["weight_mean_outer"],
            "model_path": str(MODELS / f"{tag}.txt"),
            "predictions_path": str(PRED / f"{tag}.parquet"),
        }
        records[window] = record
        log(record)
        print(json.dumps({k: record[k] for k in
                          ("recipe", "window", "rounds", "mean_rankic",
                           "best_curve_iteration", "curve_points")}), flush=True)
    return records


def pooled_ic(records: dict, panel: pd.DataFrame, windows: list[str]) -> float:
    frames = []
    for window in windows:
        frames.append(pd.read_parquet(
            Path(records[window]["predictions_path"])))
    merged = pd.concat(frames, ignore_index=True).merge(
        panel[["date", "code", "future_return"]], on=["date", "code"],
        how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--representation", default="L0")
    parser.add_argument("--recipe", default="")
    parser.add_argument("--windows", default="DEV19,DEV20")
    args = parser.parse_args()
    S3.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(S3 / "panel_ext.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    windows = args.windows.split(",")
    from . import trainers
    p5 = dict(trainers.CAPACITY_GRID["P5"])

    if args.batch == "representation":
        for name, features in (("L0_p5", BASE), ("L1_p5", L1)):
            records = train_recipe(panel, name, features, p5, windows)
            print(name, "DEV19", round(records["DEV19"]["mean_rankic"], 5),
                  "DEV20", round(records["DEV20"]["mean_rankic"], 5),
                  "pooled", round(pooled_ic(records, panel, windows), 5))
    elif args.batch == "capacity":
        names = args.recipe.split(",") if args.recipe else list(CAPACITY)
        features = feature_sets(args.representation)["base"]
        for name in names:
            overrides = dict(p5)
            overrides.update(CAPACITY[name])
            records = train_recipe(panel, f"{name}_{args.representation}", features,
                                   overrides, windows)
            print(name, args.representation,
                  "pooled", round(pooled_ic(records, panel, windows), 5))
    elif args.batch == "paths":
        names = args.recipe.split(",") if args.recipe else ["G_sep", "G_path", "joint"]
        sets = feature_sets(args.representation)
        overrides = dict(p5)
        overrides.update(CAPACITY["L3"])      # batch-1 winner (two-year consistent)
        for name in names:
            records = train_recipe(panel, f"{name}_{args.representation}", sets[name],
                                   overrides, windows)
            print(name, args.representation,
                  "pooled", round(pooled_ic(records, panel, windows), 5))
    elif args.batch == "seeds":
        name = args.recipe or f"L3_{args.representation}"
        features = feature_sets(args.representation)["base"]
        overrides = dict(p5)
        overrides.update(CAPACITY["L3"])
        for seed in (20261017, 20261029):
            records = train_recipe(panel, name, features, overrides, windows, seed=seed)
            print(name, seed, "pooled", round(pooled_ic(records, panel, windows), 5))
    else:
        raise SystemExit(f"unknown batch {args.batch}")


if __name__ == "__main__":
    main()
