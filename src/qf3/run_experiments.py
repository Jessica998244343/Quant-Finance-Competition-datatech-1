"""Batch experiment runner for the phase-1 baseline / strong-baseline search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_cache, cv, metrics, trainers
from .constants import F0_COLUMNS, G1_COLUMNS, G2_COLUMNS, G3_COLUMNS

PRED_DIR = Path("artifacts/beginning/predictions")
LOG_PATH = Path("artifacts/beginning/experiments.jsonl")

FEATURE_SETS = {
    "F0": list(F0_COLUMNS),
    "F0G1": list(F0_COLUMNS) + list(G1_COLUMNS),
    "F0G1G2": list(F0_COLUMNS) + list(G1_COLUMNS) + list(G2_COLUMNS),
    "F0G1G2G3": list(F0_COLUMNS) + list(G1_COLUMNS) + list(G2_COLUMNS) + list(G3_COLUMNS),
    "G1": list(G1_COLUMNS),
    "G2": list(G2_COLUMNS),
    "G3": list(G3_COLUMNS),
    "G1G2": list(G1_COLUMNS) + list(G2_COLUMNS),
    "G1G2G3": list(G1_COLUMNS) + list(G2_COLUMNS) + list(G3_COLUMNS),
    "G2G3": list(G2_COLUMNS) + list(G3_COLUMNS),
}


def _summarize(preds: pd.DataFrame, labels: pd.DataFrame) -> dict:
    merged = preds.merge(labels[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    summary["yearly"] = metrics.yearly_summary(rankic).to_dict("records")
    return summary, rankic, merged


def run_one(panel: pd.DataFrame, labels: pd.DataFrame, *, experiment_id: str,
            window: str, features: list[str], model: str = "lgbm",
            params: dict | None = None, seed: int = 20261005, threads: int = 8,
            target: str = "rank", tag: str = "", save: bool = True,
            num_rounds: int = 3000, target_mode: str = "rank",
            half_life: float | None = None) -> dict:
    split = cv.fold_split(panel, window)
    started = time.time()
    extra = {}
    if model == "lgbm":
        result = trainers.train_lgbm_fold(panel, features, split, params, seed, threads,
                                          num_rounds=num_rounds, target_mode=target_mode,
                                          half_life=half_life)
        extra["best_iteration"] = result["best_iteration"]
        extra["es_score"] = result["es_score"]
        extra["n_internal"] = result["n_internal"]
        extra["n_es"] = result["n_es"]
        extra["n_outer"] = result["n_outer"]
    elif model == "ridge":
        result = trainers.train_ridge_fold(panel, features, split)
        extra["alpha"] = result["alpha"]
    elif model == "catboost":
        result = trainers.train_catboost_fold(panel, features, split,
                                              **(params or {}), seed=seed, threads=threads)
        extra["best_iteration"] = result["best_iteration"]
    else:
        raise ValueError(model)
    preds = result["predictions"]
    summary, rankic, merged = _summarize(preds, labels)
    seconds = time.time() - started
    record = {
        "experiment_id": experiment_id,
        "window": window,
        "model": model,
        "features": features,
        "n_features": len(features),
        "params": params or {},
        "seed": seed,
        "threads": threads,
        "target": target,
        "target_mode": target_mode,
        "half_life": half_life,
        "train_end": split["train_end"],
        "es_start": split["es_start"],
        "tag": tag,
        "seconds": round(seconds, 1),
        "summary": {k: v for k, v in summary.items() if k != "yearly"},
        "yearly": summary["yearly"],
        **extra,
    }
    if save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_DIR / f"{experiment_id}__{window}.parquet", index=False)
        rankic.to_parquet(PRED_DIR / f"{experiment_id}__{window}__rankic.parquet", index=False)
        trainers.log_experiment(LOG_PATH, record)
    return record


def show(records: list[dict]) -> None:
    frame = pd.DataFrame([{**{k: r["summary"].get(k) for k in
                              ("mean_rankic", "rankic_ir", "positive_ratio", "n_days")},
                           "experiment_id": r["experiment_id"], "window": r["window"],
                           "n_features": r["n_features"], "iter": r.get("best_iteration"),
                           "sec": r["seconds"]} for r in records])
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-set", default="F0G1G2G3")
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    windows = args.windows.split(",")
    panel = build_cache.load()
    labels = pd.read_parquet("artifacts/beginning/cache/labels.parquet")
    records = []
    batch = args.batch
    if batch == "b0":
        for window in windows:
            records.append(run_one(panel, labels, experiment_id="B0", window=window,
                                   features=FEATURE_SETS["F0"], model="lgbm",
                                   params={}, threads=args.threads, tag="frozen-baseline"))
    elif batch == "r0":
        for window in windows:
            records.append(run_one(panel, labels, experiment_id="R0", window=window,
                                   features=FEATURE_SETS["F0"], model="ridge", tag="aux"))
    elif batch == "sa":
        for name in ("F0G1", "F0G1G2", "F0G1G2G3"):
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=f"SA_{name}",
                                       window=window, features=FEATURE_SETS[name],
                                       model="lgbm", params={}, threads=args.threads,
                                       tag="feature-group-increment"))
    elif batch == "sb":
        ids = args.ids.split(",") if args.ids else ["P1", "P2", "P3", "P4", "P5", "P6"]
        for name in ids:
            for window in windows:
                records.append(run_one(panel, labels,
                                       experiment_id=f"SB{args.feature_set}_{name}",
                                       window=window, features=FEATURE_SETS[args.feature_set],
                                       model="lgbm", params=trainers.CAPACITY_GRID[name],
                                       threads=args.threads, tag="capacity"))
    elif batch == "sbx":
        variants = {
            "SBF0G1_X1_lr02": {"feature_set": "F0G1", "rounds": 5000,
                               "params": {"learning_rate": 0.02}},
            "SBF0G1_X2_lr05": {"feature_set": "F0G1", "rounds": 2000,
                               "params": {"learning_rate": 0.05}},
            "SBF0G1_X3_leaves47": {"feature_set": "F0G1", "rounds": 3000,
                                   "params": {"num_leaves": 47, "min_data_in_leaf": 500,
                                              "lambda_l1": 0.1, "lambda_l2": 20.0,
                                              "feature_fraction": 0.8, "bagging_fraction": 0.7}},
            "SBF0G1_X4_leaves85": {"feature_set": "F0G1", "rounds": 3000,
                                   "params": {"num_leaves": 85, "min_data_in_leaf": 700,
                                              "lambda_l1": 0.1, "lambda_l2": 20.0,
                                              "feature_fraction": 0.8, "bagging_fraction": 0.7}},
            "SBF0G1_X5_maxbin255": {"feature_set": "F0G1", "rounds": 3000,
                                    "params": {"max_bin": 255}},
            "SBF0G1G2_P5": {"feature_set": "F0G1G2", "rounds": 3000,
                            "params": dict(trainers.CAPACITY_GRID["P5"])},
            "SBF0G1G2G3_P5": {"feature_set": "F0G1G2G3", "rounds": 3000,
                              "params": dict(trainers.CAPACITY_GRID["P5"])},
        }
        ids = args.ids.split(",") if args.ids else list(variants)
        for name in ids:
            cfg = variants[name]
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=name, window=window,
                                       features=FEATURE_SETS[cfg["feature_set"]],
                                       model="lgbm", params=cfg["params"],
                                       threads=args.threads, tag="capacity-extension",
                                       num_rounds=cfg["rounds"]))
    elif batch == "sc":
        variants = {
            "SC_d6_l2_5": {"depth": 6, "l2_leaf_reg": 5.0},
            "SC_d6_l2_15": {"depth": 6, "l2_leaf_reg": 15.0},
            "SC_d8_l2_5": {"depth": 8, "l2_leaf_reg": 5.0},
            "SC_d8_l2_15": {"depth": 8, "l2_leaf_reg": 15.0},
        }
        ids = args.ids.split(",") if args.ids else list(variants)
        for name in ids:
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=name, window=window,
                                       features=FEATURE_SETS[args.feature_set],
                                       model="catboost", params=variants[name],
                                       threads=args.threads, tag="catboost-architecture"))
    elif batch == "sd":
        variants = {
            "SD_winsor": {"target_mode": "winsor", "half_life": None},
            "SD_hl504": {"target_mode": "rank", "half_life": 504.0},
            "SD_hl1008": {"target_mode": "rank", "half_life": 1008.0},
        }
        ids = args.ids.split(",") if args.ids else list(variants)
        for name in ids:
            cfg = variants[name]
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=name, window=window,
                                       features=FEATURE_SETS[args.feature_set],
                                       model="lgbm", params=trainers.CAPACITY_GRID["P5"],
                                       threads=args.threads, tag="objective-time-weight",
                                       target_mode=cfg["target_mode"],
                                       half_life=cfg["half_life"]))
    elif batch == "seeds":
        for seed in (20261005, 20261017, 20261029):
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=f"B0_s{seed}",
                                       window=window, features=FEATURE_SETS["F0"],
                                       model="lgbm", params={}, seed=seed,
                                       threads=args.threads, tag="seed-stability"))
            for window in windows:
                records.append(run_one(panel, labels, experiment_id=f"P5_s{seed}",
                                       window=window, features=FEATURE_SETS["F0G1"],
                                       model="lgbm", params=trainers.CAPACITY_GRID["P5"],
                                       seed=seed, threads=args.threads, tag="seed-stability"))
    elif batch == "conf21":
        for window in windows:
            records.append(run_one(panel, labels, experiment_id="B0", window=window,
                                   features=FEATURE_SETS["F0"], model="lgbm", params={},
                                   threads=args.threads, tag="frozen-baseline-confirm"))
        for window in windows:
            records.append(run_one(panel, labels, experiment_id="P5_s20261005", window=window,
                                   features=FEATURE_SETS["F0G1"], model="lgbm",
                                   params=trainers.CAPACITY_GRID["P5"],
                                   threads=args.threads, tag="candidate-confirm"))
        for window in windows:
            records.append(run_one(panel, labels, experiment_id="SC_d8_l2_15", window=window,
                                   features=FEATURE_SETS["F0G1"], model="catboost",
                                   params={"depth": 8, "l2_leaf_reg": 15.0},
                                   threads=args.threads, tag="candidate-confirm"))
    else:
        raise SystemExit(f"unknown batch {batch}")
    show(records)


if __name__ == "__main__":
    main()
