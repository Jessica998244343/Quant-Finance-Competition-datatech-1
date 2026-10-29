"""Build ALT_v1 (tree-only, no torch) and its full-data members."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
ALT_MODELS = FUS / "alt_models"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
L3_ROUNDS = int(np.median([334, 816, 448]))     # 448
C1_TREES = int(np.median([1325, 825, 875]))     # 875
W_L3, W_C1 = 0.5, 0.5
L5_ROUNDS = int(np.median([311, 408, 122]))     # 311


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def train_l3(data: pd.DataFrame, seed: int, threads: int = 6) -> dict:
    import lightgbm as lgb
    from . import trainers

    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    w = s2r_train.normalise_within(raw_w, finite)
    params = trainers.lgb_params(trainers.CAPACITY_GRID["P5"], seed, threads)
    params.update({"num_leaves": 63, "min_data_in_leaf": 300, "lambda_l2": 20.0})
    params.pop("_num_rounds")
    dataset = lgb.Dataset(x[finite], label=z[finite], weight=w[finite],
                          feature_name=FEATURES)
    started = time.time()
    booster = lgb.train(params, dataset, num_boost_round=L3_ROUNDS)
    ALT_MODELS.mkdir(parents=True, exist_ok=True)
    path = ALT_MODELS / f"l3_seed{seed}.txt"
    booster.save_model(str(path))
    return {"member": "l3", "seed": seed, "rounds": L3_ROUNDS,
            "seconds": round(time.time() - started, 1), "path": str(path)}


def train_c1(data: pd.DataFrame, seed: int, threads: int = 6) -> dict:
    from catboost import CatBoostRegressor, Pool

    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    w = s2r_train.normalise_within(raw_w, finite)
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=C1_TREES, learning_rate=0.03, depth=8,
        l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
        bootstrap_type="Bernoulli", subsample=0.8, thread_count=threads,
        random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    started = time.time()
    model.fit(Pool(x[finite], label=z[finite], weight=w[finite], feature_names=FEATURES))
    ALT_MODELS.mkdir(parents=True, exist_ok=True)
    path = ALT_MODELS / f"c1_seed{seed}.cbm"
    model.save_model(str(path), format="cbm")
    return {"member": "c1", "seed": seed, "trees": C1_TREES,
            "actual_tree_count": int(model.tree_count_),
            "seconds": round(time.time() - started, 1), "path": str(path)}


def train_l5(data: pd.DataFrame, seed: int, threads: int = 6) -> dict:
    import lightgbm as lgb
    from . import trainers

    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    w = s2r_train.normalise_within(raw_w, finite)
    params = trainers.lgb_params(trainers.CAPACITY_GRID["P5"], seed, threads)
    params.update({"num_leaves": 127, "min_data_in_leaf": 800, "lambda_l2": 40.0})
    params.pop("_num_rounds")
    dataset = lgb.Dataset(x[finite], label=z[finite], weight=w[finite],
                          feature_name=FEATURES)
    started = time.time()
    booster = lgb.train(params, dataset, num_boost_round=L5_ROUNDS)
    ALT_MODELS.mkdir(parents=True, exist_ok=True)
    path = ALT_MODELS / f"l5_seed{seed}.txt"
    booster.save_model(str(path))
    return {"member": "l5", "seed": seed, "rounds": L5_ROUNDS,
            "seconds": round(time.time() - started, 1), "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="train")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    data = panel()
    log_path = FUS / "alt_train.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"members": []}
    if args.stage == "train":
        for seed in SEEDS:
            for fn in (train_l3, train_l5, train_c1):
                record = fn(data, seed, args.threads)
                report["members"] = [m for m in report["members"]
                                     if not (m["member"] == record["member"]
                                             and m["seed"] == seed)]
                report["members"].append(record)
                report["round_rules"] = {"l3_rounds": L3_ROUNDS, "c1_trees": C1_TREES}
                log_path.write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
