"""Reproduce TREE_v1 model/ from the official 2016-2022 data (no torch)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_panel
from alpha_constants import F0_COLUMNS, G1_COLUMNS

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
L5_ROUNDS = 311
C1_TREES = 875
SEEDS = (20261005, 20261017, 20261029)


def centred_rank_target(frame: pd.DataFrame) -> np.ndarray:
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    return ((ranks - (counts + 1.0) / 2.0) / counts).to_numpy(dtype=float)


def normalised_weights(frame: pd.DataFrame) -> np.ndarray:
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    weights = 1.0 / counts
    finite = label.notna().to_numpy()
    return (weights / weights[finite].mean()).to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="model")
    parser.add_argument("--threads", type=int, default=6,
                        help="must equal the thread count used for the shipped model; "
                             "LightGBM is not bit-identical across thread counts")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features, labels = alpha_panel.build_panel(args.data_dir, membership_filter=True)
    panel = alpha_panel.merge_labels(features, labels)
    finite = panel["future_return"].notna().to_numpy()
    z = centred_rank_target(panel)
    w = normalised_weights(panel)
    x = panel[FEATURES].to_numpy(dtype=np.float32)

    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool

    base = {"objective": "regression", "metric": "None", "learning_rate": 0.03,
            "num_leaves": 127, "min_data_in_leaf": 800, "feature_fraction": 0.8,
            "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l1": 0.1,
            "lambda_l2": 40.0, "max_bin": 127, "deterministic": True,
            "force_row_wise": True, "verbose": -1, "num_threads": args.threads}
    for seed in SEEDS:
        params = dict(base)
        params.update({"seed": seed, "bagging_seed": seed,
                       "feature_fraction_seed": seed, "data_random_seed": seed})
        dataset = lgb.Dataset(x[finite], label=z[finite], weight=w[finite],
                              feature_name=FEATURES)
        lgb.train(params, dataset, num_boost_round=L5_ROUNDS).save_model(
            str(out / f"l5_seed{seed}.txt"))
        model = CatBoostRegressor(
            loss_function="RMSE", iterations=C1_TREES, learning_rate=0.03, depth=8,
            l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
            bootstrap_type="Bernoulli", subsample=0.8, thread_count=args.threads,
            random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
        model.fit(Pool(x[finite], label=z[finite], weight=w[finite],
                       feature_names=FEATURES))
        model.save_model(str(out / f"c1_seed{seed}.cbm"), format="cbm")
        print("done seed", seed, flush=True)


if __name__ == "__main__":
    main()
