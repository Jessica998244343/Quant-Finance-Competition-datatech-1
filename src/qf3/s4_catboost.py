"""D1 §7.2: CatBoost with IC-based tree-count selection (C1-C3) plus control C0."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import cv, metrics, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
MODELS = FUS / "models_cat"
PRED = FUS / "predictions_cat"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
STEP = 25
MAX_TREES = 3000

CONFIGS = {
    "C1": {"depth": 8, "l2_leaf_reg": 15.0},
    "C2": {"depth": 6, "l2_leaf_reg": 15.0},
    "C3": {"depth": 8, "l2_leaf_reg": 30.0},
}


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def train_one(data: pd.DataFrame, window: str, name: str, params: dict,
              seed: int = 20261005, threads: int = 6) -> dict:
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)

    base = dict(loss_function="RMSE", iterations=MAX_TREES, learning_rate=0.03,
                border_count=128, random_strength=1.0, bootstrap_type="Bernoulli",
                subsample=0.8, thread_count=threads, random_seed=seed,
                nan_mode="Min", verbose=False, allow_writing_files=False, **params)
    started = time.time()
    model = CatBoostRegressor(**base)
    model.fit(Pool(x[internal], label=y[internal], weight=w_internal[internal],
                   feature_names=FEATURES),
              eval_set=Pool(x[early], label=y[early], weight=w_internal[early],
                            feature_names=FEATURES),
              use_best_model=False)
    fit_seconds = time.time() - started

    valid_index = np.nonzero(early)[0]
    checkpoints, curve = [], []
    for trees in range(STEP, MAX_TREES + 1, STEP):
        preds = model.predict(x[early], ntree_end=trees)
        curve.append(cv.daily_spearman(dates[valid_index],
                                       raw_returns[valid_index], preds))
        checkpoints.append(trees)
    curve = np.array(curve, dtype=float)
    best_index = int(np.nanargmax(curve))
    best_value = float(curve[best_index])
    near = np.nonzero(curve >= best_value - 0.0005)[0]
    selected = int(checkpoints[int(near[0])])          # earliest near-optimal checkpoint

    final_params = dict(base)
    final_params.update(iterations=selected, od_type=None, od_wait=None)
    final = CatBoostRegressor(**final_params)
    started = time.time()
    final.fit(Pool(x[outer], label=y[outer], weight=w_outer[outer],
                   feature_names=FEATURES))
    refit_seconds = time.time() - started
    preds = np.asarray(final.predict(x[predict]), dtype=float).ravel()
    frame = pd.DataFrame({"date": dates[predict].astype("int64"),
                          "code": data.loc[predict, "code"].to_numpy(),
                          "score": preds})
    MODELS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{seed}__{window}"
    final.save_model(str(MODELS / f"{tag}.cbm"), format="cbm")
    frame.to_parquet(PRED / f"{tag}.parquet", index=False)
    merged = frame.merge(data[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    summary = metrics.summarize_rankic(metrics.daily_rankic(merged))
    record = {"recipe": name, "seed": seed, "window": window,
              "depth": params["depth"], "l2_leaf_reg": params["l2_leaf_reg"],
              "best_checkpoint": checkpoints[best_index],
              "best_inner_ic": best_value,
              "selected_trees": selected,
              "near_optimal_last": int(checkpoints[int(near[-1])]),
              "actual_tree_count": int(final.tree_count_),
              "fit_seconds": round(fit_seconds, 1),
              "refit_seconds": round(refit_seconds, 1),
              "mean_rankic": summary["mean_rankic"], "n_days": summary["n_days"],
              "model_path": str(MODELS / f"{tag}.cbm")}
    return record, curve, checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", default="C1,C2,C3")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    FUS.mkdir(parents=True, exist_ok=True)
    data = panel()
    log_path = FUS / "d1_catboost.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"records": []}
    for name in args.recipes.split(","):
        for window in args.windows.split(","):
            record, curve, checkpoints = train_one(data, window, name,
                                                   CONFIGS[name], threads=args.threads)
            report["records"] = [r for r in report["records"]
                                 if not (r["recipe"] == name and r["window"] == window
                                         and r["seed"] == 20261005)]
            report["records"].append(record)
            pd.DataFrame({"trees": checkpoints, "inner_daily_ic": curve}).to_parquet(
                MODELS / f"{name}_20261005__{window}__curve.parquet", index=False)
            log_path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps(record, default=str), flush=True)
    print(pd.DataFrame(report["records"]).to_string(index=False))


if __name__ == "__main__":
    main()
