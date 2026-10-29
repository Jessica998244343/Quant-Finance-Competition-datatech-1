"""Model training under the frozen forward-year expanding protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import cv

B0_PARAMS = {
    "boosting": "gbdt",
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "seed": 20261005,
    "num_threads": 8,
}

LGBM_FIXED = {
    "metric": "None",
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

CAPACITY_GRID = {
    "P1": {"num_leaves": 15, "min_data_in_leaf": 300, "lambda_l1": 0.0,
           "lambda_l2": 10.0, "feature_fraction": 0.9, "bagging_fraction": 0.8},
    "P2": {"num_leaves": 31, "min_data_in_leaf": 100, "lambda_l1": 0.0,
           "lambda_l2": 2.0, "feature_fraction": 0.9, "bagging_fraction": 0.8},
    "P3": {"num_leaves": 31, "min_data_in_leaf": 400, "lambda_l1": 0.1,
           "lambda_l2": 10.0, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    "P4": {"num_leaves": 63, "min_data_in_leaf": 200, "lambda_l1": 0.0,
           "lambda_l2": 10.0, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    "P5": {"num_leaves": 63, "min_data_in_leaf": 500, "lambda_l1": 0.1,
           "lambda_l2": 20.0, "feature_fraction": 0.8, "bagging_fraction": 0.7},
    "P6": {"num_leaves": 127, "min_data_in_leaf": 500, "lambda_l1": 0.5,
           "lambda_l2": 30.0, "feature_fraction": 0.7, "bagging_fraction": 0.8},
}


def lgb_params(overrides: dict | None = None, seed: int = 20261005,
               threads: int = 8, num_rounds: int = 3000) -> dict:
    params = dict(B0_PARAMS)
    params.update(LGBM_FIXED)
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed
    params["data_random_seed"] = seed
    params["num_threads"] = threads
    params["_num_rounds"] = num_rounds
    if overrides:
        params.update(overrides)
    return params


def winsor_target(frame: pd.DataFrame, label_col: str = "future_return",
                  limits=(0.01, 0.99)):
    """Daily 1%/99% winsorised, demeaned raw H=1 return with date-equal weights."""
    label = pd.to_numeric(frame[label_col], errors="coerce")
    grouped = label.groupby(frame["date"])
    lower = grouped.transform(lambda s: s.quantile(limits[0]))
    upper = grouped.transform(lambda s: s.quantile(limits[1]))
    clipped = label.clip(lower=lower, upper=upper)
    means = clipped.groupby(frame["date"]).transform("mean")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    y = clipped - means
    weights = (1.0 / counts)
    finite = label.notna().to_numpy()
    if finite.any():
        weights = weights / weights[finite].mean()
    return y.to_numpy(dtype=float), weights.to_numpy(dtype=float), finite


def aging_weights(dates: np.ndarray, cutoff: int, half_life: float | None):
    if half_life is None:
        return np.ones_like(dates, dtype=float)
    unique = np.sort(np.unique(dates[np.isfinite(dates)]))
    positions = {int(day): idx for idx, day in enumerate(unique)}
    cutoff_pos = positions.get(int(cutoff))
    if cutoff_pos is None:
        cutoff_pos = len(unique) - 1
    age = np.array([max(0, cutoff_pos - positions[int(day)]) for day in dates])
    return np.power(2.0, -age / float(half_life))


def train_lgbm_fold(panel: pd.DataFrame, features: list[str], split: dict,
                    overrides: dict | None = None, seed: int = 20261005,
                    threads: int = 8, num_rounds: int = 3000,
                    early_stopping: int = 200, label_col: str = "future_return",
                    verbose: bool = False, target_mode: str = "rank",
                    half_life: float | None = None) -> dict:
    """Early-stop on the internal region, then refit on the full outer window."""
    dates = cv.to_int_date(panel["date"])
    if target_mode == "rank":
        y, w, finite = cv.make_target(panel, label_col)
    elif target_mode == "winsor":
        y, w, finite = winsor_target(panel, label_col)
    else:
        raise ValueError(target_mode)
    if half_life is not None:
        w = w * aging_weights(dates, split["train_end"], half_life)
        w = w / np.nanmean(w[finite])
    x = panel[features].to_numpy(dtype=np.float32, copy=False)
    params = lgb_params(overrides, seed, threads, num_rounds)
    rounds = int(params.pop("_num_rounds"))

    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    valid_idx = np.nonzero(early)[0]

    def feval(preds, dataset):
        value = cv.daily_spearman(dates[valid_idx], y[valid_idx], preds)
        return "daily_spearman", value, True

    train_set = lgb.Dataset(x[internal], label=y[internal], weight=w[internal],
                            feature_name=features, free_raw_data=False)
    valid_set = lgb.Dataset(x[early], label=y[early], weight=w[early],
                            reference=train_set, feature_name=features,
                            free_raw_data=False)

    started = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
        feval=feval,
        callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best_iter = int(booster.best_iteration or rounds)
    es_time = time.time() - started
    es_score = float(booster.best_score["valid_0"]["daily_spearman"]) \
        if booster.best_score else float("nan")

    final_set = lgb.Dataset(x[outer], label=y[outer], weight=w[outer],
                            feature_name=features, free_raw_data=False)
    started = time.time()
    final = lgb.train(params, final_set, num_boost_round=best_iter)
    refit_time = time.time() - started

    preds = final.predict(x[predict], num_iteration=best_iter)
    pred_frame = pd.DataFrame({
        "date": cv.to_int_date(panel.loc[predict, "date"]),
        "code": panel.loc[predict, "code"].to_numpy(),
        "score": preds,
    })
    importance = pd.Series(final.feature_importance("gain"), index=features).sort_values(
        ascending=False)
    return {
        "predictions": pred_frame,
        "best_iteration": best_iter,
        "es_score": es_score,
        "es_seconds": es_time,
        "refit_seconds": refit_time,
        "n_internal": int(internal.sum()),
        "n_es": int(early.sum()),
        "n_outer": int(outer.sum()),
        "n_predict": int(predict.sum()),
        "feature_importance_gain": importance.to_dict(),
        "booster": final,
    }


def train_ridge_fold(panel: pd.DataFrame, features: list[str], split: dict,
                     alphas=(1.0, 10.0, 100.0, 1000.0),
                     label_col: str = "future_return") -> dict:
    from sklearn.linear_model import Ridge

    y, w, finite = cv.make_target(panel, label_col)
    x = panel[features].fillna(0.0).to_numpy(dtype=np.float64)
    dates = cv.to_int_date(panel["date"])
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    best_alpha, best_score = None, -np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x[internal], y[internal], sample_weight=w[internal])
        score = cv.daily_spearman(dates[early], y[early], model.predict(x[early]))
        if np.isfinite(score) and score > best_score:
            best_alpha, best_score = alpha, score

    model = Ridge(alpha=best_alpha)
    model.fit(x[outer], y[outer], sample_weight=w[outer])
    preds = model.predict(x[predict])
    return {
        "predictions": pd.DataFrame({
            "date": dates[predict],
            "code": panel.loc[predict, "code"].to_numpy(),
            "score": preds,
        }),
        "alpha": best_alpha,
        "es_score": best_score,
        "n_outer": int(outer.sum()),
    }


def train_catboost_fold(panel: pd.DataFrame, features: list[str], split: dict,
                        depth: int = 6, l2_leaf_reg: float = 5.0,
                        iterations: int = 3000, learning_rate: float = 0.03,
                        seed: int = 20261005, threads: int = 8,
                        od_wait: int = 200, label_col: str = "future_return",
                        eval_metric: str = "RMSE") -> dict:
    from catboost import CatBoostRegressor, Pool

    y, w, finite = cv.make_target(panel, label_col)
    x = panel[features].to_numpy(dtype=np.float32, copy=False)
    dates = cv.to_int_date(panel["date"])
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    params = dict(
        loss_function="RMSE", iterations=iterations, learning_rate=learning_rate,
        depth=depth, l2_leaf_reg=l2_leaf_reg, border_count=128,
        random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.8,
        thread_count=threads, random_seed=seed, nan_mode="Min",
        od_type="Iter", od_wait=od_wait, verbose=False,
        allow_writing_files=False,
    )
    train_pool = Pool(x[internal], label=y[internal], weight=w[internal],
                      feature_names=features)
    valid_pool = Pool(x[early], label=y[early], weight=w[early],
                      feature_names=features)
    model = CatBoostRegressor(**params)
    started = time.time()
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    fit_seconds = time.time() - started
    best_iter = int(model.get_best_iteration() or iterations)

    full_params = dict(params)
    full_params.update(iterations=max(1, best_iter), od_type=None, od_wait=None)
    final = CatBoostRegressor(**full_params)
    started = time.time()
    final.fit(Pool(x[outer], label=y[outer], weight=w[outer], feature_names=features))
    refit_seconds = time.time() - started
    preds = final.predict(x[predict])
    preds = np.asarray(preds, dtype=float).ravel()
    return {
        "predictions": pd.DataFrame({
            "date": dates[predict],
            "code": panel.loc[predict, "code"].to_numpy(),
            "score": preds,
        }),
        "best_iteration": best_iter,
        "fit_seconds": fit_seconds,
        "refit_seconds": refit_seconds,
        "model": final,
        "feature_importance": final.get_feature_importance().tolist(),
    }


def log_experiment(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
