"""REV1 corrected trainer: fold-local weight normalisation and CatBoost round semantics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import cv

FOLD_LOCAL = "fold_local"


def rank_target_unnormalised(frame: pd.DataFrame,
                             label_col: str = "future_return"):
    """Daily rank target plus *unnormalised* date-equal weights."""
    label = pd.to_numeric(frame[label_col], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    z = (ranks - (counts + 1.0) / 2.0) / counts
    raw_weights = 1.0 / counts
    return z.to_numpy(dtype=float), raw_weights.to_numpy(dtype=float), finite


def normalise_within(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise so the *fitting sample* has mean weight 1 (CORR-2)."""
    out = np.array(weights, dtype=float, copy=True)
    mean = float(np.nanmean(out[mask]))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid weight mean inside fitting sample")
    out = out / mean
    return out


def train_lgbm_fold_local(panel: pd.DataFrame, features: list[str], split: dict,
                          overrides: dict | None = None, seed: int = 20261005,
                          threads: int = 8, num_rounds: int = 3000,
                          early_stopping: int = 200,
                          label_col: str = "future_return") -> dict:
    """P5-style fold training with CORR-2; early stopping scores raw labels."""
    import lightgbm as lgb
    from . import trainers

    dates = cv.to_int_date(panel["date"])
    z, raw_w, finite = rank_target_unnormalised(panel, label_col)
    raw_returns = pd.to_numeric(panel[label_col], errors="coerce").to_numpy(dtype=float)
    x = panel[features].to_numpy(dtype=np.float32, copy=False)
    params = trainers.lgb_params(overrides, seed, threads, num_rounds)
    rounds = int(params.pop("_num_rounds"))

    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    w_internal = normalise_within(raw_w, internal)
    w_outer = normalise_within(raw_w, outer)

    valid_idx = np.nonzero(early)[0]

    def feval(preds, dataset):
        value = cv.daily_spearman(dates[valid_idx], raw_returns[valid_idx], preds)
        return "daily_spearman", value, True

    train_set = lgb.Dataset(x[internal], label=z[internal], weight=w_internal[internal],
                            feature_name=features, free_raw_data=False)
    valid_set = lgb.Dataset(x[early], label=z[early], weight=w_internal[early],
                            reference=train_set, feature_name=features,
                            free_raw_data=False)
    started = time.time()
    history = {}
    booster = lgb.train(params, train_set, num_boost_round=rounds,
                        valid_sets=[valid_set], feval=feval,
                        callbacks=[lgb.early_stopping(early_stopping,
                                                      first_metric_only=True, verbose=False),
                                   lgb.record_evaluation(history),
                                   lgb.log_evaluation(0)])
    best_iter = int(booster.best_iteration or rounds)
    es_seconds = time.time() - started
    final_set = lgb.Dataset(x[outer], label=z[outer], weight=w_outer[outer],
                            feature_name=features, free_raw_data=False)
    started = time.time()
    final = lgb.train(params, final_set, num_boost_round=best_iter)
    refit_seconds = time.time() - started
    preds = final.predict(x[predict], num_iteration=best_iter)
    predictions = pd.DataFrame({"date": dates[predict].astype("int64"),
                                "code": panel.loc[predict, "code"].to_numpy(),
                                "score": preds})
    return {"predictions": predictions, "booster": final, "rounds": best_iter,
            "es_seconds": es_seconds, "refit_seconds": refit_seconds,
            "history": history,
            "weight_mean_internal": float(w_internal[internal].mean()),
            "weight_mean_outer": float(w_outer[outer].mean())}


def train_catboost_fold_local(panel: pd.DataFrame, features: list[str], split: dict,
                              depth: int = 8, l2_leaf_reg: float = 15.0,
                              iterations: int = 3000, learning_rate: float = 0.03,
                              seed: int = 20261005, threads: int = 8,
                              od_wait: int = 200,
                              label_col: str = "future_return") -> dict:
    """CatBoost fold training with CORR-1 (index vs tree count) and CORR-2."""
    from catboost import CatBoostRegressor, Pool

    z, raw_w, finite = rank_target_unnormalised(panel, label_col)
    x = panel[features].to_numpy(dtype=np.float32, copy=False)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    w_internal = normalise_within(raw_w, internal)
    w_outer = normalise_within(raw_w, outer)

    base = dict(loss_function="RMSE", iterations=iterations,
                learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
                border_count=128, random_strength=1.0, bootstrap_type="Bernoulli",
                subsample=0.8, thread_count=threads, random_seed=seed,
                nan_mode="Min", verbose=False, allow_writing_files=False,
                od_type="Iter", od_wait=od_wait)
    model = CatBoostRegressor(**base)
    started = time.time()
    model.fit(Pool(x[internal], label=z[internal], weight=w_internal[internal],
                   feature_names=features),
              eval_set=Pool(x[early], label=z[early], weight=w_internal[early],
                            feature_names=features),
              use_best_model=True)
    fit_seconds = time.time() - started
    best_index = model.get_best_iteration()
    if best_index is None:
        tree_count = iterations
        fallback = True
    else:
        tree_count = int(best_index) + 1        # CORR-1
        fallback = False

    final_params = dict(base)
    final_params.update(iterations=max(1, tree_count), od_type=None, od_wait=None)
    final = CatBoostRegressor(**final_params)
    started = time.time()
    final.fit(Pool(x[outer], label=z[outer], weight=w_outer[outer],
                   feature_names=features))
    refit_seconds = time.time() - started
    preds = np.asarray(final.predict(x[predict]), dtype=float).ravel()
    predictions = pd.DataFrame({"date": cv.to_int_date(panel.loc[predict, "date"]).astype("int64"),
                                "code": panel.loc[predict, "code"].to_numpy(),
                                "score": preds})
    return {"predictions": predictions, "model": final, "rounds": int(tree_count),
            "best_index": None if best_index is None else int(best_index),
            "fallback_used": fallback, "tree_count": int(final.tree_count_),
            "fit_seconds": fit_seconds, "refit_seconds": refit_seconds,
            "weight_mean_internal": float(w_internal[internal].mean()),
            "weight_mean_outer": float(w_outer[outer].mean())}
