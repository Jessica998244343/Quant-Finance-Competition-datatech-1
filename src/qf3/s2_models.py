"""A-stage: refit year-fold reference models and predict under both input semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

OUT = Path("artifacts/stage2_ic_ir")
MODEL_DIR = OUT / "models_fold"

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)


def fit_lgbm(panel: pd.DataFrame, split: dict, *, rounds: int | None = None,
             seed: int = 20261005, threads: int = 8) -> dict:
    """Refit the P5 recipe; when ``rounds`` is None, recover it by early stopping."""
    if rounds is None:
        result = trainers.train_lgbm_fold(panel, FEATURES, split,
                                          trainers.CAPACITY_GRID["P5"], seed, threads)
        return {"booster": result["booster"], "rounds": int(result["best_iteration"]),
                "es_score": result["es_score"]}
    dates = cv.to_int_date(panel["date"])
    y, w, finite = cv.make_target(panel)
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    outer = split["outer_train"] & finite
    params = trainers.lgb_params(trainers.CAPACITY_GRID["P5"], seed, threads)
    params.pop("_num_rounds")
    import lightgbm as lgb
    dataset = lgb.Dataset(x[outer], label=y[outer], weight=w[outer],
                          feature_name=FEATURES)
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    return {"booster": booster, "rounds": rounds, "es_score": None}


def fit_catboost(panel: pd.DataFrame, split: dict, *, rounds: int | None = None,
                 seed: int = 20261005, threads: int = 8) -> dict:
    if rounds is None:
        result = trainers.train_catboost_fold(panel, FEATURES, split, depth=8,
                                              l2_leaf_reg=15.0, seed=seed,
                                              threads=threads)
        return {"model": result["model"], "rounds": int(result["best_iteration"])}
    from catboost import CatBoostRegressor, Pool
    y, w, finite = cv.make_target(panel)
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    outer = split["outer_train"] & finite
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=rounds, learning_rate=0.03, depth=8,
        l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
        bootstrap_type="Bernoulli", subsample=0.8, thread_count=threads,
        random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    model.fit(Pool(x[outer], label=y[outer], weight=w[outer], feature_names=FEATURES))
    return {"model": model, "rounds": rounds}


def predict_lgbm(booster, panel: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    x = panel.loc[rows, FEATURES].to_numpy(dtype=np.float32)
    return booster.predict(x)


def predict_catboost(model, panel: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    x = panel.loc[rows, FEATURES].to_numpy(dtype=np.float32)
    return np.asarray(model.predict(x), dtype=float).ravel()


def centered_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def blend(lgbm_scores: np.ndarray, cat_scores: np.ndarray, weight: float = 0.25) -> np.ndarray:
    return (1.0 - weight) * centered_rank(lgbm_scores) + weight * centered_rank(cat_scores)


def save_booster(booster, name: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.txt"
    booster.save_model(str(path))
    return str(path)


def save_catboost(model, name: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.cbm"
    model.save_model(str(path), format="cbm")
    return str(path)


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
