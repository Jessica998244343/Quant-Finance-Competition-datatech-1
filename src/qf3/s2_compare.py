"""A-stage three-column comparison: legacy OOF / original model + real input / aligned retrain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, metrics, s2_models

OUT = Path("artifacts/stage2_ic_ir")
PRED = OUT / "predictions"
LEGACY_PRED = Path("artifacts/beginning/predictions")
WINDOWS = ["DEV19", "DEV20", "CONF21"]


def _rows(panel: pd.DataFrame, split: dict) -> np.ndarray:
    return np.nonzero(split["predict"])[0]


def evaluate(preds: pd.DataFrame, panel_platform: pd.DataFrame, tag: str) -> dict:
    merged = preds.merge(
        panel_platform.loc[:, ["date", "code", "future_return"]],
        on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    summary["tag"] = tag
    return summary, rankic


def run_window(legacy: pd.DataFrame, platform: pd.DataFrame, window: str,
               verify_legacy: bool = True) -> dict:
    split_legacy = cv.fold_split(legacy, window)
    split_platform = cv.fold_split(platform, window)
    rows_legacy = _rows(legacy, split_legacy)
    rows_platform = _rows(platform, split_platform)
    dates = legacy.loc[rows_legacy, "date"].to_numpy()
    codes = legacy.loc[rows_legacy, "code"].to_numpy()
    assert list(dates) == list(platform.loc[rows_platform, "date"].to_numpy())
    assert list(codes) == list(platform.loc[rows_platform, "code"].to_numpy())

    record = {"window": window, "n_signal_days": int(len(set(dates.tolist())))}

    # --- 1. legacy OOF (stage-1 artifacts) --------------------------------
    legacy_p5 = pd.read_parquet(LEGACY_PRED / f"P5_s20261005__{window}.parquet")
    legacy_cat = pd.read_parquet(LEGACY_PRED / f"SC_d8_l2_15__{window}.parquet")
    record["legacy_oof"] = {
        "p5": evaluate(legacy_p5, platform, "B0G1-legacy-oof")[0],
        "catboost": evaluate(legacy_cat, platform, "catboost-legacy-oof")[0],
    }

    # --- 2. refit on legacy to recover the fold boosters -------------------
    p5_fit = s2_models.fit_lgbm(legacy, split_legacy)
    cat_fit = s2_models.fit_catboost(legacy, split_legacy)
    p5_refit = s2_models.predict_lgbm(p5_fit["booster"], legacy, rows_legacy)
    cat_refit = s2_models.predict_catboost(cat_fit["model"], legacy, rows_legacy)
    record["refit_rounds"] = {"p5": p5_fit["rounds"], "catboost": cat_fit["rounds"]}
    if verify_legacy:
        record["refit_matches_legacy_oof"] = {
            "p5_max_abs_diff": float(np.max(np.abs(
                p5_refit - legacy_p5.sort_values(["date", "code"])["score"].to_numpy()))),
            "catboost_max_abs_diff": float(np.max(np.abs(
                cat_refit - legacy_cat.sort_values(["date", "code"])["score"].to_numpy()))),
        }

    # --- 3. original fold models applied to platform-semantics inputs ------
    p5_real = s2_models.predict_lgbm(p5_fit["booster"], platform, rows_platform)
    cat_real = s2_models.predict_catboost(cat_fit["model"], platform, rows_platform)
    real_frame = pd.DataFrame({
        "date": dates, "code": codes,
        "p5": p5_real, "catboost": cat_real,
        "score": s2_models.blend(p5_real, cat_real, 0.25)})
    record["original_model_real_input"] = {
        "p5": evaluate(real_frame.assign(score=real_frame["p5"]), platform, "p5")[0],
        "catboost": evaluate(real_frame.assign(score=real_frame["catboost"]), platform,
                             "catboost")[0],
        "blend": evaluate(real_frame, platform, "blend")[0],
    }
    pred_frame = real_frame.rename(columns={"p5": "score_lgbm_p5",
                                            "catboost": "score_catboost"})
    pred_frame[["date", "code", "score"]].to_parquet(
        PRED / f"origmodel_realinput__{window}.parquet", index=False)
    pred_frame.to_parquet(PRED / f"origmodel_realinput_members__{window}.parquet",
                          index=False)

    # --- 4. aligned retrain on platform semantics -------------------------
    p5_al = s2_models.fit_lgbm(platform, split_platform)
    cat_al = s2_models.fit_catboost(platform, split_platform)
    p5_al_scores = s2_models.predict_lgbm(p5_al["booster"], platform, rows_platform)
    cat_al_scores = s2_models.predict_catboost(cat_al["model"], platform, rows_platform)
    aligned_frame = pd.DataFrame({
        "date": dates, "code": codes,
        "score": s2_models.blend(p5_al_scores, cat_al_scores, 0.25)})
    record["aligned_retrain"] = {
        "rounds": {"p5": p5_al["rounds"], "catboost": cat_al["rounds"]},
        "p5": evaluate(aligned_frame.assign(score=p5_al_scores), platform, "p5")[0],
        "blend": evaluate(aligned_frame, platform, "blend")[0],
    }
    aligned_frame.to_parquet(PRED / f"aligned_retrain__{window}.parquet", index=False)
    pd.DataFrame({"date": dates, "code": codes,
                  "score_lgbm_p5": p5_al_scores, "score_catboost": cat_al_scores}).to_parquet(
        PRED / f"aligned_retrain_members__{window}.parquet", index=False)
    s2_models.save_booster(p5_al["booster"], f"aligned_p5_{window}")
    s2_models.save_catboost(cat_al["model"], f"aligned_catboost_{window}")
    s2_models.save_booster(p5_fit["booster"], f"legacy_p5_{window}")
    s2_models.save_catboost(cat_fit["model"], f"legacy_catboost_{window}")
    return record


def main() -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_parquet("artifacts/beginning/cache/panel.parquet")
    platform = pd.read_parquet(OUT / "panel_platform.parquet")
    results = {}
    for window in WINDOWS:
        print(f"== {window}", flush=True)
        results[window] = run_window(legacy, platform, window)
    s2_models.write_json(OUT / "three_column_comparison.json", results)
    rows = []
    for window, record in results.items():
        for source, key in (("legacy OOF", "legacy_oof"),
                            ("original model + real input", "original_model_real_input"),
                            ("aligned retrain", "aligned_retrain")):
            block = record[key]
            for member in ("p5", "catboost", "blend"):
                if member in block:
                    rows.append({"window": window, "source": source, "member": member,
                                 "mean_rankic": block[member]["mean_rankic"],
                                 "rankic_ir": block[member]["rankic_ir"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "three_column_comparison.csv", index=False)
    print(table.to_string(index=False))
    for window, record in results.items():
        print(window, "refit vs legacy:", record.get("refit_matches_legacy_oof"))


if __name__ == "__main__":
    main()
