"""B2: seed averaging and slow-information experts, measured against the B1 reference."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import cv, s2b_common, s2r_blend, s2r_train, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

ALL_FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
FAST_FEATURES = ["ret_1", "ret_2", "hl_range", "close_pos", "overnight", "intraday",
                 "daily_return", "change_ratio"]
SLOW_FEATURES = [f for f in ALL_FEATURES if f not in FAST_FEATURES]
REFERENCE_WEIGHT = 0.5
SEEDS = [20261005, 20261017, 20261029]

SLOW_CONFIGS = {
    "E1_p5params": dict(trainers.CAPACITY_GRID["P5"]),
    "E2_conservative": {"num_leaves": 31, "min_data_in_leaf": 700, "lambda_l1": 0.1,
                        "lambda_l2": 30.0, "feature_fraction": 0.8,
                        "bagging_fraction": 0.7},
}


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def train_seed_variants(data: pd.DataFrame, windows) -> None:
    for seed in SEEDS[1:]:
        for window in windows:
            split = cv.fold_split(data, window)
            result = s2r_train.train_lgbm_fold_local(
                data, ALL_FEATURES, split, trainers.CAPACITY_GRID["P5"],
                seed=seed, threads=8)
            result["predictions"].to_parquet(
                s2b_common.PRED / f"seed{seed}_lgbm__{window}.parquet", index=False)
            s2b_common.log({"stage": "B2-seed", "seed": seed, "window": window,
                            "rounds": result["rounds"]})
            print(f"  seed {seed} {window}: rounds={result['rounds']}", flush=True)


def seed_average(windows) -> dict:
    out = {}
    for window in windows:
        members = []
        for seed in SEEDS:
            if seed == SEEDS[0]:
                path = (s2b_common.REV / "predictions" /
                        f"ref_corrected_lgbm__{window}.parquet")
            else:
                path = s2b_common.PRED / f"seed{seed}_lgbm__{window}.parquet"
            frame = pd.read_parquet(path).rename(columns={"score": f"lgbm_{seed}"})
            members.append(frame)
        merged = members[0]
        for frame in members[1:]:
            merged = merged.merge(frame, on=["date", "code"], validate="one_to_one")
        columns = [f"lgbm_{seed}" for seed in SEEDS]
        ranked = pd.DataFrame({"date": merged["date"], "code": merged["code"]})
        accumulated = np.zeros(len(merged))
        for column in columns:
            series = merged[column].astype(float)
            centered = ((series.rank(method="average")
                         - (len(series) + 1) / 2.0) / len(series))
            accumulated += centered.to_numpy() / len(columns)
        ranked["lgbm_seedavg"] = accumulated

        cat = pd.read_parquet(s2b_common.REV / "predictions" /
                              f"ref_corrected_catboost__{window}.parquet")
        both = ranked.rename(columns={"lgbm_seedavg": "lgbm_p5"}).merge(
            cat.rename(columns={"score": "catboost"}), on=["date", "code"],
            validate="one_to_one")
        blended = s2r_blend.blend_frame(both, ["lgbm_p5", "catboost"],
                                        [1 - REFERENCE_WEIGHT, REFERENCE_WEIGHT])
        blended.to_parquet(s2b_common.PRED / f"b2_seedavg__{window}.parquet", index=False)
        out[window] = blended
    return out


def train_slow_experts(data: pd.DataFrame, windows) -> dict:
    results = {}
    for name, params in SLOW_CONFIGS.items():
        for window in windows:
            split = cv.fold_split(data, window)
            result = s2r_train.train_lgbm_fold_local(data, SLOW_FEATURES, split,
                                                     params, seed=20261005, threads=8)
            result["predictions"].to_parquet(
                s2b_common.PRED / f"b2_{name}__{window}.parquet", index=False)
            results[(name, window)] = result["rounds"]
            s2b_common.log({"stage": "B2-slow", "expert": name, "window": window,
                            "rounds": result["rounds"], "n_features": len(SLOW_FEATURES)})
            print(f"  {name} {window}: rounds={result['rounds']}", flush=True)
    return results


def blend_with_reference(expert: str, alpha: float, windows) -> dict:
    out = {}
    for window in windows:
        ref = pd.read_parquet(s2b_common.PRED /
                              f"ref_corrected_cat{REFERENCE_WEIGHT:g}__{window}.parquet")
        exp = pd.read_parquet(s2b_common.PRED / f"b2_{expert}__{window}.parquet")
        both = ref.rename(columns={"score": "reference"}).merge(
            exp.rename(columns={"score": "expert"}), on=["date", "code"],
            validate="one_to_one")
        blended = s2r_blend.blend_frame(both, ["reference", "expert"],
                                        [1 - alpha, alpha])
        blended.to_parquet(s2b_common.PRED /
                           f"b2_{expert}_a{alpha:g}__{window}.parquet", index=False)
        out[window] = blended
    return out


def evaluate_frames(frames: dict, label: str, panel_frame: pd.DataFrame) -> dict:
    rows = []
    for window, frame in frames.items():
        summary, _ = s2b_common.ic_summary(frame, panel_frame)
        rows.append({"label": label, "window": window, "mean_rankic": summary["mean_rankic"],
                     "rankic_ir": summary["rankic_ir"], "n_days": summary["n_days"]})
    return {"label": label, "windows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", action="store_true")
    parser.add_argument("--train-slow", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    s2b_common.PRED.mkdir(parents=True, exist_ok=True)
    data = panel()
    windows = ["DEV19", "DEV20", "CONF21"]
    if args.train_seeds:
        train_seed_variants(data, windows)
    if args.train_slow:
        train_slow_experts(data, windows)
    if args.evaluate:
        report = {"slow_feature_count": len(SLOW_FEATURES),
                  "removed_features": FAST_FEATURES}
        seed_frames = seed_average(windows)
        report["seed_average"] = evaluate_frames(seed_frames, "seed_average", data)
        for expert in SLOW_CONFIGS:
            solo = {w: pd.read_parquet(s2b_common.PRED / f"b2_{expert}__{w}.parquet")
                    for w in windows}
            report[f"{expert}_solo"] = evaluate_frames(solo, expert, data)
            for alpha in (0.1, 0.2):
                frames = blend_with_reference(expert, alpha, windows)
                report[f"{expert}_a{alpha:g}"] = evaluate_frames(
                    frames, f"{expert} a={alpha}", data)
        s2b_common.write_json(s2b_common.B / "b2_ic.json", report)
        for key, value in report.items():
            if isinstance(value, dict) and "windows" in value:
                print(key, [(r["window"], round(r["mean_rankic"], 5)) for r in value["windows"]])


if __name__ == "__main__":
    main()
