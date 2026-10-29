"""D1 §6: round-count protocols T0/T1/T2 with strict label-maturity boundaries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, metrics, s2r_train, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
MODELS = FUS / "models_rounds"
PRED = FUS / "predictions_rounds"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
L3 = {"num_leaves": 63, "min_data_in_leaf": 300, "lambda_l1": 0.1,
      "lambda_l2": 20.0, "feature_fraction": 0.8, "bagging_fraction": 0.7}
GRID = [50, 75, 100, 150, 200, 300, 450, 650, 900, 1200, 1600, 2200, 3000]
PATIENCE = 200
MAX_ROUNDS = 3000


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def segments(data: pd.DataFrame, window: str) -> list[dict]:
    """Two consecutive 63-signal-day inner validation blocks, strict maturity."""
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    label_end = cv.to_int_date(data["label_end_date"])
    finite = data["future_return"].notna().to_numpy()
    outer = split["outer_train"] & finite
    train_dates = np.sort(pd.unique(dates[outer]))
    if len(train_dates) < 130:
        raise ValueError(f"{window}: not enough training days for two segments")
    blocks = [(train_dates[-126:-63], train_dates[-63:])]
    out = []
    for index, (first_block, second_block) in enumerate(blocks):
        for j, validation in enumerate((first_block, second_block)):
            start = int(validation[0])
            end = int(validation[-1])
            fit = outer & (label_end < start)
            valid = outer & (dates >= start) & (dates <= end)
            boundaries = {
                "segment": f"{index + 1}.{j + 1}",
                "validation_start": start, "validation_end": end,
                "n_fit": int(fit.sum()), "n_valid": int(valid.sum()),
                "max_fit_label_end": int(np.nanmax(label_end[fit])),
                "max_valid_label_end": int(np.nanmax(label_end[valid])),
                "fit_strictly_before_validation": bool(
                    np.nanmax(label_end[fit]) < start),
                "valid_labels_mature_by_train_end": bool(
                    np.nanmax(label_end[valid]) <= split["train_end"]),
                "violations": int((fit & (label_end >= start)).sum()),
            }
            out.append({"window": window, "fit": fit, "valid": valid,
                        "boundaries": boundaries})
    return out


def train_trajectory(data: pd.DataFrame, seg: dict, seed: int = 20261005,
                     threads: int = 6) -> dict:
    """Train to the common cap, recording the per-iteration validation daily IC."""
    import lightgbm as lgb

    dates = cv.to_int_date(data["date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    fit = seg["fit"]
    valid = seg["valid"]
    w = s2r_train.normalise_within(raw_w, fit)
    params = trainers.lgb_params(L3, seed, threads)
    params.pop("_num_rounds")
    valid_index = np.nonzero(valid)[0]

    def feval(preds, dataset):
        return ("daily_spearman",
                cv.daily_spearman(dates[valid_index], raw_returns[valid_index], preds),
                True)

    history = {}
    train_set = lgb.Dataset(x[fit], label=y[fit], weight=w[fit],
                            feature_name=FEATURES, free_raw_data=False)
    valid_set = lgb.Dataset(x[valid], label=y[valid], weight=w[valid],
                            reference=train_set, feature_name=FEATURES,
                            free_raw_data=False)
    started = time.time()
    booster = lgb.train(params, train_set, num_boost_round=MAX_ROUNDS,
                        valid_sets=[valid_set], feval=feval,
                        callbacks=[lgb.record_evaluation(history),
                                   lgb.log_evaluation(0)])
    elapsed = time.time() - started
    curve = np.array(history["valid_0"]["daily_spearman"], dtype=float)
    return {"booster": booster, "curve": curve, "seconds": elapsed,
            "n_fit": int(fit.sum()), "n_valid": int(valid.sum())}


def t1_rounds(curve: np.ndarray, patience: int = PATIENCE) -> int:
    """Simulate patience-based early stopping offline."""
    best_value, best_iteration, rounds = -np.inf, 0, 0
    for index, value in enumerate(curve, start=1):
        if np.isfinite(value) and value > best_value:
            best_value, best_iteration = value, index
        rounds = index
        if index - best_iteration >= patience:
            break
    return int(best_iteration)


def t2_rounds(curves: list[np.ndarray], grid=GRID, tolerance: float = 0.001) -> dict:
    averaged = []
    for point in grid:
        values = [curve[point - 1] if point <= len(curve) else np.nan for curve in curves]
        averaged.append(float(np.nanmean(values)))
    smoothed = []
    for index, value in enumerate(averaged):
        neighbours = []
        if index > 0:
            neighbours.append(averaged[index - 1])
        if index < len(averaged) - 1:
            neighbours.append(averaged[index + 1])
        smoothed.append(float(np.nanmean([value, *neighbours])))
    best = max(smoothed)
    selected = next(point for point, value in zip(grid, smoothed)
                    if value >= best - tolerance)
    return {"grid": grid, "averaged": averaged, "smoothed": smoothed,
            "selected_rounds": int(selected), "best_smoothed": best}


def refit_and_predict(data: pd.DataFrame, window: str, rounds: int,
                      seed: int = 20261005, tag: str = "", threads: int = 6) -> dict:
    import lightgbm as lgb

    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    outer = split["outer_train"] & finite
    w = s2r_train.normalise_within(raw_w, outer)
    params = trainers.lgb_params(L3, seed, threads)
    params.pop("_num_rounds")
    dataset = lgb.Dataset(x[outer], label=y[outer], weight=w[outer],
                          feature_name=FEATURES)
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    predict = split["predict"]
    preds = booster.predict(x[predict], num_iteration=rounds)
    frame = pd.DataFrame({"date": dates[predict].astype("int64"),
                          "code": data.loc[predict, "code"].to_numpy(),
                          "score": preds})
    MODELS.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / f"{tag}_{window}.txt"))
    frame.to_parquet(PRED / f"{tag}_{window}.parquet", index=False)
    merged = frame.merge(data[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    summary = metrics.summarize_rankic(metrics.daily_rankic(merged))
    return {"rounds": rounds, "mean_rankic": summary["mean_rankic"],
            "n_days": summary["n_days"]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--segments-only", action="store_true",
                        help="train missing segment trajectories and stop")
    args = parser.parse_args()
    threads = args.threads
    FUS.mkdir(parents=True, exist_ok=True)
    data = panel()
    segments_dir = FUS / "rounds_curves"
    segments_dir.mkdir(parents=True, exist_ok=True)
    meta_path = FUS / "rounds_segments.json"
    report = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "segments": [], "protocols": {}}
    for window in ("DEV19", "DEV20"):
        segs = segments(data, window)
        trajectories = []
        for seg in segs:
            curve_path = segments_dir / f"{window}_{seg['boundaries']['segment']}.npy"
            if curve_path.exists():
                curve = np.load(curve_path)
                print(f"{window} {seg['boundaries']['segment']}: cached "
                      f"({len(curve)} points)", flush=True)
                trajectories.append({"curve": curve})
                continue
            result = train_trajectory(data, seg, threads=threads)
            trajectories.append(result)
            np.save(curve_path, result["curve"])
            report["segments"] = [s for s in report["segments"]
                                  if not (s.get("window") == window
                                          and s.get("segment") ==
                                          seg["boundaries"]["segment"])]
            report["segments"].append({"window": window, **seg["boundaries"],
                                       "seconds": round(result["seconds"], 1),
                                       "threads": threads})
            meta_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"{window} {seg['boundaries']['segment']}: "
                  f"fit {result['n_fit']} valid {result['n_valid']} "
                  f"{result['seconds']:.0f}s (checkpoint saved)", flush=True)
        if args.segments_only:
            continue
        curves = [t["curve"] for t in trajectories]
        t1 = [t1_rounds(curve) for curve in curves]
        t2 = t2_rounds(curves)
        t1_selected = int(np.floor(np.median(t1)))
        refit_t1 = refit_and_predict(data, window, t1_selected, tag="T1",
                                     threads=threads)
        refit_t2 = refit_and_predict(data, window, t2["selected_rounds"], tag="T2",
                                     threads=threads)
        t0_preds = pd.read_parquet(
            Path("artifacts/stage3_ic_first/predictions") /
            f"L3_L0_20261005__{window}.parquet")
        merged = t0_preds.merge(data[["date", "code", "future_return"]],
                                on=["date", "code"], how="left", validate="one_to_one")
        t0 = metrics.summarize_rankic(metrics.daily_rankic(merged))
        report["protocols"][window] = {
            "T0": {"rounds": 334 if window == "DEV19" else 816,
                   "mean_rankic": t0["mean_rankic"]},
            "T1": {"segment_rounds": t1, "selected_rounds": t1_selected, **refit_t1},
            "T2": {"segment_curves": t2, **refit_t2},
        }
        print(json.dumps(report["protocols"][window], default=str), flush=True)
        meta_path.write_text(json.dumps(report, indent=2, default=str))
    (FUS / "d1_rounds.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
