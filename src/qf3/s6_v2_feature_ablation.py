"""§11: C1 feature-group ablation on the frozen control protocol.

Each configuration trains C1 (depth8/l2=15, per-seed inner IC tree selection,
fold-local weights, 6 threads) on BASE 61 columns plus one feature group, and is
compared with the frozen C1 control (seed 20261005) on the two DEV years.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS
from .s6_v2_features import GROUPS

V2 = Path("artifacts/stage4_verified_ic/v2")
FUS = Path("artifacts/stage3_fusion")
BASE_FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEED = 20261005
STEP, MAX_TREES, NEAR_TOL, THREADS = 25, 3000, 0.0005, 6
CONFIGS = {"BASE": (), "F1": GROUPS["F1"], "F2": GROUPS["F2"], "F3": GROUPS["F3"],
           "F123": GROUPS["F1"] + GROUPS["F2"] + GROUPS["F3"]}


def train_c1(data: pd.DataFrame, window: str, features: list[str], seed: int = SEED) -> dict:
    from catboost import CatBoostRegressor, Pool

    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    x = data[features].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)
    base = dict(loss_function="RMSE", iterations=MAX_TREES, learning_rate=0.03, depth=8,
                l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
                bootstrap_type="Bernoulli", subsample=0.8, thread_count=THREADS,
                random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    started = time.time()
    inner = CatBoostRegressor(**base)
    inner.fit(Pool(x[internal], label=y[internal], weight=w_internal[internal],
                   feature_names=features),
              eval_set=Pool(x[early], label=y[early], weight=w_internal[early],
                            feature_names=features),
              use_best_model=False)
    valid = np.nonzero(early)[0]
    checkpoints = list(range(STEP, MAX_TREES + 1, STEP))
    curve = np.array([cv.daily_spearman(dates[valid], raw_returns[valid],
                                        inner.predict(x[early], ntree_end=trees))
                      for trees in checkpoints])
    best = int(np.nanargmax(curve))
    near = np.nonzero(curve >= float(curve[best]) - NEAR_TOL)[0]
    selected = int(checkpoints[int(near[0])])
    final = CatBoostRegressor(**{**base, "iterations": selected, "od_type": None, "od_wait": None})
    final.fit(Pool(x[outer], label=y[outer], weight=w_outer[outer], feature_names=features))
    predict = split["predict"]
    preds = np.asarray(final.predict(x[predict]), dtype=float).ravel()
    frame = pd.DataFrame({"date": dates[predict].astype("int64"),
                          "code": data.loc[predict, "code"].to_numpy(), "score": preds})
    return {"selected_trees": selected, "best_checkpoint": int(checkpoints[best]),
            "best_inner_ic": float(curve[best]), "seconds": round(time.time() - started, 1),
            "predictions": frame}


def daily_ic(frame: pd.DataFrame) -> float:
    values = []
    for _, block in frame.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values.append(float(np.corrcoef(y, p)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="BASE,F1,F2,F3,F123")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--out", default=str(V2 / "v2_feature_ablation.json"))
    args = parser.parse_args()
    panel = pd.read_parquet(V2 / "panel_v2.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    labels = panel[["date", "code", "future_return"]]
    report = {"seed": SEED, "records": []}
    for window in args.windows.split(","):
        anchor = pd.read_parquet(
            Path("artifacts/stage4_verified_ic") / f"v1_anchor__{window}.parquet")
        control = pd.read_parquet(FUS / "predictions_cat" / f"C1_{SEED}__{window}.parquet")
        control_frame = control.merge(labels, on=["date", "code"], how="left",
                                      validate="one_to_one")
        control_ic = daily_ic(control_frame)
        for name in args.configs.split(","):
            features = BASE_FEATURES + list(CONFIGS[name])
            result = train_c1(panel, window, features)
            frame = result["predictions"].merge(labels, on=["date", "code"], how="left",
                                                validate="one_to_one")
            standalone = daily_ic(frame)
            fused = anchor.rename(columns={"score": "anchor"}).merge(
                result["predictions"].rename(columns={"score": "new"}),
                on=["date", "code"], validate="one_to_one").merge(
                labels, on=["date", "code"], how="left", validate="one_to_one")
            pieces = []
            for _, block in fused.groupby("date", sort=True):
                def centred(values):
                    ranks = pd.Series(values).rank(method="average")
                    return ((ranks - (len(ranks) + 1) / 2.0) / len(ranks)).to_numpy()
                value = 0.9 * block["anchor"].to_numpy() + 0.1 * centred(block["new"].to_numpy())
                pieces.append(pd.DataFrame({"date": block["date"].to_numpy(),
                                            "code": block["code"].to_numpy(),
                                            "score": centred(value),
                                            "future_return": block["future_return"].to_numpy()}))
            fused_ic = daily_ic(pd.concat(pieces, ignore_index=True))
            record = {"window": window, "config": name, "n_features": len(features),
                      "selected_trees": result["selected_trees"],
                      "best_checkpoint": result["best_checkpoint"],
                      "seconds": result["seconds"], "standalone_rankic": standalone,
                      "control_rankic": control_ic, "delta_vs_control": standalone - control_ic,
                      "anchor_plus_alpha0.10_rankic": fused_ic}
            report["records"].append(record)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in record.items()}), flush=True)
    V2.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
