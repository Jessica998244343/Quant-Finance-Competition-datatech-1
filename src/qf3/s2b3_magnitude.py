"""B3: rank target mixed with a robust return-magnitude term (lambda in {0.1, 0.25})."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import cv, s2b_common, s2r_train, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
LAMBDAS = [0.1, 0.25]


def mixed_target(frame: pd.DataFrame, lam: float, label_col: str = "future_return"):
    label = pd.to_numeric(frame[label_col], errors="coerce")
    finite = label.notna().to_numpy()
    grouped = label.groupby(frame["date"])
    ranks = grouped.rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    z_rank = (ranks - (counts + 1.0) / 2.0) / counts

    median = grouped.transform("median")
    deviation = (label - median).abs().groupby(frame["date"]).transform("median") * 1.4826
    scale_ok = deviation.notna() & (deviation > 1e-8)
    z_mag = pd.Series(0.0, index=frame.index)
    scaled = ((label - median) / deviation).clip(-3.0, 3.0) / 6.0
    z_mag.loc[scale_ok] = scaled.loc[scale_ok]
    target = (1.0 - lam) * z_rank + lam * z_mag
    weights = 1.0 / counts
    return (target.to_numpy(dtype=float), weights.to_numpy(dtype=float), finite,
            float(scale_ok.mean()))


def train(panel: pd.DataFrame, window: str, lam: float, threads: int = 8) -> dict:
    import lightgbm as lgb

    split = cv.fold_split(panel, window)
    dates = cv.to_int_date(panel["date"])
    y, raw_w, finite, scale_ok_share = mixed_target(panel, lam)
    raw_returns = pd.to_numeric(panel["future_return"], errors="coerce").to_numpy(dtype=float)
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)

    params = trainers.lgb_params(trainers.CAPACITY_GRID["P5"], 20261005, threads)
    rounds = int(params.pop("_num_rounds"))
    valid_idx = np.nonzero(early)[0]

    def feval(preds, dataset):
        return ("daily_spearman",
                cv.daily_spearman(dates[valid_idx], raw_returns[valid_idx], preds), True)

    train_set = lgb.Dataset(x[internal], label=y[internal], weight=w_internal[internal],
                            feature_name=FEATURES, free_raw_data=False)
    valid_set = lgb.Dataset(x[early], label=y[early], weight=w_internal[early],
                            reference=train_set, feature_name=FEATURES,
                            free_raw_data=False)
    started = time.time()
    booster = lgb.train(params, train_set, num_boost_round=rounds,
                        valid_sets=[valid_set], feval=feval,
                        callbacks=[lgb.early_stopping(200, first_metric_only=True,
                                                      verbose=False),
                                   lgb.log_evaluation(0)])
    best_iter = int(booster.best_iteration or rounds)
    es_seconds = time.time() - started
    final_set = lgb.Dataset(x[outer], label=y[outer], weight=w_outer[outer],
                            feature_name=FEATURES, free_raw_data=False)
    final = lgb.train(params, final_set, num_boost_round=best_iter)
    preds = final.predict(x[predict], num_iteration=best_iter)
    predictions = pd.DataFrame({"date": dates[predict].astype("int64"),
                                "code": panel.loc[predict, "code"].to_numpy(),
                                "score": preds})
    return {"predictions": predictions, "rounds": best_iter,
            "es_seconds": es_seconds, "scale_ok_share": scale_ok_share}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--alpha", type=float, default=None,
                        help="mix this trained expert with the B1 reference")
    args = parser.parse_args()
    s2b_common.PRED.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    report = {}
    for lam in LAMBDAS:
        for window in ("DEV19", "DEV20", "CONF21"):
            tag = f"b3_lam{lam:g}"
            path = s2b_common.PRED / f"{tag}__{window}.parquet"
            if args.train or not path.exists():
                result = train(panel, window, lam)
                result["predictions"].to_parquet(path, index=False)
                s2b_common.log({"stage": "B3", "lambda": lam, "window": window,
                                "rounds": result["rounds"],
                                "scale_ok_share": result["scale_ok_share"]})
                print(f"  lam={lam} {window}: rounds={result['rounds']} "
                      f"scale_ok={result['scale_ok_share']:.3f}", flush=True)
            frame = pd.read_parquet(path)
            summary, _ = s2b_common.ic_summary(frame, panel)
            report.setdefault(f"lam{lam:g}", {})[window] = summary["mean_rankic"]
    s2b_common.write_json(s2b_common.B / "b3_ic.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
