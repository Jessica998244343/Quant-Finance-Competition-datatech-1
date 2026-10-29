"""A-stage: portfolio backtests for the new input-semantics score sets."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_runner

OUT = Path("artifacts/stage2_ic_ir")
PRED = OUT / "predictions"
BT = OUT / "backtests"
PRED_V1 = Path("artifacts/beginning/predictions")

RUNS = {
    "bt_realinput_dev19_20": {
        "files": ["origmodel_realinput__DEV19.parquet", "origmodel_realinput__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "bt_realinput_conf21": {
        "files": ["origmodel_realinput__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "bt_aligned_dev19_20": {
        "files": ["aligned_retrain__DEV19.parquet", "aligned_retrain__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "bt_aligned_conf21": {
        "files": ["aligned_retrain__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "bt_catboost_dev19_20": {
        "files": ["../beginning/predictions/SC_d8_l2_15__DEV19.parquet",
                  "../beginning/predictions/SC_d8_l2_15__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "bt_catboost_conf21": {
        "files": ["../beginning/predictions/SC_d8_l2_15__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "bt_p5_realinput_dev19_20": {
        "files": ["origmodel_realinput_members__DEV19.parquet",
                  "origmodel_realinput_members__DEV20.parquet"],
        "score_column": "score_lgbm_p5",
        "start": "2019-01-02", "end": "2020-12-31"},
    "bt_catboost_realinput_dev19_20": {
        "files": ["origmodel_realinput_members__DEV19.parquet",
                  "origmodel_realinput_members__DEV20.parquet"],
        "score_column": "score_catboost",
        "start": "2019-01-02", "end": "2020-12-31"},
}

REFERENCE = {
    "bt_SSTAR_dev19_20": Path("artifacts/beginning/backtests/bt_SSTAR_dev19_20"),
    "bt_SSTAR_conf21": Path("artifacts/beginning/backtests/bt_SSTAR_conf21"),
}


def load_scores(files: list[str], score_column: str = "score") -> pd.DataFrame:
    frames = []
    for name in files:
        path = PRED / name
        if not path.exists():
            path = OUT / name
        frame = pd.read_parquet(path)
        frames.append(pd.DataFrame({"date": frame["date"], "code": frame["code"],
                                    "score": frame[score_column]}))
    scores = pd.concat(frames, ignore_index=True)
    return scores.drop_duplicates(["date", "code"], keep="last")


def run(name: str, spec: dict) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    scores = load_scores(spec["files"], spec.get("score_column", "score"))
    inference = backtest_runner.score_lookup_inference(scores)
    (out_dir / "config.json").write_text(json.dumps(
        {**spec, "data_view": "Source/Data", "contract":
         "configs/beginning/evaluation_contract.json"}, indent=2))
    started = time.time()
    metrics_out, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=spec["start"],
        end_date=spec["end"], inference=inference, output_dir=None,
        async_inference=False)
    elapsed = time.time() - started
    backtest_runner.save_run(out_dir, metrics_out, tables, account)
    record = {"run": name, "seconds": round(elapsed, 1), "metrics": metrics_out}
    (out_dir / "run_summary.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"run": name, "seconds": round(elapsed, 1),
                      "IC": metrics_out["mean_rankic"],
                      "IR": metrics_out["information_ratio"],
                      "ann_excess": metrics_out["annualized_excess_return"],
                      "TE": metrics_out["tracking_error"],
                      "turnover": metrics_out["turnover"],
                      "cost": metrics_out["transaction_cost"],
                      "failed": metrics_out["failed_orders"]}, default=str))
    return record


def constraint_delta(new_dir: Path, reference_dir: Path) -> dict:
    if not (reference_dir / "target_weights.csv").exists():
        return {}
    left = pd.read_csv(new_dir / "target_weights.csv").sort_values(["signal_date", "code"])
    right = pd.read_csv(reference_dir / "target_weights.csv").sort_values(["signal_date", "code"])
    if len(left) != len(right) or not (left["signal_date"].to_numpy()
                                       == right["signal_date"].to_numpy()).all():
        return {"comparable": False}
    dw = (left["target_weight"].to_numpy() - right["target_weight"].to_numpy())
    lo = pd.read_csv(new_dir / "orders.csv")
    ro = pd.read_csv(reference_dir / "orders.csv")
    rejected_new = int((lo["status"] == "REJECTED").sum())
    rejected_ref = int((ro["status"] == "REJECTED").sum())
    return {
        "comparable": True,
        "max_abs_target_weight_delta": float(np.max(np.abs(dw))),
        "n_target_rows": int(len(left)),
        "n_relweight_changed_gt_1e-6": int((np.abs(dw) > 1e-6).sum()),
        "rejected_new": rejected_new, "rejected_reference": rejected_ref,
        "rejected_relative_change": (abs(rejected_new - rejected_ref)
                                     / max(1, rejected_ref)),
    }


def main() -> None:
    results = {}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    for name in names:
        spec = RUNS[name]
        results[name] = run(name, spec)
    pairs = [("bt_realinput_dev19_20", "bt_SSTAR_dev19_20"),
             ("bt_realinput_conf21", "bt_SSTAR_conf21"),
             ("bt_aligned_dev19_20", "bt_SSTAR_dev19_20")]
    checks = {}
    for new_name, ref_name in pairs:
        checks[f"{new_name}_vs_{ref_name}"] = constraint_delta(
            BT / new_name, REFERENCE[ref_name])
    s2 = json.dumps({"runs": {k: v["metrics"] for k, v in results.items()},
                     "constraint_checks": checks}, indent=2, default=str)
    (OUT / "backtests_new_input.json").write_text(s2)
    print("== constraint deltas ==")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
