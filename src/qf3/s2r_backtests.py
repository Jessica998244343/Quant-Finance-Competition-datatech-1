"""REV1 backtests with new run IDs; old polluted products are never overwritten."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from . import backtest_runner

REV = Path("artifacts/stage2_ic_ir/a_revision1")
PRED = REV / "predictions"
BT = REV / "backtests"
A_PRED = Path("artifacts/stage2_ic_ir/predictions")
V1_PRED = Path("artifacts/beginning/predictions")

RUNS = {
    "rev1_bt_realinput_dev19_20": {
        "files": [PRED / "origmodel_realinput__DEV19.parquet",
                  PRED / "origmodel_realinput__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "rev1_bt_realinput_conf21": {
        "files": [PRED / "origmodel_realinput__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "rev1_bt_aligned_dev19_20": {
        "files": [PRED / "aligned_retrain__DEV19.parquet",
                  PRED / "aligned_retrain__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "rev1_bt_aligned_conf21": {
        "files": [PRED / "aligned_retrain__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "rev1_bt_p5_realinput_conf21": {
        "files": [A_PRED / "origmodel_realinput_members__CONF21.parquet"],
        "score_column": "score_lgbm_p5",
        "start": "2021-01-04", "end": "2021-12-31"},
    "rev1_bt_catboost_realinput_conf21": {
        "files": [A_PRED / "origmodel_realinput_members__CONF21.parquet"],
        "score_column": "score_catboost",
        "start": "2021-01-04", "end": "2021-12-31"},
    "rev1_bt_refcorrected_dev19_20": {
        "files": [PRED / "ref_corrected__DEV19.parquet",
                  PRED / "ref_corrected__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "rev1_bt_refcorrected_conf21": {
        "files": [PRED / "ref_corrected__CONF21.parquet"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "rev1_bt_refcorrected_p5_dev19_20": {
        "files": [PRED / "ref_corrected_lgbm__DEV19.parquet",
                  PRED / "ref_corrected_lgbm__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "rev1_bt_refcorrected_catboost_dev19_20": {
        "files": [PRED / "ref_corrected_catboost__DEV19.parquet",
                  PRED / "ref_corrected_catboost__DEV20.parquet"],
        "start": "2019-01-02", "end": "2020-12-31"},
}

# carried over unchanged: single-model runs are not affected by the blend bug
CARRIED = {
    "a_bt_p5_realinput_dev19_20": Path(
        "artifacts/stage2_ic_ir/backtests/bt_p5_realinput_dev19_20"),
    "a_bt_catboost_realinput_dev19_20": Path(
        "artifacts/stage2_ic_ir/backtests/bt_catboost_realinput_dev19_20"),
    "a_bt_p5_legacy_dev19_20": Path("artifacts/beginning/backtests/bt_P5_dev19_20"),
    "a_bt_catboost_legacy_dev19_20": Path(
        "artifacts/stage2_ic_ir/backtests/bt_catboost_dev19_20"),
    "a_bt_p5_legacy_conf21": Path("artifacts/beginning/backtests/bt_P5_conf21"),
    "a_bt_catboost_legacy_conf21": Path(
        "artifacts/stage2_ic_ir/backtests/bt_catboost_conf21"),
    "v1_bt_ref_dev19_20": Path("artifacts/beginning/backtests/bt_SSTAR_dev19_20"),
    "v1_bt_ref_conf21": Path("artifacts/beginning/backtests/bt_SSTAR_conf21"),
    "v1_bt_b0_dev19_20": Path("artifacts/beginning/backtests/bt_B0_dev19_20"),
    "v1_bt_ref_zerocost_dev19_20": Path(
        "artifacts/beginning/backtests/bt_SSTAR_dev19_20_zerocost"),
}


def load_scores(files: list[Path], score_column: str = "score") -> pd.DataFrame:
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(pd.DataFrame({"date": frame["date"], "code": frame["code"],
                                    "score": frame[score_column]}))
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["date", "code"]).any():
        raise ValueError("duplicate scores across windows")
    return scores


def run(name: str, spec: dict) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run {out_dir}")
    out_dir.mkdir(parents=True)
    scores = load_scores(spec["files"], spec.get("score_column", "score"))
    inference = backtest_runner.score_lookup_inference(scores)
    (out_dir / "config.json").write_text(json.dumps(
        {"files": [str(p) for p in spec["files"]],
         "score_column": spec.get("score_column", "score"),
         "start": spec["start"], "end": spec["end"],
         "data_view": "artifacts/beginning/data_zero_st",
         "contract": "configs/beginning/evaluation_contract.json"}, indent=2))
    started = time.time()
    metrics_out, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=spec["start"],
        end_date=spec["end"], inference=inference, output_dir=None,
        async_inference=False)
    elapsed = time.time() - started
    backtest_runner.save_run(out_dir, metrics_out, tables, account)
    record = {"run": name, "seconds": round(elapsed, 1), "metrics": metrics_out,
              "scores": str(spec["files"])}
    (out_dir / "run_summary.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"run": name, "IC": metrics_out["mean_rankic"],
                      "IR": metrics_out["information_ratio"],
                      "excess": metrics_out["annualized_excess_return"],
                      "TE": metrics_out["tracking_error"],
                      "turnover": metrics_out["turnover"],
                      "cost": metrics_out["transaction_cost"]}, default=str))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    for name in names:
        run(name, RUNS[name])


if __name__ == "__main__":
    main()
