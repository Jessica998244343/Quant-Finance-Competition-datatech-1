"""D3 §13.2: main-cost backtests for the frozen candidates (IR reported only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import backtest_runner

FUS = Path("artifacts/stage3_fusion")
BT = FUS / "backtests"

RUNS = {
    "d3_bt_final_v1_dev19_20": {
        "prefix": "FINAL_v1", "windows": ["DEV19", "DEV20"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "d3_bt_final_v1_conf21": {
        "prefix": "FINAL_v1", "windows": ["CONF21"],
        "start": "2021-01-04", "end": "2021-12-31"},
    "d3_bt_alt_v1_dev19_20": {
        "prefix": "CATFUSION_a0.5", "windows": ["DEV19", "DEV20"],
        "start": "2019-01-02", "end": "2020-12-31"},
    "d3_bt_alt_v1_conf21": {
        "prefix": "CATFUSION_a0.5", "windows": ["CONF21"],
        "start": "2021-01-04", "end": "2021-12-31"},
}


def run(name: str, spec: dict) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    frames = [pd.read_parquet(FUS / f"{spec['prefix']}__{w}.parquet")
              for w in spec["windows"]]
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["date", "code"]).any():
        raise ValueError("duplicate scores")
    (out_dir / "config.json").write_text(json.dumps(
        {"candidate": spec["prefix"], "windows": spec["windows"],
         "start": spec["start"], "end": spec["end"],
         "data_view": "artifacts/beginning/data_zero_st",
         "contract": "configs/beginning/evaluation_contract.json",
         "ir_policy": "reported only; not used for any decision"}, indent=2))
    inference = backtest_runner.score_lookup_inference(scores)
    m, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=spec["start"],
        end_date=spec["end"], inference=inference, output_dir=None,
        async_inference=False)
    backtest_runner.save_run(out_dir, m, tables, account)
    (out_dir / "run_summary.json").write_text(json.dumps(
        {"run": name, "metrics": m}, indent=2, default=str))
    print(json.dumps({"run": name, "IC": m["mean_rankic"], "IR": m["information_ratio"],
                      "excess": m["annualized_excess_return"], "TE": m["tracking_error"],
                      "turnover": m["turnover"], "cost": m["transaction_cost"],
                      "failed": m["failed_orders"]}, default=str), flush=True)
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    out = {}
    for name in names:
        out[name] = run(name, RUNS[name])
    path = FUS / "d3_backtests.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(out)
    path.write_text(json.dumps(existing, indent=2, default=str))


if __name__ == "__main__":
    main()
