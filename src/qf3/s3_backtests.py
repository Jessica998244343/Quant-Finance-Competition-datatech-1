"""S0: main-cost backtests for the corrected seed ensemble (new run IDs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import backtest_runner

S3 = Path("artifacts/stage3_ic_first")
BT = S3 / "backtests"

RUNS = {
    "s3_bt_seed_ensemble_dev19_20": (["DEV19", "DEV20"], "2019-01-02", "2020-12-31"),
    "s3_bt_seed_ensemble_conf21": (["CONF21"], "2021-01-04", "2021-12-31"),
}


def run(name: str, windows: list[str], start: str, end: str) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    frames = [pd.read_parquet(S3 / f"seed_ensemble_fixed__{w}.parquet") for w in windows]
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["date", "code"]).any():
        raise ValueError("duplicate scores")
    (out_dir / "config.json").write_text(json.dumps(
        {"windows": windows, "start": start, "end": end,
         "source": "artifacts/stage3_ic_first/seed_ensemble_fixed__*.parquet",
         "data_view": "artifacts/beginning/data_zero_st",
         "contract": "configs/beginning/evaluation_contract.json"}, indent=2))
    inference = backtest_runner.score_lookup_inference(scores)
    m, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=start, end_date=end,
        inference=inference, output_dir=None, async_inference=False)
    backtest_runner.save_run(out_dir, m, tables, account)
    (out_dir / "run_summary.json").write_text(json.dumps(
        {"run": name, "metrics": m}, indent=2, default=str))
    print(json.dumps({"run": name, "IC": m["mean_rankic"], "IR": m["information_ratio"],
                      "excess": m["annualized_excess_return"],
                      "TE": m["tracking_error"], "turnover": m["turnover"],
                      "cost": m["transaction_cost"]}, default=str))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    for name in names:
        windows, start, end = RUNS[name]
        run(name, windows, start, end)


if __name__ == "__main__":
    main()
