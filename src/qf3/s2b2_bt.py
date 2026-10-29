"""B2 portfolio backtests for the seed average and slow-expert blends."""

from __future__ import annotations

import argparse

import pandas as pd

from . import s2b_common

RUNS = {
    "b2_seedavg": {"prefix": "b2_seedavg", "windows": ["DEV19", "DEV20"],
                   "start": "2019-01-02", "end": "2020-12-31"},
    "b2_seedavg_conf21": {"prefix": "b2_seedavg", "windows": ["CONF21"],
                          "start": "2021-01-04", "end": "2021-12-31"},
    "b2_E1_a0.1": {"prefix": "b2_E1_p5params_a0.1", "windows": ["DEV19", "DEV20"],
                   "start": "2019-01-02", "end": "2020-12-31"},
    "b2_E1_a0.1_conf21": {"prefix": "b2_E1_p5params_a0.1", "windows": ["CONF21"],
                          "start": "2021-01-04", "end": "2021-12-31"},
    "b2_E2_a0.1": {"prefix": "b2_E2_conservative_a0.1", "windows": ["DEV19", "DEV20"],
                   "start": "2019-01-02", "end": "2020-12-31"},
    "b2_E1_a0.2": {"prefix": "b2_E1_p5params_a0.2", "windows": ["DEV19", "DEV20"],
                   "start": "2019-01-02", "end": "2020-12-31"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    rows = []
    for name in names:
        spec = RUNS[name]
        frames = [pd.read_parquet(s2b_common.PRED / f"{spec['prefix']}__{w}.parquet")
                  for w in spec["windows"]]
        frame = pd.concat(frames, ignore_index=True)
        metrics_out = s2b_common.run_backtest(name, frame, spec["start"], spec["end"])
        row = {"run": name, "mean_rankic": metrics_out["mean_rankic"],
               "annualized_excess_return": metrics_out["annualized_excess_return"],
               "tracking_error": metrics_out["tracking_error"],
               "information_ratio": metrics_out["information_ratio"],
               "turnover": metrics_out["turnover"],
               "transaction_cost": metrics_out["transaction_cost"]}
        rows.append(row)
        s2b_common.log({"stage": "B2-backtest", **row})
        print(row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(s2b_common.B / "b2_backtests.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
