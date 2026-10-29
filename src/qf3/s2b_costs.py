"""B: cost scenarios (0x / 0.5x / 1x / 1.5x) for the retained candidates."""

from __future__ import annotations

import argparse

import pandas as pd

from . import s2b_common

CANDIDATES = {
    "reference_cat0.5": ["ref_corrected_cat0.5__DEV19", "ref_corrected_cat0.5__DEV20"],
    "seed_average": ["b2_seedavg__DEV19", "b2_seedavg__DEV20"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="reference_cat0.5,seed_average")
    parser.add_argument("--multipliers", default="0,0.5,1.5")
    args = parser.parse_args()
    rows = []
    for name in args.candidates.split(","):
        frames = [pd.read_parquet(s2b_common.PRED / f"{tag}.parquet")
                  for tag in CANDIDATES[name]]
        frame = pd.concat(frames, ignore_index=True)
        for multiplier in [float(x) for x in args.multipliers.split(",")]:
            run_name = f"b2cost_{name}_x{multiplier:g}"
            out_dir = s2b_common.BT / run_name
            if out_dir.exists():
                print(f"skip existing {run_name}")
                continue
            out_dir.mkdir(parents=True)
            inference = s2b_common.backtest_runner.score_lookup_inference(frame)
            metrics_out, tables, account = s2b_common.backtest_runner.run_backtest(
                data_dir="artifacts/beginning/data_zero_st", start_date="2019-01-02",
                end_date="2020-12-31", inference=inference, output_dir=None,
                async_inference=False,
                zero_cost=(multiplier == 0.0),
                cost_multiplier=multiplier if multiplier else 1.0)
            s2b_common.backtest_runner.save_run(out_dir, metrics_out, tables, account)
            row = {"run": run_name, "candidate": name, "multiplier": multiplier,
                   "mean_rankic": metrics_out["mean_rankic"],
                   "annualized_excess_return": metrics_out["annualized_excess_return"],
                   "information_ratio": metrics_out["information_ratio"],
                   "transaction_cost": metrics_out["transaction_cost"]}
            rows.append(row)
            s2b_common.log({"stage": "B-cost", **row})
            print(row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(s2b_common.B / "b_cost_scenarios.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
