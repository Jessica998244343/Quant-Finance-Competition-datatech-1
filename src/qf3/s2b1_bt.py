"""B1 portfolio backtests for the CatBoost weight grid."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import s2b_common

WEIGHTS = [0.0, 0.125, 0.25, 0.5, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="")
    parser.add_argument("--windows", default="dev19_20")
    args = parser.parse_args()
    weights = [float(x) for x in args.weights.split(",")] if args.weights else WEIGHTS
    rows = []
    for cat_weight in weights:
        tag = f"ref_corrected_cat{cat_weight:g}"
        if args.windows == "conf21":
            path = s2b_common.PRED / f"{tag}__CONF21.parquet"
            name = f"b1_bt_cat{cat_weight:g}_conf21"
            start, end = "2021-01-04", "2021-12-31"
        else:
            dev19 = pd.read_parquet(s2b_common.PRED / f"{tag}__DEV19.parquet")
            dev20 = pd.read_parquet(s2b_common.PRED / f"{tag}__DEV20.parquet")
            path = pd.concat([dev19, dev20], ignore_index=True)
            name = f"b1_bt_cat{cat_weight:g}_dev19_20"
            start, end = "2019-01-02", "2020-12-31"
        frame = pd.read_parquet(path) if not isinstance(path, pd.DataFrame) else path
        metrics_out = s2b_common.run_backtest(name, frame, start, end)
        row = {"run": name, "cat_weight": cat_weight,
               "mean_rankic": metrics_out["mean_rankic"],
               "annualized_excess_return": metrics_out["annualized_excess_return"],
               "tracking_error": metrics_out["tracking_error"],
               "information_ratio": metrics_out["information_ratio"],
               "turnover": metrics_out["turnover"],
               "transaction_cost": metrics_out["transaction_cost"],
               "failed_orders": metrics_out["failed_orders"]}
        rows.append(row)
        s2b_common.log({"stage": "B1-backtest", **row})
        print(json.dumps(row, default=str))
    out = pd.DataFrame(rows)
    out.to_csv(s2b_common.B / f"b1_backtests_{args.windows}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
