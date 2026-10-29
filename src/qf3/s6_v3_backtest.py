"""V3 §15: main-cost smoke backtest of the packaged full-data model.

The model is trained on ``label_end_date <= 2022-12-30``, so every window here
is **in-training**: the numbers prove the packaged platform entry runs and let us
report the requested IC/IR, but they are *not* out-of-sample performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import backtest_runner

BT = Path("artifacts/stage4_verified_ic/backtests")
RUNS = {
    "v3_full_pkg_conf21": ("2021-01-04", "2021-12-31"),
    "v3_full_pkg_dev19_20": ("2019-01-02", "2020-12-31"),
}


def run(name: str, start: str, end: str, submission: str, data_view: str) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"submission_dir": submission, "start": start, "end": end,
         "data_view": data_view, "contract": "configs/beginning/evaluation_contract.json",
         "label": "in-training smoke backtest (model trained through 2022)"}, indent=2))
    m, tables, account = backtest_runner.run_backtest(
        data_dir=data_view, start_date=start, end_date=end,
        submission_dir=submission, output_dir=None, async_inference=False)
    backtest_runner.save_run(out_dir, m, tables, account)
    print(json.dumps({"run": name, "IC": m["mean_rankic"], "IR": m["information_ratio"],
                      "rankic_ir": m["rankic_ir"], "excess": m["annualized_excess_return"],
                      "TE": m["tracking_error"], "turnover": m["turnover"],
                      "cost": m["transaction_cost"]}, default=str))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-view", default="artifacts/beginning/data_zero_st")
    parser.add_argument("--runs", default="")
    args = parser.parse_args()
    names = args.runs.split(",") if args.runs else list(RUNS)
    for name in names:
        start, end = RUNS[name]
        run(name, start, end, args.submission, args.data_view)


if __name__ == "__main__":
    main()
