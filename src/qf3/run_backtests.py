"""Run the locked Barra-contract backtests for B0 and the candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from . import backtest_runner, metrics

PRED = Path("artifacts/beginning/predictions")
OUT = Path("artifacts/beginning/backtests")
ZERO_ST = "artifacts/beginning/data_zero_st"
SOURCE = "Source/Data"


def load_scores(experiment_ids, windows) -> pd.DataFrame:
    frames = []
    for experiment_id in experiment_ids:
        for window in windows:
            path = PRED / f"{experiment_id}__{window}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no predictions for {experiment_ids} {windows}")
    scores = pd.concat(frames, ignore_index=True)
    return scores.drop_duplicates(["date", "code"], keep="last")


RUNS = {
    "bt_B0_dev19_20": dict(ids=["B0"], windows=["DEV19", "DEV20"],
                           start="2019-01-02", end="2020-12-31", view=ZERO_ST),
    "bt_SSTAR_dev19_20": dict(ids=["SSTAR"], windows=["DEV19", "DEV20"],
                              start="2019-01-02", end="2020-12-31", view=ZERO_ST),
    "bt_P5_dev19_20": dict(ids=["P5_s20261005"], windows=["DEV19", "DEV20"],
                           start="2019-01-02", end="2020-12-31", view=ZERO_ST),
    "bt_B0_conf21": dict(ids=["B0"], windows=["CONF21"],
                         start="2021-01-04", end="2021-12-31", view=ZERO_ST),
    "bt_SSTAR_conf21": dict(ids=["SSTAR"], windows=["CONF21"],
                            start="2021-01-04", end="2021-12-31", view=ZERO_ST),
    "bt_P5_conf21": dict(ids=["P5_s20261005"], windows=["CONF21"],
                         start="2021-01-04", end="2021-12-31", view=ZERO_ST),
    "bt_B0_dev19_20_zerocost": dict(ids=["B0"], windows=["DEV19", "DEV20"],
                                    start="2019-01-02", end="2020-12-31",
                                    view=ZERO_ST, zero_cost=True),
    "bt_SSTAR_dev19_20_zerocost": dict(ids=["SSTAR"], windows=["DEV19", "DEV20"],
                                       start="2019-01-02", end="2020-12-31",
                                       view=ZERO_ST, zero_cost=True),
    "bt_B0_dev19_20_storig": dict(ids=["B0"], windows=["DEV19", "DEV20"],
                                  start="2019-01-02", end="2020-12-31", view=SOURCE),
    "bt_SSTAR_dev19_20_storig": dict(ids=["SSTAR"], windows=["DEV19", "DEV20"],
                                     start="2019-01-02", end="2020-12-31", view=SOURCE),
}


def run(name: str, spec: dict) -> dict:
    out_dir = OUT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    scores = load_scores(spec["ids"], spec["windows"])
    inference = backtest_runner.score_lookup_inference(scores)
    config = {k: v for k, v in spec.items() if k not in ("ids",)}
    (out_dir / "config.json").write_text(json.dumps(
        {**config, "ids": spec["ids"], "data_view": spec["view"],
         "contract": "configs/beginning/evaluation_contract.json"}, indent=2))
    started = time.time()
    result = backtest_runner.run_backtest(
        data_dir=spec["view"], start_date=spec["start"], end_date=spec["end"],
        inference=inference, output_dir=None,
        zero_cost=bool(spec.get("zero_cost")), async_inference=False)
    metrics_out, tables, account = result
    elapsed = time.time() - started
    backtest_runner.save_run(out_dir, metrics_out, tables, account)
    rankic = metrics.summarize_rankic(metrics.daily_rankic(tables["predictions"]))
    record = {"run": name, "seconds": round(elapsed, 1), "metrics": metrics_out,
              "rankic": rankic, "config": config}
    (out_dir / "run_summary.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"run": name, "seconds": round(elapsed, 1),
                      "IC": metrics_out["mean_rankic"],
                      "IR": metrics_out["information_ratio"],
                      "ann_excess": metrics_out["annualized_excess_return"],
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
