"""V1 batch: corrected TabM folds (skips folds that already have an atom manifest)."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from . import s6_nn_v1 as S

JOBS = [("DEV19", 20261005), ("DEV20", 20261005), ("CONF21", 20261005),
        ("DEV19", 20261017), ("DEV20", 20261017), ("CONF21", 20261017),
        ("DEV19", 20261029), ("DEV20", 20261029), ("CONF21", 20261029)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="")
    parser.add_argument("--max-seconds", type=float, default=3000.0)
    args = parser.parse_args()
    wanted = ([(w, int(s)) for w, s in
               (item.split(":") for item in args.jobs.split(","))] if args.jobs else JOBS)
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    started = time.time()
    for window, seed in wanted:
        if time.time() - started > args.max_seconds:
            print("TIME_BUDGET_REACHED", flush=True)
            break
        out = S.OUT / f"tabm_pc_{seed}__{window}"
        if (out / "atom_manifest.json").exists():
            print(f"skip {window} {seed}", flush=True)
            continue
        meta = S.train_fold(data, window, seed)
        print(f"done {window} {seed} outer_IC {meta['outer_mean_rankic']:.6f} "
              f"epochs {meta['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
