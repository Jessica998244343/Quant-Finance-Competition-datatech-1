"""D1 §7.1: supplement the retained L5 capacity expert with two more seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import s3_s1

FUS = Path("artifacts/stage3_fusion")


def main() -> None:
    panel = pd.read_parquet("artifacts/stage3_ic_first/panel_ext.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    overrides = dict(s3_s1.trainers.CAPACITY_GRID["P5"])
    overrides.update(s3_s1.CAPACITY["L5"])
    out = {}
    for seed in (20261017, 20261029):
        for window in ("DEV19", "DEV20", "CONF21"):
            records = s3_s1.train_recipe(panel, "L5_L0", s3_s1.BASE, overrides,
                                         [window], seed=seed)
            out[f"L5_{seed}_{window}"] = records[window]["mean_rankic"]
            print(f"L5 seed {seed} {window}: {records[window]['mean_rankic']:.6f}",
                  flush=True)
    (FUS / "d1_l5_seeds.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
