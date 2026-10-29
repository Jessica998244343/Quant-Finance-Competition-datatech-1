"""V3 §C: full-data (2016-2022 mature labels) three-member diagnosis models.

Only the legal training cutoff is extended; features, architecture, objective,
preprocessing, tail-window selection and outer refit are unchanged.  No 2022
outer evaluation is produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cv
from .s6_v2_deploy_models import custom_split, train_catboost, train_tabm
from .s6_v2_tcn import build_sequences, train_fold
V2 = Path("artifacts/stage4_verified_ic/v2")
OUT = V2 / "full_2022"
TRAIN_END = 20221230
SEEDS = (20261005, 20261017, 20261029)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v3_full_fit.json"))
    parser.add_argument("--tcn-only", action="store_true",
                        help="skip C1/TabM (already saved) and train only the TCN members")
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    split = custom_split(data, TRAIN_END, 20230101, 20231231)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"train_label_cutoff": TRAIN_END, "outer_metrics_computed": False,
              "seeds": [int(s) for s in SEEDS], "models": {}}
    started = time.time()
    if args.tcn_only:
        previous = json.loads((V2 / "v3_full_fit.json").read_text()) \
            if (V2 / "v3_full_fit.json").exists() else {"models": {}}
        report["models"].update(previous.get("models", {}))
    else:
        for seed in SEEDS:
            record = train_catboost(data, split, seed, OUT, evaluate=False)
            tabm = train_tabm(data, split, seed, OUT, evaluate=False)
            report["models"][f"c1_{seed}"] = record
            report["models"][f"tabm_{seed}"] = tabm
            print("full fit", seed, "catboost trees", record.get("actual_tree_count"),
                  "tabm epoch", tabm.get("best_epoch"), flush=True)
    sequences = build_sequences()
    for seed in SEEDS:
        manifest = train_fold(data, sequences["sequences"], sequences["keys"], "FULL22",
                              seed, True, False, evaluate=False, split=split)
        report["models"][f"tcn_{seed}"] = {k: v for k, v in manifest.items()
                                           if k != "predictions"}
        print("full fit", seed, "tcn epoch", manifest["best_epoch"], flush=True)
    report["seconds"] = round(time.time() - started, 1)
    report["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)}
                       for p in sorted(OUT.iterdir()) if p.is_file()}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"models": list(report["models"]), "seconds": report["seconds"]},
                     indent=2))


if __name__ == "__main__":
    main()
