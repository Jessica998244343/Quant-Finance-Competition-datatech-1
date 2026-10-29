"""D3: member ablation and a bounded coarse coordinate scan over fusion weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, s3_ranks

FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first/predictions")
NN = FUS / "predictions_nn"
CAT = FUS / "predictions_cat"
WINDOWS = ("DEV19", "DEV20")


def expert_columns() -> dict[str, dict[str, Path]]:
    return {
        "l3": {w: S3 / f"MAIN_SEED_ENSEMBLE__{w}.parquet" for w in WINDOWS},
        "c1": {w: FUS / f"EXPERT_C1_seedmean__{w}.parquet" for w in WINDOWS},
        "l5": {w: FUS / f"EXPERT_L5_seedmean__{w}.parquet" for w in WINDOWS},
        "n0": {w: FUS / f"EXPERT_N0_seedmean__{w}.parquet" for w in WINDOWS},
        "tabm": {w: NN / f"TABM_20261005__{w}.parquet" for w in WINDOWS},
    }


def load_matrix(experts: dict, w: str) -> pd.DataFrame:
    frame = None
    for name, mapping in experts.items():
        column = pd.read_parquet(mapping[w]).rename(columns={"score": name})
        frame = column if frame is None else frame.merge(
            column, on=["date", "code"], validate="one_to_one")
    return frame


def evaluate(frame: pd.DataFrame, weights: dict, panel: pd.DataFrame) -> float:
    names = [n for n in weights if weights[n] > 0]
    fused = s3_ranks.fuse(frame, names, [weights[n] for n in names], label="scan")
    merged = fused.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def main() -> None:
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    experts = expert_columns()
    frames = {w: load_matrix(experts, w) for w in WINDOWS}

    def dev_ic(weights: dict) -> float:
        parts = []
        for w in WINDOWS:
            parts.append(frames[w].assign(date=frames[w]["date"]))
        pooled = pd.concat(parts, ignore_index=True)
        return evaluate(pooled, weights, panel)

    report = {"single": {}, "ablation": {}, "scan": []}
    for name in experts:
        report["single"][name] = dev_ic({name: 1.0})

    # current best candidate from D1+D2
    best = {"l3": 0.375, "c1": 0.375, "tabm": 0.25}
    report["scan"].append({"pass": 0, "weights": dict(best), "dev_ic": dev_ic(best)})

    # coordinate scan, two passes, each member tested at a few weights
    grid = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    for pass_index in (1, 2):
        improved = False
        for name in ("l3", "c1", "l5", "n0", "tabm"):
            for target in grid:
                trial = dict(best)
                others = {k: v for k, v in trial.items() if k != name and v > 0}
                rest = 1.0 - target
                if not others:
                    continue
                total = sum(others.values())
                trial = {k: rest * v / total for k, v in others.items()}
                trial[name] = target
                score = dev_ic(trial)
                report["scan"].append({"pass": pass_index, "member": name,
                                       "target": target, "weights": trial,
                                       "dev_ic": score})
                if score > report["scan"][0]["dev_ic"] and score > best_ic(best, report):
                    pass
        # keep the best found in this pass
        pass_rows = [r for r in report["scan"] if r["pass"] == pass_index]
        if pass_rows:
            top = max(pass_rows, key=lambda r: r["dev_ic"])
            if top["dev_ic"] > best_ic(best, report):
                best = dict(top["weights"])
                improved = True
        if not improved:
            break
    report["frozen"] = {"weights": best, "dev_ic": dev_ic(best)}

    # leave-one-out ablation on the frozen combination
    for name in list(best):
        trial = {k: v for k, v in best.items() if k != name}
        total = sum(trial.values())
        trial = {k: v / total for k, v in trial.items()}
        report["ablation"][name] = {"without": dev_ic(trial),
                                    "delta": dev_ic(best) - dev_ic(trial)}
    (FUS / "d3_fusion_scan.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: round(v, 6) for k, v in report["single"].items()}, indent=2))
    print("frozen", json.dumps(report["frozen"], default=str))
    print(json.dumps(report["ablation"], indent=2, default=str))


def best_ic(weights: dict, report: dict) -> float:
    for row in report["scan"]:
        if row["weights"] == weights:
            return row["dev_ic"]
    return -np.inf


if __name__ == "__main__":
    main()
