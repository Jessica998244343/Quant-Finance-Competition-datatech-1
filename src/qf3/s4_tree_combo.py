"""D1 §7.1: same-architecture capacity experts fused with the frozen L3 trunk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import metrics, s3_ranks

FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first/predictions")
ALPHAS = [0.0, 0.05, 0.10, 0.25, 0.50]
WINDOWS = ("DEV19", "DEV20")


def load(name: str, window: str) -> pd.DataFrame:
    return pd.read_parquet(S3 / f"{name}__{window}.parquet")


def fuse_alpha(trunk: pd.DataFrame, expert: pd.DataFrame, alpha: float) -> pd.DataFrame:
    both = trunk.rename(columns={"score": "trunk"}).merge(
        expert.rename(columns={"score": "expert"}), on=["date", "code"],
        validate="one_to_one")
    return s3_ranks.fuse(both, ["trunk", "expert"], [1 - alpha, alpha],
                         label=f"alpha{alpha}")


def evaluate(frame: pd.DataFrame, panel: pd.DataFrame) -> dict:
    merged = frame.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def main() -> None:
    FUS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    report = {}
    for expert_name in ("L4_L0_20261005", "L5_L0_20261005"):
        rows = []
        for alpha in ALPHAS:
            per_year = {}
            frames = []
            for window in WINDOWS:
                fused = fuse_alpha(load("L3_L0_20261005", window),
                                   load(expert_name, window), alpha)
                per_year[window] = evaluate(fused, panel)
                frames.append(fused)
            pooled = evaluate(pd.concat(frames, ignore_index=True), panel)
            rows.append({"alpha": alpha, "DEV19": per_year["DEV19"],
                         "DEV20": per_year["DEV20"], "dev_pooled": pooled})
            fused.to_parquet(FUS / f"combo_{expert_name}_a{alpha:g}.parquet",
                             index=False)
        trunk = evaluate(pd.concat([load("L3_L0_20261005", w) for w in WINDOWS],
                                   ignore_index=True), panel)
        for row in rows:
            row["delta_vs_alpha0"] = row["dev_pooled"] - trunk
        report[expert_name] = {"trunk_pooled": trunk, "grid": rows}
        print(expert_name, json.dumps(rows, default=str), flush=True)
    (FUS / "d1_tree_combo.json").write_text(json.dumps(report, indent=2, default=str))
    print(pd.DataFrame(report["L4_L0_20261005"]["grid"]).to_string(index=False))
    print(pd.DataFrame(report["L5_L0_20261005"]["grid"]).to_string(index=False))


if __name__ == "__main__":
    main()
