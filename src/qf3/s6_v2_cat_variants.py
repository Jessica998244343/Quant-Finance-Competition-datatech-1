"""§10.1: evaluate the reused C2 (depth6) / C3 (depth8, l2=30) atoms.

Uses the existing legal DEV predictions (seed 20261005) and the frozen graph:
the new member is added to the frozen V1 anchor with alpha in {0, .05, .10, .25, .50}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
V2 = S4 / "v2"
WINDOWS = ("DEV19", "DEV20")
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)


def _rank(values) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return (ranks - (len(series) + 1) / 2.0) / len(series)


def _daily_ic(frame: pd.DataFrame) -> float:
    values = []
    for _, block in frame.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values.append(float(np.corrcoef(y, p)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_cat_variants.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    report = {"alphas": list(ALPHAS), "windows": {}}
    for window in WINDOWS:
        anchor = pd.read_parquet(S4 / f"v1_anchor__{window}.parquet")
        entry = {}
        for recipe in ("C2", "C3", "C1"):
            frame = pd.read_parquet(
                FUS / "predictions_cat" / f"{recipe}_20261005__{window}.parquet")
            merged = anchor.merge(frame.rename(columns={"score": "new"}),
                                  on=["date", "code"], validate="one_to_one").merge(
                labels, on=["date", "code"], how="left", validate="one_to_one")
            standalone = _daily_ic(
                merged[["date", "code", "future_return"]].assign(score=merged["new"]))
            curve = {}
            for alpha in ALPHAS:
                pieces = []
                for date, block in merged.groupby("date", sort=True):
                    value = ((1.0 - alpha) * block["score"].to_numpy()
                             + alpha * _rank(block["new"]).to_numpy())
                    pieces.append(pd.DataFrame({
                        "date": date, "code": block["code"].to_numpy(),
                        "score": _rank(value).to_numpy(),
                        "future_return": block["future_return"].to_numpy()}))
                curve[str(alpha)] = _daily_ic(pd.concat(pieces, ignore_index=True))
            entry[recipe] = {
                "standalone_rankic": standalone,
                "delta_vs_anchor": {key: curve[key] - curve["0.0"] for key in curve},
                "ic_by_alpha": curve}
        best = max(("C2", "C3"), key=lambda r: entry[r]["delta_vs_anchor"]["0.1"])
        entry["best_second_cat"] = best
        entry["best_delta_at_0.10"] = entry[best]["delta_vs_anchor"]["0.1"]
        report["windows"][window] = entry
    report["note"] = ("reused legal DEV predictions; DEV-only decision, "
                      "increment measured against the frozen V1 anchor")
    V2.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
