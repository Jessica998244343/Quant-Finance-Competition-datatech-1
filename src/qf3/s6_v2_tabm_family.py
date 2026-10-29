"""§10.2 gate: three-seed dropout-0.2 TabM family vs the frozen V1 anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
V2 = S4 / "v2"
SEEDS = (20261005, 20261017, 20261029)
WINDOWS = ("DEV19", "DEV20")
WEIGHTS = {"c1": 0.5, "tabm": 0.5}


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def seed_mean(paths, column="score") -> pd.DataFrame:
    merged = None
    for index, path in enumerate(paths):
        frame = pd.read_parquet(path).rename(columns={column: f"s{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(paths)):
            value += _rank(block[f"s{index}"].reset_index(drop=True)) / len(paths)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "value": value}))
    return pd.concat(pieces, ignore_index=True)


def fence(c1: pd.DataFrame, tabm: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = c1.merge(tabm, on=["date", "code"], validate="one_to_one").merge(
        labels, on=["date", "code"], how="left", validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = (WEIGHTS["c1"] * block["value_x"].to_numpy()
                 + WEIGHTS["tabm"] * block["value_y"].to_numpy())
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": _rank(value),
                                    "future_return": block["future_return"].to_numpy()}))
    return pd.concat(pieces, ignore_index=True)


def daily_ic(frame: pd.DataFrame) -> float:
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
    parser.add_argument("--out", default=str(V2 / "v2_tabm_family.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    report = {"seeds": list(SEEDS), "windows": {}}
    for window in WINDOWS:
        base = pd.read_parquet(S4 / f"v1_tabm_seedmean__{window}.parquet").rename(
            columns={"tabm": "value"})
        variant_paths = [V2 / f"tabm_mse_d0.2_{seed}__{window}" / "predictions.parquet"
                         for seed in SEEDS]
        variant = seed_mean(variant_paths)
        c1 = pd.read_parquet(S4 / f"v1_c1_seedmean__{window}.parquet").rename(
            columns={"c1": "value"})
        base_ic = daily_ic(base.merge(labels, on=["date", "code"], how="left",
                                      validate="one_to_one").assign(score=lambda d: d["value"]))
        variant_ic = daily_ic(variant.merge(labels, on=["date", "code"], how="left",
                                            validate="one_to_one").assign(score=lambda d: d["value"]))
        frozen_anchor = pd.read_parquet(S4 / f"v1_anchor__{window}.parquet")
        new_anchor = fence(c1, variant, labels)
        frozen_ic = daily_ic(frozen_anchor.merge(labels, on=["date", "code"], how="left",
                                                 validate="one_to_one"))
        new_ic = daily_ic(new_anchor)
        report["windows"][window] = {
            "tabm_base_seedmean": base_ic, "tabm_d02_seedmean": variant_ic,
            "delta_tabm_family": variant_ic - base_ic,
            "frozen_anchor": frozen_ic, "d02_anchor": new_ic,
            "delta_anchor": new_ic - frozen_ic}
        new_anchor.to_parquet(V2 / f"v2_anchor_d02__{window}.parquet", index=False)
    dev_family = float(np.average([report["windows"][w]["delta_tabm_family"]
                                   for w in WINDOWS], weights=[244, 243]))
    dev_anchor = float(np.average([report["windows"][w]["delta_anchor"]
                                   for w in WINDOWS], weights=[244, 243]))
    report["dev_merged_delta_tabm_family"] = dev_family
    report["dev_merged_delta_anchor"] = dev_anchor
    report["both_years_nonneg_family"] = all(
        report["windows"][w]["delta_tabm_family"] >= 0 for w in WINDOWS)
    report["decision"] = ("retain dropout0.2 as a TabM-family candidate"
                          if dev_family >= 0.0005 and report["both_years_nonneg_family"]
                          else "keep the frozen anchor")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
