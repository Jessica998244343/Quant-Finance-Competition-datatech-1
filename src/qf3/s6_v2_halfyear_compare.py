"""§A2: persistent annual-vs-half-year comparison for the three combination vectors.

Vectors (all flat graph):
  V1         {c1: 0.5,  tabm: 0.5}
  3-member   {c1: 0.375, tabm: 0.375, tcn: 0.25}
  2-member   {c1: 0.6,  tcn: 0.4}          (the prescribed drop-TabM ablation)

Saves per-day IC and ΔIC, the common date index and the valid-label mask rule, and
records the SHA-256 of every input prediction file.  No training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s6_fusion import family_q, fuse

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
V2 = S4 / "v2"
SPEC = {
    "2019H2": {"annual": "DEV19", "year": 2019, "seeds": (20261005,), "start": 20190701},
    "2020H2": {"annual": "DEV20", "year": 2020, "seeds": (20261005, 20261017, 20261029),
               "start": 20200701},
    "2021H2": {"annual": "CONF21", "year": 2021, "seeds": (20261005, 20261017, 20261029),
               "start": 20210701},
}
VECTORS = {
    "V1": {"c1": 0.5, "tabm": 0.5},
    "three_member": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25},
    "two_member": {"c1": 0.6, "tcn": 0.4},
}
LABEL_MASK_RULE = "future_return finite (panel_platform label, adjusted-open O2O) on the same legal pool"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _annual_paths(window: str, seed: int) -> dict:
    return {"c1": FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet",
            "tabm": S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "predictions.parquet",
            "tcn": V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "predictions.parquet"}


def _half_paths(window: str, seed: int) -> dict:
    tagged = V2 / f"halfyear_{window}_s{seed}"
    base = tagged if tagged.exists() else V2 / f"halfyear_{window}"
    return {"c1": base / f"catboost_c1_seed{seed}__predictions.parquet",
            "tabm": base / f"tabm_seed{seed}__predictions.parquet",
            "tcn": V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "predictions.parquet"}


def _family(paths: dict, seeds) -> dict:
    out = {}
    for family in ("c1", "tabm", "tcn"):
        frames, hashes = [], []
        for seed in seeds:
            path = paths[seed][family]
            if not path.exists():
                raise FileNotFoundError(f"missing {family} {seed}: {path}")
            frames.append(pd.read_parquet(path))
            hashes.append({"seed": int(seed), "path": str(path), "sha256": _sha(path)})
        out[family] = family_q(frames, "score", label=f"{family}")
        out[f"{family}_hashes"] = hashes
    return out


def _daily(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    merged = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    values = {}
    for date, block in merged.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values[date] = float(np.corrcoef(y, p)[0, 1])
    return pd.Series(values).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_completion_halfyear_compare.json"))
    parser.add_argument("--daily-out", default=str(V2 / "v2_completion_halfyear_daily.parquet"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    report = {"vectors": VECTORS, "label_mask_rule": LABEL_MASK_RULE, "windows": {}}
    daily_rows = []
    for window, spec in SPEC.items():
        seeds = spec["seeds"]
        starts = {"annual": {s: _annual_paths(spec["annual"], s) for s in seeds},
                  "half": {s: _half_paths(window, s) for s in seeds}}
        entry = {"annual_window": spec["annual"], "year": spec["year"],
                 "seeds": [int(s) for s in seeds], "starts": {}}
        series = {}
        for start, paths in starts.items():
            families = _family(paths, seeds)
            entry["starts"][start] = {f"{family}_hashes": families[f"{family}_hashes"]
                                      for family in ("c1", "tabm", "tcn")}
            start_entry = {}
            for name, vector in VECTORS.items():
                components = {key: families[key] for key in vector}
                fused = fuse(components, vector)
                values = _daily(fused, labels)
                values = values[values.index >= spec["start"]]
                series[(start, name)] = values
                start_entry[name] = {"rankic": float(values.mean()), "days": int(len(values))}
            entry["starts"][start]["combinations"] = start_entry
        common = None
        for key in series:
            index = series[key].index
            common = index if common is None else common.intersection(index)
        entry["common_days"] = int(len(common))
        entry["common_dates"] = [int(d) for d in common]
        for start in starts:
            for name in VECTORS:
                values = series[(start, name)].reindex(common)
                base = series[(start, "V1")].reindex(common)
                entry["starts"][start]["combinations"][name]["daily_ic"] = [
                    float(v) for v in values.to_numpy()]
                entry["starts"][start]["combinations"][name]["delta_vs_v1_daily"] = [
                    float(v) for v in (values - base).to_numpy()]
                daily_rows.append(pd.DataFrame({
                    "window": window, "start": start, "combination": name,
                    "date": list(common), "daily_ic": values.to_numpy(),
                    "daily_delta_vs_v1": (values - base).to_numpy()}))
        diffs = {}
        for name in ("three_member", "two_member"):
            for start in starts:
                a = series[(start, name)].reindex(common)
                b = series[(start, "V1")].reindex(common)
                c = series[(start, "three_member")].reindex(common)
                diffs[f"{name}_minus_v1_{start}"] = float((a - b).mean())
                diffs[f"{name}_minus_three_{start}"] = float((a - c).mean())
        entry["deltas"] = diffs
        entry["start_sensitivity"] = float(
            (series[("half", "three_member")].reindex(common)
             - series[("half", "V1")].reindex(common)).mean()
            - (series[("annual", "three_member")].reindex(common)
               - series[("annual", "V1")].reindex(common)).mean())
        report["windows"][window] = entry
        print(window, {k: round(v, 9) for k, v in diffs.items()},
              "start_sensitivity", round(entry["start_sensitivity"], 9), flush=True)
    pd.concat(daily_rows, ignore_index=True).to_parquet(args.daily_out, index=False)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"windows": list(report["windows"]),
                      "daily": args.daily_out}, indent=2))


if __name__ == "__main__":
    main()
