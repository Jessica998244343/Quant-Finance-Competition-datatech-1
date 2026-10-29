"""§4.6/§5.1: coarse family-weight comparison, deletion ablation and intervals.

Uses only saved predictions (no training).  The weight grid is selected on the
two DEV years only; 2021 is reported as a frozen robustness check.  Paired daily
Delta-IC intervals use the fixed-length moving-block draw from `s4_bootstrap`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s4_bootstrap import _draw_indices

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
SEEDS = (20261005, 20261017, 20261029)
WINDOWS = ("DEV19", "DEV20", "CONF21")
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOT_SEED = 20261005
BOOT_DRAWS = 2000
FLAT_THRESHOLD = 0.0003
KEEP_THRESHOLD = 0.0005


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def _daily_ic(frame: pd.DataFrame, column: str) -> pd.Series:
    out = {}
    for day, block in frame.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block[column])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, column].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        out[day] = float(np.corrcoef(y, p)[0, 1])
    return pd.Series(out).sort_index()


def _family(window: str, family: str, seeds=SEEDS) -> pd.DataFrame:
    """Seed-mean of per-seed daily centred ranks for one family."""
    merged = None
    for index, seed in enumerate(seeds):
        if family == "c1":
            frame = pd.read_parquet(FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet")
        else:
            frame = pd.read_parquet(
                S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "predictions.parquet")
        frame = frame.rename(columns={"score": f"s{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(seeds)):
            value += _rank(block[f"s{index}"].reset_index(drop=True)) / len(seeds)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    family: value}))
    return pd.concat(pieces, ignore_index=True)


def _fuse(c1: pd.DataFrame, tabm: pd.DataFrame, w_tabm: float,
          labels: pd.DataFrame) -> pd.DataFrame:
    merged = c1.merge(tabm, on=["date", "code"], validate="one_to_one")
    merged = merged.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        u = (1.0 - w_tabm) * block["c1"].to_numpy() + w_tabm * block["tabm"].to_numpy()
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": _rank(u),
                                    "future_return": block["future_return"].to_numpy()}))
    return pd.concat(pieces, ignore_index=True)


def _merged_ic(ics: list[pd.Series]) -> float:
    joined = pd.concat(ics)
    return float(joined.mean())


def _interval(delta: pd.Series, block: int) -> dict:
    """Per-year moving-block draws; each year keeps its own day count (task §9.1)."""
    years = pd.Series(delta.index).astype("int64") // 10000
    groups = [delta.to_numpy(dtype=float)[(years == year).to_numpy()]
              for year in sorted(years.unique())]
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(BOOT_DRAWS)
    for index in range(BOOT_DRAWS):
        pieces = [values[_draw_indices(len(values), block, rng)] for values in groups]
        draws[index] = np.concatenate(pieces).mean()
    return {"days": int(len(delta)), "block": block,
            "years": {int(year): int((years == year).sum()) for year in sorted(years.unique())},
            "point": float(delta.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(S4 / "v1_anchor_compare.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    families = {window: {"c1": _family(window, "c1"), "tabm": _family(window, "tabm")}
                for window in WINDOWS}
    report: dict = {"grid": list(GRID), "windows": {},
                    "flat_threshold": FLAT_THRESHOLD, "keep_threshold": KEEP_THRESHOLD}
    ic_by_window: dict = {}
    for window in WINDOWS:
        entry = {}
        for w in GRID:
            fused = _fuse(families[window]["c1"], families[window]["tabm"], w, labels)
            entry[str(w)] = float(_daily_ic(fused, "score").mean())
        report["windows"][window] = entry
        ic_by_window[window] = entry

    dev_grid = {w: float(np.average(
        [ic_by_window[window][str(w)] for window in ("DEV19", "DEV20")],
        weights=[244, 243])) for w in GRID}
    best_w = max(dev_grid, key=dev_grid.get)
    report["dev_merged_ic"] = {str(w): dev_grid[w] for w in GRID}
    report["dev_best_w"] = best_w
    report["dev_best_minus_default"] = dev_grid[best_w] - dev_grid[0.5]
    if report["dev_best_minus_default"] < FLAT_THRESHOLD:
        report["frozen_weight"] = 0.5
        report["weight_decision"] = "flat: keep the simple 0.5/0.5 fixed anchor"
    else:
        report["frozen_weight"] = best_w
        report["weight_decision"] = (
            "selected on DEV only; V1_SELECTED_ANCHOR (>=0.0005)" if
            report["dev_best_minus_default"] >= KEEP_THRESHOLD else
            "selected but below the keep threshold")
    report["conf21_at_frozen_weight"] = ic_by_window["CONF21"][str(report["frozen_weight"])]
    report["conf21_at_default_weight"] = ic_by_window["CONF21"]["0.5"]

    frozen_w = report["frozen_weight"]
    deletion = {}
    for window in WINDOWS:
        base = _fuse(families[window]["c1"], families[window]["tabm"], frozen_w, labels)
        base_ic = float(_daily_ic(base, "score").mean())
        row = {}
        for family in ("c1", "tabm"):
            for seed in SEEDS:
                seeds = tuple(item for item in SEEDS if item != seed)
                re_c1 = families[window]["c1"] if family == "c1" else _family(window, "c1", seeds)
                re_tabm = families[window]["tabm"] if family == "tabm" else _family(window, "tabm", seeds)
                candidate = _fuse(re_c1, re_tabm, frozen_w, labels)
                row[f"drop_{family}_{seed}"] = float(
                    _daily_ic(candidate, "score").mean() - base_ic)
        deletion[window] = row
    report["deletion_delta_ic"] = deletion

    controls = {}
    for window in WINDOWS:
        anchor = _fuse(families[window]["c1"], families[window]["tabm"], frozen_w, labels)
        anchor_ic = _daily_ic(anchor, "score")
        row = {}
        for name, series in (("c1_seedmean", _daily_ic(_fuse(
                families[window]["c1"], families[window]["tabm"], 0.0, labels), "score")),
                ("tabm_seedmean", _daily_ic(_fuse(
                    families[window]["c1"], families[window]["tabm"], 1.0, labels), "score"))):
            joined = pd.concat([anchor_ic, series], axis=1, keys=["anchor", "control"]).dropna()
            delta = joined["anchor"] - joined["control"]
            row[name] = {f"block{block}": _interval(delta, block)
                         for block in (10, 20, 40)}
        controls[window] = row
    dev_anchor = pd.concat([_daily_ic(_fuse(families[w]["c1"], families[w]["tabm"],
                                            frozen_w, labels), "score")
                            for w in ("DEV19", "DEV20")])
    dev_row = {}
    for name, w_value in (("c1_seedmean", 0.0), ("tabm_seedmean", 1.0)):
        series = pd.concat([_daily_ic(_fuse(families[w]["c1"], families[w]["tabm"],
                                            w_value, labels), "score")
                            for w in ("DEV19", "DEV20")])
        joined = pd.concat([dev_anchor, series], axis=1, keys=["anchor", "control"]).dropna()
        delta = joined["anchor"] - joined["control"]
        dev_row[name] = {f"block{block}": _interval(delta, block)
                         for block in (10, 20, 40)}
    controls["DEV_merged"] = dev_row
    report["anchor_vs_controls"] = controls
    report["note"] = ("conditional evidence on already-used development data; "
                      "no 2022 performance was computed")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
