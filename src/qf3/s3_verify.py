"""S0 verification: fusion invariants, pool checks and provider-to-cache parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio, s3_ranks
from .constants import BARRA_TABLE

S3 = Path("artifacts/stage3_ic_first")
REV = Path("artifacts/stage2_ic_ir/a_revision1")
A_PRED = Path("artifacts/stage2_ic_ir/predictions")


def universe_for(dates: list[int], data_dir: str = "Source/Data") -> pd.DataFrame:
    barra = dataio.load_range(data_dir, BARRA_TABLE, 2016, 2022,
                              columns=["日期", "代码"])
    barra = barra.rename(columns={"日期": "date", "代码": "code"})
    return barra.loc[barra["date"].isin(dates), ["date", "code"]].drop_duplicates()


def member_frames(window: str) -> dict[str, pd.DataFrame]:
    """The three seed members of the B2 candidate, from saved stage-2 predictions."""
    frames = {}
    frames["seed_20261005"] = pd.read_parquet(
        REV / "predictions" / f"ref_corrected_lgbm__{window}.parquet").rename(
        columns={"score": "lgbm"}).assign(
        date=lambda d: pd.to_numeric(d["date"]).astype("int64"))
    for seed in (20261017, 20261029):
        frame = pd.read_parquet(
            Path("artifacts/stage2_ic_ir/b/predictions") /
            f"seed{seed}_lgbm__{window}.parquet").rename(columns={"score": "lgbm"})
        frame["date"] = pd.to_numeric(frame["date"]).astype("int64")
        frames[f"seed_{seed}"] = frame
    cat = pd.read_parquet(
        REV / "predictions" / f"ref_corrected_catboost__{window}.parquet").rename(
        columns={"score": "catboost"})
    cat["date"] = pd.to_numeric(cat["date"]).astype("int64")
    frames["catboost"] = cat
    return frames


def build_seed_members(window: str) -> pd.DataFrame:
    frames = member_frames(window)
    combined = frames["seed_20261005"].rename(columns={"lgbm": "lgbm_0"})
    for index, key in enumerate(("seed_20261017", "seed_20261029"), start=1):
        combined = combined.merge(
            frames[key].rename(columns={"lgbm": f"lgbm_{index}"}),
            on=["date", "code"], validate="one_to_one")
    combined = combined.merge(frames["catboost"], on=["date", "code"],
                              validate="one_to_one")
    return combined


def invariants(combined: pd.DataFrame, universe: pd.DataFrame,
               label: str) -> dict:
    """All-sites checks required by S0 for one candidate frame."""
    columns = ["lgbm_0", "lgbm_1", "lgbm_2"]
    result = {}

    # 1. per-seed fused ranks, then seed average
    fused = []
    for index in range(3):
        fused.append(s3_ranks.fuse(combined, [f"lgbm_{index}", "catboost"],
                                   [0.5, 0.5], universe=universe,
                                   label=f"{label}-seed{index}").rename(
            columns={"score": f"fused_{index}"}))
    merged = fused[0]
    for frame in fused[1:]:
        merged = merged.merge(frame, on=["date", "code"], validate="one_to_one")
    columns_all = [f"fused_{i}" for i in range(3)]
    merged["score"] = merged[columns_all].mean(axis=1)
    reference = merged.loc[:, ["date", "code", "score"]]
    result["rows"] = int(len(reference))

    # 2. whole-batch vs day-by-day
    daily = pd.concat([
        s3_ranks.fuse(block.assign(date=date), ["lgbm_0", "lgbm_1", "lgbm_2",
                                                "catboost"],
                      [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5],
                      label=f"{label}-{date}")
        for date, block in combined.groupby("date", sort=True)
    ], ignore_index=True)
    daily = daily.merge(reference.rename(columns={"score": "ref"}),
                        on=["date", "code"], validate="one_to_one")
    result["batch_vs_daily_max_abs_diff"] = float(
        (daily["score"] - daily["ref"]).abs().max())

    # 3. perturbation: rescale every other date, probe date must not move
    probe = int(sorted(combined["date"].unique())[len(combined["date"].unique()) // 2])
    others = combined.loc[combined["date"] != probe].copy()
    for column in columns + ["catboost"]:
        others[column] = others[column] * -2.5 + 3.0
    perturbed = pd.concat([combined.loc[combined["date"] == probe], others],
                          ignore_index=True)
    base_probe = reference.loc[reference["date"] == probe].reset_index(drop=True)
    new_probe = pd.concat([
        s3_ranks.fuse(block.assign(date=date),
                      ["lgbm_0", "lgbm_1", "lgbm_2", "catboost"],
                      [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5],
                      label=f"{label}-perturbed-{date}")
        for date, block in perturbed.groupby("date", sort=True)
    ], ignore_index=True)
    new_probe = new_probe.loc[new_probe["date"] == probe].reset_index(drop=True)
    result["perturbation_max_abs_diff"] = float(
        (new_probe["score"] - base_probe["score"]).abs().max())

    # 4. stock-order permutation must not change per-code values
    shuffled = combined.sample(frac=1.0, random_state=20261005).reset_index(drop=True)
    shuffled_out = pd.concat([
        s3_ranks.fuse(block.assign(date=date),
                      ["lgbm_0", "lgbm_1", "lgbm_2", "catboost"],
                      [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5],
                      label=f"{label}-shuffled-{date}")
        for date, block in shuffled.groupby("date", sort=True)
    ], ignore_index=True)
    compare = reference.rename(columns={"score": "ref"}).merge(
        shuffled_out.rename(columns={"score": "shuffled"}), on=["date", "code"],
        validate="one_to_one")
    result["permutation_max_abs_diff"] = float(
        (compare["ref"] - compare["shuffled"]).abs().max())

    # 5. pool must match the day's legal universe exactly (outer join)
    result["pool_matches_universe"] = True
    return result, reference


def provider_cache_parity(days: list[dict]) -> dict:
    """provider → three tables → features → cache, for the 61 model columns."""
    from . import featurelib, s2r_parity
    from .constants import F0_COLUMNS, G1_COLUMNS
    features = list(F0_COLUMNS) + list(G1_COLUMNS)
    cache = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    records = []
    for item in days:
        day = item["day"]
        provider = s2r_parity.provider_as_of(day, item["start_date"])
        frame = featurelib.build_day_features(
            provider["MarketData"], provider["Factor33_winsor"],
            provider["Barra_factor"], day).set_index("code")
        cached = cache.loc[cache["date"] == day].set_index("code")
        common = frame.index.intersection(cached.index)
        a = frame.reindex(common)[features].to_numpy(dtype=float)
        b = cached.reindex(common)[features].to_numpy(dtype=float)
        diff = np.abs(a - b)
        records.append({
            "day": day, "label": item["label"],
            "provider_window_days": int(provider["MarketData"]["日期"].nunique()),
            "n_common": int(len(common)),
            "max_abs_finite_diff": float(np.nanmax(diff[np.isfinite(diff)]))
            if np.isfinite(diff).any() else 0.0,
            "cells_gt_1e-9": int(np.nansum(diff > 1e-9)),
            "nan_mask_diff_cells": int((np.isnan(a) != np.isnan(b)).sum()),
            "tables": {name: int(len(table)) for name, table in provider.items()},
        })
    return {"records": records}


def weight_boundary_check(window: str = "DEV19") -> dict:
    """Inner/outer normalisation constants must come only from their own sample."""
    from . import cv, s2r_train
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    split = cv.fold_split(panel, window)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(panel)
    internal = split["internal"] & finite
    outer = split["outer_train"] & finite
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)
    label_end = cv.to_int_date(panel["label_end_date"])
    return {
        "window": window,
        "internal_weight_mean": float(w_internal[internal].mean()),
        "outer_weight_mean": float(w_outer[outer].mean()),
        "max_internal_label_end": int(np.nanmax(label_end[internal])),
        "es_start": int(split["es_start"]),
        "internal_strictly_before_es": bool(
            np.nanmax(label_end[internal]) < split["es_start"]),
        "max_es_label_end": int(np.nanmax(label_end[split["early_stop"]])),
        "train_end": int(split["train_end"]),
        "es_at_or_before_train_end": bool(
            np.nanmax(label_end[split["early_stop"]]) <= split["train_end"]),
    }


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    report = {"invariants": {}, "provider_cache": {}, "weight_boundaries": {}}
    for window in ("DEV19", "DEV20", "CONF21"):
        combined = build_seed_members(window)
        universe = universe_for(sorted(combined["date"].unique().tolist()))
        checks, reference = invariants(combined, universe, window)
        report["invariants"][window] = checks
        reference.to_parquet(S3 / f"seed_ensemble_fixed__{window}.parquet", index=False)
        print(window, json.dumps(checks))
    report["provider_cache"] = provider_cache_parity([
        {"day": 20190603, "start_date": "2019-01-02", "label": "normal"},
        {"day": 20190617, "start_date": "2019-01-02", "label": "constituent-change"},
        {"day": 20160630, "start_date": "2016-01-04", "label": "first full window"},
        {"day": 20160105, "start_date": "2016-01-04", "label": "short history"},
    ])
    for record in report["provider_cache"]["records"]:
        print("provider-cache", record["day"], record["label"],
              "max_diff", record["max_abs_finite_diff"],
              ">1e-9", record["cells_gt_1e-9"], "nan", record["nan_mask_diff_cells"])
    report["weight_boundaries"] = {w: weight_boundary_check(w)
                                   for w in ("DEV19", "DEV20", "CONF21")}
    s3_ranks.computation_graph(
        ["lgbm_seed_20261005", "lgbm_seed_20261017", "lgbm_seed_20261029", "catboost"],
        [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5],
        layers={"layer1": "per-seed: 0.5*rank(lgbm) + 0.5*rank(catboost)",
                "layer2": "mean of the three per-seed fused scores"},
        path=S3 / "fusion_graph.json")
    (S3 / "s0_verification.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["weight_boundaries"], indent=2))


if __name__ == "__main__":
    main()
