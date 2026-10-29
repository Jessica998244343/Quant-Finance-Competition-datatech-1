"""D1 §4 fixes: correct reference table, hashes, curve report, short-history N/A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first")
REV = Path("artifacts/stage2_ic_ir/a_revision1/predictions")
B = Path("artifacts/stage2_ic_ir/b/predictions")

WINDOWS = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}

REFERENCES = {
    "R_SINGLE": {w: REV / f"ref_corrected_lgbm__{w}.parquet" for w in WINDOWS},
    "R_WORK": {w: B / f"ref_corrected_cat0.5__{w}.parquet" for w in WINDOWS},
    "R_DEPLOY": {w: REV / f"origmodel_realinput__{w}.parquet" for w in WINDOWS},
    "MAIN_SINGLE_v1": {w: S3 / "predictions" / f"L3_L0_20261005__{w}.parquet"
                       for w in WINDOWS},
    "MAIN_SEED_ENSEMBLE_v1": {w: S3 / "predictions" /
                              f"MAIN_SEED_ENSEMBLE__{w}.parquet" for w in WINDOWS},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_year_ic(panel: pd.DataFrame, mapping: dict) -> dict:
    out = {}
    for window, year in WINDOWS.items():
        preds = pd.read_parquet(mapping[window])
        days = pd.unique(preds["date"])
        scope = panel.loc[panel["date"].isin(days), ["date", "code", "future_return"]]
        merged = preds.merge(scope,
                             on=["date", "code"], how="outer", indicator=True,
                             validate="one_to_one")
        missing = int((merged["_merge"] != "both").sum())
        matched = merged.loc[merged["_merge"] == "both"]
        summary = metrics.summarize_rankic(metrics.daily_rankic(matched))
        out[year] = {"mean_rankic": summary["mean_rankic"],
                     "n_days": summary["n_days"],
                     "key_mismatch": missing,
                     "rows": int(len(preds))}
    dev = []
    for window in ("DEV19", "DEV20"):
        preds = pd.read_parquet(mapping[window])
        dev.append(preds.merge(panel[["date", "code", "future_return"]],
                               on=["date", "code"], how="left", validate="one_to_one"))
    dev_frame = pd.concat(dev, ignore_index=True)
    out["dev_pooled"] = metrics.summarize_rankic(metrics.daily_rankic(dev_frame))["mean_rankic"]
    three = []
    for window in WINDOWS:
        preds = pd.read_parquet(mapping[window])
        three.append(preds.merge(panel[["date", "code", "future_return"]],
                                 on=["date", "code"], how="left", validate="one_to_one"))
    three_frame = pd.concat(three, ignore_index=True)
    out["three_year_pooled"] = metrics.summarize_rankic(
        metrics.daily_rankic(three_frame))["mean_rankic"]
    return out


def curve_report() -> pd.DataFrame:
    rows = []
    for path in sorted((S3 / "models").glob("*__curve.parquet")):
        curve = pd.read_parquet(path)
        values = curve["daily_spearman"].to_numpy(dtype=float)
        if not len(values):
            continue
        best = int(np.nanargmax(values)) + 1
        best_value = float(values[best - 1])
        end_value = float(values[-1])
        tolerance = best_value - 0.001
        near = np.nonzero(values >= tolerance)[0]
        rows.append({
            "model": path.stem.replace("__curve", ""),
            "evaluated_points": int(len(values)),
            "best_iteration": best,
            "best_daily_spearman": best_value,
            "end_iteration": int(len(values)),
            "end_daily_spearman": end_value,
            "near_optimal_first": int(near[0]) + 1,
            "near_optimal_last": int(near[-1]) + 1,
            "near_optimal_span": int(near[-1] - near[0] + 1),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def short_history_report() -> dict:
    """Empty cache intersection must be reported as N/A, never as a pass."""
    from . import s2r_parity
    records = []
    for day, start, label in ((20160105, "2016-01-04", "2-day short history"),
                              (20160201, "2016-01-04", "20-day history")):
        provider = s2r_parity.provider_as_of(day, start)
        from . import featurelib
        frame = featurelib.build_day_features(
            provider["MarketData"], provider["Factor33_winsor"],
            provider["Barra_factor"], day)
        barra = provider["Barra_factor"]
        universe = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
        finite = np.isfinite(frame[list(featurelib.FEATURE_COLUMNS)].to_numpy(dtype=float))
        cache = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
        cached = cache.loc[cache["date"] == day]
        records.append({
            "day": day, "label": label,
            "provider_window_days": int(provider["MarketData"]["日期"].nunique()),
            "provider_rows_per_table": {k: int(len(v)) for k, v in provider.items()},
            "cache_rows_for_day": int(len(cached)),
            "cache_intersection": int(len(cached)),
            "comparison": "N/A (no cache rows for this day)" if cached.empty
                          else "compared",
            "pool_matches_universe": bool(set(frame["code"]) == set(universe)),
            "scored_rows": int(len(frame)),
            "all_finite": bool(finite.all()),
            "n_finite_features_total": int(finite.sum()),
        })
    return {"records": records,
            "note": "empty intersections are N/A and are never counted as a pass"}


def main() -> None:
    FUS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    report = {"reference_ic": {}, "reference_hashes": {}, "curves": {}, "short_history": {}}
    for name, mapping in REFERENCES.items():
        report["reference_ic"][name] = per_year_ic(panel, mapping)
        report["reference_hashes"][name] = {w: sha256(p) for w, p in mapping.items()}
    curves = curve_report()
    curves.to_csv(FUS / "d1_curve_report.csv", index=False)
    report["curves"] = curves.to_dict("records")
    report["short_history"] = short_history_report()
    (FUS / "d1_fixes.json").write_text(json.dumps(report, indent=2, default=str))
    table = pd.DataFrame({name: {**{str(k): v["mean_rankic"] if isinstance(v, dict) else v
                                    for k, v in stats.items()}}
                          for name, stats in report["reference_ic"].items()}).T
    print(table.to_string())
    print("\n", curves.to_string(index=False))
    print(json.dumps(report["short_history"], indent=2, default=str))


if __name__ == "__main__":
    main()
