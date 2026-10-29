"""CPU acceptance tests for the extracted release packages."""

from __future__ import annotations

import importlib.util
import json
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio
from .constants import BARRA_TABLE, FACTOR_TABLE, FEATURE_COLUMNS, MARKET_TABLE

OUT = Path("artifacts/beginning/cpu_acceptance")


def extract(zip_path: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    return target / "submission"


def load_inference(submission_dir: Path):
    """Mirror `skd_backtest.SubmissionRunner.from_submission` loading exactly."""
    for name in list(sys.modules):
        if name.startswith("_skd_acc_") or name.startswith("alpha_"):
            del sys.modules[name]
    pkg = "_skd_acc_" + str(abs(hash(str(submission_dir))))
    spec = importlib.util.spec_from_file_location(
        pkg, submission_dir / "inference.py",
        submodule_search_locations=[str(submission_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def package_features(module):
    return importlib.import_module(module.__name__ + ".alpha_features")


def window_for(tables: dict, day: int, lookback: int = 120) -> dict:
    calendar = sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
    pos = calendar.index(day)
    first = calendar[max(0, pos - lookback + 1)]
    return {name: table.loc[(table["日期"] >= first) & (table["日期"] <= day)]
            for name, table in tables.items()}


def as_iso(day: int) -> str:
    text = str(day)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def check_frame(out: pd.DataFrame, as_of: str, universe: list[str]) -> dict:
    return {
        "columns_ok": list(out.columns) == ["date", "code", "score"],
        "date_ok": bool((out["date"] == as_of).all()),
        "universe_exact": set(out["code"]) == set(universe) and len(out) == len(universe),
        "unique_codes": not bool(out["code"].duplicated().any()),
        "finite_scores": bool(np.isfinite(out["score"].to_numpy(dtype=float)).all()),
    }


def run_package(zip_path: Path, data_dir: str, name: str,
                dates: list[int] | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    submission = extract(Path(zip_path), OUT / f"extract_{name}")
    module = load_inference(submission)
    model = module.InferenceModel(str(submission / "model"))
    panel = pd.read_parquet("artifacts/beginning/cache/panel.parquet")
    tables = dataio.load_all(data_dir)
    dates = dates or [20160630, 20190603, 20210315]

    report = {"package": str(zip_path), "submission_dir": str(submission), "cases": {}}
    for day in dates:
        window = window_for(tables, day)
        as_of = as_iso(day)
        out = model.predict(as_of, window)
        barra_day = tables[BARRA_TABLE].loc[tables[BARRA_TABLE]["日期"] == day, "代码"]
        checks = check_frame(out, as_of, barra_day.drop_duplicates().tolist())

        offline = panel.loc[panel["date"] == day]
        if not offline.empty:
            pkg_features = package_features(module)
            rebuilt = pkg_features.build_day_features(
                window[MARKET_TABLE], window[FACTOR_TABLE], window[BARRA_TABLE], day)
            left = rebuilt.set_index("code").reindex(offline["code"]).reset_index()
            right = offline.reset_index(drop=True)
            diff = np.abs(left[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
                          - right[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
            finite = np.isfinite(diff)
            checks["feature_parity_max_abs_diff"] = float(diff[finite].max()) if finite.any() else 0.0
            checks["feature_parity_nan_mask_identical"] = bool(
                (np.isnan(left[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
                 == np.isnan(right[list(FEATURE_COLUMNS)].to_numpy(dtype=float))).all())
            checks["offline_row_count_matches"] = bool(len(out) == len(offline))
        checks["n_rows"] = int(len(out))
        report["cases"][as_of] = checks

    # --- invariance: identifiers/banned column cannot change scores --------
    day = 20190603
    window = window_for(tables, day)
    base = model.predict(as_iso(day), window)
    tampered = {k: v.copy() for k, v in window.items()}
    tampered[MARKET_TABLE]["名称"] = "XXXX"
    tampered[FACTOR_TABLE]["名称"] = "YYYY"
    tampered[FACTOR_TABLE]["is_st"] = 1 - tampered[FACTOR_TABLE]["is_st"].fillna(0).astype(int)
    changed = model.predict(as_iso(day), tampered)
    diff = float(np.max(np.abs(base["score"].to_numpy() - changed["score"].to_numpy()))) \
        if len(base) == len(changed) else float("nan")
    report["identifier_invariance_max_abs_diff"] = diff
    report["identifier_invariance_ok"] = bool(diff == 0.0)

    # --- future-data isolation -------------------------------------------
    truncated = {k: v.loc[v["日期"] <= day] for k, v in window.items()}
    trunc_out = model.predict(as_iso(day), truncated)
    trunc_diff = float(np.max(np.abs(base["score"].to_numpy() - trunc_out["score"].to_numpy())))
    report["future_truncation_ok"] = bool(len(base) == len(trunc_out) and trunc_diff == 0.0)

    calendar = sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
    pos = calendar.index(day)
    if pos >= 120:
        old = calendar[pos - 120]
        older = {k: v.copy() for k, v in window.items()}
        mask = older[MARKET_TABLE]["日期"] == old
        older[MARKET_TABLE].loc[mask, "close"] = (
            older[MARKET_TABLE].loc[mask, "close"] * 1.5)
        old_out = model.predict(as_iso(day), older)
        old_diff = float(np.max(np.abs(base["score"].to_numpy() - old_out["score"].to_numpy())))
        report["history_outside_window_ok"] = bool(
            len(base) == len(old_out) and old_diff == 0.0)

    # --- repeat / reconstruction determinism ------------------------------
    again = model.predict(as_iso(day), window)
    report["repeat_call_ok"] = bool(
        np.array_equal(base["score"].to_numpy(), again["score"].to_numpy()))
    model2 = module.InferenceModel(str(submission / "model"))
    rebuilt = model2.predict(as_iso(day), window)
    report["rebuilt_instance_ok"] = bool(
        np.array_equal(base["score"].to_numpy(), rebuilt["score"].to_numpy()))

    # --- short-history validity -------------------------------------------
    for short_day in (20160105, 20160115):
        window_short = window_for(tables, short_day)
        out = model.predict(as_iso(short_day), window_short)
        barra_day = tables[BARRA_TABLE].loc[tables[BARRA_TABLE]["日期"] == short_day, "代码"]
        report[f"short_history_{short_day}"] = check_frame(
            out, as_iso(short_day), barra_day.drop_duplicates().tolist())

    # --- suspension coverage ----------------------------------------------
    susp = tables[MARKET_TABLE].loc[tables[MARKET_TABLE]["is_suspend"] == 1]
    susp = susp.loc[susp["日期"] // 10000 == 2019]
    if len(susp):
        susp_day = int(susp["日期"].iloc[0])
        window_s = window_for(tables, susp_day)
        out = model.predict(as_iso(susp_day), window_s)
        code = susp["代码"].iloc[0]
        report["suspended_stock_scored"] = bool(
            code in set(out["code"]) and np.isfinite(
                float(out.loc[out["code"] == code, "score"].iloc[0])))

    # --- feature-deficient constituent still gets a finite score ----------
    scan_day = 20190603
    window_n = window_for(tables, scan_day)
    pkg_features = package_features(module)
    features_n = pkg_features.build_day_features(
        window_n[MARKET_TABLE], window_n[FACTOR_TABLE], window_n[BARRA_TABLE], scan_day)
    gap = features_n[list(FEATURE_COLUMNS)].isna().sum(axis=1)
    worst = features_n.loc[gap.idxmax()]
    out = model.predict(as_iso(scan_day), window_n)
    report["feature_deficient_stock"] = {
        "code": str(worst["code"]),
        "nan_features": int(gap.max()),
        "scored": bool(worst["code"] in set(out["code"])),
        "score_finite": bool(np.isfinite(
            float(out.loc[out["code"] == worst["code"], "score"].iloc[0]))),
    }
    return report


def timing_run(zip_path: Path, data_dir: str, name: str,
               start: int = 20210601, days: int = 40) -> dict:
    submission = extract(Path(zip_path), OUT / f"extract_{name}_timing")
    module = load_inference(submission)
    model = module.InferenceModel(str(submission / "model"))
    tables = dataio.load_all(data_dir)
    calendar = sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
    pos = calendar.index(start)
    selected = calendar[pos:pos + days]
    durations = []
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for day in selected:
        window = window_for(tables, day)
        t0 = time.perf_counter()
        out = model.predict(as_iso(day), window)
        durations.append(time.perf_counter() - t0)
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    days_arr = np.array(durations)
    return {
        "package": str(zip_path),
        "n_days": len(days_arr),
        "first_day_seconds": float(days_arr[0]),
        "steady_median_seconds": float(np.median(days_arr[1:])),
        "total_seconds": float(days_arr.sum()),
        "peak_rss_mb": peak / 1024.0,
    }


def main() -> None:
    data_dir = "artifacts/beginning/data_zero_st"
    results = {}
    for zip_path, name in ((Path("releases/baseline_v1.zip"), "baseline_v1"),
                           (Path("releases/strong_baseline_v1.zip"), "strong_baseline_v1")):
        results[name] = run_package(zip_path, data_dir, name)
        results[name + "_timing"] = timing_run(zip_path, data_dir, name)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_report.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
