"""Assemble the D2 result tables, manifests and candidate records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, metrics

OUT = Path("artifacts/beginning")
PRED = OUT / "predictions"
WINDOWS = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}
VERSIONS = ["B0", "R0", "P5_s20261005", "SC_d8_l2_15", "SSTAR"]


def rankic_for(version: str, window: str) -> pd.DataFrame | None:
    path = PRED / f"{version}__{window}__rankic.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pred_path = PRED / f"{version}__{window}.parquet"
    if not pred_path.exists():
        return None
    labels = pd.read_parquet(OUT / "cache" / "labels.parquet")
    preds = pd.read_parquet(pred_path)
    merged = preds.merge(labels[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    rankic.to_parquet(PRED / f"{version}__{window}__rankic.parquet", index=False)
    return rankic


def results_table() -> pd.DataFrame:
    rows = []
    for version in VERSIONS:
        for window in WINDOWS:
            rankic = rankic_for(version, window)
            if rankic is None:
                continue
            summary = metrics.summarize_rankic(rankic)
            rows.append({"version": version, "window": window,
                         "mean_rankic": summary["mean_rankic"],
                         "rankic_std": summary["rankic_std"],
                         "rankic_ir": summary["rankic_ir"],
                         "positive_ratio": summary["positive_ratio"],
                         "n_days": summary["n_days"]})
    return pd.DataFrame(rows)


def paired_table() -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        base = rankic_for("B0", window)
        if base is None:
            continue
        for version in ("P5_s20261005", "SC_d8_l2_15", "SSTAR"):
            cand = rankic_for(version, window)
            if cand is None:
                continue
            stat = metrics.paired_ic_difference(cand, base, block=20, n_boot=2000)
            stat10 = metrics.paired_ic_difference(cand, base, block=10, n_boot=2000)
            stat40 = metrics.paired_ic_difference(cand, base, block=40, n_boot=2000)
            rows.append({"window": window, "version": version,
                         "delta_ic": stat["mean_diff"],
                         "ci95_low": stat["ci95"][0], "ci95_high": stat["ci95"][1],
                         "ci90_low": stat["ci90"][0], "ci90_high": stat["ci90"][1],
                         "block10_ci95": [stat10["ci95"][0], stat10["ci95"][1]],
                         "block40_ci95": [stat40["ci95"][0], stat40["ci95"][1]],
                         "n_days": stat["n"]})
    return pd.DataFrame(rows)


def backtest_table() -> pd.DataFrame:
    rows = []
    for path in sorted((OUT / "backtests").glob("*/run_summary.json")):
        record = json.loads(path.read_text())
        m = record["metrics"]
        rows.append({
            "run": record["run"], "seconds": record["seconds"],
            "mean_rankic": m["mean_rankic"], "rankic_ir": m["rankic_ir"],
            "annualized_excess_return": m["annualized_excess_return"],
            "tracking_error": m["tracking_error"],
            "information_ratio": m["information_ratio"],
            "max_drawdown": m["maximum_drawdown"], "turnover": m["turnover"],
            "transaction_cost": m["transaction_cost"],
            "failed_orders": m["failed_orders"],
            "positive_ratio": m["positive_rankic_ratio"],
        })
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_manifest() -> dict:
    manifest = {"packages": {}, "artifacts": {}}
    for name in ("baseline_v1", "strong_baseline_v1"):
        zip_path = Path("releases") / f"{name}.zip"
        package_dir = Path("releases/build") / name
        if zip_path.exists():
            manifest["packages"][name] = {
                "zip": str(zip_path),
                "bytes": zip_path.stat().st_size,
                "sha256": sha256(zip_path),
                "uncompressed_bytes": sum(p.stat().st_size for p in package_dir.rglob("*")
                                          if p.is_file()),
                "model_files": {p.name: p.stat().st_size
                                for p in (package_dir / "model").iterdir()},
            }
    for name, path in (("acceptance", OUT / "cpu_acceptance" / "acceptance_report.json"),
                       ("reproduction", OUT / "cpu_acceptance" / "reproduction_report.json"),
                       ("b0_freeze", OUT / "B0_FREEZE.md"),
                       ("decisions", OUT / "decisions_pre_conf21.md"),
                       ("feature_catalog", OUT / "feature_catalog.csv"),
                       ("splits", OUT / "splits.json"),
                       ("experiments", OUT / "experiments.jsonl"),
                       ("panel_cache", OUT / "cache" / "panel.parquet")):
        if Path(path).exists():
            manifest["artifacts"][name] = {"path": str(path),
                                           "bytes": Path(path).stat().st_size}
    return manifest


def main() -> None:
    log = analysis.load_log()
    log.to_csv(OUT / "experiments.csv", index=False)
    results = results_table()
    results.to_csv(OUT / "results_by_window.csv", index=False)
    paired = paired_table()
    paired.to_csv(OUT / "paired_vs_B0.csv", index=False)
    backtests = backtest_table()
    backtests.to_csv(OUT / "backtests.csv", index=False)
    manifest = release_manifest()
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 200)
    print("== results by window ==\n", results.to_string(index=False))
    print("\n== paired vs B0 ==\n", paired.to_string(index=False))
    print("\n== backtests ==\n", backtests.to_string(index=False))
    print("\n== manifest ==\n", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
