"""REV1: assemble the corrected comparison table, run manifest and invalidation list."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REV = Path("artifacts/stage2_ic_ir/a_revision1")
BT = REV / "backtests"
V1_BT = Path("artifacts/beginning/backtests")
A_BT = Path("artifacts/stage2_ic_ir/backtests")

VALID_RUNS = {
    "v1_bt_b0_dev19_20": V1_BT / "bt_B0_dev19_20",
    "v1_bt_ref_legacy_dev19_20": V1_BT / "bt_SSTAR_dev19_20",
    "v1_bt_ref_legacy_conf21": V1_BT / "bt_SSTAR_conf21",
    "v1_bt_p5_legacy_dev19_20": V1_BT / "bt_P5_dev19_20",
    "v1_bt_p5_legacy_conf21": V1_BT / "bt_P5_conf21",
    "v1_bt_ref_zerocost_dev19_20": V1_BT / "bt_SSTAR_dev19_20_zerocost",
    "a_bt_p5_realinput_dev19_20": A_BT / "bt_p5_realinput_dev19_20",
    "a_bt_catboost_realinput_dev19_20": A_BT / "bt_catboost_realinput_dev19_20",
    "a_bt_catboost_legacy_dev19_20": A_BT / "bt_catboost_dev19_20",
    "a_bt_catboost_legacy_conf21": A_BT / "bt_catboost_conf21",
    "rev1_bt_realinput_dev19_20": BT / "rev1_bt_realinput_dev19_20",
    "rev1_bt_realinput_conf21": BT / "rev1_bt_realinput_conf21",
    "rev1_bt_aligned_dev19_20": BT / "rev1_bt_aligned_dev19_20",
    "rev1_bt_aligned_conf21": BT / "rev1_bt_aligned_conf21",
    "rev1_bt_p5_realinput_conf21": BT / "rev1_bt_p5_realinput_conf21",
    "rev1_bt_catboost_realinput_conf21": BT / "rev1_bt_catboost_realinput_conf21",
    "rev1_bt_refcorrected_dev19_20": BT / "rev1_bt_refcorrected_dev19_20",
    "rev1_bt_refcorrected_conf21": BT / "rev1_bt_refcorrected_conf21",
    "rev1_bt_refcorrected_p5_dev19_20": BT / "rev1_bt_refcorrected_p5_dev19_20",
    "rev1_bt_refcorrected_catboost_dev19_20": BT / "rev1_bt_refcorrected_catboost_dev19_20",
}

INVALIDATED = {
    "reason": ("A-stage whole-array ranking bug in s2_models.centered_rank; the saved "
               "blend score column and everything derived from it are void"),
    "files": [
        "artifacts/stage2_ic_ir/predictions/origmodel_realinput__*.parquet",
        "artifacts/stage2_ic_ir/predictions/aligned_retrain__*.parquet",
        "artifacts/stage2_ic_ir/backtests/bt_realinput_dev19_20",
        "artifacts/stage2_ic_ir/backtests/bt_realinput_conf21",
        "artifacts/stage2_ic_ir/backtests/bt_aligned_dev19_20",
        "artifacts/stage2_ic_ir/backtests/bt_aligned_conf21",
    ],
    "still_valid": [
        "artifacts/stage2_ic_ir/predictions/*_members__*.parquet (member raw scores)",
        "artifacts/stage2_ic_ir/diagnostics/* (re-superseded by rev1 diagnostics)",
    ],
}


def collect() -> pd.DataFrame:
    rows = []
    for name, path in VALID_RUNS.items():
        summary = path / "run_summary.json"
        if not summary.exists():
            continue
        metrics = json.loads(summary.read_text())["metrics"]
        rows.append({
            "run": name,
            "mean_rankic": metrics["mean_rankic"],
            "rankic_ir": metrics["rankic_ir"],
            "annualized_excess_return": metrics["annualized_excess_return"],
            "tracking_error": metrics["tracking_error"],
            "information_ratio": metrics["information_ratio"],
            "max_drawdown": metrics["maximum_drawdown"],
            "turnover": metrics["turnover"],
            "transaction_cost": metrics["transaction_cost"],
            "failed_orders": metrics["failed_orders"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    table = collect()
    table.to_csv(REV / "rev1_backtest_table.csv", index=False)
    manifest = {
        "revision": "a_revision1",
        "frozen_at": "2026-10-05",
        "invalidated_products": INVALIDATED,
        "valid_run_dirs": {k: str(v) for k, v in VALID_RUNS.items()},
        "note": ("all runs are separate directories with distinct run IDs; the rev1 script "
                 "refuses to overwrite an existing run directory"),
    }
    (REV / "rev1_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    print(json.dumps(manifest["invalidated_products"], indent=2))


if __name__ == "__main__":
    main()
