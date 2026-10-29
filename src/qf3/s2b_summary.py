"""B: consolidate every stage-B run into one summary table and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import s2b_common

REV = Path("artifacts/stage2_ic_ir/a_revision1")

RUNS = {
    "REF_legacy_v1": Path("artifacts/beginning/backtests/bt_SSTAR_dev19_20"),
    "REF_corrected_cat0": s2b_common.BT / "b1_bt_cat0_dev19_20",
    "REF_corrected_cat0.125": s2b_common.BT / "b1_bt_cat0.125_dev19_20",
    "REF_corrected_cat0.25": s2b_common.BT / "b1_bt_cat0.25_dev19_20",
    "REF_corrected_cat0.5": s2b_common.BT / "b1_bt_cat0.5_dev19_20",
    "REF_corrected_cat1": s2b_common.BT / "b1_bt_cat1_dev19_20",
    "B2_seed_average": s2b_common.BT / "b2_seedavg",
    "B2_E1_a0.1": s2b_common.BT / "b2_E1_a0.1",
    "B2_E1_a0.2": s2b_common.BT / "b2_E1_a0.2",
    "B2_E2_a0.1": s2b_common.BT / "b2_E2_a0.1",
    "B3_lam0.1": s2b_common.BT / "b3_bt_b3_lam0.1_dev19_20",
    "B3_lam0.25": s2b_common.BT / "b3_bt_b3_lam0.25_dev19_20",
    "B3_lam0.25_a0.2": s2b_common.BT / "b3_bt_b3_lam0.25_a0.2_dev19_20",
}

CONF21 = {
    "REF_legacy_v1": Path("artifacts/beginning/backtests/bt_SSTAR_conf21"),
    "REF_corrected_cat0": s2b_common.BT / "b1_bt_cat0_conf21",
    "REF_corrected_cat0.125": s2b_common.BT / "b1_bt_cat0.125_conf21",
    "REF_corrected_cat0.25": s2b_common.BT / "b1_bt_cat0.25_conf21",
    "REF_corrected_cat0.5": s2b_common.BT / "b1_bt_cat0.5_conf21",
    "B2_seed_average": s2b_common.BT / "b2_seedavg_conf21",
    "B2_E1_a0.1": s2b_common.BT / "b2_E1_a0.1_conf21",
    "B3_lam0.25": s2b_common.BT / "b3_bt_b3_lam0.25_conf21",
    "B3_lam0.25_a0.2": s2b_common.BT / "b3_bt_b3_lam0.25_a0.2_conf21",
}


def collect(mapping: dict, period: str) -> pd.DataFrame:
    rows = []
    for name, path in mapping.items():
        summary = path / "run_summary.json"
        if not summary.exists():
            continue
        metrics = json.loads(summary.read_text())["metrics"]
        rows.append({"candidate": name, "period": period,
                     "mean_rankic": metrics["mean_rankic"],
                     "annualized_excess_return": metrics["annualized_excess_return"],
                     "tracking_error": metrics["tracking_error"],
                     "information_ratio": metrics["information_ratio"],
                     "turnover": metrics["turnover"],
                     "transaction_cost": metrics["transaction_cost"],
                     "run_dir": str(path)})
    return pd.DataFrame(rows)


def main() -> None:
    dev = collect(RUNS, "DEV19_20")
    conf = collect(CONF21, "CONF21")
    table = pd.concat([dev, conf], ignore_index=True)
    table.to_csv(s2b_common.B / "b_summary.csv", index=False)
    manifest = {
        "stage": "B",
        "frozen_at": "2026-10-05",
        "reference": "REF-corrected (platform semantics + fold-local weights + CORR-1)",
        "working_reference_after_B1": "REF-corrected, CatBoost weight 0.5",
        "runs": {row["candidate"] + "_" + row["period"]: row["run_dir"]
                 for row in table.to_dict("records")},
        "note": "every run is a separate directory with a unique run ID; nothing overwritten",
    }
    (s2b_common.B / "b_manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 200)
    print(table.drop(columns=["run_dir"]).to_string(index=False))
    print(json.dumps(manifest["runs"], indent=2))


if __name__ == "__main__":
    main()
