"""S0: engine-alignment checks, fixed yearly decomposition and corrected tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

S3 = Path("artifacts/stage3_ic_first")
B = Path("artifacts/stage2_ic_ir/b")


def engine_alignment(run_dir: Path) -> dict:
    """Recomputed point estimates from daily returns must equal the engine output."""
    metrics_json = json.loads((run_dir / "metrics.json").read_text())
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    p = pd.to_numeric(equity["portfolio_return"], errors="coerce").to_numpy()
    b = pd.to_numeric(equity["benchmark_return"], errors="coerce").to_numpy()
    n = len(p)
    ann_p = float(np.prod(1.0 + p) ** (252.0 / n) - 1.0)
    ann_b = float(np.prod(1.0 + b) ** (252.0 / n) - 1.0)
    te = float(np.std(p - b, ddof=1) * np.sqrt(252))
    return {
        "run": run_dir.name,
        "recomputed_excess": ann_p - ann_b,
        "engine_excess": metrics_json["annualized_excess_return"],
        "abs_diff_excess": abs((ann_p - ann_b)
                               - metrics_json["annualized_excess_return"]),
        "recomputed_ir": (ann_p - ann_b) / te,
        "engine_ir": metrics_json["information_ratio"],
        "abs_diff_ir": abs((ann_p - ann_b) / te - metrics_json["information_ratio"]),
        "active_return_identity_ok": bool(np.allclose(
            pd.to_numeric(equity["active_return"], errors="coerce").to_numpy(),
            p - b, atol=1e-12)),
    }


def yearly_from_returns(run_dir: Path) -> dict:
    """Calendar years compounded from that year's own daily returns (first year =
    account start), never from the first day's closing NAV."""
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    equity["year"] = pd.to_datetime(equity["date"]).dt.year
    out = {}
    for year, block in equity.groupby("year"):
        p = pd.to_numeric(block["portfolio_return"], errors="coerce").to_numpy()
        b = pd.to_numeric(block["benchmark_return"], errors="coerce").to_numpy()
        n = len(p)
        if n < 2:
            continue
        ann_p = float(np.prod(1.0 + p) ** (252.0 / n) - 1.0)
        ann_b = float(np.prod(1.0 + b) ** (252.0 / n) - 1.0)
        te = float(np.std(p - b, ddof=1) * np.sqrt(252))
        out[int(year)] = {"days": int(n), "annualized_excess_return": ann_p - ann_b,
                          "tracking_error": te,
                          "information_ratio": (ann_p - ann_b) / te if te else None}
    return out


def corrected_b_tables() -> dict:
    """Fix the misreported B3 row and add the R_DEPLOY bridge column."""
    summary = pd.read_csv(B / "b_summary.csv")
    ref_dev = summary.loc[(summary["candidate"] == "REF_corrected_cat0.5")
                          & (summary["period"] == "DEV19_20")].iloc[0]
    ref_conf = summary.loc[(summary["candidate"] == "REF_corrected_cat0.5")
                           & (summary["period"] == "CONF21")].iloc[0]
    rows = []
    for candidate in ("B2_seed_average", "B3_lam0.25", "B3_lam0.25_a0.2",
                      "REF_legacy_v1"):
        dev = summary.loc[(summary["candidate"] == candidate)
                          & (summary["period"] == "DEV19_20")]
        conf = summary.loc[(summary["candidate"] == candidate)
                           & (summary["period"] == "CONF21")]
        row = {"candidate": candidate}
        if len(dev):
            row.update({
                "dev_delta_ic_vs_R_WORK": float(dev.iloc[0]["mean_rankic"]
                                                - ref_dev["mean_rankic"]),
                "dev_delta_ir_vs_R_WORK": float(dev.iloc[0]["information_ratio"]
                                                - ref_dev["information_ratio"]),
                "dev_delta_excess_pp_vs_R_WORK": float(
                    100 * (dev.iloc[0]["annualized_excess_return"]
                           - ref_dev["annualized_excess_return"])),
            })
        if len(conf):
            row.update({
                "conf21_delta_ic_vs_R_WORK": float(conf.iloc[0]["mean_rankic"]
                                                   - ref_conf["mean_rankic"]),
                "conf21_delta_ir_vs_R_WORK": float(conf.iloc[0]["information_ratio"]
                                                   - ref_conf["information_ratio"]),
                "conf21_delta_excess_pp_vs_R_WORK": float(
                    100 * (conf.iloc[0]["annualized_excess_return"]
                           - ref_conf["annualized_excess_return"])),
            })
        rows.append(row)
    return {"rows": rows,
            "note": ("IR columns are recorded only; per user instruction this stage "
                     "does not use IR for any selection or promotion decision")}


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    runs = {
        "R_WORK_dev": B / "backtests" / "b1_bt_cat0.5_dev19_20",
        "R_WORK_conf21": B / "backtests" / "b1_bt_cat0.5_conf21",
        "seed_ensemble_dev": S3 / "backtests" / "s3_bt_seed_ensemble_dev19_20",
        "seed_ensemble_conf21": S3 / "backtests" / "s3_bt_seed_ensemble_conf21",
        "R_DEPLOY_dev": Path("artifacts/stage2_ic_ir/a_revision1/backtests/"
                             "rev1_bt_realinput_dev19_20"),
    }
    report = {"engine_alignment": {}, "yearly": {}, "corrected_b_tables": {},
              "ic_panel": {}}
    for name, path in runs.items():
        if path.exists():
            report["engine_alignment"][name] = engine_alignment(path)
            report["yearly"][name] = yearly_from_returns(path)
    report["corrected_b_tables"] = corrected_b_tables()

    # panel-based IC for the corrected seed ensemble vs R_WORK and R_DEPLOY
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    sources = {
        "seed_ensemble_fixed": {w: S3 / f"seed_ensemble_fixed__{w}.parquet"
                                for w in ("DEV19", "DEV20", "CONF21")},
        "R_WORK": {w: B / "predictions" / f"ref_corrected_cat0.5__{w}.parquet"
                   for w in ("DEV19", "DEV20", "CONF21")},
        "R_DEPLOY": {w: Path("artifacts/stage2_ic_ir/a_revision1/predictions") /
                     f"origmodel_realinput__{w}.parquet"
                     for w in ("DEV19", "DEV20", "CONF21")},
    }
    for label, mapping in sources.items():
        for windows, tag in ((("DEV19", "DEV20"), "dev"), (("CONF21",), "conf21")):
            frames = [pd.read_parquet(mapping[w]) for w in windows]
            merged = pd.concat(frames, ignore_index=True).merge(
                panel[["date", "code", "future_return"]], on=["date", "code"],
                how="left", validate="one_to_one")
            summary = metrics.summarize_rankic(metrics.daily_rankic(merged))
            report["ic_panel"][f"{label}_{tag}"] = summary["mean_rankic"]
    (S3 / "s0_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["engine_alignment"], indent=2))
    print(json.dumps(report["yearly"], indent=2))
    print(json.dumps(report["ic_panel"], indent=2))
    print(json.dumps(report["corrected_b_tables"], indent=2))


if __name__ == "__main__":
    main()
