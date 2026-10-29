"""S1 finalisation: per-year IC, paired intervals, registry and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

S3 = Path("artifacts/stage3_ic_first")
PRED = S3 / "predictions"
A_PRED = Path("artifacts/stage2_ic_ir/a_revision1/predictions")
B_PRED = Path("artifacts/stage2_ic_ir/b/predictions")
D_PRED = Path("artifacts/stage2_ic_ir/a_revision1/predictions")

SOURCES = {
    "R_SINGLE_L0P5": {w: A_PRED / f"ref_corrected_lgbm__{w}.parquet"
                      for w in ("DEV19", "DEV20", "CONF21")},
    "R_WORK_cat0.5": {w: B_PRED / f"ref_corrected_cat0.5__{w}.parquet"
                      for w in ("DEV19", "DEV20", "CONF21")},
    "R_DEPLOY": {w: D_PRED / f"origmodel_realinput__{w}.parquet"
                 for w in ("DEV19", "DEV20", "CONF21")},
    "MAIN_SINGLE_L3": {w: PRED / f"L3_L0_20261005__{w}.parquet"
                       for w in ("DEV19", "DEV20", "CONF21")},
    "MAIN_SEED_ENSEMBLE": {w: PRED / f"MAIN_SEED_ENSEMBLE__{w}.parquet"
                           for w in ("DEV19", "DEV20", "CONF21")},
}


def rankic(mapping: dict, windows) -> pd.DataFrame:
    frames = [pd.read_parquet(mapping[w]) for w in windows]
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    merged = pd.concat(frames, ignore_index=True).merge(
        panel[["date", "code", "future_return"]], on=["date", "code"],
        how="left", validate="one_to_one")
    return metrics.daily_rankic(merged)


def main() -> None:
    report = {"ic_by_year": {}, "paired": {}}
    for label, mapping in SOURCES.items():
        row = {}
        for window, year in (("DEV19", 2019), ("DEV20", 2020), ("CONF21", 2021)):
            rk = rankic(mapping, (window,))
            row[year] = metrics.summarize_rankic(rk)["mean_rankic"]
        row["dev_pooled"] = metrics.summarize_rankic(rankic(mapping, ("DEV19", "DEV20")))["mean_rankic"]
        row["all_pooled"] = metrics.summarize_rankic(
            rankic(mapping, ("DEV19", "DEV20", "CONF21")))["mean_rankic"]
        report["ic_by_year"][label] = row

    pairs = [("MAIN_SINGLE_L3", "R_SINGLE_L0P5"), ("MAIN_SINGLE_L3", "R_WORK_cat0.5"),
             ("MAIN_SEED_ENSEMBLE", "R_WORK_cat0.5"), ("MAIN_SEED_ENSEMBLE", "R_SINGLE_L0P5")]
    for candidate, reference in pairs:
        for windows, tag in ((("DEV19", "DEV20"), "dev"),
                             (("CONF21",), "conf21"),
                             (("DEV19", "DEV20", "CONF21"), "all")):
            stat = metrics.paired_ic_difference(
                rankic(SOURCES[candidate], windows), rankic(SOURCES[reference], windows),
                block=20, n_boot=2000)
            report["paired"][f"{candidate}_vs_{reference}_{tag}"] = stat

    registry = []
    for line in (S3 / "experiments.jsonl").read_text().splitlines():
        if line.strip():
            registry.append(json.loads(line))
    pd.DataFrame(registry).to_csv(S3 / "experiments.csv", index=False)
    manifest = {
        "stage": "S0+S1",
        "identity": {
            "R_SINGLE": "REF-corrected pipeline, F0+G1 (L0), LightGBM P5 (63/500/λ2=20)",
            "R_WORK": "REF-corrected daily rank fusion, CatBoost weight 0.5",
            "R_DEPLOY": "v1 year models + real inputs + corrected daily fusion (bridge)",
            "MAIN_SINGLE_v1": "L0, LightGBM 63/300/λ2=20 (L3), seed 20261005",
            "MAIN_SEED_ENSEMBLE_v1": "L3 recipe, mean of per-day ranks over seeds "
                                     "20261005/20261017/20261029",
        },
        "official_identity": "strong_baseline_v1.zip only "
                             "(sha256 b34712933d952da40ff903804096e0f104ad501b598d643b335a7b31dc0f87c5)",
        "ir_policy": "IR is computed and recorded but is NOT used in any S0/S1 decision",
        "registry_rows": len(registry),
    }
    (S3 / "s1_manifest.json").write_text(json.dumps(manifest, indent=2))
    (S3 / "s1_finalize.json").write_text(json.dumps(report, indent=2, default=str))
    print(pd.DataFrame(report["ic_by_year"]).T.to_string())
    for key, value in report["paired"].items():
        print(key, "dIC=%.5f" % value["mean_diff"],
              "95%%CI", [round(x, 5) for x in value["ci95"]], "n", value["n"])


if __name__ == "__main__":
    main()
