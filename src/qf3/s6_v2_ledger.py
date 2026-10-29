"""§A4: machine-generated ledger with four separate layers and asserted counts.

Layers: recipes / runs / complete weight vectors / access years.  Every count is
computed from the detail lists; mismatches raise instead of being written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .s6_fusion import push_down_alpha

V2 = Path("artifacts/stage4_verified_ic/v2")
S4 = Path("artifacts/stage4_verified_ic")
SEEDS = (20261005, 20261017, 20261029)
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
PARENT = {"c1": 0.5, "tabm": 0.5}
SCREEN_MEMBERS = ("C2", "C3", "tabm_dropout0.2", "tabm_corr_loss", "resmlp_n0",
                  "tcn", "static", "shuffle")


def recipes() -> list[dict]:
    rows = [
        {"id": "C2_depth6", "family": "catboost", "status": "reused_existing_predictions"},
        {"id": "C3_depth8_l2_30", "family": "catboost", "status": "reused_existing_predictions"},
        {"id": "tabm_dropout0.2", "family": "tabm", "status": "new"},
        {"id": "tabm_corr_loss", "family": "tabm", "status": "new"},
        {"id": "F1_features", "family": "features", "status": "new"},
        {"id": "F2_features", "family": "features", "status": "new"},
        {"id": "F3_features", "family": "features", "status": "new"},
        {"id": "F123_joint", "family": "features", "status": "history_not_extended"},
        {"id": "TCN_W20", "family": "tcn", "status": "new"},
        {"id": "TCN_static_control", "family": "tcn", "status": "new"},
        {"id": "TCN_time_shuffle", "family": "tcn", "status": "diagnostic_only"},
        {"id": "tabm_recent504", "family": "tabm", "status": "new_no_full_artifacts"},
        {"id": "resmlp_n0", "family": "resmlp", "status": "reused_and_recomputed_v1"},
    ]
    return rows


def runs() -> list[dict]:
    rows = []
    for path in sorted(V2.glob("tcn_rev1_tcn_s*_*")):
        rows.append({"kind": "tcn_train", "path": str(path)})
    for path in sorted(V2.glob("tcn_rev1_static_s*_*")):
        rows.append({"kind": "tcn_static_train", "path": str(path)})
    for path in sorted(V2.glob("tcn_rev1_shuffle_s*_*")):
        rows.append({"kind": "tcn_shuffle_diagnostic", "path": str(path)})
    for path in sorted(V2.glob("halfyear_2*H2_s*")):
        rows.append({"kind": "halfyear_control", "path": str(path)})
    for path in sorted(V2.glob("features_rev1/*")):
        rows.append({"kind": "feature_ablation", "path": str(path)})
    for path in sorted(V2.glob("tabm_mse_*")) + sorted(V2.glob("tabm_recent504_*")):
        rows.append({"kind": "tabm_variant", "path": str(path)})
    for superseded in sorted(V2.glob("halfyear_*__superseded_mixedseeds")):
        rows.append({"kind": "superseded_run", "path": str(superseded)})
    if (V2 / "deploy_2021cutoff").exists():
        rows.append({"kind": "deploy_2021cutoff", "path": str(V2 / "deploy_2021cutoff")})
    return rows


def vectors() -> list[dict]:
    rows = []
    for member in SCREEN_MEMBERS:
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue
            weights = push_down_alpha(PARENT, member, alpha)
            rows.append({"purpose": "member_screening", "member": member, "alpha": alpha,
                         "weights": weights})
    for alpha in ALPHAS:
        rows.append({"purpose": "tcn_final_selection",
                     "weights": push_down_alpha(PARENT, "tcn", alpha)})
    rows.append({"purpose": "erroneous_ablation_0.5_0.5_drop_c1",
                 "weights": {"tabm": 0.5, "tcn": 0.5}, "status": "error_kept"})
    rows.append({"purpose": "erroneous_ablation_0.5_0.5_drop_tabm",
                 "weights": {"c1": 0.5, "tcn": 0.5}, "status": "error_kept"})
    rows.append({"purpose": "corrected_ablation_drop_c1",
                 "weights": {"tabm": 0.6, "tcn": 0.4}})
    rows.append({"purpose": "corrected_ablation_drop_tabm",
                 "weights": {"c1": 0.6, "tcn": 0.4}})
    rows.append({"purpose": "corrected_ablation_drop_tcn_v1", "weights": {"c1": 0.5, "tabm": 0.5}})
    rows.append({"purpose": "frozen_candidate", "weights": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_completion_ledger_v2.json"))
    args = parser.parse_args()
    recipe_rows, run_rows, vector_rows = recipes(), runs(), vectors()
    registry = json.loads((S4 / "access_registry.json").read_text())
    canonical = {}
    for row in vector_rows:
        key = json.dumps({k: round(v, 6) for k, v in sorted(row["weights"].items())},
                         separators=(",", ":"))
        canonical.setdefault(key, []).append(row["purpose"])
    counts = {
        "recipes_total": len(recipe_rows),
        "recipes_used_for_selection": sum(1 for r in recipe_rows
                                          if r["status"] not in ("history_not_extended",
                                                                 "diagnostic_only")),
        "recipes_new": sum(1 for r in recipe_rows if r["status"] == "new"),
        "runs_total": len(run_rows),
        "vectors_rows": len(vector_rows),
        "vectors_distinct": len(canonical),
        "vectors_for_final_selection": sum(1 for r in vector_rows
                                           if r["purpose"] == "tcn_final_selection"),
        "access_entries_2021": len(registry["year_2021"]),
        "access_entries_2022": len(registry["year_2022"]),
    }
    assert counts["recipes_used_for_selection"] <= counts["recipes_total"]
    assert counts["vectors_distinct"] <= counts["vectors_rows"]
    assert counts["vectors_distinct"] <= 40, "joint-fusion vector budget exceeded"
    report = {"counts": counts, "recipes": recipe_rows, "runs": run_rows,
              "vectors": vector_rows,
              "distinct_vector_identities": canonical,
              "access_years": {"2021": len(registry["year_2021"]),
                               "2022_performance": "not opened"},
              "notes": [
                  "member screening vectors and final-selection vectors are counted separately",
                  "the erroneous 0.5/0.5 ablations are kept as recorded attempts",
                  "tabm_recent504 has predictions+manifest only (no model/preprocessing/curve)",
                  "F123 is history only and was not extended"]}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
