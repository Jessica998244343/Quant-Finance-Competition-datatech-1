"""Summarise finished reproduction and supplemental original-curve comparisons."""
import sys
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_repro_resume as r


def main():
    binding = r.fingerprint()
    progress = json.loads((r.RUN / "progress.json").read_text())
    assert progress["complete"] and progress["binding"] == binding
    gate = json.loads((r.ROOT / "artifacts/stage4_verified_ic/v3_train_repro.json").read_text())
    assert gate["passed"], "strict gate must pass before emitting the receipt"
    assert Path(gate["repro_dir"]).resolve() == (r.RUN / "model").resolve()
    reference = r.SUB / "model"
    members = []
    for seed in r.original.SEEDS:
        for family in ("c1", "tabm", "tcn"):
            tag = f"{family}_{seed}"
            record = json.loads((r.RUN / "stages" / f"{tag}.json").read_text())
            name = f"catboost_c1_seed{seed}.cbm" if family == "c1" else f"{family}_seed{seed}.npz"
            model = r.RUN / "model" / name
            assert r.verified_file(model, binding)
            row = {"family": family, "seed": seed, "file": name,
                   "selected_rounds": record.get("best_epoch", record.get("selected_trees")),
                   "seconds_final_attempt": record["seconds_this_attempt"],
                   "reference_sha256": r.sha(reference / name), "repro_sha256": r.sha(model)}
            row["file_bytes_identical"] = row["reference_sha256"] == row["repro_sha256"]
            if family == "c1":
                curve_a = r.RUN / "model" / f"catboost_c1_seed{seed}__curve.parquet"
                curve_b = r.ROOT / "artifacts/stage4_verified_ic/v2/full_2022" / curve_a.name
                row["tree_structure_exact"] = gate["trees"][name]["tree_structure_exact"]
            else:
                curve_a = r.RUN / "stages" / f"{tag}_inner_curve.parquet"
                if family == "tabm":
                    curve_b = r.ROOT / "artifacts/stage4_verified_ic/v2/full_2022" / f"inner_curve_seed{seed}.parquet"
                else:
                    curve_b = r.ROOT / "artifacts/stage4_verified_ic/v2" / f"tcn_rev1_tcn_s{seed}_FULL22/inner_curve.parquet"
                with np.load(reference / name, allow_pickle=False) as a, np.load(model, allow_pickle=False) as b:
                    assert set(a.files) == set(b.files)
                    row["parameter_max_abs_diff"] = max(float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files)
            a, b = pd.read_parquet(curve_a), pd.read_parquet(curve_b)
            row["full_selection_curve_exact"] = a.equals(b)
            row["selection_curve_rows"] = len(a)
            row["curve_reference_sha256"] = r.sha(curve_b)
            row["curve_repro_sha256"] = r.sha(curve_a)
            members.append(row)
    receipt = {"completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "binding": binding, "original_zip_sha256": r.sha(r.ROOT / "releases/stage4_verified_ic/V3_full_pkg.zip"),
               "scope": "Full raw-data rebuild and all nine members, via semantically equivalent epoch-resumable audit runner; not one uninterrupted original CLI process.",
               "actual_interruption": "TabM seed20261017 restored at inner epoch8; after-restoration curve verified exact.",
               "strict_gate_passed": True, "members": members,
               "all_selection_curves_exact": all(v["full_selection_curve_exact"] for v in members),
               "end_to_end": gate["end_to_end"]["summary"],
               "negative_controls": gate["negative_controls"],
               "evidence_sha256": {str(p.relative_to(r.ROOT)):r.sha(p) for p in
                    [r.RUN/"identity.json", r.RUN/"resume_equivalence_selftest.json",
                     r.RUN/"real_resume_precheck.json", r.RUN/"full_panel_alignment.json",
                     r.ROOT/"artifacts/stage4_verified_ic/v3_train_repro.json",
                     Path(r.__file__), r.ROOT/"src/qf3/s6_v3_repro_check.py"]}}
    r.write_json(r.RUN / "FINAL_REPRO_RECEIPT.json", receipt)
    print(json.dumps({"strict_gate_passed":True,"all_selection_curves_exact":receipt["all_selection_curves_exact"],
                      "members":len(members),"end_to_end":receipt["end_to_end"]},indent=2))


if __name__ == "__main__":
    main()
