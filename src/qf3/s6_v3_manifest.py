"""V3 §A4: persistent builder for the confirmation manifest.

Extends the approved V2 manifest with the online-input, target/weight, IO,
loader, preflight and L5-training sources plus the control structures and the
structured adoption rule.  Supersedes (and records) the approved V2 manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

V2 = Path("artifacts/stage4_verified_ic/v2")
SUPERSEDED = V2 / "v2_completion_frozen_manifest_v2.json"
EXTRA = [("online_input_validation", "src/qf3/s6_v2_input_acceptance.py"),
         ("target_and_weights", "src/qf3/s2r_train.py"),
         ("data_reading", "src/qf3/dataio.py"),
         ("run_dir_policy", "src/qf3/s6_io.py"),
         ("confirmation_loader", "src/qf3/s6_v3_models.py"),
         ("preflight_gate", "src/qf3/s6_v3_preflight.py"),
         ("confirmation_entry", "src/qf3/s6_v3_confirm.py"),
         ("manifest_builder", "src/qf3/s6_v3_manifest.py"),
         ("l5_training_program", "src/qf3/s4_tree_train_src.py"),
         ("annual_fixture_evidence", "artifacts/stage4_verified_ic/v1_anchor__CONF21.parquet"),
         ("annual_fixture_evidence", "artifacts/stage4_verified_ic/v1_anchor__DEV19.parquet"),
         ("annual_fixture_evidence", "artifacts/stage4_verified_ic/v1_anchor__DEV20.parquet"),
         ("deploy_summary", "artifacts/stage4_verified_ic/v2/v2_rev1_deploy_models_deploy.json"),
         ("tcn_deploy_summary", "artifacts/stage4_verified_ic/v2/v2_rev1_tcn_deploy2021.json"),
         ("l5_deploy_summary", "artifacts/stage4_verified_ic/v2/v2_rev1_l5_cutoff2021.json")]
ADOPTION = {
    "order": ["TCN_W20_a0.25", "V1_FIXED_ANCHOR"],
    "primary_comparison": {"candidate": "TCN_W20_a0.25", "reference": "V1_FIXED_ANCHOR"},
    "support": "candidate - V1 year dIC > 0 AND both halves >= -0.001 AND the same vs TREE",
    "fallback_check": "if the candidate fails, check V1 - TREE with the same condition",
    "controls": {"C1_fixed": "catboost x3", "TREE_structure": "0.5 x L5 x3 + 0.5 x C1 x3"},
    "if_all_fail": "keep the existing successfully submitted version; no re-tuning",
    "intervals": {"blocks": [10, 20, 40], "draws": 2000, "seed": 20261005,
                  "rule": "per-year resampling preserving each year's day count"},
    "ir_role": "report only",
    "no_recall": "no feature/seed/alpha/architecture change after opening 2022",
    "two_member_ablation": "not promoted",
}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items()
               if k not in ("manifest_hash", "manifest_file_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def build() -> dict:
    old = json.loads(SUPERSEDED.read_text())
    manifest = {k: v for k, v in old.items() if k not in ("manifest_hash", "required_paths")}
    manifest["created"] = "2026-10-07 (Asia/Shanghai)"
    manifest["supersedes"] = {"path": str(SUPERSEDED), "manifest_hash": old["manifest_hash"],
                              "file_sha256": sha(SUPERSEDED)}
    manifest["adoption_rule"] = ADOPTION
    manifest["candidates"] = {
        "primary": {"id": "TCN_W20_a0.25",
                    "weights": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25},
                    "note": "keeps the half-year starting-point sensitivity warning"},
        "fallback": {"id": "V1_FIXED_ANCHOR", "weights": {"c1": 0.5, "tabm": 0.5}}}
    manifest["controls"] = {
        "C1_fixed": {"members": "catboost_c1_seed*.cbm x3"},
        "TREE_structure": {"rule": "0.5 x L5 family + 0.5 x C1 family",
                           "l5": "l5_seed*.txt x3", "c1": "catboost_c1_seed*.cbm x3"}}
    seen = {e["path"] for e in manifest["files"]}
    for role, path in EXTRA:
        if path in seen:
            continue
        target = Path(path)
        if not target.exists():
            raise SystemExit(f"required freeze artefact missing: {path}")
        manifest["files"].append({"role": role, "path": path,
                                  "bytes": target.stat().st_size, "sha256": sha(target)})
        seen.add(path)
    manifest["required_paths"] = sorted(e["path"] for e in manifest["files"])
    manifest["manifest_hash"] = canonical(manifest)
    return manifest


def main() -> None:
    manifest = build()
    out = V2 / "v3_confirmation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(out), "files": len(manifest["files"]),
                      "unique": len({e['path'] for e in manifest['files']}),
                      "manifest_hash": manifest["manifest_hash"],
                      "file_sha256": sha(out)}, indent=2))


if __name__ == "__main__":
    main()
