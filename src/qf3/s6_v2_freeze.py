"""§C1/C2: complete executable freeze identity and a self-checking verifier.

The manifest binds every prediction-determining artefact (weights, all
preprocessing, ordered features/design/channels, training/selection code, date
boundaries and the dependency lock), carries the structured adoption rule, and
records both a canonical JSON hash and the manifest file's own SHA-256.
``verify`` refuses: any changed file, any missing required file, and any manifest
whose content no longer matches its recorded canonical hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .constants import F0_COLUMNS, G1_COLUMNS, OUTER_WINDOWS
from .s6_fusion import GRAPH

# TCN input constants, duplicated here so the freeze verifier runs without torch
# (kept in sync with `s6_v2_tcn.STEPS/PRICE_POINTS/CHANNELS`).
STEPS, PRICE_POINTS, CHANNELS = 20, 21, 8

ROOT = Path(".")
S4 = Path("artifacts/stage4_verified_ic")
V2 = S4 / "v2"
SEEDS = (20261005, 20261017, 20261029)

CODE_FILES = (
    ("fusion_source", "src/qf3/s6_fusion.py"),
    ("tcn_source", "src/qf3/s6_v2_tcn.py"),
    ("tabm_trainer", "src/qf3/s6_v2_tabm_variants.py"),
    ("tabm_architecture", "src/qf3/s4_tabm_wrapper_src.py"),
    ("deploy_trainer", "src/qf3/s6_v2_deploy_models.py"),
    ("feature_definition", "src/qf3/s6_v2_features.py"),
    ("static_feature_code", "src/qf3/featurelib.py"),
    ("constants_and_windows", "src/qf3/constants.py"),
    ("fold_definitions", "src/qf3/cv.py"),
    ("completion_report_entry", "src/qf3/s6_v2_completion.py"),
    ("freeze_verifier", "src/qf3/s6_v2_freeze.py"),
)
LOCK_FILES = (("dependency_lock", "requirements.lock"),
              ("train_requirements", "artifacts/stage4_verified_ic/deps/V1_train_requirements.txt"))

ADOPTION_RULE = {
    "order": ["TCN_W20_a0.25", "V1_FIXED_ANCHOR"],
    "primary_comparison": {"candidate": "TCN_W20_a0.25", "reference": "V1_FIXED_ANCHOR"},
    "support": "candidate - V1 year dIC > 0 AND both halves >= -0.001 AND the same holds vs TREE",
    "fallback_check": "if the candidate fails, check V1 - TREE with the same condition",
    "controls": {"C1_fixed": "deploy_2021cutoff catboost x3",
                 "TREE_structure": "0.5 x L5 x3 + 0.5 x x3 C1"},
    "if_all_fail": "keep the existing successfully submitted version; do not re-tune weights/seed",
    "intervals": {"blocks": [10, 20, 40], "draws": 2000, "seed": 20261005,
                  "rule": "per-year resampling, each year keeps its day count"},
    "ir_role": "report only, never decides",
    "recency_warning": ("the three-member candidate keeps a half-year starting-point "
                        "sensitivity warning; the B4 automatic-elimination clause was "
                        "withdrawn by the main agent and is recorded, not passed"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items()
               if key not in ("manifest_hash", "manifest_file_sha256")}
    return _digest_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build() -> dict:
    entries, seen = [], set()

    def add(role: str, path: Path, required: bool = True) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        if not path.exists():
            if required:
                raise FileNotFoundError(f"required freeze artefact missing: {path}")
            entries.append({"role": role, "path": key, "missing": True})
            return
        entries.append({"role": role, "path": key, "bytes": path.stat().st_size,
                        "sha256": _sha(path)})

    for seed in SEEDS:
        add("c1_weight", V2 / f"deploy_2021cutoff/catboost_c1_seed{seed}.cbm")
        add("tabm_weight", V2 / f"deploy_2021cutoff/tabm_seed{seed}.safetensors")
        add("tabm_design", V2 / f"deploy_2021cutoff/design_seed{seed}.json")
        add("l5_weight", V2 / f"deploy_2021cutoff/l5_seed{seed}.txt")
        add("tcn_weight", V2 / f"tcn_rev1_tcn_s{seed}_RES22/model.safetensors")
        add("tcn_preprocessing", V2 / f"tcn_rev1_tcn_s{seed}_RES22/preprocessing.json")
    for seed in (20261005,):
        add("c1_curve", V2 / f"deploy_2021cutoff/catboost_c1_seed{seed}__curve.parquet")
        add("tabm_curve", V2 / f"deploy_2021cutoff/inner_curve_seed{seed}.parquet")
    for role, path in CODE_FILES:
        add(role, ROOT / path)
    for role, path in LOCK_FILES:
        add(role, ROOT / path)
    manifest = {
        "created": "2026-10-07 (Asia/Shanghai)",
        "graph": GRAPH,
        "families": {"c1": {"seeds": list(SEEDS), "weight_pattern": "catboost_c1_seed{seed}.cbm",
                            "preprocessing": "none (tree model, nan_mode=Min)"},
                     "tabm": {"seeds": list(SEEDS), "weight_pattern": "tabm_seed{seed}.safetensors",
                              "preprocessing_pattern": "design_seed{seed}.json"},
                     "tcn": {"seeds": list(SEEDS), "weight_pattern": "tcn_rev1_tcn_s{seed}_RES22/model.safetensors",
                             "preprocessing_pattern": "tcn_rev1_tcn_s{seed}_RES22/preprocessing.json"}},
        "weight_vectors": {"V1_FIXED_ANCHOR": {"c1": 0.5, "tabm": 0.5},
                           "TCN_W20_a0.25": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25}},
        "features": {"F0": list(F0_COLUMNS), "G1": list(G1_COLUMNS),
                     "ordered_61": list(F0_COLUMNS) + list(G1_COLUMNS),
                     "ordered_61_sha256": _digest_text("\n".join(list(F0_COLUMNS) + list(G1_COLUMNS))),
                     "design_order": "2 x 61 blocks of (scaled value, missing mask) then 19 G1 daily ranks",
                     "design_141_sha256": _digest_text("2x61_scaled_mask + 19_g1_ranks")},
        "tcn_channels": {"order": ["close-close log return", "open-close log return",
                                   "high/low amplitude", "close position", "log1p volume",
                                   "log1p amount", "suspension flag", "presence mask"],
                         "steps": STEPS, "price_points": PRICE_POINTS, "channels": CHANNELS,
                         "channel_order_sha256": _digest_text(
                             "|".join(["cc_ret", "oc_ret", "hl_amp", "close_pos",
                                       "log_volume", "log_amount", "suspend", "mask"]))},
        "training_config": {"catboost": "depth8/l2=15 RMSE lr0.03 3000-cap 25-tree inner IC selection",
                            "tabm": "k8/width128/2 blocks/dropout0.1 AdamW lr1e-3 wd1e-3 clip1 batch8days max100/patience12",
                            "tcn": "8ch/20 steps/32ch/2 residual blocks/dilation 1,2 AdamW lr1e-3 wd1e-3 clip1 batch4days max80/patience10",
                            "selection": "inner daily-Spearman selection (tree: 25-tree checkpoints, near-opt 0.0005 earliest; NN: best validation epoch with patience)",
                            "train_label_cutoff": 20211231,
                            "windows": OUTER_WINDOWS},
        "adoption_rule": ADOPTION_RULE,
        "files": entries,
    }
    manifest["required_paths"] = sorted(entry["path"] for entry in entries)
    manifest["manifest_hash"] = _canonical(manifest)
    return manifest


def verify(manifest: dict, root: Path = ROOT) -> dict:
    problems = []
    if manifest.get("manifest_hash") != _canonical(manifest):
        problems.append({"path": "<manifest>", "issue": "canonical hash mismatch"})
    listed = sorted(entry["path"] for entry in manifest["files"])
    required = manifest.get("required_paths")
    if required is None:
        problems.append({"path": "<manifest>", "issue": "required_paths missing"})
    elif listed != sorted(required):
        problems.append({"path": "<manifest>", "issue": "file set differs",
                         "missing": sorted(set(required) - set(listed)),
                         "extra": sorted(set(listed) - set(required))})
    for entry in manifest["files"]:
        path = root / entry["path"]
        if entry.get("missing"):
            problems.append({"path": entry["path"], "issue": "required artefact marked missing"})
            continue
        if not path.exists():
            problems.append({"path": entry["path"], "issue": "missing"})
            continue
        if _sha(path) != entry["sha256"]:
            problems.append({"path": entry["path"], "issue": "sha256 mismatch"})
        if path.stat().st_size != entry["bytes"]:
            problems.append({"path": entry["path"], "issue": "size mismatch"})
    return {"checked": len(manifest["files"]), "problems": problems,
            "passed": not problems}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", default="")
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--out", default=str(V2 / "v2_completion_frozen_manifest_v2.json"))
    args = parser.parse_args()
    if args.build:
        manifest = build()
        Path(args.out).write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"manifest": args.out, "manifest_hash": manifest["manifest_hash"],
                          "file_sha256": _sha(args.out), "files": len(manifest["files"])},
                         indent=2))
        return
    if args.verify:
        manifest = json.loads(Path(args.verify).read_text())
        result = verify(manifest)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.fixtures:
        manifest = build()
        results = {}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for entry in manifest["files"]:
                target = base / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / entry["path"], target)
            results["clean_copy_passes"] = verify(manifest, base)["passed"]
            weight = next(e for e in manifest["files"] if e["role"] == "tabm_weight")
            target = base / weight["path"]
            payload = bytearray(target.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            target.write_bytes(bytes(payload))
            results["modified_file_rejected"] = not verify(manifest, base)["passed"]
            shutil.copy(ROOT / weight["path"], target)

            tampered = json.loads(json.dumps(manifest))
            tampered["training_config"]["train_label_cutoff"] = 20991231
            results["tampered_config_rejected"] = not verify(tampered, base)["passed"]

            dropped = json.loads(json.dumps(manifest))
            dropped["files"] = [e for e in dropped["files"] if e["role"] != "tcn_preprocessing"]
            results["deleted_required_file_rejected"] = not verify(dropped, base)["passed"]

            missing_declared = json.loads(json.dumps(manifest))
            for entry in missing_declared["files"]:
                if entry["role"] == "tcn_preprocessing":
                    entry.clear()
                    entry.update({"role": "tcn_preprocessing",
                                  "path": "artifacts/stage4_verified_ic/v2/does_not_exist.json",
                                  "missing": True})
            missing_declared["required_paths"] = sorted(
                e["path"] for e in missing_declared["files"])
            missing_declared["manifest_hash"] = _canonical(missing_declared)
            results["missing_marked_entry_rejected"] = not verify(missing_declared, base)["passed"]
        results["all_rejected"] = bool(results["clean_copy_passes"] and all(
            value for key, value in results.items() if key != "clean_copy_passes"))
        Path(V2 / "v2_completion_freeze_fixtures_v2.json").write_text(
            json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return
    raise SystemExit("choose --build, --verify <file> or --fixtures")


if __name__ == "__main__":
    main()
