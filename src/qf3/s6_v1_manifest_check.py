"""§6.1: machine-check the V1 package manifest against the actual files.

Checks are grouped so that a missing/extra key, a wrong identity (family, seed,
weight key, fusion graph, training cutoff, validation year) or a mismatched
weight file all fail.  ``all_pass`` is computed from the individual results.
The packaging flow calls :func:`check_package` and aborts on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from .constants import BANNED_COLUMNS, F0_COLUMNS, G1_COLUMNS

FAMILIES = ("c1", "tabm")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed_of(name: str):
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def check_package(meta: dict, model_dir: Path, expected_seeds, expected_cutoff: int,
                  expected_year: int, expected_graph: str) -> dict:
    model_dir = Path(model_dir)
    members = meta.get("members", {})
    names = list(members.get("c1", [])) + list(members.get("tabm", []))
    expected_names = {f"catboost_c1_seed{seed}.cbm" for seed in expected_seeds} | \
                     {f"tabm_seed{seed}.npz" for seed in expected_seeds}
    c1_seeds = [_seed_of(name) for name in members.get("c1", [])]
    tabm_seeds = [_seed_of(name) for name in members.get("tabm", [])]
    declared = meta.get("member_hashes", {})
    actual = {name: (_sha(model_dir / name) if (model_dir / name).exists() else None)
              for name in names}
    weights = meta.get("weights", {})
    features = list(meta.get("features", []))
    expected_features = list(F0_COLUMNS) + list(G1_COLUMNS)
    config_path = model_dir / "tabm_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    design_path = model_dir / "tabm_design.json"
    weight_keys = set(weights)
    return {
        "member_files_exist": bool(names) and all(
            (model_dir / name).exists() for name in names),
        "member_names_exact": set(names) == expected_names,
        "member_hash_keys_exact": set(declared) == set(names) and bool(names),
        "hashes_match_files": all(actual.get(name) == declared.get(name) for name in names),
        "hashes_unique": (len(set(declared.values())) == len(names) == len(
            {value for value in actual.values() if value})),
        "families_exact": set(members) == set(FAMILIES),
        "c1_seed_identity": sorted(seed for seed in c1_seeds if seed is not None)
                              == sorted(expected_seeds),
        "tabm_seed_identity": sorted(seed for seed in tabm_seeds if seed is not None)
                                == sorted(expected_seeds),
        "seed_identity_matches": c1_seeds == tabm_seeds,
        "weight_keys_exact": weight_keys == set(FAMILIES),
        "weights_nonneg_sum1": (bool(weights) and all(value >= 0 for value in weights.values())
                                and abs(sum(weights.values()) - 1.0) < 1e-12),
        "fusion_graph_exact": meta.get("fusion_graph") == expected_graph,
        "fusion_graph_no_extra_family_rank": "family rank" not in str(
            meta.get("fusion_graph", "")).replace("family weight", ""),
        "training_cutoff_exact": int(meta.get("train_label_cutoff", -1)) == int(expected_cutoff),
        "validation_year_exact": int(meta.get("validation_year", -1)) == int(expected_year),
        "preprocessing_cutoff_exact": int(meta.get("preprocessing_fit_cutoff", -1))
                                      == int(expected_cutoff),
        "feature_count_61": len(features) == 61,
        "feature_order_matches_contract": features == expected_features,
        "no_banned_fields": not any(field in features for field in BANNED_COLUMNS),
        "tabm_config_d_in": int(config.get("d_in", 0)) == 2 * len(FEATURES) + len(G1_COLUMNS),
        "tabm_config_architecture": all(config.get(key) == value for key, value in
                                        (("k", 8), ("width", 128), ("blocks", 2),
                                         ("dropout", 0.1))),
        "design_hash_matches": (design_path.exists()
                                and meta.get("design_hash") == _sha(design_path)),
    }


FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)


def all_true(checks: dict) -> bool:
    return bool(checks) and all(checks.values())


def regression_fixtures(meta: dict, model_dir: Path, expected_seeds, expected_cutoff: int,
                        expected_year: int, expected_graph: str) -> dict:
    def fails(mutated: dict) -> bool:
        return not all_true(check_package(mutated, model_dir, expected_seeds,
                                          expected_cutoff, expected_year, expected_graph))

    def clone() -> dict:
        return json.loads(json.dumps(meta))

    fixtures = {}
    missing = clone()
    missing["members"]["tabm"] = missing["members"]["tabm"][:-1]
    fixtures["missing_member_detected"] = fails(missing)
    empty_hashes = clone()
    empty_hashes["member_hashes"] = {}
    fixtures["empty_hash_map_detected"] = fails(empty_hashes)
    wrong_graph = clone()
    wrong_graph["fusion_graph"] = "atom rank -> seed mean -> family rank -> family weight -> final rank"
    fixtures["extra_family_rank_detected"] = fails(wrong_graph)
    wrong_weights = clone()
    wrong_weights["weights"] = {"unknown": 1.0}
    fixtures["unknown_weight_key_detected"] = fails(wrong_weights)
    illegal = clone()
    illegal["weights"] = {"c1": 0.5, "tabm": 0.6}
    fixtures["illegal_weights_detected"] = fails(illegal)
    wrong_cutoff = clone()
    wrong_cutoff["train_label_cutoff"] = 20991231
    fixtures["wrong_training_cutoff_detected"] = fails(wrong_cutoff)
    duplicate = clone()
    first, second = duplicate["members"]["c1"][0], duplicate["members"]["c1"][1]
    duplicate["member_hashes"][second] = duplicate["member_hashes"][first]
    fixtures["duplicate_seed_hash_detected"] = fails(duplicate)
    wrong_features = clone()
    wrong_features["features"] = list(reversed(wrong_features["features"]))
    fixtures["feature_order_detected"] = fails(wrong_features)
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="releases/stage4_verified_ic/build/V1_pkg")
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v1_manifest_check.json")
    parser.add_argument("--expected-cutoff", type=int, default=20201231)
    parser.add_argument("--expected-year", type=int, default=2021)
    parser.add_argument("--expected-graph",
                        default="atom rank -> seed mean -> family weight -> final rank")
    parser.add_argument("--seeds", default="20261005,20261017,20261029")
    args = parser.parse_args()
    model_dir = Path(args.build) / "model"
    meta = json.loads((model_dir / "meta.json").read_text())
    expected_seeds = [int(seed) for seed in args.seeds.split(",")]

    checks = check_package(meta, model_dir, expected_seeds, args.expected_cutoff,
                           args.expected_year, args.expected_graph)
    fixtures = regression_fixtures(meta, model_dir, expected_seeds, args.expected_cutoff,
                                   args.expected_year, args.expected_graph)
    report = {"checks": checks, "negative_fixtures": fixtures,
              "all_pass": all_true(checks) and all_true(fixtures)}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
