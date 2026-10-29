"""V3 §A: preflight checks before any 2022 evaluation.

One validator (`validate_day`) is used by both the normal acceptance and the
negative fixtures, so "the negative cases fail" is proven by the same gate that
accepts the good days.  The verdict is a conjunction; nothing is hand-written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .s6_fusion import FusionContractError, daily_centred_rank
from .s6_v3_models import FEATURES, SEEDS, annual_models, deploy_models
from .s6_v2_tcn import build_sequences

V2 = Path("artifacts/stage4_verified_ic/v2")
S4 = Path("artifacts/stage4_verified_ic")
ATOL, RTOL = 1e-5, 1e-4
CHANGE_DAYS = {"DEV19": (20190617, 20190102), "DEV20": (20200615, 20200102),
               "CONF21": (20210615, 20210104)}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_day(codes, atoms: dict, reference: dict, expected_codes,
                 families: dict | None = None, weights: dict | None = None) -> dict:
    """The single gate: coverage, pool, uniqueness, finiteness, tolerance, ranks."""
    problems = []
    if len(set(codes)) != len(list(codes)):
        problems.append("duplicated codes")
    if set(codes) != set(expected_codes):
        problems.append(f"pool mismatch ({len(set(expected_codes) - set(codes))} missing, "
                        f"{len(set(codes) - set(expected_codes))} extra)")
    for name, values in atoms.items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(list(codes)) or not np.isfinite(values).all():
            problems.append(f"atom {name} invalid (len/finiteness)")
    max_diff = 0.0
    rank_ok = True
    for name, truth in reference.items():
        if name not in atoms:
            problems.append(f"reference {name} missing from atoms")
            continue
        diff = np.abs(np.asarray(atoms[name]) - np.asarray(truth))
        max_diff = max(max_diff, float(np.nanmax(diff)))
        if not np.allclose(atoms[name], truth, atol=ATOL, rtol=RTOL):
            rank_ok &= bool(np.array_equal(
                pd.Series(atoms[name]).rank().to_numpy(),
                pd.Series(truth).rank().to_numpy()))
    # review §6: raw tolerance and rank agreement are separate, both required
    if max_diff > ATOL:
        problems.append(f"raw tolerance exceeded (max {max_diff:.3e} > atol {ATOL})")
    if not rank_ok:
        problems.append("daily ranks differ from the reference")
    if families is not None and weights is not None:
        value = np.zeros(len(list(codes)))
        for name, weight in weights.items():
            value += weight * families[name]
        if not np.isfinite(value).all():
            problems.append("fusion produced non-finite values")
    return {"passed": not problems, "problems": problems, "max_abs_diff": max_diff,
            "rank_ok": bool(rank_ok), "rows": len(list(codes))}


def _reference(window: str, day: int, codes) -> dict:
    out = {}
    for seed in SEEDS:
        c1 = pd.read_parquet(S4 / ".." / "stage3_fusion" / "predictions_cat"
                             / f"C1_{seed}__{window}.parquet")
        out[f"c1_{seed}"] = c1.loc[c1["date"] == day].set_index(
            "code")["score"].reindex(codes).to_numpy()
        tabm = pd.read_parquet(S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}"
                               / "predictions.parquet")
        out[f"tabm_{seed}"] = tabm.loc[tabm["date"] == day].set_index(
            "code")["score"].reindex(codes).to_numpy()
        tcn = pd.read_parquet(V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "predictions.parquet")
        out[f"tcn_{seed}"] = tcn.loc[tcn["date"] == day].set_index(
            "code")["score"].reindex(codes).to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v3_preflight.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    report: dict = {"interpreter": {"python": sys.version.split()[0],
                                    "platform": platform.platform(),
                                    "executable": sys.executable}}
    checks = {}

    import catboost, lightgbm
    report["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                          "torch": torch.__version__, "catboost": catboost.__version__,
                          "lightgbm": lightgbm.__version__,
                          "torch_threads": torch.get_num_threads()}
    checks["environment_importable"] = True

    deploy = deploy_models()
    checks["confirmation_models_loaded"] = bool(
        len(deploy.cat) == len(deploy.tabm) == len(deploy.tcn) == len(deploy.l5) == 3)
    report["confirmation_atoms"] = {"c1": list(deploy.cat), "tabm": list(deploy.tabm),
                                    "tcn": list(deploy.tcn), "l5": list(deploy.l5)}
    dep = json.loads((V2 / "v2_rev1_deploy_models_deploy.json").read_text())["deploy"]
    tcn_dep = json.loads((V2 / "v2_rev1_tcn_deploy2021.json").read_text())
    l5_dep = json.loads((V2 / "v2_rev1_l5_cutoff2021.json").read_text())["l5_cutoff2021"]
    report["cutoffs"] = {"deploy": dep["train_label_cutoff"],
                         "deploy_max_outer_label_end": dep["max_outer_label_end"],
                         "tcn": tcn_dep["train_label_cutoff"], "l5": l5_dep["train_label_cutoff"],
                         "all_le_20211231": all(v <= 20211231 for v in (
                             dep["train_label_cutoff"], tcn_dep["train_label_cutoff"],
                             l5_dep["train_label_cutoff"]))}
    checks["cutoff_le_20211231"] = report["cutoffs"]["all_le_20211231"]
    checks["no_2022_metrics_in_artifacts"] = bool(
        dep["predict_window_evaluated"] is False
        and tcn_dep["outer_metrics_computed"] is False
        and l5_dep["outer_metrics_computed"] is False)

    built = build_sequences()
    sequence_map, codes_by_day = built["sequences"], built["keys"]
    equivalence = {}
    negative = {}
    for window, days in CHANGE_DAYS.items():
        loader = annual_models(window)
        equivalence[window] = {}
        for day in days:
            block = panel.loc[panel["date"] == day].reset_index(drop=True)
            lookup = {code: index for index, code in enumerate(codes_by_day.get(day, []))}
            block = block.loc[block["code"].map(lookup).notna()].reset_index(drop=True)
            rows = np.stack([sequence_map[day][lookup[code]] for code in block["code"]])
            atoms = loader.atoms(block, rows)
            reference = _reference(window, day, block["code"].tolist())
            families = {family: np.mean([daily_centred_rank(atoms[f"{family}_{s}"])
                                         for s in SEEDS], axis=0)
                        for family in ("c1", "tabm", "tcn")}
            result = validate_day(block["code"].tolist(), atoms, reference,
                                  block["code"].tolist(), families,
                                  {"c1": 0.375, "tabm": 0.375, "tcn": 0.25})
            equivalence[window][int(day)] = result
        if window == "CONF21":
            day = days[0]
            block = panel.loc[panel["date"] == day].reset_index(drop=True)
            lookup = {code: index for index, code in enumerate(codes_by_day[day])}
            block = block.loc[block["code"].map(lookup).notna()].reset_index(drop=True)
            rows = np.stack([sequence_map[day][lookup[code]] for code in block["code"]])
            good = loader.atoms(block, rows)
            reference = _reference(window, day, block["code"].tolist())
            broken = {key: value.copy() for key, value in good.items()}
            broken[f"c1_{SEEDS[0]}"][0] = np.nan
            negative["nan_atom"] = validate_day(block["code"].tolist(), broken, reference,
                                                block["code"].tolist())
            negative["missing_stock"] = validate_day(
                block["code"].tolist()[1:], {k: v[1:] for k, v in good.items()},
                {k: v[1:] for k, v in reference.items()}, block["code"].tolist())
            tampered = {key: value.copy() for key, value in reference.items()}
            tampered[f"c1_{SEEDS[0]}"][0] += 0.05
            negative["tampered_reference"] = validate_day(
                block["code"].tolist(), good, tampered, block["code"].tolist())
    checks["loader_equivalence_ok"] = all(
        result["passed"] for window in equivalence for result in equivalence[window].values())
    checks["negative_cases_rejected"] = all(
        not result["passed"] for result in negative.values())
    report["equivalence"] = equivalence
    report["negative_cases"] = negative

    manifest_path = V2 / "v3_confirmation_manifest.json"
    frozen = {}
    if manifest_path.exists():
        from .s6_v2_freeze import verify as verify_freeze
        frozen = verify_freeze(json.loads(manifest_path.read_text()))
    report["confirmation_manifest"] = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
        "sha256": _sha(manifest_path) if manifest_path.exists() else None,
        "manifest_hash": (json.loads(manifest_path.read_text())["manifest_hash"]
                          if manifest_path.exists() else None),
        "freeze_verified": bool(frozen.get("passed")),
        "freeze_problems": frozen.get("problems", [])}
    checks["confirmation_manifest_present"] = manifest_path.exists()
    checks["confirmation_manifest_self_consistent"] = bool(
        report["confirmation_manifest"]["freeze_verified"])

    report["checks"] = checks
    report["passed"] = bool(all(checks.values()))
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"checks": checks, "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
