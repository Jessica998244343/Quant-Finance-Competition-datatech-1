"""§14.1 layer 3: check that a train.py rerun reproduces the shipped package.

Uses the *shipped* package code for both model directories.  The scorer is
single-day by contract, so it is called once per trading day (complete pool) and
the results are concatenated; the fused result is compared with the frozen
annual anchor as well as with the shipped package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (20261005, 20261017, 20261029)
VOLATILE_METADATA = ("model_guid", "train_finish_time")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {key: _strip_volatile(value) for key, value in obj.items()
                if key not in VOLATILE_METADATA}
    if isinstance(obj, list):
        return [_strip_volatile(value) for value in obj]
    return obj


def catboost_digest(path: Path) -> str:
    """SHA-256 of the CatBoost model JSON without CatBoost's volatile metadata."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(path))
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "model.json"
        model.save_model(str(target), format="json")
        payload = json.loads(target.read_text())
    canonical = json.dumps(_strip_volatile(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_package(root: Path, alias: str):
    spec = importlib.util.spec_from_file_location(
        alias, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return importlib.import_module(f"{alias}.alpha_models").Scorer


def npz_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def daily_rankic(frame: pd.DataFrame) -> tuple[float, int]:
    values = []
    for _, block in frame.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values.append(float(np.corrcoef(y, p)[0, 1]))
    return (float(np.mean(values)) if values else float("nan")), len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="releases/stage4_verified_ic/build/V1_pkg")
    parser.add_argument("--reference", default=None,
                        help="shipped model dir (default: <package>/model)")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--panel", default="artifacts/stage2_ic_ir/panel_platform.parquet")
    parser.add_argument("--anchor", default="artifacts/stage4_verified_ic/v1_anchor__CONF21.parquet")
    parser.add_argument("--window", default="CONF21")
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v1_repro_check.json")
    args = parser.parse_args()

    package = Path(args.package)
    reference = Path(args.reference) if args.reference else package / "model"
    candidate = Path(args.candidate)
    Scorer = load_package(package, "v1_repro_pkg")

    sys.path.insert(0, "src")
    from qf3 import cv  # noqa: E402

    panel = pd.read_parquet(args.panel).sort_values(["date", "code"]).reset_index(drop=True)
    split = cv.fold_split(panel, args.window)
    window = panel.loc[split["predict"]].sort_values(["date", "code"]).reset_index(drop=True)

    ref_scorer, cand_scorer = Scorer(str(reference)), Scorer(str(candidate))
    report = {"reference": str(reference), "candidate": str(candidate),
              "panel": args.panel, "window": args.window,
              "n_predict_rows": int(len(window)),
              "n_days": int(window["date"].nunique()), "seeds": {}}

    report["design_hash_equal"] = _sha(reference / "tabm_design.json") == _sha(
        candidate / "tabm_design.json")
    report["design_hash"] = _sha(reference / "tabm_design.json")
    report["config_equal"] = (json.loads((reference / "tabm_config.json").read_text())
                              == json.loads((candidate / "tabm_config.json").read_text()))

    for seed in SEEDS:
        ref_cbm, cand_cbm = reference / f"catboost_c1_seed{seed}.cbm", candidate / f"catboost_c1_seed{seed}.cbm"
        ref_npz, cand_npz = npz_arrays(reference / f"tabm_seed{seed}.npz"), npz_arrays(candidate / f"tabm_seed{seed}.npz")
        report["seeds"][str(seed)] = {
            "cbm_bytes_identical": _sha(ref_cbm) == _sha(cand_cbm),
            "cbm_tree_content_identical": catboost_digest(ref_cbm) == catboost_digest(cand_cbm),
            "npz_bytes_identical": _sha(reference / f"tabm_seed{seed}.npz") == _sha(
                candidate / f"tabm_seed{seed}.npz"),
            "npz_parameter_max_abs_diff": float(max(
                np.max(np.abs(ref_npz[name].astype(np.float64) - cand_npz[name].astype(np.float64)))
                for name in ref_npz)),
        }

    raw_names = [f"{family}_{seed}" for family in ("c1", "tabm") for seed in SEEDS]
    raw_diff = {name: 0.0 for name in raw_names}
    atom_max = {"c1": 0.0, "tabm": 0.0}
    fused_max = 0.0
    ranks_identical = True
    ref_scores, cand_scores = [], []
    for _, day in window.groupby("date", sort=True):
        day = day.reset_index(drop=True)
        ref_raw, cand_raw = ref_scorer.member_raw(day), cand_scorer.member_raw(day)
        if set(ref_raw) != set(raw_names) or set(cand_raw) != set(raw_names):
            raise RuntimeError(f"unexpected atom names: {sorted(set(ref_raw) | set(cand_raw))}")
        for name in raw_names:
            raw_diff[name] = max(raw_diff[name], float(np.max(np.abs(ref_raw[name] - cand_raw[name]))))
        ref_atoms, cand_atoms = ref_scorer.atom_scores(day), cand_scorer.atom_scores(day)
        for family in ("c1", "tabm"):
            atom_max[family] = max(atom_max[family],
                                   float(np.max(np.abs(ref_atoms[family] - cand_atoms[family]))))
        ref_score, cand_score = ref_scorer.score(day), cand_scorer.score(day)
        fused_max = max(fused_max, float(np.max(np.abs(ref_score - cand_score))))
        ranks_identical &= bool(np.array_equal(
            pd.Series(ref_score).rank().to_numpy(), pd.Series(cand_score).rank().to_numpy()))
        ref_scores.append(ref_score)
        cand_scores.append(cand_score)
    report["member_raw_max_abs_diff"] = raw_diff
    report["atom_max_abs_diff"] = atom_max
    report["fused_max_abs_diff"] = fused_max
    report["fused_daily_rank_identical"] = bool(ranks_identical)

    ref_frame = window.assign(score=np.concatenate(ref_scores))
    cand_frame = window.assign(score=np.concatenate(cand_scores))
    report["rankic_reference"], report["rankic_days"] = daily_rankic(ref_frame)
    report["rankic_candidate"], _ = daily_rankic(cand_frame)

    anchor = pd.read_parquet(args.anchor)
    merged = ref_frame.merge(anchor.rename(columns={"score": "anchor"}),
                             on=["date", "code"], how="outer", validate="one_to_one")
    report["anchor_rows"] = int(len(anchor))
    report["anchor_keys_match"] = bool(len(merged) == len(anchor) == len(ref_frame)
                                       and merged["score"].notna().all()
                                       and merged["anchor"].notna().all())
    report["max_abs_diff_vs_anchor"] = float(np.nanmax(
        np.abs(merged["score"] - merged["anchor"]))) if report["anchor_keys_match"] else None
    anchor_frame = anchor.merge(window[["date", "code", "future_return"]],
                                on=["date", "code"], how="left", validate="one_to_one")
    report["rankic_anchor"], _ = daily_rankic(anchor_frame)

    mixed = window.reset_index(drop=True)
    report["mixed_date_rejected"] = False
    try:
        ref_scorer.score(mixed)
    except ValueError:
        report["mixed_date_rejected"] = True

    report["passed"] = bool(
        report["design_hash_equal"] and report["config_equal"]
        and all(v["cbm_tree_content_identical"] and v["npz_parameter_max_abs_diff"] == 0.0
                for v in report["seeds"].values())
        and all(value == 0.0 for value in raw_diff.values())
        and all(value == 0.0 for value in atom_max.values())
        and fused_max == 0.0 and ranks_identical
        and report["anchor_keys_match"] and report["max_abs_diff_vs_anchor"] == 0.0
        and abs(report["rankic_reference"] - report["rankic_anchor"]) < 1e-12
        and report["mixed_date_rejected"])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
