"""V3 final-compliance P0-2: strict gate for the ``train.py`` reproduction.

Upgraded after the review (``REVIEW_V3_FINAL_COMPLIANCE_20261010.md``) proved the
first version leaked: a candidate whose TCN standardisation was shifted by +1000
and whose fusion weights were replaced by 1/0/0 still reported
``catboost_all_bitequal=true`` / ``nn_all_within_tolerance=true`` with exit 0.

This gate therefore:

* binds the original ZIP SHA and requires the exact member set
  (C1 x3, TabM x3, TCN x3 over the frozen seeds) plus the three preprocessing
  files and the fusion meta; missing/extra/misnamed members, wrong counts,
  NaN/Inf or shape/dtype anomalies fail immediately;
* compares tree structure (tree count, leaf values, borders) and NN tensors with
  an explicit elementwise tolerance ``abs(a-b) <= atol + rtol*abs(reference)``;
* compares the two model directories end-to-end through the *packaged* scorer on
  a continuous signal span plus cross-year / short-history / constituent-change
  days: 9 atom raws, per-seed daily ranks, family means, fusion weights and the
  final fused score, reporting the max diff, rank-change ratio, key identity and
  the local-IC difference;
* includes five negative controls (perturbed standardisation, changed fusion
  weights, deleted member, perturbed tensor, shuffled feature order) that the
  gate must reject;
* emits one single ``passed`` flag and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ATOL, RTOL = 1e-5, 1e-4
SEEDS = (20261005, 20261017, 20261029)
FAMILIES = ("c1", "tabm", "tcn")
EXPECTED_MEMBERS = {
    "c1": [f"catboost_c1_seed{s}.cbm" for s in SEEDS],
    "tabm": [f"tabm_seed{s}.npz" for s in SEEDS],
    "tcn": [f"tcn_seed{s}.npz" for s in SEEDS],
}
REQUIRED_AUX = ("tabm_design.json", "tcn_design.json", "tcn_channel_stats.json")
TREE_SUFFIXES = ("__curve.parquet", ".complete.json")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elementwise(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_ok": False, "max_abs_diff": float("nan"), "tolerance_pass": False}
    diff = np.abs(left - right)
    return {"shape_ok": True, "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "tolerance_pass": bool(np.isfinite(left).all() and np.isfinite(right).all()
                                   and (diff <= ATOL + RTOL * np.abs(right)).all())}


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def _rank_compare(left, right) -> dict:
    left, right = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    rl, rr = _rank(left), _rank(right)
    changed = int(np.sum(rl != rr))
    return {"rank_identical": bool(changed == 0), "rank_changed_rows": changed,
            "rank_change_ratio": float(changed / rl.size) if rl.size else 0.0,
            "spearman": float(np.corrcoef(rl, rr)[0, 1]) if rl.size > 1 else float("nan")}


def _model_files(root: Path) -> set:
    return {p.name for p in root.iterdir() if p.is_file()}


def identity_checks(reference: Path, candidate: Path, zip_sha: str | None) -> tuple[list, dict]:
    """Member set, counts, meta semantics, aux files; the hard identity gate."""
    problems: list[str] = []
    info: dict = {}
    ref_meta = json.loads((reference / "meta.json").read_text())
    if not (candidate / "meta.json").is_file():
        return ["candidate is missing meta.json"], info
    cand_meta = json.loads((candidate / "meta.json").read_text())

    ref_files, cand_files = _model_files(reference), _model_files(candidate)
    expected = {name for group in EXPECTED_MEMBERS.values() for name in group} | set(REQUIRED_AUX) | {"meta.json"}
    missing = sorted(expected - cand_files)
    if missing:
        problems.append(f"candidate missing packaged files: {missing}")
    extra = sorted(f for f in cand_files - expected
                   if not any(f.endswith(suffix) for suffix in TREE_SUFFIXES))
    if extra:
        problems.append(f"candidate has unexpected files: {extra}")
    for name in cand_files:
        if name.endswith(".complete.json"):
            receipt = json.loads((candidate / name).read_text())
            target = candidate / name.removesuffix(".complete.json")
            if not target.is_file() or receipt.get("sha256") != _sha(target):
                problems.append(f"invalid completion receipt: {name}")
    absent_in_ref = sorted(cand_files - ref_files)
    info["candidate_only_files"] = absent_in_ref

    for family, names in EXPECTED_MEMBERS.items():
        listed = list(cand_meta.get("members", {}).get(family, []))
        if sorted(listed) != sorted(names):
            problems.append(f"{family}: meta members {listed} != expected {names}")
        present = [n for n in names if n in cand_files]
        if len(present) != 3:
            problems.append(f"{family}: expected 3 members, found {len(present)}")
    if cand_meta.get("members") != ref_meta.get("members"):
        problems.append("meta members differ from the frozen reference")
    if cand_meta.get("weights") != ref_meta.get("weights"):
        problems.append(f"fusion weights differ: {cand_meta.get('weights')} vs {ref_meta.get('weights')}")
    if abs(sum(float(v) for v in cand_meta.get("weights", {}).values()) - 1.0) > 1e-9:
        problems.append("fusion weights do not sum to 1")
    if list(cand_meta.get("features", [])) != list(ref_meta.get("features", [])):
        problems.append("feature list/order differs from the frozen reference")
    if cand_meta.get("train_label_cutoff") != ref_meta.get("train_label_cutoff"):
        problems.append("train_label_cutoff differs from the frozen reference")
    if cand_meta.get("fusion_graph") != ref_meta.get("fusion_graph"):
        problems.append("fusion graph differs from the frozen reference")
    for family in EXPECTED_MEMBERS.values():
        for name in family:
            path = candidate / name
            if path.exists() and cand_meta.get("member_hashes", {}).get(name) != _sha(path):
                problems.append(f"candidate meta hash does not bind {name}")
    if zip_sha is not None and cand_meta.get("members") != ref_meta.get("members"):
        problems.append("zip-linked member identity mismatch")
    for name in REQUIRED_AUX:
        if not (candidate / name).is_file():
            problems.append(f"missing preprocessing/meta file {name}")
    info["weights"] = cand_meta.get("weights")
    info["features"] = len(cand_meta.get("features", []))
    return problems, info


def preprocessing_checks(reference: Path, candidate: Path) -> tuple[list, dict]:
    problems: list[str] = []
    info: dict = {}
    for name in REQUIRED_AUX:
        ref = json.loads((reference / name).read_text())
        cand = json.loads((candidate / name).read_text())
        info[name] = {"exact_equal": json.dumps(ref, sort_keys=True) == json.dumps(cand, sort_keys=True)}
        if not info[name]["exact_equal"]:
            problems.append(f"{name}: preprocessing JSON is not exactly equal")
        if name == "tcn_channel_stats.json":
            for key, value in ref.items():
                other = cand.get(key)
                if other is None or len(value) != len(other) or any(
                        abs(float(a) - float(b)) > 1e-9 for a, b in zip(value, other)):
                    problems.append(f"tcn_channel_stats[{key}] differs")
        elif name == "tcn_design.json":
            for field in ("median", "iqr"):
                for key, value in ref[field].items():
                    other = cand.get(field, {}).get(key)
                    if other is None or abs(float(value) - float(other)) > 1e-9:
                        problems.append(f"tcn_design.{field}[{key}] differs")
            if list(ref.get("features", [])) != list(cand.get("features", [])):
                problems.append("tcn_design feature order differs")
        else:  # tabm_design.json
            for field in ("median", "iqr"):
                for key, value in ref[field].items():
                    other = cand.get(field, {}).get(key)
                    if other is None or abs(float(value) - float(other)) > 1e-9:
                        problems.append(f"tabm_design.{field}[{key}] differs")
            if list(ref.get("features", [])) != list(cand.get("features", [])):
                problems.append("tabm_design feature order differs")
    return problems, info


def tensor_checks(reference: Path, candidate: Path) -> tuple[list, dict]:
    problems: list[str] = []
    info: dict = {}
    for name in EXPECTED_MEMBERS["tabm"] + EXPECTED_MEMBERS["tcn"]:
        ref_path, cand_path = reference / name, candidate / name
        if not cand_path.is_file():
            problems.append(f"{name}: missing")
            continue
        with np.load(ref_path, allow_pickle=False) as a, np.load(cand_path, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                problems.append(f"{name}: key set differs")
                continue
            entry = {}
            bad = []
            for key in sorted(a.files):
                left, right = a[key], b[key]
                if left.dtype != right.dtype or left.shape != right.shape:
                    problems.append(f"{name}[{key}]: dtype/shape differs "
                                    f"({left.dtype}{left.shape} vs {right.dtype}{right.shape})")
                    bad.append(key)
                    continue
                if left.dtype != np.float32:
                    problems.append(f"{name}[{key}]: dtype {left.dtype} != float32")
                if not np.isfinite(left).all() or not np.isfinite(right).all():
                    problems.append(f"{name}[{key}]: non-finite values")
                result = _elementwise(right, left)
                entry[key] = result
                if not result["tolerance_pass"]:
                    bad.append(key)
            info[name] = {"per_tensor": entry, "failing_keys": bad,
                          "all_within_tolerance": not bad}
            if bad:
                problems.append(f"{name}: tensors outside tolerance: {bad}")
    return problems, info


def tree_checks(reference: Path, candidate: Path) -> tuple[list, dict]:
    from catboost import CatBoostRegressor

    problems: list[str] = []
    info: dict = {}
    for name in EXPECTED_MEMBERS["c1"]:
        cand_path = candidate / name
        if not cand_path.is_file():
            problems.append(f"{name}: missing")
            continue
        ref, cand = CatBoostRegressor(), CatBoostRegressor()
        ref.load_model(str(reference / name))
        cand.load_model(str(cand_path))
        entry = {"trees_reference": int(ref.tree_count_), "trees_candidate": int(cand.tree_count_),
                 "file_sha_equal": _sha(reference / name) == _sha(cand_path),
                 "file_sha_reference": _sha(reference / name),
                 "file_sha_candidate": _sha(cand_path)}
        if ref.tree_count_ != cand.tree_count_:
            problems.append(f"{name}: tree count {cand.tree_count_} != {ref.tree_count_}")
        leaf = _elementwise(cand.get_leaf_values(), ref.get_leaf_values())
        entry["leaf_values"] = leaf
        if not leaf["tolerance_pass"]:
            problems.append(f"{name}: leaf values differ (max {leaf['max_abs_diff']})")
        borders_ref, borders_cand = ref.get_borders(), cand.get_borders()
        border_problem = False
        if set(borders_ref) != set(borders_cand):
            border_problem = True
        else:
            for key in borders_ref:
                if not np.array_equal(np.asarray(borders_ref[key], dtype=float),
                                      np.asarray(borders_cand[key], dtype=float)):
                    border_problem = True
                    break
        entry["borders_equal"] = not border_problem
        if border_problem:
            problems.append(f"{name}: split borders differ")
        with tempfile.TemporaryDirectory(prefix="qf3_tree_audit_") as folder:
            rp, cp = Path(folder) / "reference.json", Path(folder) / "candidate.json"
            ref.save_model(str(rp), format="json")
            cand.save_model(str(cp), format="json")
            rj, cj = json.loads(rp.read_text()), json.loads(cp.read_text())
        fields = ("oblivious_trees", "features_info", "scale_and_bias")
        entry["tree_structure_exact"] = all(rj.get(k) == cj.get(k) for k in fields)
        if not entry["tree_structure_exact"]:
            problems.append(f"{name}: full split/leaf/feature/scale structure differs")
        entry["feature_names_equal"] = ref.feature_names_ == cand.feature_names_
        if not entry["feature_names_equal"]:
            problems.append(f"{name}: feature names/order differ")
        info[name] = entry
    return problems, info


def _prepare_days(data_dir: str, panel_path: str | None) -> tuple[dict, list]:
    from . import acceptance
    from .constants import BARRA_TABLE, MARKET_TABLE

    tables = acceptance.dataio.load_all(data_dir)
    membership = tables[BARRA_TABLE].loc[:, ["日期", "代码"]]
    for name in ("MarketData", "Factor33_winsor"):
        tables[name] = tables[name].merge(membership, on=["日期", "代码"], how="inner", sort=False)
    calendar = sorted(pd.unique(tables[MARKET_TABLE]["\u65e5\u671f"]).tolist())
    span = [day for day in calendar if 20221201 <= day <= 20221228][:20]
    barra = tables[BARRA_TABLE]
    by_day = {day: set(g) for day, g in barra.groupby("\u65e5\u671f")["\u4ee3\u7801"]}
    change_day = None
    for previous, current in zip(calendar, calendar[1:]):
        if current >= 20190101 and current in by_day and previous in by_day and by_day[current] != by_day[previous]:
            change_day = current
            break
    days = span + [20160105, 20190617, 20200615, 20210615]
    if change_day:
        days.append(change_day)
    # A constituent-change date can also be one of the fixed probes. Each
    # signal day must occur once in both the per-day table and IC mean.
    days = list(dict.fromkeys(days))
    labels = None
    if panel_path and Path(panel_path).exists():
        panel = pd.read_parquet(panel_path, columns=["date", "code", "future_return"])
        labels = {(int(d), str(c)): float(v) for d, c, v in
                  zip(panel["date"], panel["code"], panel["future_return"])}
    return {"tables": tables, "days": days, "labels": labels,
            "constituent_change_day": change_day}, days


def _score_all(scorer, frame: pd.DataFrame, sequence: np.ndarray) -> dict:
    atoms = scorer.member_raw(frame, sequence)
    families = scorer.atom_scores(frame, sequence)
    final = scorer.score(frame, sequence)
    return {"atoms": atoms, "families": families, "final": final}


def end_to_end_checks(reference: Path, candidate: Path, context: dict,
                      days: list, module) -> tuple[list, dict]:
    from . import acceptance
    from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE

    feat = importlib.import_module(module.__name__ + ".alpha_features")
    seqmod = importlib.import_module(module.__name__ + ".alpha_tcn_seq")
    scorer_ref = module.Scorer(str(reference))
    scorer_cand = module.Scorer(str(candidate))
    inference_ref = module.InferenceModel(str(reference))
    inference_cand = module.InferenceModel(str(candidate))
    tables, labels = context["tables"], context["labels"]
    problems: list[str] = []
    per_day: dict = {}
    worst = {"atom_raw": 0.0, "family_raw": 0.0, "final_raw": 0.0}
    rank_bad = 0
    ic_ref, ic_cand = [], []
    for day in days:
        window = acceptance.window_for(tables, day)
        barra = window[BARRA_TABLE]
        universe = barra.loc[barra["\u65e5\u671f"] == day, "\u4ee3\u7801"].drop_duplicates().tolist()
        frame = feat.build_day_features(window[MARKET_TABLE], window[FACTOR_TABLE], barra, day)
        frame = frame.set_index("code").reindex(universe).reset_index()
        codes, sequence = seqmod.build_sequence(window[MARKET_TABLE], barra, day)
        assert list(codes) == list(frame["code"])
        left = _score_all(scorer_ref, frame, sequence)
        right = _score_all(scorer_cand, frame, sequence)
        entry = {"date": acceptance.as_iso(day), "rows": int(len(frame)),
                 "keys_identical": list(left["atoms"]) == list(right["atoms"])}
        out_ref = inference_ref.predict(entry["date"], window)
        out_cand = inference_cand.predict(entry["date"], window)
        entry["reference_frame_checks"] = acceptance.check_frame(out_ref, entry["date"], universe)
        entry["candidate_frame_checks"] = acceptance.check_frame(out_cand, entry["date"], universe)
        if not all(entry["reference_frame_checks"].values()) or not all(entry["candidate_frame_checks"].values()):
            problems.append(f"{entry['date']}: inference output contract failed")
        ordered_ref = out_ref.set_index("code").reindex(universe).score.to_numpy()
        ordered_cand = out_cand.set_index("code").reindex(universe).score.to_numpy()
        entry["entry_matches_scorer"] = bool(np.array_equal(ordered_ref, left["final"])
                                            and np.array_equal(ordered_cand, right["final"]))
        if not entry["entry_matches_scorer"]:
            problems.append(f"{entry['date']}: real inference entry differs from scorer")
        for name, group in (("atoms", left["atoms"]), ("families", left["families"])):
            for key in group:
                result = _elementwise(right[name][key], group[key])
                result.update(_rank_compare(right[name][key], group[key]))
                entry[f"{name}.{key}"] = result
                bucket = "atom_raw" if name == "atoms" else "family_raw"
                worst[bucket] = max(worst[bucket], result["max_abs_diff"])
                if not result["tolerance_pass"]:
                    problems.append(f"{entry['date']} {name}.{key}: raw outside tolerance")
                if not result["rank_identical"]:
                    rank_bad += 1
        final = _elementwise(right["final"], left["final"])
        final.update(_rank_compare(right["final"], left["final"]))
        entry["final"] = final
        worst["final_raw"] = max(worst["final_raw"], final["max_abs_diff"])
        if not final["tolerance_pass"]:
            problems.append(f"{entry['date']} final: raw outside tolerance")
        if not final["rank_identical"]:
            rank_bad += 1
            problems.append(f"{entry['date']} final: rank order changed")
        if labels is not None:
            codes_arr = frame["code"].to_numpy()
            values = np.array([labels.get((int(day), str(code)), np.nan) for code in codes_arr])
            keep = np.isfinite(values)
            def _daily_ic(score):
                if keep.sum() < 2:
                    return np.nan
                return float(np.corrcoef(_rank(values[keep]), _rank(score[keep]))[0, 1])
            a, b = _daily_ic(left["final"]), _daily_ic(right["final"])
            entry["daily_ic"] = {"reference": a, "candidate": b}
            if np.isfinite(a) and np.isfinite(b):
                ic_ref.append(a), ic_cand.append(b)
        per_day[entry["date"]] = entry
    summary = {
        "days": len(per_day),
        "max_atom_raw_diff": worst["atom_raw"],
        "max_family_raw_diff": worst["family_raw"],
        "max_final_raw_diff": worst["final_raw"],
        "rank_mismatch_objects": rank_bad,
        "mean_daily_ic_reference": float(np.mean(ic_ref)) if ic_ref else None,
        "mean_daily_ic_candidate": float(np.mean(ic_cand)) if ic_cand else None,
        "mean_daily_ic_diff": (float(np.mean(ic_cand) - np.mean(ic_ref)) if ic_ref else None),
        "constituent_change_day": context["constituent_change_day"],
    }
    return problems, {"summary": summary, "per_day": per_day}


def _perturb_negative_controls(reference: Path, root: Path) -> dict:
    """Build five intentionally broken copies of the reference model directory."""
    controls = {}
    for name in ("standardisation", "fusion_weights", "deleted_member",
                 "tensor", "feature_order"):
        target = root / f"NEG_{name}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(reference, target)
        if name == "standardisation":
            path = target / "tcn_channel_stats.json"
            stats = json.loads(path.read_text())
            stats["0"] = [float(stats["0"][0]) + 1000.0, float(stats["0"][1])]
            path.write_text(json.dumps(stats))
        elif name == "fusion_weights":
            path = target / "meta.json"
            meta = json.loads(path.read_text())
            meta["weights"] = {"c1": 1.0, "tabm": 0.0, "tcn": 0.0}
            path.write_text(json.dumps(meta))
        elif name == "deleted_member":
            (target / EXPECTED_MEMBERS["tcn"][-1]).unlink()
        elif name == "tensor":
            path = target / EXPECTED_MEMBERS["tabm"][0]
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            arrays["W0"] = arrays["W0"] + np.float32(1e-3)
            np.savez(path, **arrays)
        elif name == "feature_order":
            path = target / "meta.json"
            meta = json.loads(path.read_text())
            meta["features"] = list(reversed(meta["features"]))
            path.write_text(json.dumps(meta))
        controls[name] = str(target)
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-model", required=True,
                        help="the packaged model/ directory (read-only reference)")
    parser.add_argument("--repro", required=True, help="the train.py output directory")
    parser.add_argument("--submission", required=True, help="extracted submission dir")
    parser.add_argument("--data-dir", default="Source/Data")
    parser.add_argument("--panel", default="artifacts/stage2_ic_ir/panel_platform.parquet")
    parser.add_argument("--zip-sha256", default=None)
    parser.add_argument("--zip", default="releases/stage4_verified_ic/V3_full_pkg.zip")
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v3_train_repro.json")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()

    reference = Path(args.reference_model)
    candidate = Path(args.repro)
    report: dict = {"reference_model": str(reference), "repro_dir": str(candidate),
                    "zip_sha256": args.zip_sha256, "problems": []}
    problems: list[str] = []

    if not reference.is_dir() or not candidate.is_dir():
        raise SystemExit("reference-model and repro must both be directories")

    actual_sha = _sha(Path(args.zip))
    if args.zip_sha256 is None or actual_sha != args.zip_sha256:
        raise SystemExit("original ZIP hash not explicitly bound or does not match")
    with zipfile.ZipFile(args.zip) as archive:
        if archive.testzip() is not None:
            raise SystemExit("original ZIP CRC error")
        for name in archive.namelist():
            path = Path(args.submission) / Path(name).relative_to("submission")
            if not path.is_file() or hashlib.sha256(archive.read(name)).hexdigest() != _sha(path):
                raise SystemExit(f"extracted source/model differs from original ZIP: {name}")
        expected_files = {str(Path(name).relative_to("submission/model"))
                          for name in archive.namelist() if name.startswith("submission/model/")}
        if _model_files(reference) != expected_files:
            raise SystemExit("reference model file set differs from original ZIP")
        for name in expected_files:
            if _sha(reference / name) != hashlib.sha256(archive.read("submission/model/" + name)).hexdigest():
                raise SystemExit("reference model does not match original ZIP")

    p, info = identity_checks(reference, candidate, args.zip_sha256)
    report["identity"] = info
    problems += p
    # Identity failures are a hard gate: do not load incomplete or wrong models.
    if problems:
        report["problems"] = problems
        report["passed"] = False
        Path(args.out).write_text(json.dumps(report, indent=2))
        raise SystemExit(1)
    p, info = preprocessing_checks(reference, candidate)
    report["preprocessing"] = info
    problems += p
    p, info = tree_checks(reference, candidate)
    report["trees"] = info
    problems += p
    tensor_problems, info = tensor_checks(reference, candidate)
    report["tensors"] = info
    problems += tensor_problems

    context, days = _prepare_days(args.data_dir, args.panel)
    from . import acceptance
    module = acceptance.load_inference(Path(args.submission))
    p, info = end_to_end_checks(reference, candidate, context, days, module)
    report["end_to_end"] = info
    problems += p

    negative = {}
    if args.negative_controls:
        with tempfile.TemporaryDirectory(prefix="qf3_neg_") as tmp:
            controls = _perturb_negative_controls(reference, Path(tmp))
            small_days = days[:3]
            for name, path in controls.items():
                sub_problems = []
                sub_problems += identity_checks(reference, Path(path), None)[0]
                if not sub_problems:
                    sub_problems += preprocessing_checks(reference, Path(path))[0]
                    sub_problems += tree_checks(reference, Path(path))[0]
                    sub_problems += tensor_checks(reference, Path(path))[0]
                if not sub_problems:
                    sub_problems += end_to_end_checks(reference, Path(path), context,
                                                      small_days, module)[0]
                negative[name] = {"rejected": bool(sub_problems),
                                  "problems": sub_problems[:5]}
                if not sub_problems:
                    problems.append(f"negative control '{name}' was NOT rejected")
    report["negative_controls"] = negative
    report["problems"] = problems
    report["passed"] = not problems
    Path(args.out).write_text(json.dumps(report, indent=2))
    summary = report["end_to_end"]["summary"]
    print(json.dumps({"passed": report["passed"], "problems": problems[:10],
                      "problem_count": len(problems),
                      "max_atom_raw_diff": summary["max_atom_raw_diff"],
                      "max_final_raw_diff": summary["max_final_raw_diff"],
                      "mean_daily_ic_reference": summary["mean_daily_ic_reference"],
                      "mean_daily_ic_candidate": summary["mean_daily_ic_candidate"],
                      "negative_controls": {k: v["rejected"] for k, v in negative.items()}},
                     indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
