"""V3 §C acceptance: platform-entry checks and package<->research equivalence.

This is an execution-agent self-review harness, not a "suggest upload" verdict.
It reuses the offline acceptance helpers (extract / load_inference / window_for)
and adds the two items the review asked for explicitly: the **raw tolerance** and
the **rank agreement** are computed and reported *separately* (neither substitutes
for the other), and the packaged NumPy pipeline is compared against the research
PyTorch/CatBoost pipeline on real one-day inputs.
"""

from __future__ import annotations

import importlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import acceptance
from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE
from .s6_v3_models import ConfirmationModels
from .s6_v3_package import SEEDS, V2, FULL, WEIGHTS, BUILD

ATOL, RTOL = 1e-5, 1e-4


def _research_reference() -> ConfirmationModels:
    deploy = V2 / "deploy_2021cutoff"
    return ConfirmationModels(
        c1_path=lambda seed: FULL / f"catboost_c1_seed{seed}.cbm",
        tabm_dir=lambda seed: FULL,
        tcn_dir=lambda seed: V2 / f"tcn_rev1_tcn_s{seed}_FULL22",
        l5_path=lambda seed: deploy / f"l5_seed{seed}.txt", seeds=SEEDS)


def _close(left, right) -> dict:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    diff = np.abs(left - right)
    tol = ATOL + RTOL * np.abs(right)
    return {"max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "tolerance_pass": bool((diff <= tol).all())}


def _rank_agreement(left, right) -> dict:
    left, right = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    rl = pd.Series(left).rank(method="average").to_numpy()
    rr = pd.Series(right).rank(method="average").to_numpy()
    return {"identical_rank_order": bool(np.array_equal(rl, rr)),
            "max_rank_gap": float(np.max(np.abs(rl - rr))) if rl.size else 0.0,
            "spearman": float(np.corrcoef(rl, rr)[0, 1]) if rl.size > 1 else float("nan")}


def compare_day(model, module, tables, day: int, reference: ConfirmationModels) -> dict:
    feat = importlib.import_module(module.__name__ + ".alpha_features")
    seqmod = importlib.import_module(module.__name__ + ".alpha_tcn_seq")
    window = acceptance.window_for(tables, day)
    barra = window[BARRA_TABLE]
    universe = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
    frame = feat.build_day_features(window[MARKET_TABLE], window[FACTOR_TABLE], barra, day)
    frame = frame.set_index("code").reindex(universe).reset_index()
    codes, sequence = seqmod.build_sequence(window[MARKET_TABLE], barra, day)
    assert list(codes) == list(frame["code"])

    pkg_raw = model.scorer.member_raw(frame, sequence)
    ref_raw = reference.atoms(frame, sequence)
    atoms = {}
    for key in sorted(pkg_raw):
        if key not in ref_raw:
            raise KeyError(f"research reference is missing atom {key}")
        atoms[key] = {"raw": _close(pkg_raw[key], ref_raw[key]),
                      "rank": _rank_agreement(pkg_raw[key], ref_raw[key])}
    pkg_fam = model.scorer.atom_scores(frame, sequence)
    ref_fam = reference.family(frame, sequence)
    families = {name: {"raw": _close(pkg_fam[name], ref_fam[name]),
                       "rank": _rank_agreement(pkg_fam[name], ref_fam[name])}
                for name in ("c1", "tabm", "tcn")}
    pkg_final = model.scorer.score(frame, sequence)
    ref_final = reference.fusion(ref_fam, WEIGHTS)
    final = {"raw": _close(pkg_final, ref_final), "rank": _rank_agreement(pkg_final, ref_final),
             "reference_final_is_rank": True}
    out = model.predict(acceptance.as_iso(day), window)
    final["platform_entry_matches_scorer"] = bool(np.array_equal(
        out.sort_values("code")["score"].to_numpy(),
        pd.DataFrame({"code": frame["code"], "score": pkg_final}).sort_values("code")["score"].to_numpy()))
    return {"date": acceptance.as_iso(day), "n": len(frame), "atoms": atoms,
            "families": families, "final": final,
            "frame_checks": acceptance.check_frame(out, acceptance.as_iso(day), universe)}


def timing_and_rss(model, tables, start: int, days: int) -> dict:
    calendar = sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
    pos = calendar.index(start)
    selected = calendar[pos:pos + days]
    durations, peak = [], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for day in selected:
        window = acceptance.window_for(tables, day)
        t0 = time.perf_counter()
        model.predict(acceptance.as_iso(day), window)
        durations.append(time.perf_counter() - t0)
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    arr = np.array(durations)
    warm = arr[1:]
    return {"n_days": len(arr), "cold_start_seconds": float(arr[0]),
            "warm_p50_seconds": float(np.percentile(warm, 50)),
            "warm_p95_seconds": float(np.percentile(warm, 95)),
            "warm_max_seconds": float(warm.max()),
            "peak_rss_mb": peak / 1024.0}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--data-dir", default="Source/Data")
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v3_acceptance.json")
    parser.add_argument("--timing-start", type=int, default=20220601)
    parser.add_argument("--timing-days", type=int, default=40)
    args = parser.parse_args()

    submission = acceptance.extract(Path(args.zip), Path("artifacts/stage4_verified_ic/accept_extract"))
    module = acceptance.load_inference(submission)
    t_load = time.perf_counter()
    model = module.InferenceModel(str(submission / "model"))
    load_seconds = time.perf_counter() - t_load
    tables = acceptance.dataio.load_all(args.data_dir)
    reference = _research_reference()

    report = {"package": args.zip, "model_load_seconds": load_seconds, "days": {}}
    for day in (20220104, 20220613, 20221228):
        report["days"][str(day)] = compare_day(model, module, tables, day, reference)

    base_day = 20220613
    window = acceptance.window_for(tables, base_day)
    base = model.predict(acceptance.as_iso(base_day), window)
    tampered = {k: v.copy() for k, v in window.items()}
    tampered[MARKET_TABLE]["名称"] = "XXXX"
    tampered[FACTOR_TABLE]["is_st"] = 1 - tampered[FACTOR_TABLE]["is_st"].fillna(0).astype(int)
    changed = model.predict(acceptance.as_iso(base_day), tampered)
    report["identifier_invariance_max_abs_diff"] = float(
        np.max(np.abs(base["score"].to_numpy() - changed["score"].to_numpy())))
    truncated = {k: v.loc[v["日期"] <= base_day] for k, v in window.items()}
    trunc = model.predict(acceptance.as_iso(base_day), truncated)
    report["future_truncation_ok"] = bool(
        np.array_equal(base["score"].to_numpy(), trunc["score"].to_numpy()))
    rng = np.random.default_rng(0)
    shuffled = {k: v.iloc[rng.permutation(len(v))].reset_index(drop=True)
                for k, v in window.items()}
    perm = model.predict(acceptance.as_iso(base_day), shuffled)
    merged = base.merge(perm, on="code", suffixes=("_a", "_b"))
    report["row_permutation_ok"] = bool(len(merged) == len(base) and np.allclose(
        merged["score_a"].to_numpy(), merged["score_b"].to_numpy(), rtol=0, atol=0))
    report["timing"] = timing_and_rss(model, tables, args.timing_start, args.timing_days)
    susp = tables[MARKET_TABLE]
    susp = susp.loc[(susp["is_suspend"] == 1) & (susp["日期"] // 10000 == 2019)]
    if len(susp):
        sday = int(susp["日期"].iloc[0])
        scode = susp["代码"].iloc[0]
        sout = model.predict(acceptance.as_iso(sday), acceptance.window_for(tables, sday))
        report["suspended_stock_scored"] = bool(
            scode in set(sout["code"]) and np.isfinite(
                float(sout.loc[sout["code"] == scode, "score"].iloc[0])))
    feat = importlib.import_module(module.__name__ + ".alpha_features")
    fday = 20190603
    fwin = acceptance.window_for(tables, fday)
    fframe = feat.build_day_features(fwin[MARKET_TABLE], fwin[FACTOR_TABLE],
                                     fwin[BARRA_TABLE], fday)
    gaps = fframe[list(feat.FEATURE_COLUMNS)].isna().sum(axis=1)
    worst = fframe.loc[gaps.idxmax(), "code"]
    fout = model.predict(acceptance.as_iso(fday), fwin)
    report["feature_deficient_stock"] = {
        "code": str(worst), "nan_features": int(gaps.max()),
        "scored": bool(worst in set(fout["code"])),
        "score_finite": bool(np.isfinite(
            float(fout.loc[fout["code"] == worst, "score"].iloc[0])))}
    report["package_bytes"] = Path(args.zip).stat().st_size
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"load_seconds": load_seconds, "timing": report["timing"],
                      "final": {d: report["days"][d]["final"] for d in report["days"]}},
                     indent=2))


if __name__ == "__main__":
    main()
