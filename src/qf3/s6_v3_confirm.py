"""V3 §B: the single 2022 time-out-of-sample confirmation.

Evaluates the frozen set (three-member candidate, V1 fallback, fixed C1, actual
TREE structure control) with the cut-off-2021 models, on the same dates, pool and
valid-label mask.  No selection happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .s4_bootstrap import _draw_indices
from .s6_v3_models import V2, deploy_models
from .s6_v2_tcn import build_sequences

S4 = Path("artifacts/stage4_verified_ic")
COMBOS = {"three_member": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25},
          "V1": {"c1": 0.5, "tabm": 0.5},
          "C1_fixed": {"c1": 1.0},
          "TREE_structure": {"l5": 0.5, "c1": 0.5}}
BOOT = {"draws": 2000, "seed": 20261005, "blocks": (10, 20, 40)}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interval(delta: pd.Series, block: int) -> dict:
    values = delta.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOT["seed"])
    draws = np.empty(BOOT["draws"])
    for index in range(BOOT["draws"]):
        draws[index] = values[_draw_indices(len(values), block, rng)].mean()
    return {"days": int(len(values)), "block": block, "point": float(values.mean()),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--out", default=str(V2 / "v3_confirmation_2022.json"))
    parser.add_argument("--atoms-out", default=str(V2 / "v3_confirmation_atoms.parquet"))
    args = parser.parse_args()
    manifest_path = V2 / "v3_confirmation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # review §6: the entry must enforce the freeze check, not just record the hash
    import sys
    sys.path.insert(0, "src")
    from qf3.s6_v2_freeze import verify as verify_freeze
    frozen = verify_freeze(manifest)
    preflight = json.loads((V2 / "v3_preflight.json").read_text())
    if not frozen["passed"]:
        raise SystemExit(f"freeze verification failed: {frozen['problems']}")
    if preflight["confirmation_manifest"]["manifest_hash"] != manifest["manifest_hash"]:
        raise SystemExit(
            "preflight was computed under a different manifest identity; re-run preflight")
    opening = {
        "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "manifest_file_sha256": _sha(manifest_path),
        "confirmation_source_sha256": {
            name: _sha(Path(name)) for name in
            ("src/qf3/s6_v3_confirm.py", "src/qf3/s6_v3_models.py",
             "src/qf3/s6_v3_preflight.py")},
        "preflight": preflight["passed"],
        "freeze_verified": True,
        "freeze_checked": frozen["checked"],
    }
    if not opening["preflight"]:
        raise SystemExit("preflight not passed; refusing to open 2022")

    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    window = panel.loc[panel["date"] // 10000 == args.year].reset_index(drop=True)
    days = sorted(int(d) for d in window.loc[window["future_return"].notna(), "date"].unique())
    opening["evaluation_days"] = len(days)
    opening["first_day"], opening["last_day"] = days[0], days[-1]
    print(f"2022 legal signal days with mature labels: {len(days)} "
          f"({days[0]} .. {days[-1]})", flush=True)

    built = build_sequences()
    sequence_map, codes_by_day = built["sequences"], built["keys"]
    models = deploy_models()
    rows_out = []
    for day in days:
        block = window.loc[window["date"] == day].reset_index(drop=True)
        lookup = {code: index for index, code in enumerate(codes_by_day.get(day, []))}
        block = block.loc[block["code"].map(lookup).notna()].reset_index(drop=True)
        if block.empty:
            continue
        sequence = np.stack([sequence_map[day][lookup[code]] for code in block["code"]])
        families = models.family(block, sequence)
        record = {"date": day, "code": block["code"].to_numpy(),
                  "future_return": block["future_return"].to_numpy()}
        for name, weights in COMBOS.items():
            record[name] = models.fusion(families, weights)
        rows_out.append(record)
    report = {"opening": opening, "combos": COMBOS, "boot": BOOT, "days": len(rows_out)}
    daily = {}
    for name in COMBOS:
        series = {}
        for record in rows_out:
            keep = np.isfinite(record["future_return"]) & np.isfinite(record[name])
            if keep.sum() < 2:
                continue
            y = pd.Series(record["future_return"][keep]).rank(method="average")
            p = pd.Series(record[name][keep]).rank(method="average")
            if y.nunique() == 1 or p.nunique() == 1:
                continue
            series[record["date"]] = float(np.corrcoef(y, p)[0, 1])
        daily[name] = pd.Series(series).sort_index()
    report["rankic"] = {name: float(values.mean()) for name, values in daily.items()}
    halves = {}
    for name in COMBOS:
        month = pd.Series(daily[name].index).astype("int64") % 10000 // 100
        halves[name] = {"H1": float(daily[name].to_numpy()[month.between(1, 6).to_numpy()].mean()),
                        "H2": float(daily[name].to_numpy()[month.between(7, 12).to_numpy()].mean())}
    report["halves"] = halves
    deltas = {}
    for name in ("three_member", "V1", "TREE_structure", "C1_fixed"):
        for ref in ("V1", "TREE_structure", "C1_fixed"):
            if name == ref:
                continue
            joined = pd.concat([daily[name], daily[ref]], axis=1,
                               keys=["a", "b"]).dropna()
            if joined.empty:
                continue
            delta = joined["a"] - joined["b"]
            deltas[f"{name}_minus_{ref}"] = {
                "point": float(delta.mean()),
                "intervals": {f"block{b}": _interval(delta, b) for b in BOOT["blocks"]}}
    report["deltas"] = deltas
    rule = manifest["adoption_rule"]
    primary = deltas.get("three_member_minus_V1", {}).get("point", float("nan"))
    primary_tree = deltas.get("three_member_minus_TREE_structure", {}).get("point", float("nan"))
    half_deltas = {"H1": halves["three_member"]["H1"] - halves["V1"]["H1"],
                   "H2": halves["three_member"]["H2"] - halves["V1"]["H2"]}
    halves_ok = bool(half_deltas["H1"] >= -0.001 and half_deltas["H2"] >= -0.001)
    # review §6: the TREE half gate was missing from the first decision code
    tree_half_deltas = {
        "H1": halves["three_member"]["H1"] - halves["TREE_structure"]["H1"],
        "H2": halves["three_member"]["H2"] - halves["TREE_structure"]["H2"]}
    tree_halves_ok = bool(tree_half_deltas["H1"] >= -0.001
                          and tree_half_deltas["H2"] >= -0.001)
    report["decision"] = {
        "rule": rule["support"],
        "primary_delta": primary, "primary_vs_tree": primary_tree,
        "half_deltas_vs_V1": half_deltas,
        "halves_ok": bool(halves_ok),
        "half_deltas_vs_TREE": tree_half_deltas,
        "tree_halves_ok": bool(tree_halves_ok),
        "three_member_supported": bool(primary > 0 and primary_tree > 0
                                       and halves_ok and tree_halves_ok),
    }
    if not report["decision"]["three_member_supported"]:
        fallback = deltas.get("V1_minus_TREE_structure", {}).get("point", float("nan"))
        fallback_halves = (halves["V1"]["H1"] - halves["TREE_structure"]["H1"],
                           halves["V1"]["H2"] - halves["TREE_structure"]["H2"])
        report["decision"]["fallback_delta"] = fallback
        report["decision"]["fallback_halves"] = [float(v) for v in fallback_halves]
        report["decision"]["V1_supported"] = bool(
            fallback > 0 and all(v >= -0.001 for v in fallback_halves))
    pd.DataFrame([{**{"date": r["date"], "code": c},
                   **{name: r[name][i] for name in COMBOS},
                   "future_return": r["future_return"][i]}
                  for r in rows_out for i, c in enumerate(r["code"])]).to_parquet(
        args.atoms_out, index=False)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("deltas",)},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
