"""V2 completion: reproducible report/stats/ledger from saved predictions only.

Every frozen field is derived from **one** α row, so the family report can no
longer contradict itself.  No training happens here; missing required windows
raise instead of triggering a silent workaround.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s6_fusion import as_family, daily_centred_rank, family_q, fuse, push_down_alpha
from .s4_bootstrap import _draw_indices

S4 = Path("artifacts/stage4_verified_ic")
V2 = S4 / "v2"
SEEDS = (20261005, 20261017, 20261029)
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
DEV_WINDOWS = ("DEV19", "DEV20")
YEAR = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}
DEV_DAYS = {"DEV19": 244, "DEV20": 243}
PARENT = {"c1": 0.5, "tabm": 0.5}
BOOT = {"draws": 2000, "seed": 20261005, "blocks": (10, 20, 40)}


def _panel() -> pd.DataFrame:
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def _pool(panel: pd.DataFrame, window: str) -> pd.DataFrame:
    return panel.loc[panel["date"] // 10000 == YEAR[window], ["date", "code"]]


def parents_for(window: str, panel: pd.DataFrame) -> dict:
    return {
        "c1": as_family(pd.read_parquet(S4 / f"v1_c1_seedmean__{window}.parquet"),
                        column="c1", expected_keys=_pool(panel, window), label="c1"),
        "tabm": as_family(pd.read_parquet(S4 / f"v1_tabm_seedmean__{window}.parquet"),
                          column="tabm", expected_keys=_pool(panel, window), label="tabm")}


def member_family(window: str, label: str, panel: pd.DataFrame) -> pd.DataFrame:
    seeds = (20261005,) if label == "shuffle" else SEEDS
    prefix = "tcn_rev1_shuffle_s20261005" if label == "shuffle" else f"tcn_rev1_{label}_s"
    frames = []
    for seed in seeds:
        path = (V2 / f"{prefix}_{window}" / "predictions.parquet") if label == "shuffle" \
            else (V2 / f"{prefix}{seed}_{window}" / "predictions.parquet")
        if not path.exists():
            raise FileNotFoundError(f"missing required artifact: {path}")
        frames.append(pd.read_parquet(path))
    return family_q(frames, "score", expected_keys=_pool(panel, window), label=f"{label}_{window}")


def _ic(frame: pd.DataFrame, labels: pd.DataFrame) -> float:
    merged = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    values = []
    for _, block in merged.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values.append(float(np.corrcoef(y, p)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def daily_ic_series(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    merged = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    out = {}
    for date, block in merged.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        out[date] = float(np.corrcoef(y, p)[0, 1])
    return pd.Series(out).sort_index()


def _interval(delta: pd.Series, block: int) -> dict:
    years = pd.Series(delta.index).astype("int64") // 10000
    groups = [delta.to_numpy(dtype=float)[(years == year).to_numpy()]
              for year in sorted(years.unique())]
    rng = np.random.default_rng(BOOT["seed"])
    draws = np.empty(BOOT["draws"])
    for index in range(BOOT["draws"]):
        pieces = [values[_draw_indices(len(values), block, rng)] for values in groups]
        draws[index] = np.concatenate(pieces).mean()
    return {"days": int(len(delta)), "block": block,
            "years": {int(y): int((years == y).sum()) for y in sorted(years.unique())},
            "point": float(delta.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5))}


def drop_member(weights: dict[str, float], member: str) -> dict[str, float]:
    """Remove one member and renormalise the remaining weights (review §3.1)."""
    if member not in weights:
        raise KeyError(member)
    remaining = {key: value for key, value in weights.items()
                 if key != member and value > 0}
    total = float(sum(remaining.values()))
    if total <= 0:
        raise ValueError("nothing left after deletion")
    return {key: value / total for key, value in remaining.items()}


def family_report(windows, panel) -> dict:
    labels = panel[["date", "code", "future_return"]]
    report = {"graph": "flat", "alphas": list(ALPHAS), "windows": {}}
    for window in windows:
        pool = _pool(panel, window)
        parents = parents_for(window, panel)
        base = fuse(parents, PARENT)
        entry = {"days": int(pool["date"].nunique()),
                 "parent_rankic": _ic(base, labels), "members": {}}
        if window in DEV_DAYS:
            assert entry["days"] == DEV_DAYS[window], \
                f"{window}: expected {DEV_DAYS[window]} days, found {entry['days']}"
        for label in ("tcn", "static", "shuffle"):
            try:
                family = member_family(window, label, panel)
            except FileNotFoundError as error:
                if label == "shuffle":
                    entry["members"][label] = {"skipped": str(error),
                                               "reason": "single-seed DEV diagnostic only"}
                    continue
                raise
            components = dict(parents)
            components[label] = family
            curve = {}
            for alpha in ALPHAS:
                weights = push_down_alpha(PARENT, label, alpha)
                curve[str(alpha)] = _ic(fuse(components, weights), labels)
            entry["members"][label] = {
                "family_standalone": _ic(family.rename(columns={"q": "score"}), labels),
                "ic_by_alpha": curve,
                "delta_vs_parent": {k: curve[k] - curve["0.0"] for k in curve}}
        report["windows"][window] = entry
    if all(w in report["windows"] for w in DEV_WINDOWS):
        report["dev_merged_delta"] = {}
        for label in ("tcn", "static", "shuffle"):
            report["dev_merged_delta"][label] = {
                str(a): float(np.average(
                    [report["windows"][w]["members"][label]["delta_vs_parent"][str(a)]
                     for w in DEV_WINDOWS], weights=[244, 243])) for a in ALPHAS}
        candidates = {}
        for a in ALPHAS:
            if a == 0.0:
                continue
            key = str(a)
            merged = report["dev_merged_delta"]["tcn"][key]
            dev19 = report["windows"]["DEV19"]["members"]["tcn"]["delta_vs_parent"][key]
            dev20 = report["windows"]["DEV20"]["members"]["tcn"]["delta_vs_parent"][key]
            candidates[key] = {"dev_merged": merged, "DEV19": dev19, "DEV20": dev20,
                               "eligible": bool(merged >= 0.0005 and dev19 >= 0 and dev20 >= 0)}
        report["alpha_selection"] = candidates
        eligible = {k: v for k, v in candidates.items() if v["eligible"]}
        if eligible:
            frozen = max(eligible, key=lambda k: eligible[k]["dev_merged"])
            report["frozen_alpha"] = float(frozen)
            row = candidates[frozen]
            report["frozen_dev_merged"] = row["dev_merged"]
            report["frozen_dev19"] = row["DEV19"]
            report["frozen_dev20"] = row["DEV20"]
            report["frozen_both_dev_nonneg"] = bool(row["DEV19"] >= 0 and row["DEV20"] >= 0)
            if "CONF21" in report["windows"]:
                report["frozen_conf21_delta"] = report["windows"]["CONF21"]["members"]["tcn"][
                    "delta_vs_parent"][frozen]
                report["frozen_conf21_nonneg"] = bool(report["frozen_conf21_delta"] >= 0)
            else:
                report["frozen_conf21_delta"] = None
                report["frozen_conf21_nonneg"] = None
            report["passed_initial_gate"] = bool(
                report["frozen_dev_merged"] >= 0.0005
                and report["frozen_both_dev_nonneg"]
                and report["frozen_conf21_nonneg"] is not False)
        else:
            report["frozen_alpha"] = None
            report["passed_initial_gate"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", choices=("family", "final-stats", "deletion"),
                        default="family")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--out", default="")
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    panel = _panel()
    windows = args.windows.split(",")
    missing = [w for w in windows if w not in YEAR]
    if missing:
        raise SystemExit(f"unknown window(s): {missing}")
    out = Path(args.out) if args.out else V2 / f"v2_completion_{args.command}.json"
    if args.command == "family":
        absent = [w for w in DEV_WINDOWS if w not in windows]
        if absent:
            raise SystemExit(f"missing required DEV window(s): {absent}; "
                             f"pass --windows DEV19,DEV20[,CONF21]")
        report = family_report(windows, panel)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))
        return
    labels = panel[["date", "code", "future_return"]]
    if args.command == "final-stats":
        out_frames = {}
        daily_rows = []
        for window in windows:
            parents = parents_for(window, panel)
            tcn = member_family(window, "tcn", panel)
            components = dict(parents)
            components["tcn"] = tcn
            weights = push_down_alpha(PARENT, "tcn", args.alpha)
            parent_ic = daily_ic_series(fuse(parents, PARENT), labels)
            tcn_ic = daily_ic_series(fuse(components, weights), labels)
            joined = pd.concat([parent_ic, tcn_ic], axis=1, keys=["parent", "tcn"]).dropna()
            delta = joined["tcn"] - joined["parent"]
            daily_rows.append(pd.DataFrame({
                "window": window, "date": joined.index,
                "parent_daily_ic": joined["parent"].to_numpy(),
                "tcn_daily_ic": joined["tcn"].to_numpy(),
                "daily_delta": delta.to_numpy(),
                "valid_label_mask_rule": "future_return finite on the same legal pool"}))
            entry = {"window": window, "alpha": args.alpha, "days": int(len(joined)),
                     "parent_rankic": float(joined["parent"].mean()),
                     "tcn_rankic": float(joined["tcn"].mean()),
                     "delta": float(delta.mean()),
                     "intervals": {f"block{b}": _interval(delta, b) for b in BOOT["blocks"]}}
            halves = {}
            for half, months in (("H1", (1, 6)), ("H2", (7, 12))):
                month = pd.Series(delta.index).astype("int64") % 10000 // 100
                mask = month.between(*months).to_numpy()
                halves[half] = float(delta.to_numpy()[mask].mean())
            entry["halves"] = halves
            out_frames[window] = entry
        report = {"alpha": args.alpha, "windows": out_frames,
                  "block_rule": "per-year moving block, day counts preserved",
                  "boot": BOOT}
        if all(w in out_frames for w in DEV_WINDOWS):
            report["dev_merged_delta"] = float(np.average(
                [out_frames[w]["delta"] for w in DEV_WINDOWS], weights=[244, 243]))
            merged_delta = []
            for window in DEV_WINDOWS:
                parents = parents_for(window, panel)
                components = dict(parents)
                components["tcn"] = member_family(window, "tcn", panel)
                weights = push_down_alpha(PARENT, "tcn", args.alpha)
                parent_ic = daily_ic_series(fuse(parents, PARENT), labels)
                tcn_ic = daily_ic_series(fuse(components, weights), labels)
                joined = pd.concat([parent_ic, tcn_ic], axis=1,
                                   keys=["parent", "tcn"]).dropna()
                merged_delta.append(joined["tcn"] - joined["parent"])
            combined = pd.concat(merged_delta).sort_index()
            report["dev_merged_intervals"] = {
                f"block{b}": _interval(combined, b) for b in BOOT["blocks"]}
        out.write_text(json.dumps(report, indent=2))
        daily_path = Path(str(out).replace(".json", "_daily.parquet"))
        pd.concat(daily_rows, ignore_index=True).to_parquet(daily_path, index=False)
        report["daily_artifact"] = str(daily_path)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "block_rule"}, indent=2))
        return
    if args.command == "deletion":
        report = {"alpha": args.alpha, "vectors": {}, "windows": {}}
        for window in windows:
            parents = parents_for(window, panel)
            tcn = member_family(window, "tcn", panel)
            full = dict(parents)
            full["tcn"] = tcn
            weights = push_down_alpha(PARENT, "tcn", args.alpha)
            variants = {
                "full": dict(weights),
                "drop_c1": drop_member(weights, "c1"),
                "drop_tabm": drop_member(weights, "tabm"),
                "drop_tcn": drop_member(weights, "tcn"),
            }
            # explicit assertions on the prescribed vectors (review §3.1)
            assert variants["drop_c1"] == {"tabm": 0.6, "tcn": 0.4}, variants["drop_c1"]
            assert variants["drop_tabm"] == {"c1": 0.6, "tcn": 0.4}, variants["drop_tabm"]
            assert variants["drop_tcn"] == {"c1": 0.5, "tabm": 0.5}, variants["drop_tcn"]
            entry = {}
            base = None
            for name, vector in variants.items():
                components = {k: full[k] for k in vector}
                value = _ic(fuse(components, vector), labels)
                entry[name] = {"weights": vector, "rankic": value}
                if name == "full":
                    base = value
            for name in entry:
                entry[name]["delta_vs_full"] = entry[name]["rankic"] - base
            report["windows"][window] = entry
        if all(w in report["windows"] for w in DEV_WINDOWS):
            report["dev_merged"] = {name: float(np.average(
                [report["windows"][w][name]["delta_vs_full"] for w in DEV_WINDOWS],
                weights=[244, 243])) for name in report["windows"]["DEV19"]}
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return


if __name__ == "__main__":
    main()
