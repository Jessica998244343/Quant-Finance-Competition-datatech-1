"""§5 correction: member scan under the canonical flat graph (no intermediate rank).

Parent = the frozen anchor expressed as its family stage: {c1: 0.5, tabm: 0.5},
where c1/tabm are the seed-mean family daily ranks.  Adding a member uses the
full weight vector ``(1-a)*parent + a*new``; the old nested-rank variant is
recomputed only to document the difference.  Writes a new artifact, never
overwrites the superseded V2 files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s6_fusion import as_family, daily_centred_rank, family_q, fuse, push_down_alpha

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
V2 = S4 / "v2"
WINDOWS = ("DEV19", "DEV20")
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
PARENT = {"c1": 0.5, "tabm": 0.5}
SEEDS = (20261005, 20261017, 20261029)


def daily_ic(frame: pd.DataFrame) -> float:
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
    return float(np.mean(values)) if values else float("nan")


def parent_frames(window: str) -> dict[str, pd.DataFrame]:
    c1 = as_family(pd.read_parquet(S4 / f"v1_c1_seedmean__{window}.parquet"),
                   column="c1", label=f"c1_{window}")
    tabm = as_family(pd.read_parquet(S4 / f"v1_tabm_seedmean__{window}.parquet"),
                     column="tabm", label=f"tabm_{window}")
    return {"c1": c1, "tabm": tabm}


def candidate_specs(window: str) -> dict[str, list[Path]]:
    return {
        "C2": [FUS / "predictions_cat" / f"C2_20261005__{window}.parquet"],
        "C3": [FUS / "predictions_cat" / f"C3_20261005__{window}.parquet"],
        "tabm_dropout0.2": [V2 / f"tabm_mse_d0.2_{seed}__{window}" / "predictions.parquet"
                            for seed in SEEDS],
        "tabm_corr_loss": [V2 / f"tabm_mse_corr_d0.1_20261005__{window}" / "predictions.parquet"],
        "resmlp_n0": [S4 / "v1_resmlp" / f"resmlp_n0_20261005__{window}_predictions.parquet"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_rev1_member_scan.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    report = {"graph": "flat: u = sum_g w_g q_g ; output = rank_t(u)",
              "parent_weights": PARENT, "alphas": list(ALPHAS), "windows": {}}
    for window in WINDOWS:
        parents = parent_frames(window)
        entry = {"parent_rankic": daily_ic(fuse(parents, PARENT).merge(
            labels, on=["date", "code"], how="left", validate="one_to_one")), "members": {}}
        for name, paths in candidate_specs(window).items():
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                entry["members"][name] = {"missing_artifacts": missing}
                continue
            new_q = family_q([pd.read_parquet(path) for path in paths], "score").rename(
                columns={"q": "q"})
            components = dict(parents)
            components[name] = new_q
            curve = {}
            for alpha in ALPHAS:
                weights = push_down_alpha(PARENT, name, alpha)
                fused = fuse(components, weights).merge(
                    labels, on=["date", "code"], how="left", validate="one_to_one")
                curve[str(alpha)] = daily_ic(fused)
            base = curve["0.0"]
            entry["members"][name] = {
                "seeds": len(paths), "ic_by_alpha": curve,
                "delta_vs_parent": {key: curve[key] - base for key in curve}}
        report["windows"][window] = entry
    dev = {}
    for name in report["windows"]["DEV19"]["members"]:
        if "delta_vs_parent" not in report["windows"]["DEV19"]["members"][name]:
            continue
        dev[name] = {key: float(np.average(
            [report["windows"][w]["members"][name]["delta_vs_parent"][key]
             for w in WINDOWS], weights=[244, 243])) for key in
            report["windows"]["DEV19"]["members"][name]["delta_vs_parent"]}
    report["dev_merged_delta"] = dev
    report["retention_rule"] = ("flat-graph DEV-merged dIC >= 0.0005 with both DEV years "
                                "non-negative (or a negative year not below -0.001 with a "
                                "documented complementarity)")
    report["retained"] = [name for name, deltas in dev.items()
                          if max(deltas.values()) >= 0.0005]
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"parent_ic": {w: report["windows"][w]["parent_rankic"] for w in WINDOWS},
                      "dev_merged_delta": {k: {a: round(v, 6) for a, v in d.items()}
                                           for k, d in dev.items()},
                      "retained": report["retained"]}, indent=2))


if __name__ == "__main__":
    main()
