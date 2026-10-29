"""§A3: negative fixtures for the fusion contract + a real-data regression check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s6_fusion import (FusionContractError, as_family, check_atom, family_q, fuse,
                        push_down_alpha)

S4 = Path("artifacts/stage4_verified_ic")
V2 = S4 / "v2"
SEEDS = (20261005, 20261017, 20261029)
ALPHA = 0.25
WINDOW_YEAR = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}


def _toy(days=(20200102, 20200103), codes=("A", "B", "C", "D"), seeds=2):
    rng = np.random.default_rng(7)
    frames = []
    for index in range(seeds):
        rows = [{"date": day, "code": code,
                 "score": float(rng.normal() + index * 0.05)}
                for day in days for code in codes]
        frames.append(pd.DataFrame(rows))
    return frames


def _rejects(function) -> bool:
    try:
        function()
    except (FusionContractError, ValueError):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_rev1_fusion_contract.json"))
    args = parser.parse_args()
    good = _toy()
    pool = good[0][["date", "code"]]
    good_q = family_q(good, "score", expected_keys=pool, label="toy")
    toy_components = {"a": good_q, "b": family_q(
        _toy(seeds=1), "score", expected_keys=pool, label="toy_b")}

    fixtures = {}
    missing = good[1].iloc[:-1]
    fixtures["missing_row_rejected"] = _rejects(
        lambda: family_q([good[0], missing], "score", expected_keys=pool, label="missing"))
    duplicate = pd.concat([good[1], good[1].iloc[[0]]], ignore_index=True)
    fixtures["duplicate_key_rejected"] = _rejects(
        lambda: check_atom(duplicate, "score", expected_keys=pool, label="dup"))
    extra = pd.concat([good[1], pd.DataFrame([{"date": 20200102, "code": "ZZ",
                                               "score": 0.1}])], ignore_index=True)
    fixtures["extra_code_rejected"] = _rejects(
        lambda: check_atom(extra, "score", expected_keys=pool, label="extra"))
    nonfinite = good[1].copy()
    nonfinite.loc[0, "score"] = np.nan
    fixtures["nan_atom_rejected"] = _rejects(
        lambda: check_atom(nonfinite, "score", expected_keys=pool, label="nan"))
    infinite = good[1].copy()
    infinite.loc[0, "score"] = np.inf
    fixtures["inf_atom_rejected"] = _rejects(
        lambda: check_atom(infinite, "score", expected_keys=pool, label="inf"))
    fixtures["nan_weight_rejected"] = _rejects(
        lambda: fuse(toy_components, {"a": np.nan, "b": 1.0}))
    fixtures["inf_weight_rejected"] = _rejects(
        lambda: fuse(toy_components, {"a": np.inf, "b": 0.0}))
    fixtures["negative_weight_rejected"] = _rejects(
        lambda: fuse(toy_components, {"a": -0.1, "b": 1.1}))
    fixtures["weight_sum_rejected"] = _rejects(
        lambda: fuse(toy_components, {"a": 0.4, "b": 0.4}))
    untagged = good_q.copy()
    untagged.attrs = {}
    fixtures["untagged_family_rejected"] = _rejects(
        lambda: fuse({"a": untagged, "b": toy_components["b"]}, {"a": 0.5, "b": 0.5}))
    renamed_final = good_q.copy().rename(columns={"q": "score"})
    fixtures["renamed_final_score_rejected"] = _rejects(
        lambda: fuse({"a": renamed_final, "b": toy_components["b"]}, {"a": 0.5, "b": 0.5}))
    mismatched = family_q(_toy(days=(20200102,), seeds=1), "score", label="toy_c")
    fixtures["component_pool_mismatch_rejected"] = _rejects(
        lambda: fuse({"a": good_q, "b": mismatched}, {"a": 0.5, "b": 0.5}))
    # review §4.3: without expected_keys the family stage must still compare seeds
    fixtures["no_expected_keys_shrink_rejected"] = _rejects(
        lambda: family_q([good[0], good[1].iloc[:-1]], "score", label="shrink"))
    fixtures["no_expected_keys_shrink_reversed_rejected"] = _rejects(
        lambda: family_q([good[1].iloc[:-1], good[0]], "score", label="shrink_rev"))
    fixtures["empty_input_rejected"] = _rejects(
        lambda: family_q([], "score", label="empty"))
    fixtures["bad_alpha_rejected"] = _rejects(lambda: push_down_alpha({"a": 1.0}, "b", 1.5))
    fixtures["all_rejected"] = all(fixtures.values())

    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    report = {"fixtures": fixtures}
    regression = {}
    for window in ("DEV19", "DEV20", "CONF21"):
        pool_frame = panel.loc[panel["date"] // 10000 == WINDOW_YEAR[window], ["date", "code"]]
        parents = {"c1": as_family(pd.read_parquet(
            S4 / f"v1_c1_seedmean__{window}.parquet"), column="c1", label="c1"),
            "tabm": as_family(pd.read_parquet(
                S4 / f"v1_tabm_seedmean__{window}.parquet"), column="tabm", label="tabm")}
        tcn = family_q([pd.read_parquet(
            V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "predictions.parquet")
            for seed in SEEDS], "score", expected_keys=pool_frame, label="tcn")
        components = dict(parents)
        components["tcn"] = tcn
        weights = push_down_alpha({"c1": 0.5, "tabm": 0.5}, "tcn", ALPHA)
        fused = fuse(components, weights)
        base = fuse(parents, {"c1": 0.5, "tabm": 0.5})
        labels = panel[["date", "code", "future_return"]]

        def ic(frame):
            merged = frame.merge(labels, on=["date", "code"], how="left",
                                 validate="one_to_one")
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
            return float(np.mean(values))
        regression[window] = {"parent_rankic": ic(base), "tcn_fusion_rankic": ic(fused),
                              "delta": ic(fused) - ic(base)}
    dev = float(np.average([regression[w]["delta"] for w in ("DEV19", "DEV20")],
                           weights=[244, 243]))
    report["regression"] = regression
    report["dev_merged_delta"] = dev
    report["matches_review_reference"] = abs(dev - 0.001748582) < 5e-9
    report["passed"] = bool(fixtures["all_rejected"] and report["matches_review_reference"])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
