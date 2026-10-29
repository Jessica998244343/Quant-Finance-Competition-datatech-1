"""§5 check: the canonical flat fusion is consistent and detects the nested-rank error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .s6_fusion import daily_centred_rank, family_q, fuse, push_down_alpha


def _toy():
    rng = np.random.default_rng(20261006)
    frames = {"c1": [], "tabm": []}
    for family, seeds in (("c1", 3), ("tabm", 3)):
        for seed in range(seeds):
            rows = []
            for date in (20200102, 20200103):
                for code in ("A", "B", "C", "D"):
                    rows.append({"date": date, "code": code,
                                 "score": float(rng.normal() + seed * 0.1)})
            frames[family].append(pd.DataFrame(rows))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/stage4_verified_ic/v2/v2_fusion_check.json")
    args = parser.parse_args()
    frames = _toy()
    q = {name: family_q(per_seed, "score") for name, per_seed in frames.items()}

    # reference: recompute q and u by hand
    reference = {}
    for family, per_seed in frames.items():
        long = pd.concat([frame.assign(seed=index) for index, frame in enumerate(per_seed)],
                         ignore_index=True)
        counts = long.groupby(["date", "seed"])["score"].transform("count")
        ranks = long.groupby(["date", "seed"])["score"].rank(method="average")
        long["centred"] = (ranks - (counts + 1.0) / 2.0) / counts
        reference[family] = long.groupby(["date", "code"], sort=True)["centred"].mean().reset_index(
            name="q")
    q_matches = True
    for name in q:
        produced = q[name].sort_values(["date", "code"]).reset_index(drop=True)["q"].to_numpy()
        expected = reference[name].sort_values(["date", "code"]).reset_index(drop=True)["q"].to_numpy()
        q_matches &= bool(np.allclose(produced, expected))

    flat = fuse(q, {"c1": 0.5, "tabm": 0.5})
    parent_sum = None
    for name in ("c1", "tabm"):
        part = q[name].rename(columns={"q": name})
        parent_sum = part if parent_sum is None else parent_sum.merge(
            part, on=["date", "code"], validate="one_to_one")
    manual = []
    for date, block in parent_sum.groupby("date", sort=True):
        u = 0.5 * block["c1"].to_numpy() + 0.5 * block["tabm"].to_numpy()
        manual.append(pd.DataFrame({"date": date, "code": block["code"],
                                    "score": daily_centred_rank(u)}))
    manual = pd.concat(manual, ignore_index=True)
    flat_matches_manual = bool(np.allclose(
        flat.sort_values(["date", "code"])["score"].to_numpy(),
        manual.sort_values(["date", "code"])["score"].to_numpy()))

    new_frame = frames["c1"][0].rename(columns={"score": "score"}).assign(
        score=lambda d: d["score"] * 2.0 + 0.3)
    q_new = family_q([new_frame], "score").rename(columns={"q": "new"})[["date", "code", "new"]]
    weights = push_down_alpha({"c1": 0.5, "tabm": 0.5}, "new", 0.10)
    components = {"c1": q["c1"], "tabm": q["tabm"], "new": q_new[["date", "code"]].assign(
        q=q_new["new"])}
    flat_add = fuse(components, weights)

    nested = []
    merged = parent_sum.merge(q_new, on=["date", "code"], validate="one_to_one")
    for date, block in merged.groupby("date", sort=True):
        u = 0.5 * block["c1"].to_numpy() + 0.5 * block["tabm"].to_numpy()
        nested.append(pd.DataFrame({
            "date": date, "code": block["code"],
            "score": daily_centred_rank(0.9 * daily_centred_rank(u)
                                        + 0.1 * block["new"].to_numpy())}))
    nested = pd.concat(nested, ignore_index=True)
    nested_self_consistent = bool(np.array_equal(
        nested.sort_values(["date", "code"])["score"].to_numpy(),
        daily_centred_rank(nested.sort_values(["date", "code"])["score"].to_numpy())))
    differs = not bool(np.allclose(
        flat_add.sort_values(["date", "code"])["score"].to_numpy(),
        nested.sort_values(["date", "code"])["score"].to_numpy()))

    report = {
        "graph": "flat: u = sum_g w_g q_g ; output = rank_t(u)",
        "family_stage_matches_reference": q_matches,
        "flat_fusion_matches_manual": flat_matches_manual,
        "alpha_weight_vector": weights,
        "nested_rank_differs_from_flat": differs,
        "nested_rank_is_self_consistent_but_a_different_algorithm": nested_self_consistent,
    }
    report["passed"] = bool(q_matches and flat_matches_manual and differs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
