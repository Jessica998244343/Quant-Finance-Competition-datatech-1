"""REV1: rebuild both new blends correctly and verify the fix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import s2r_blend

REV = Path("artifacts/stage2_ic_ir/a_revision1")
PRED = REV / "predictions"
A_PRED = Path("artifacts/stage2_ic_ir/predictions")
V1_PRED = Path("artifacts/beginning/predictions")
WINDOWS = ["DEV19", "DEV20", "CONF21"]
WEIGHTS = [0.75, 0.25]


def _members(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, ["date", "code", *columns]].copy()
    out["date"] = out["date"].astype("int64")
    return out


def main() -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    report = {"windows": {}, "legacy_blend_verification": {}, "bug_effect": {}}

    for window in WINDOWS:
        # --- stage-1 legacy members ---------------------------------------
        p5 = pd.read_parquet(V1_PRED / f"P5_s20261005__{window}.parquet")
        cat = pd.read_parquet(V1_PRED / f"SC_d8_l2_15__{window}.parquet")
        legacy_members = p5.merge(cat, on=["date", "code"], suffixes=("_p5", "_cat"),
                                  validate="one_to_one").rename(
            columns={"score_p5": "lgbm_p5", "score_cat": "catboost"})
        legacy_members = _members(legacy_members, ["lgbm_p5", "catboost"])
        legacy_fixed = s2r_blend.blend_frame(legacy_members, ["lgbm_p5", "catboost"], WEIGHTS)
        saved = pd.read_parquet(V1_PRED / f"SSTAR__{window}.parquet")
        check = legacy_fixed.rename(columns={"score": "a"}).merge(
            saved.rename(columns={"score": "b"}), on=["date", "code"], validate="one_to_one")
        report["legacy_blend_verification"][window] = {
            "max_abs_diff": float((check["a"] - check["b"]).abs().max()),
            "identical": bool(np.allclose(check["a"], check["b"], atol=0, rtol=0)),
            "note": "stage-1 legacy blend (SSTAR) is per-day; the fix reproduces it exactly",
        }
        legacy_fixed.to_parquet(PRED / f"legacy_oof_blend__{window}.parquet", index=False)

        # --- A-stage members produced under real platform inputs -----------
        for tag, source in (("origmodel_realinput",
                             A_PRED / f"origmodel_realinput_members__{window}.parquet"),
                            ("aligned_retrain",
                             A_PRED / f"aligned_retrain_members__{window}.parquet")):
            frame = pd.read_parquet(source)
            frame["date"] = pd.to_numeric(frame["date"]).astype("int64")
            members = _members(frame, ["score_lgbm_p5", "score_catboost"]).rename(
                columns={"score_lgbm_p5": "lgbm_p5", "score_catboost": "catboost"})
            fixed = s2r_blend.blend_frame(members, ["lgbm_p5", "catboost"], WEIGHTS)
            if "score" in frame.columns:
                wrong = pd.DataFrame({"date": members["date"], "code": members["code"],
                                      "score": frame["score"].to_numpy()})
            else:
                wrong = s2r_blend.blend_whole_frame_wrong(
                    members, ["lgbm_p5", "catboost"], WEIGHTS)
            fixed.to_parquet(PRED / f"{tag}__{window}.parquet", index=False)
            members.to_parquet(PRED / f"{tag}_members__{window}.parquet", index=False)
            effect = s2r_blend.rank_disagreement(fixed, wrong)
            report["bug_effect"][f"{tag}__{window}"] = effect

        # --- perturbation / single-day equivalence -------------------------
        probe = int(sorted(legacy_members["date"].unique())[len(legacy_members["date"].unique()) // 2])
        report["windows"].setdefault(window, {})["perturbation"] = \
            s2r_blend.perturbation_check(
                _members(pd.read_parquet(
                    A_PRED / f"origmodel_realinput_members__{window}.parquet").assign(
                        date=lambda d: pd.to_numeric(d["date"]).astype("int64")),
                    ["score_lgbm_p5", "score_catboost"]).rename(
                    columns={"score_lgbm_p5": "lgbm_p5", "score_catboost": "catboost"}),
                ["lgbm_p5", "catboost"], WEIGHTS, probe)

    s2r_blend.write_json(REV / "blend_rebuild_verification.json", report)
    print(json.dumps(report["legacy_blend_verification"], indent=2))
    print(json.dumps(report["bug_effect"], indent=2))
    for window, value in report["windows"].items():
        print(window, json.dumps(value["perturbation"]))


if __name__ == "__main__":
    main()
