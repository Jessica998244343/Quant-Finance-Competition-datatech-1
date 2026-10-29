"""REV1: verify CORR-2 (fold-local weights) and CORR-1 (CatBoost rounds)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, s2r_train, trainers

REV = Path("artifacts/stage2_ic_ir/a_revision1")
FEATURES = None


def _sample(panel: pd.DataFrame, window: str) -> pd.DataFrame:
    split = cv.fold_split(panel, window)
    mask = split["internal"] & panel["future_return"].notna().to_numpy()
    frame = panel.loc[mask, ["date", "code", "future_return"]].copy()
    return frame, split


def verify(window: str = "DEV19") -> dict:
    full = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    full = full.sort_values(["date", "code"]).reset_index(drop=True)
    split = cv.fold_split(full, window)

    # real perturbations: drop everything after the fitting cutoff, and separately
    # invalidate labels of those future rows
    cutoff = int(split["train_end"])
    future_rows = int((pd.to_numeric(full["date"]) > cutoff).sum())
    truncated = full.loc[pd.to_numeric(full["date"]) <= cutoff].reset_index(drop=True)
    invalidated = full.copy()
    invalidated.loc[pd.to_numeric(invalidated["date"]) > cutoff, "future_return"] = np.nan

    def weights_local(frame: pd.DataFrame) -> pd.Series:
        s = cv.fold_split(frame, window)
        z, raw_w, finite = s2r_train.rank_target_unnormalised(frame)
        mask = s["internal"] & finite
        w = s2r_train.normalise_within(raw_w, mask)
        return pd.Series(w[mask], index=pd.MultiIndex.from_arrays(
            [frame.loc[mask, "date"], frame.loc[mask, "code"]]), name="w_local")

    def weights_global(frame: pd.DataFrame) -> pd.Series:
        s = cv.fold_split(frame, window)
        z, raw_w, finite = s2r_train.rank_target_unnormalised(frame)
        w = raw_w / np.nanmean(raw_w[finite])          # the old, future-dependent constant
        mask = s["internal"] & finite
        return pd.Series(w[mask], index=pd.MultiIndex.from_arrays(
            [frame.loc[mask, "date"], frame.loc[mask, "code"]]), name="w_global")

    base_local = weights_local(full)
    base_global = weights_global(full)
    out = {
        "window": window,
        "cutoff": cutoff,
        "perturbed_future_rows": future_rows,
        "n_internal_samples": int(len(base_local)),
        "old_global_normalisation_constant": float(np.nanmean(
            s2r_train.rank_target_unnormalised(full)[1][
                full["future_return"].notna().to_numpy()])),
        "internal_weight_mean_old": float(base_global.mean()),
    }
    for label, frame in (("drop_future_rows", truncated),
                         ("invalidate_future_labels", invalidated)):
        local = weights_local(frame)
        glob = weights_global(frame)
        joined_local = pd.concat([base_local.rename("base"), local.rename("new")], axis=1,
                                 join="inner")
        joined_glob = pd.concat([base_global.rename("base"), glob.rename("new")], axis=1,
                                join="inner")
        out[label] = {
            "common_samples": int(len(joined_local)),
            "fold_local_max_abs_weight_diff": float(
                (joined_local["base"] - joined_local["new"]).abs().max())
            if len(joined_local) else None,
            "old_global_max_abs_weight_diff": float(
                (joined_glob["base"] - joined_glob["new"]).abs().max())
            if len(joined_glob) else None,
        }
    return out


def main() -> None:
    report = {"corr2_weight_normalisation": {w: verify(w) for w in ("DEV19", "DEV20")}}
    # CORR-1 evidence: the corrected trainer records index and tree count separately
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model("artifacts/stage2_ic_ir/models_fold/aligned_catboost_DEV19.cbm")
    report["corr1_catboost_rounds"] = {
        "old_implementation": "iterations = get_best_iteration(); final tree count = that value",
        "example_old_model_tree_count": int(model.tree_count_),
        "note": ("index is 0-based; the corrected trainer uses index+1 as the tree count "
                 "and only falls back to the full iteration budget when "
                 "get_best_iteration() is None"),
    }
    (REV / "corr_verification.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
