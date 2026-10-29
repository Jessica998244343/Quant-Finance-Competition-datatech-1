"""§6 correction: F1/F2/F3 ablation with full artifacts and the flat-graph delta.

Saves, per run: CatBoost model, inner curve, per-day predictions and a manifest
binding the feature list (hash), split boundaries and tree counts.  Marginal IC is
measured against the frozen C1 control (same seed) and, under the canonical flat
graph, as an extra member of the frozen anchor.  F1+F2+F3 (F123) is kept as
history only and is not extended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS
from .s6_fusion import as_family, family_q, fuse, push_down_alpha
from .s6_v2_features import GROUPS

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
V2 = S4 / "v2"
OUT = V2 / "features_rev1"
BASE_FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEED = 20261005
STEP, MAX_TREES, NEAR_TOL, THREADS = 25, 3000, 0.0005, 6
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train(data: pd.DataFrame, window: str, group: str) -> dict:
    from catboost import CatBoostRegressor, Pool

    features = BASE_FEATURES + list(GROUPS[group])
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    label_end = cv.to_int_date(data["label_end_date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    x = data[features].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)
    base = dict(loss_function="RMSE", iterations=MAX_TREES, learning_rate=0.03, depth=8,
                l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
                bootstrap_type="Bernoulli", subsample=0.8, thread_count=THREADS,
                random_seed=SEED, nan_mode="Min", verbose=False, allow_writing_files=False)
    started = time.time()
    inner = CatBoostRegressor(**base)
    inner.fit(Pool(x[internal], label=y[internal], weight=w_internal[internal],
                   feature_names=features),
              eval_set=Pool(x[early], label=y[early], weight=w_internal[early],
                            feature_names=features), use_best_model=False)
    valid = np.nonzero(early)[0]
    checkpoints = list(range(STEP, MAX_TREES + 1, STEP))
    curve = np.array([cv.daily_spearman(dates[valid], raw_returns[valid],
                                        inner.predict(x[early], ntree_end=trees))
                      for trees in checkpoints])
    best = int(np.nanargmax(curve))
    near = np.nonzero(curve >= float(curve[best]) - NEAR_TOL)[0]
    selected = int(checkpoints[int(near[0])])
    final = CatBoostRegressor(**{**base, "iterations": selected, "od_type": None, "od_wait": None})
    final.fit(Pool(x[outer], label=y[outer], weight=w_outer[outer], feature_names=features))
    predict = split["predict"]
    preds = np.asarray(final.predict(x[predict]), dtype=float).ravel()
    predictions = pd.DataFrame({"date": dates[predict].astype("int64"),
                                "code": data.loc[predict, "code"].to_numpy(),
                                "score": preds})
    tag = f"{group}_{SEED}__{window}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out / "model.cbm"), format="cbm")
    pd.DataFrame({"trees": checkpoints, "inner_daily_ic": curve}).to_parquet(
        out / "inner_curve.parquet", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    manifest = {
        "tag": tag, "group": group, "window": window, "seed": SEED,
        "features": features, "feature_list_sha256": _sha_text("\n".join(features)),
        "n_features": len(features), "selected_trees": selected,
        "best_checkpoint": int(checkpoints[best]), "best_inner_ic": float(curve[best]),
        "actual_tree_count": int(final.tree_count_),
        "train_label_cutoff": int(split["train_end"]), "es_start": int(split["es_start"]),
        "max_inner_label_end": int(np.nanmax(label_end[internal])),
        "max_outer_label_end": int(np.nanmax(label_end[outer])),
        "n_internal": int(internal.sum()), "n_early": int(early.sum()),
        "n_outer": int(outer.sum()), "train_threads": THREADS,
        "seconds": round(time.time() - started, 1),
        "feature_recipe": FEATURE_RECIPE}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest | {"predictions": predictions}


FEATURE_RECIPE = {
    "F1": "f1_ret_1_5 = C[t-1]/C[t-5]-1; f1_ret_5_20 = C[t-5]/C[t-20]-1; "
          "endpoints must be finite positive closes, otherwise NaN; no forward fill",
    "F2": "amt_abn = log1p(amount_t) - median(log1p(amount[t-20:t-1])) with >=16 valid "
          "history values else NaN; negative/zero amount -> NaN before log1p; "
          "turn_abn identical construction on Factor33 'turn' (turn<=0 -> NaN); "
          "interactions = centred cross-sectional rank of ret_1 (resp. ret_5) times "
          "centred rank of amt_abn within the same legal pool, no future ranking",
    "F3": "on = log(O_j/C[j-1]), intra = log(C_j/O_j); f3_on_intra_mean_5/20 = mean over "
          "the window of (on-intra) with >=min(16,window) valid days else NaN; "
          "f3_path_eff_20 = |sum(r)|/(sum|r|+1e-12), f3_downside_share_20 = "
          "sum(min(r,0)^2)/(sum(r^2)+1e-12), r = 20 one-day log returns (21 price points), "
          ">=16 valid returns else NaN",
    "suspension": "suspended or missing days stay missing (no fill); the panel key set is "
                  "identical to panel_platform (same legal pool)",
}


def frame_ic(frame: pd.DataFrame) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="F1,F2,F3")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--out", default=str(V2 / "v2_rev1_feature_ablation.json"))
    args = parser.parse_args()
    panel = pd.read_parquet(V2 / "panel_v2.parquet").sort_values(["date", "code"]).reset_index(drop=True)
    labels = panel[["date", "code", "future_return"]]
    report = {"seed": SEED, "alphas": list(ALPHAS), "records": []}
    for window in args.windows.split(","):
        parents = {"c1": as_family(pd.read_parquet(
            S4 / f"v1_c1_seedmean__{window}.parquet"), column="c1", label=f"c1_{window}"),
            "tabm": as_family(pd.read_parquet(
                S4 / f"v1_tabm_seedmean__{window}.parquet"), column="tabm",
                label=f"tabm_{window}")}
        control = pd.read_parquet(FUS / "predictions_cat" / f"C1_{SEED}__{window}.parquet")
        control_ic = frame_ic(control.merge(labels, on=["date", "code"], how="left",
                                            validate="one_to_one"))
        for group in args.groups.split(","):
            manifest = train(panel, window, group)
            frame = manifest["predictions"].merge(labels, on=["date", "code"], how="left",
                                                  validate="one_to_one")
            new_q = family_q([manifest["predictions"]], "score")
            components = dict(parents)
            components[group] = new_q
            curve = {}
            for alpha in ALPHAS:
                weights = push_down_alpha({"c1": 0.5, "tabm": 0.5}, group, alpha)
                curve[str(alpha)] = frame_ic(fuse(components, weights).merge(
                    labels, on=["date", "code"], how="left", validate="one_to_one"))
            record = {"window": window, "group": group,
                      "selected_trees": manifest["selected_trees"],
                      "standalone_rankic": frame_ic(frame), "control_rankic": control_ic,
                      "delta_vs_control": frame_ic(frame) - control_ic,
                      "delta_vs_parent": {k: curve[k] - curve["0.0"] for k in curve},
                      "manifest": str(OUT / manifest["tag"] / "manifest.json")}
            report["records"].append(record)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else
                                  ({a: round(x, 6) for a, x in v.items()} if isinstance(v, dict) else v))
                              for k, v in record.items()}), flush=True)
    dev = {}
    for group in args.groups.split(","):
        rows = [r for r in report["records"] if r["group"] == group]
        dev[group] = {
            "delta_vs_control": float(np.average([r["delta_vs_control"] for r in rows],
                                                 weights=[244, 243])) if len(rows) == 2 else None,
            "delta_vs_parent": {a: float(np.average([r["delta_vs_parent"][a] for r in rows],
                                                    weights=[244, 243])) for a in r["delta_vs_parent"]}
            if len(rows) == 2 else None}
    report["dev_merged"] = dev
    report["retained"] = [group for group, values in dev.items()
                          if values["delta_vs_parent"]
                          and max(values["delta_vs_parent"].values()) >= 0.0005]
    report["note"] = ("flat-graph increments; F123 (three-group joint) is kept as history "
                      "only and is not re-trained or extended")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"dev_merged": dev, "retained": report["retained"]}, indent=2))


if __name__ == "__main__":
    main()
