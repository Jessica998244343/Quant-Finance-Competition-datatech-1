"""§4.4 / §8.1: true cut-off-2021 control models and half-year sensitivity.

``--deploy`` fits the frozen control (C1 x3 + TabM x3) with
``label_end_date <= 20211231`` and **saves models/preprocessing only** - the
2022 outer window is never evaluated, so no 2022 performance is produced.

``--halfyear`` trains the same control on labels mature before the end of June
and evaluates July-December, for the three windows required by §8.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from .s6_io import new_run_dir

from . import cv, s2r_train
from .s6_fusion import family_q  # noqa: F401  (kept for provenance of the graph)

V2 = Path("artifacts/stage4_verified_ic/v2")
SEEDS = (20261005, 20261017, 20261029)
ALLOW_EXISTING = False
EARLY_STOP_DAYS = 126
STEP, MAX_TREES, NEAR_TOL, THREADS = 25, 3000, 0.0005, 6
from .constants import F0_COLUMNS, G1_COLUMNS  # noqa: E402

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
DEPLOY = {"window": "RES22", "train_end": 20211231, "predict_start": 20220101,
          "predict_end": 20221231}
HALF_YEARS = {
    "2019H2": {"train_end": 20190628, "predict_start": 20190701, "predict_end": 20191231},
    "2020H2": {"train_end": 20200630, "predict_start": 20200701, "predict_end": 20201231},
    "2021H2": {"train_end": 20210630, "predict_start": 20210701, "predict_end": 20211231},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def custom_split(frame: pd.DataFrame, train_end: int, predict_start: int,
                 predict_end: int) -> dict:
    dates = cv.to_int_date(frame["date"])
    label_end = cv.to_int_date(frame["label_end_date"])
    eligible = label_end <= train_end
    train_dates = np.sort(pd.unique(dates[eligible]))
    es_days = min(EARLY_STOP_DAYS, max(1, len(train_dates) // 4))
    es_start = int(train_dates[-es_days])
    return {"train_end": train_end, "es_start": es_start, "outer_train": eligible,
            "internal": eligible & (label_end < es_start),
            "early_stop": eligible & (dates >= es_start),
            "predict": (dates >= predict_start) & (dates <= predict_end)}


def train_catboost(data: pd.DataFrame, split: dict, seed: int, out: Path,
                   evaluate: bool) -> dict:
    from catboost import CatBoostRegressor, Pool

    dates = cv.to_int_date(data["date"])
    y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    w_internal = s2r_train.normalise_within(raw_w, internal)
    w_outer = s2r_train.normalise_within(raw_w, outer)
    base = dict(loss_function="RMSE", iterations=MAX_TREES, learning_rate=0.03, depth=8,
                l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
                bootstrap_type="Bernoulli", subsample=0.8, thread_count=THREADS,
                random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    inner = CatBoostRegressor(**base)
    inner.fit(Pool(x[internal], label=y[internal], weight=w_internal[internal],
                   feature_names=FEATURES),
              eval_set=Pool(x[early], label=y[early], weight=w_internal[early],
                            feature_names=FEATURES),
              use_best_model=False)
    valid = np.nonzero(early)[0]
    checkpoints = list(range(STEP, MAX_TREES + 1, STEP))
    curve = np.array([cv.daily_spearman(dates[valid], raw_returns[valid],
                                        inner.predict(x[early], ntree_end=trees))
                      for trees in checkpoints])
    best = int(np.nanargmax(curve))
    near = np.nonzero(curve >= float(curve[best]) - NEAR_TOL)[0]
    selected = int(checkpoints[int(near[0])])
    final = CatBoostRegressor(**{**base, "iterations": selected, "od_type": None, "od_wait": None})
    final.fit(Pool(x[outer], label=y[outer], weight=w_outer[outer], feature_names=FEATURES))
    final.save_model(str(out / f"catboost_c1_seed{seed}.cbm"), format="cbm")
    pd.DataFrame({"trees": checkpoints, "inner_daily_ic": curve}).to_parquet(
        out / f"catboost_c1_seed{seed}__curve.parquet", index=False)
    record = {"seed": seed, "selected_trees": selected,
              "best_checkpoint": int(checkpoints[best]),
              "best_inner_ic": float(curve[best]),
              "actual_tree_count": int(final.tree_count_),
              "curve_file": f"catboost_c1_seed{seed}__curve.parquet",
              "outer_metrics_computed": False}
    if evaluate:
        predict = split["predict"]
        preds = np.asarray(final.predict(x[predict]), dtype=float).ravel()
        frame = pd.DataFrame({"date": dates[predict].astype("int64"),
                              "code": data.loc[predict, "code"].to_numpy(),
                              "score": preds})
        frame.to_parquet(out / f"catboost_c1_seed{seed}__predictions.parquet", index=False)
    return record


def train_tabm(data: pd.DataFrame, split: dict, seed: int, out: Path,
               evaluate: bool) -> dict:
    import torch
    from .s6_v2_tabm_variants import Design, _groups, _new, _predict, _run

    torch.set_num_threads(2)
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    internal, early = split["internal"] & finite, split["early_stop"] & finite
    outer = split["outer_train"] & finite
    inner_design = Design().fit(data.loc[internal])
    x_inner = inner_design.transform(data)
    tensor_x, tensor_z = torch.from_numpy(x_inner), torch.from_numpy(np.nan_to_num(z))
    inner_groups, valid_groups = _groups(dates, internal), _groups(dates, early)
    model, optimizer, rng = _new(x_inner.shape[1], seed, 0.1)
    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, 101):
        _run(model, optimizer, rng, tensor_x, tensor_z, inner_groups, 1, "mse")
        ic = _daily_ic_wrapper(raw, valid_groups, _predict(model, x_inner, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= 12:
            break
    outer_design = Design().fit(data.loc[outer])
    x_outer = outer_design.transform(data)
    model2, optimizer2, rng2 = _new(x_outer.shape[1], seed, 0.1)
    _run(model2, optimizer2, rng2, torch.from_numpy(x_outer), tensor_z,
         _groups(dates, outer), best_epoch, "mse")
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in model2.state_dict().items()},
              str(out / f"tabm_seed{seed}.safetensors"))
    (out / f"design_seed{seed}.json").write_text(json.dumps(outer_design.to_json(), indent=2))
    pd.DataFrame(history).to_parquet(out / f"inner_curve_seed{seed}.parquet", index=False)
    record = {"seed": seed, "best_epoch": int(best_epoch), "best_inner_ic": float(best_ic),
              "inner_epochs_run": len(history), "outer_metrics_computed": False}
    if evaluate:
        predict = split["predict"]
        preds = _predict(model2, x_outer, _groups(dates, predict))
        keep = np.nonzero(predict)[0]
        pd.DataFrame({"date": dates[keep].astype("int64"),
                      "code": data.loc[keep, "code"].to_numpy(),
                      "score": preds[keep]}).to_parquet(
            out / f"tabm_seed{seed}__predictions.parquet", index=False)
    return record


def _daily_ic_wrapper(raw, groups, preds) -> float:
    values = []
    for group in groups:
        y, p = raw[group], preds[group]
        keep = np.isfinite(y)
        if keep.sum() < 2:
            continue
        ry = pd.Series(y[keep]).rank(method="average")
        rp = pd.Series(p[keep]).rank(method="average")
        if ry.nunique() == 1 or rp.nunique() == 1:
            continue
        values.append(float(np.corrcoef(ry, rp)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def _frame_daily_ic(frame: pd.DataFrame) -> float:
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
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--cat-only", action="store_true",
                        help="with --deploy: only re-run CatBoost (TabM records read from saved curves)")
    parser.add_argument("--halfyear", action="store_true")
    parser.add_argument("--hy-windows", default="2019H2,2020H2,2021H2",
                        help="subset of H2 windows to train")
    parser.add_argument("--seed", type=int, default=0,
                        help="override the seed list for the half-year runs")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--l5", action="store_true",
                        help="train the L5 (TREE structure control) boosters at cutoff 2021")
    parser.add_argument("--out", default="",
                        help="default: v2_deploy_<mode>.json (deploy and halfyear never share a file)")
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    if args.allow_existing:
        globals()["ALLOW_EXISTING"] = True
    if not args.out:
        mode = "deploy" if args.deploy else ("halfyear" if args.halfyear else "none")
        args.out = str(V2 / f"v2_rev1_deploy_models_{mode}.json")
    report = {"deploy": {}, "half_year": {}}
    if args.l5:
        import lightgbm as lgb
        spec = DEPLOY
        split = custom_split(data, spec["train_end"], spec["predict_start"],
                             spec["predict_end"])
        y, raw_w, finite = s2r_train.rank_target_unnormalised(data)
        z = y
        w = s2r_train.normalise_within(raw_w, split["outer_train"] & finite)
        x = data[FEATURES].to_numpy(dtype=np.float32)
        mask = split["outer_train"] & finite
        out = V2 / "deploy_2021cutoff"
        out = new_run_dir(out, ALLOW_EXISTING)
        params = {"objective": "regression", "metric": "None", "learning_rate": 0.03,
                  "num_leaves": 127, "min_data_in_leaf": 800, "feature_fraction": 0.8,
                  "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l1": 0.1,
                  "lambda_l2": 40.0, "max_bin": 127, "deterministic": True,
                  "force_row_wise": True, "verbose": -1, "num_threads": 6}
        records = []
        for seed in SEEDS:
            run = dict(params)
            run.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
                        "data_random_seed": seed})
            dataset = lgb.Dataset(x[mask], label=z[mask], weight=w[mask],
                                  feature_name=FEATURES)
            booster = lgb.train(run, dataset, num_boost_round=311)
            path = out / f"l5_seed{seed}.txt"
            booster.save_model(str(path))
            records.append({"seed": seed, "num_rounds": 311, "file": path.name,
                            "sha256": _sha(path)})
            print("L5 seed", seed, "done", flush=True)
        report["l5_cutoff2021"] = {
            "train_label_cutoff": spec["train_end"], "rounds": 311,
            "max_outer_label_end": int(np.nanmax(cv.to_int_date(
                data.loc[mask, "label_end_date"]))),
            "outer_metrics_computed": False, "records": records,
            "note": "TREE structure control (0.5 x L5 + 0.5 x C1); L5 fixed at 311 rounds"}
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    if args.deploy:
        spec = DEPLOY
        split = custom_split(data, spec["train_end"], spec["predict_start"],
                             spec["predict_end"])
        out = V2 / "deploy_2021cutoff"
        out = new_run_dir(out, ALLOW_EXISTING)
        started = time.time()
        trees, tabm = [], []
        for seed in SEEDS:
            trees.append(train_catboost(data, split, seed, out, evaluate=False))
            if args.cat_only:
                curve = pd.read_parquet(out / f"inner_curve_seed{seed}.parquet")
                index = int(np.nanargmax(curve["valid_daily_ic"].to_numpy()))
                tabm.append({"seed": seed,
                             "best_epoch": int(curve["epoch"].to_numpy()[index]),
                             "best_inner_ic": float(curve["valid_daily_ic"].to_numpy()[index]),
                             "inner_epochs_run": int(len(curve)),
                             "outer_metrics_computed": False,
                             "source": "rebuilt from the saved inner curve (not a new training run)"})
            else:
                tabm.append(train_tabm(data, split, seed, out, evaluate=False))
            print("deploy seed", seed, "done", flush=True)
        report["deploy"] = {
            "train_label_cutoff": spec["train_end"], "predict_window_evaluated": False,
            "max_outer_label_end": int(np.nanmax(cv.to_int_date(
                data.loc[split["outer_train"], "label_end_date"]))),
            "seconds": round(time.time() - started, 1),
            "catboost": trees, "tabm": tabm,
            "artifacts": str(out),
            "rebuilt_metadata": bool(args.cat_only),
            "rebuilt_note": ("CatBoost curves re-run with the same frozen protocol; TabM records "
                             "rebuilt from curves saved during the original run." if args.cat_only
                             else "original run"),
            "note": "trained on label_end<=20211231; 2022 outer metrics NOT computed"}
    if args.halfyear:
        hy_seeds = (args.seed,) if args.seed else (20261005,)
        for window, spec in HALF_YEARS.items():
            if window not in args.hy_windows.split(","):
                continue
            window_start = time.time()
            split = custom_split(data, spec["train_end"], spec["predict_start"],
                                 spec["predict_end"])
            labels = data[["date", "code", "future_return"]]
            scores = []
            for seed in hy_seeds:
                out = V2 / f"halfyear_{window}_s{seed}"
                out = new_run_dir(out, ALLOW_EXISTING)
                train_catboost(data, split, seed, out, evaluate=True)
                train_tabm(data, split, seed, out, evaluate=True)
                c1 = pd.read_parquet(out / f"catboost_c1_seed{seed}__predictions.parquet")
                tabm_frame = pd.read_parquet(out / f"tabm_seed{seed}__predictions.parquet")

                def centred(values):
                    ranks = pd.Series(values).rank(method="average")
                    return ((ranks - (len(ranks) + 1) / 2.0) / len(ranks)).to_numpy()
                merged = c1.rename(columns={"score": "c1"}).merge(
                    tabm_frame.rename(columns={"score": "tabm"}),
                    on=["date", "code"], validate="one_to_one").merge(
                    labels, on=["date", "code"], how="left", validate="one_to_one")
                pieces = []
                for date, block in merged.groupby("date", sort=True):
                    value = 0.5 * centred(block["c1"].to_numpy()) + 0.5 * centred(
                        block["tabm"].to_numpy())
                    pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                                "score": centred(value),
                                                "future_return": block["future_return"].to_numpy()}))
                scores.append(_frame_daily_ic(pd.concat(pieces, ignore_index=True)))
            report["half_year"][window] = {
                "train_label_cutoff": spec["train_end"], "anchor_rankic": float(np.mean(scores)),
                "seconds": round(time.time() - window_start, 1)}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
