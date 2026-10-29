"""Reproduce V3_full_pkg/model from the official data (training entry, full 2016-2022).

Run standalone:  python train.py --data-dir <official data dir> --out model

Training needs the extra packages in train_requirements.txt (torch stack);
inference needs only requirements.txt.  The three families use the frozen
recipes: CatBoost C1 (depth8/l2=15, 25-tree inner IC selection), TabM
(k8/width128/2 blocks/dropout0.1, inner IC epoch selection) and TCN W20
(8 channels, 20 steps, dilations 1/2, inner IC epoch selection).  Every member
is refit from scratch on ``label_end_date <= 2022-12-30`` and exported to the
static NumPy layout read by the package scorer.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_cv
import alpha_panel
import alpha_train_tabm
import alpha_tcn_train
from alpha_constants import F0_COLUMNS, G1_COLUMNS

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
TRAIN_END = 20221230
EARLY_STOP_DAYS = 126
CAT_STEP = 25
CAT_MAX_TREES = 3000
CAT_NEAR_TOL = 0.0005
CAT_THREADS = 6
TABM_THREADS = 2
TABM = dict(k=8, width=128, blocks=2, dropout=0.1)
WEIGHTS = {"c1": 0.375, "tabm": 0.375, "tcn": 0.25}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_split(panel: pd.DataFrame) -> dict:
    dates = alpha_cv.to_int_date(panel["date"])
    label_end = alpha_cv.to_int_date(panel["label_end_date"])
    eligible = label_end <= TRAIN_END
    train_dates = np.sort(pd.unique(dates[eligible]))
    es_days = min(EARLY_STOP_DAYS, max(1, len(train_dates) // 4))
    es_start = int(train_dates[-es_days])
    return {"train_end": TRAIN_END, "es_start": es_start, "outer_train": eligible,
            "internal": eligible & (label_end < es_start),
            "early_stop": eligible & (dates >= es_start),
            "predict": np.zeros(len(panel), dtype=bool)}


def rank_target(frame: pd.DataFrame):
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    z = ((ranks - (counts + 1.0) / 2.0) / counts).to_numpy(dtype=float)
    raw_w = (1.0 / counts).to_numpy(dtype=float)
    return z, raw_w, finite


def normalise_within(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(weights, dtype=float, copy=True)
    mean = float(np.nanmean(out[mask]))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid weight mean inside fitting sample")
    return out / mean


def train_catboost(panel: pd.DataFrame, split: dict, seed: int, threads: int, out: Path) -> dict:
    from catboost import CatBoostRegressor, Pool

    dates = alpha_cv.to_int_date(panel["date"])
    z, raw_w, finite = rank_target(panel)
    raw_returns = pd.to_numeric(panel["future_return"], errors="coerce").to_numpy(dtype=float)
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    w_internal = normalise_within(raw_w, internal)
    w_outer = normalise_within(raw_w, outer)
    base = dict(loss_function="RMSE", iterations=CAT_MAX_TREES, learning_rate=0.03,
                depth=8, l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
                bootstrap_type="Bernoulli", subsample=0.8, thread_count=threads,
                random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    inner = CatBoostRegressor(**base)
    inner.fit(Pool(x[internal], label=z[internal], weight=w_internal[internal],
                   feature_names=FEATURES),
              eval_set=Pool(x[early], label=z[early], weight=w_internal[early],
                            feature_names=FEATURES), use_best_model=False)
    valid_index = np.nonzero(early)[0]
    checkpoints = list(range(CAT_STEP, CAT_MAX_TREES + 1, CAT_STEP))
    curve = np.array([alpha_cv.daily_spearman(dates[valid_index], raw_returns[valid_index],
                                              inner.predict(x[early], ntree_end=trees))
                      for trees in checkpoints], dtype=float)
    best_index = int(np.nanargmax(curve))
    near = np.nonzero(curve >= float(curve[best_index]) - CAT_NEAR_TOL)[0]
    selected = int(checkpoints[int(near[0])])
    final = CatBoostRegressor(**{**base, "iterations": selected, "od_type": None, "od_wait": None})
    final.fit(Pool(x[outer], label=z[outer], weight=w_outer[outer], feature_names=FEATURES))
    final.save_model(str(out / f"catboost_c1_seed{seed}.cbm"), format="cbm")
    pd.DataFrame({"trees": checkpoints, "inner_daily_ic": curve}).to_parquet(
        out / f"catboost_c1_seed{seed}__curve.parquet", index=False)
    return {"seed": seed, "selected_trees": selected,
            "best_checkpoint": int(checkpoints[best_index]),
            "best_inner_ic": float(curve[best_index]),
            "actual_tree_count": int(final.tree_count_)}


def export_tabm_npz(state_dict, path: Path) -> None:
    arrays = {
        "W0": state_dict["backbone.blocks.0.0.weight"],
        "r0": state_dict["backbone.blocks.0.0.r"],
        "s0": state_dict["backbone.blocks.0.0.s"],
        "b0": state_dict["backbone.blocks.0.0.bias"],
        "W1": state_dict["backbone.blocks.1.0.weight"],
        "r1": state_dict["backbone.blocks.1.0.r"],
        "s1": state_dict["backbone.blocks.1.0.s"],
        "b1": state_dict["backbone.blocks.1.0.bias"],
        "Wh": state_dict["head.weight"],
        "bh": state_dict["head.bias"],
        "k": np.array([TABM["k"]], dtype=np.float32),
    }
    np.savez(path, **{name: np.ascontiguousarray(value, dtype=np.float32)
                      for name, value in arrays.items()})


def export_tcn_npz(state_dict, path: Path) -> None:
    arrays = {
        "static_W": state_dict["static.0.weight"], "static_b": state_dict["static.0.bias"],
        "input_W": state_dict["input.weight"], "input_b": state_dict["input.bias"],
        "fuse0_W": state_dict["fuse.0.weight"], "fuse0_b": state_dict["fuse.0.bias"],
        "fuse2_W": state_dict["fuse.2.weight"], "fuse2_b": state_dict["fuse.2.bias"],
    }
    for i in range(2):
        arrays[f"b{i}_c1w"] = state_dict[f"blocks.{i}.conv1.weight"]
        arrays[f"b{i}_c1b"] = state_dict[f"blocks.{i}.conv1.bias"]
        arrays[f"b{i}_c2w"] = state_dict[f"blocks.{i}.conv2.weight"]
        arrays[f"b{i}_c2b"] = state_dict[f"blocks.{i}.conv2.bias"]
    np.savez(path, **{name: np.ascontiguousarray(value, dtype=np.float32)
                      for name, value in arrays.items()})


def train_tabm(panel: pd.DataFrame, split: dict, seed: int, out: Path) -> dict:
    import torch

    torch.set_num_threads(TABM_THREADS)
    result = alpha_train_tabm.train_fold(panel, split, seed)
    export_tabm_npz({k: v.detach().cpu().numpy()
                     for k, v in result["model"].state_dict().items()},
                    out / f"tabm_seed{seed}.npz")
    return {"best_epoch": int(result["best_epoch"]), "best_inner_ic": float(result["best_inner_ic"]),
            "inner_epochs_run": int(len(result["history"])), "design": result["design"]}


def train_tcn(panel: pd.DataFrame, sequences, keys, split: dict, seed: int, out: Path) -> dict:
    result = alpha_tcn_train.train_fold(panel, sequences, keys, split, seed)
    export_tcn_npz({k: v.detach().cpu().numpy()
                    for k, v in result["model"].state_dict().items()},
                   out / f"tcn_seed{seed}.npz")
    return {"best_epoch": int(result["best_epoch"]), "best_inner_ic": float(result["best_inner_ic"]),
            "seconds": result["seconds"], "design": result["design"],
            "channel_stats": result["channel_stats"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="model")
    parser.add_argument("--threads", type=int, default=CAT_THREADS)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    features, labels = alpha_panel.build_panel(args.data_dir, membership_filter=True)
    panel = alpha_panel.merge_labels(features, labels)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    split = full_split(panel)
    sequences, keys = alpha_tcn_train.build_sequences(args.data_dir)

    cat_records, designs, tcn_pre, tcn_stats = {}, [], None, None
    for seed in SEEDS:
        cat_records[seed] = train_catboost(panel, split, seed, args.threads, out)
        tabm = train_tabm(panel, split, seed, out)
        tcn = train_tcn(panel, sequences, keys, split, seed, out)
        designs.append(json.dumps(tabm["design"], sort_keys=True))
        if tcn_pre is None:
            tcn_pre, tcn_stats = tcn["design"], tcn["channel_stats"]
        print("done seed", seed, "catboost", cat_records[seed]["actual_tree_count"],
              "tabm", tabm["best_epoch"], "tcn", tcn["best_epoch"], flush=True)

    if len(set(designs)) != 1:
        raise RuntimeError("TabM designs differ across seeds")
    (out / "tabm_design.json").write_text(designs[0])
    (out / "tcn_design.json").write_text(json.dumps(tcn_pre, indent=2))
    (out / "tcn_channel_stats.json").write_text(
        json.dumps({str(k): v for k, v in tcn_stats.items()}, indent=2))

    members = {"c1": [f"catboost_c1_seed{s}.cbm" for s in SEEDS],
               "tabm": [f"tabm_seed{s}.npz" for s in SEEDS],
               "tcn": [f"tcn_seed{s}.npz" for s in SEEDS]}
    hashes = {name: _sha(out / name) for group in members.values() for name in group}
    meta = {"version": "V3_full_pkg", "train_label_cutoff": TRAIN_END,
            "features": FEATURES, "weights": WEIGHTS, "members": members,
            "fusion_graph": "atom raw -> per-seed daily centred rank -> seed mean -> "
                            "family weight -> final rank",
            "train_cutoff_note": "outer fit uses label_end_date <= 2022-12-30",
            "runtime": "NumPy inference for TabM/TCN; no torch required at run time",
            "member_hashes": hashes,
            "catboost_trees": {str(s): cat_records[s]["actual_tree_count"] for s in SEEDS}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done V3_full_pkg")


if __name__ == "__main__":
    main()
