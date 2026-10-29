"""Reproduce V1_pkg/model from the official data (training entry).

Run standalone:  python train.py --data-dir <official data dir> --out model

Training needs the extra packages in train_requirements.txt (torch); inference
needs only requirements.txt.  The CatBoost members use the frozen C1 recipe with
**per-seed inner IC tree selection** (25-tree checkpoints, earliest checkpoint
within 0.0005 of the inner daily-Spearman optimum), fold-local weight
normalisation and 6 threads.  The TabM members use the corrected protocol
(fresh model/optimiser/RNG, outer-fitted preprocessing, inner-selected epoch)
and are exported to the static NumPy layout read by the package scorer.
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
from alpha_constants import F0_COLUMNS, G1_COLUMNS

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
WINDOW = "CONF21"                     # cutoff 2021: label_end_date <= 2020-12-31
CAT_STEP = 25
CAT_MAX_TREES = 3000
CAT_NEAR_TOL = 0.0005
CAT_THREADS = 6                       # fixed: CatBoost is not bit-identical across thread counts
TABM_THREADS = 2
TABM = dict(k=8, width=128, blocks=2, dropout=0.1)
WEIGHTS = {"c1": 0.5, "tabm": 0.5}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_target(frame: pd.DataFrame):
    """Daily average-tied centred rank target plus *unnormalised* 1/n weights."""
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    z = ((ranks - (counts + 1.0) / 2.0) / counts).to_numpy(dtype=float)
    raw_w = (1.0 / counts).to_numpy(dtype=float)
    return z, raw_w, finite


def normalise_within(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise so the *fitting sample* has mean weight 1 (fold-local, no PIT leak)."""
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
                            feature_names=FEATURES),
              use_best_model=False)

    valid_index = np.nonzero(early)[0]
    checkpoints = list(range(CAT_STEP, CAT_MAX_TREES + 1, CAT_STEP))
    curve = np.array([
        alpha_cv.daily_spearman(dates[valid_index], raw_returns[valid_index],
                                inner.predict(x[early], ntree_end=trees))
        for trees in checkpoints], dtype=float)
    best_index = int(np.nanargmax(curve))
    best_value = float(curve[best_index])
    near = np.nonzero(curve >= best_value - CAT_NEAR_TOL)[0]
    selected = int(checkpoints[int(near[0])])

    final = CatBoostRegressor(**{**base, "iterations": selected, "od_type": None, "od_wait": None})
    final.fit(Pool(x[outer], label=z[outer], weight=w_outer[outer], feature_names=FEATURES))
    final.save_model(str(out / f"catboost_c1_seed{seed}.cbm"), format="cbm")
    pd.DataFrame({"trees": checkpoints, "inner_daily_ic": curve}).to_parquet(
        out / f"catboost_c1_seed{seed}__curve.parquet", index=False)
    return {"seed": seed, "selected_trees": selected,
            "best_checkpoint": int(checkpoints[best_index]), "best_inner_ic": best_value,
            "near_optimal_last": int(checkpoints[int(near[-1])]),
            "actual_tree_count": int(final.tree_count_)}


def export_tabm_npz(state_dict, path: Path) -> None:
    """Static NumPy layout read by alpha_tabm_static.TabmStatic.from_npz."""
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


def train_tabm(panel: pd.DataFrame, split: dict, seed: int, out: Path) -> dict:
    import torch

    torch.set_num_threads(TABM_THREADS)
    result = alpha_train_tabm.train_fold(panel, split, seed)
    export_tabm_npz({k: v.detach().cpu().numpy()
                     for k, v in result["model"].state_dict().items()},
                    out / f"tabm_seed{seed}.npz")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="model")
    parser.add_argument("--threads", type=int, default=CAT_THREADS,
                        help="CatBoost threads; must stay 6 for bit-identical reproduction")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    features, labels = alpha_panel.build_panel(args.data_dir, membership_filter=True)
    panel = alpha_panel.merge_labels(features, labels)
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    split = alpha_cv.fold_split(panel, WINDOW)

    tree_counts = {}
    for seed in SEEDS:
        record = train_catboost(panel, split, seed, args.threads, out)
        tree_counts[seed] = record
        result = train_tabm(panel, split, seed, out)
        (out / f"design_seed{seed}.json").write_text(json.dumps(result["design"], indent=2))
        (out / f"tabm_epoch_seed{seed}.json").write_text(json.dumps(
            {"best_epoch": int(result["best_epoch"]),
             "best_inner_ic": float(result["best_inner_ic"]),
             "inner_epochs_run": int(len(result["history"]))}, indent=2))
        print("done seed", seed, "catboost trees", record["actual_tree_count"],
              "tabm epochs", result["best_epoch"], flush=True)

    design_text = []
    for seed in SEEDS:
        design_text.append((out / f"design_seed{seed}.json").read_text())
    if len(set(design_text)) != 1:
        raise RuntimeError("TabM designs differ across seeds; refusing to publish an ambiguous package")
    (out / "tabm_design.json").write_text(design_text[0])
    (out / "tabm_config.json").write_text(json.dumps(
        {"d_in": 2 * len(FEATURES) + len(G1_COLUMNS), **TABM}, indent=2))
    for seed in SEEDS:
        (out / f"design_seed{seed}.json").unlink()
        (out / f"tabm_epoch_seed{seed}.json").unlink()
        (out / f"catboost_c1_seed{seed}__curve.parquet").unlink()

    members = {"c1": [f"catboost_c1_seed{s}.cbm" for s in SEEDS],
               "tabm": [f"tabm_seed{s}.npz" for s in SEEDS]}
    hashes = {name: _sha(out / name) for group in members.values() for name in group}
    meta = {"version": "V1_pkg", "cutoff": "2021-12-31",
            "features": FEATURES, "weights": WEIGHTS, "members": members,
            "fusion_graph": "atom rank -> seed mean -> family weight -> final rank",
            "train_cutoff_note": "year models trained with label_end_date <= 2020-12-31",
            "runtime": "NumPy inference for TabM; no torch required at run time",
            "member_hashes": hashes, "design_hash": _sha(out / "tabm_design.json"),
            "catboost_trees": {str(seed): tree_counts[seed] for seed in SEEDS}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done V1_pkg")


if __name__ == "__main__":
    main()
