"""Reproduce model/ for FINAL_v1 from the official 2016-2022 data.

Run standalone:  python train.py --data-dir <official data dir> --out model
Plain absolute imports of sibling modules are used; running the script puts this
directory on sys.path automatically, so no import-system manipulation is needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import alpha_features
import alpha_panel
import alpha_tabminput
from alpha_constants import F0_COLUMNS, G1_COLUMNS
from alpha_tabm_wrapper import TabMWrapper

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
CATBOOST_TREES = 875          # median of the three inner selections (floored)
TABM_EPOCHS = 13              # median of the three inner selections (floored)
SEED = 20261005


def centred_rank_target(frame: pd.DataFrame) -> np.ndarray:
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    return ((ranks - (counts + 1.0) / 2.0) / counts).to_numpy(dtype=float)


def normalised_weights(frame: pd.DataFrame) -> np.ndarray:
    label = pd.to_numeric(frame["future_return"], errors="coerce")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    weights = 1.0 / counts
    finite = label.notna().to_numpy()
    weights = weights / weights[finite].mean()
    return weights.to_numpy(dtype=float)


def build_panel(data_dir: str) -> pd.DataFrame:
    features, labels = alpha_panel.build_panel(data_dir, membership_filter=True)
    return alpha_panel.merge_labels(features, labels)


def train_catboost(panel: pd.DataFrame, out: Path) -> dict:
    from catboost import CatBoostRegressor, Pool

    finite = panel["future_return"].notna().to_numpy()
    z = centred_rank_target(panel)
    w = normalised_weights(panel)
    x = panel[FEATURES].to_numpy(dtype=np.float32)
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=CATBOOST_TREES, learning_rate=0.03, depth=8,
        l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
        bootstrap_type="Bernoulli", subsample=0.8, thread_count=8,
        random_seed=SEED, nan_mode="Min", verbose=False, allow_writing_files=False)
    model.fit(Pool(x[finite], label=z[finite], weight=w[finite], feature_names=FEATURES))
    model.save_model(str(out / "catboost_c1.cbm"), format="cbm")
    return {"trees": CATBOOST_TREES, "actual_tree_count": int(model.tree_count_)}


def train_tabm(panel: pd.DataFrame, out: Path) -> dict:
    torch.manual_seed(SEED)
    finite = panel["future_return"].notna().to_numpy()
    z = centred_rank_target(panel)
    medians, iqrs = {}, {}
    for column in FEATURES:
        values = pd.to_numeric(panel.loc[finite, column], errors="coerce")
        medians[column] = float(values.median()) if values.notna().any() else 0.0
        q75, q25 = values.quantile(0.75), values.quantile(0.25)
        iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
        iqrs[column] = iqr if iqr > 0 else 1.0
    design = {"median": medians, "iqr": iqrs, "mode": "N0", "features": FEATURES}
    x = alpha_tabminput.transform(panel, design)
    tensor_x = torch.from_numpy(x)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    dates = pd.to_numeric(panel["date"], errors="coerce").astype("int64").to_numpy()
    index = np.nonzero(finite)[0]
    groups = [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]
    model = TabMWrapper(x.shape[1], k=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(SEED)
    for _ in range(TABM_EPOCHS):
        model.train()
        order = rng.permutation(len(groups))
        for start in range(0, len(order), 8):
            batch = [groups[i] for i in order[start:start + 8]]
            optimizer.zero_grad()
            per_member = []
            for group in batch:
                preds = model(tensor_x[group])
                target = tensor_z[group].unsqueeze(1).expand_as(preds)
                per_member.append(F.mse_loss(preds, target, reduction="none").mean(dim=0))
            torch.stack(per_member).mean(dim=0).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(out / "tabm.safetensors"))
    (out / "tabm_design.json").write_text(json.dumps(design, indent=2))
    return {"epochs": TABM_EPOCHS, "d_in": int(x.shape[1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="model")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = build_panel(args.data_dir)
    cat = train_catboost(panel, out)
    tabm = train_tabm(panel, out)
    meta = {"version": "FINAL_v1", "features": FEATURES,
            "weights": {"catboost_c1": 0.5, "tabm": 0.5},
            "fusion": "daily centred rank of each member, then 0.5/0.5, output re-ranked",
            "trained_on": "official 2016-2022 legal mature H1 labels",
            "rounds": {"catboost_c1_trees": CATBOOST_TREES, "tabm_epochs": TABM_EPOCHS},
            "note": "no network, no in-package fit at inference, no cross-call state"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "tabm_config.json").write_text(json.dumps(
        {"d_in": tabm["d_in"], "k": 8, "width": 128, "blocks": 2, "dropout": 0.1},
        indent=2))
    print("done", meta["version"], cat, tabm)


if __name__ == "__main__":
    main()
