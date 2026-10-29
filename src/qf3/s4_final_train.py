"""D3 §13.1: full-data refit of the frozen members with pre-registered round rules."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import s2r_train, s3_s1
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
OUT = FUS / "final_models"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)

# pre-registered round rules: median of the three inner selections (floored)
C1_TREES = int(np.median([1325, 825, 875]))          # = 875
TABM_EPOCHS = int(np.median([15, 13, 12]))           # = 13


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def train_catboost(data: pd.DataFrame, seed: int = 20261005, threads: int = 6) -> dict:
    from catboost import CatBoostRegressor, Pool

    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    x = data[FEATURES].to_numpy(dtype=np.float32)
    w = s2r_train.normalise_within(raw_w, finite)
    model = CatBoostRegressor(
        loss_function="RMSE", iterations=C1_TREES, learning_rate=0.03, depth=8,
        l2_leaf_reg=15.0, border_count=128, random_strength=1.0,
        bootstrap_type="Bernoulli", subsample=0.8, thread_count=threads,
        random_seed=seed, nan_mode="Min", verbose=False, allow_writing_files=False)
    started = time.time()
    model.fit(Pool(x[finite], label=z[finite], weight=w[finite], feature_names=FEATURES))
    seconds = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "catboost_c1.cbm"
    model.save_model(str(path), format="cbm")
    return {"member": "catboost_c1", "trees": C1_TREES,
            "actual_tree_count": int(model.tree_count_), "seconds": round(seconds, 1),
            "rows": int(finite.sum()), "path": str(path)}


def train_tabm(data: pd.DataFrame, seed: int = 20261005) -> dict:
    import torch
    import torch.nn.functional as F

    from . import s4_mlp, s4_tabm

    torch.manual_seed(seed)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    design = s4_mlp.build_design(data.loc[finite], "N0")
    x_all = design.transform(data)
    tensor_x = torch.from_numpy(x_all)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    dates = pd.to_numeric(data["date"], errors="coerce").astype("int64").to_numpy()
    index = np.nonzero(finite)[0]
    groups = [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]
    model = s4_tabm.TabMWrapper(x_all.shape[1], k=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    started = time.time()
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
    seconds = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tabm.pt"
    torch.save({"state_dict": model.state_dict(), "d_in": int(x_all.shape[1]), "k": 8,
                "width": 128, "blocks": 2, "dropout": 0.1}, path)
    design_path = OUT / "tabm_design.json"
    design_path.write_text(json.dumps({"median": design.median, "iqr": design.iqr,
                                       "mode": "N0", "features": FEATURES}, indent=2))
    return {"member": "tabm", "epochs": TABM_EPOCHS, "seconds": round(seconds, 1),
            "rows": int(finite.sum()), "path": str(path),
            "design": str(design_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", default="catboost_c1,tabm")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = panel()
    log_path = FUS / "d3_final_train.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"members": []}
    for member in args.members.split(","):
        if member == "catboost_c1":
            record = train_catboost(data, threads=args.threads)
        elif member == "tabm":
            record = train_tabm(data)
        else:
            raise SystemExit(f"unknown member {member}")
        report["members"] = [m for m in report["members"] if m["member"] != member]
        report["members"].append(record)
        report["round_rules"] = {"catboost_c1_trees": C1_TREES,
                                 "tabm_epochs": TABM_EPOCHS,
                                 "rule": "median of the three inner selections, floored"}
        log_path.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
