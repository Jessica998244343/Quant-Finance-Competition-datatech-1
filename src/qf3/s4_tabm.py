"""D2 §9.3: TabM comparison using the official yandex-research implementation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cv, metrics, s2r_train, s4_mlp

FUS = Path("artifacts/stage3_fusion")
PRED = FUS / "predictions_nn"
TABM_PATH = Path("vendor/tabm")


def load_tabm():
    if str(TABM_PATH.resolve()) not in sys.path:
        sys.path.insert(0, str(TABM_PATH.resolve()))
    import tabm
    return tabm


class TabMWrapper(nn.Module):
    def __init__(self, d_in: int, k: int = 8, width: int = 128, blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        tabm = load_tabm()
        self.k = k
        self.backbone = tabm.make_tabm_backbone(
            d_in=d_in, n_blocks=blocks, d_block=width, dropout=dropout,
            activation="ReLU", k=k, arch_type="tabm",
            start_scaling_init="random-signs", start_scaling_init_chunks=None)
        self.head = tabm.LinearEnsemble(width, 1, k=k)

    def forward(self, x):
        repeated = x.unsqueeze(1).expand(-1, self.k, -1)
        h = self.backbone(repeated)
        return self.head(h).squeeze(-1)          # (batch, k)


def train(data: pd.DataFrame, window: str, seed: int, k: int = 8,
          epochs: int = 100, patience: int = 12) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = cv.fold_split(data, window)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    design = s4_mlp.build_design(data.loc[internal], "N0")
    x_all = design.transform(data)
    tensor_x = torch.from_numpy(x_all)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    dates = cv.to_int_date(data["date"])

    def day_groups(mask):
        index = np.nonzero(mask)[0]
        return [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]

    train_days, valid_days = day_groups(internal), day_groups(early)
    model = TabMWrapper(x_all.shape[1], k=k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    def predict_days(groups):
        model.eval()
        out = np.zeros(len(x_all))
        with torch.no_grad():
            for group in groups:
                out[group] = model(tensor_x[group]).mean(dim=1).numpy()
        return out

    def day_ic(groups, preds):
        values = []
        for group in groups:
            y, p = raw_returns[group], preds[group]
            keep = np.isfinite(y)
            if keep.sum() < 2:
                continue
            rk_y = pd.Series(y[keep]).rank(method="average")
            rk_p = pd.Series(p[keep]).rank(method="average")
            if rk_p.nunique() == 1 or rk_y.nunique() == 1:
                continue
            values.append(float(np.corrcoef(rk_y, rk_p)[0, 1]))
        return float(np.mean(values)) if values else float("nan")

    rng = np.random.default_rng(seed)
    best_ic, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_days))
        for start in range(0, len(order), 8):
            batch = [train_days[i] for i in order[start:start + 8]]
            optimizer.zero_grad()
            per_member = []
            for group in batch:
                preds = model(tensor_x[group])                 # (n, k)
                target = tensor_z[group].unsqueeze(1).expand_as(preds)
                per_member.append(F.mse_loss(preds, target, reduction="none").mean(dim=0))
            loss = torch.stack(per_member).mean(dim=0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ic = day_ic(valid_days, predict_days(valid_days))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    for _ in range(best_epoch):
        model.train()
        order = rng.permutation(len(day_groups(outer)))
        groups = day_groups(outer)
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
    out = predict_days(day_groups(predict))
    keep = np.nonzero(predict)[0]
    frame = pd.DataFrame({"date": dates[keep].astype("int64"),
                          "code": data.loc[keep, "code"].to_numpy(),
                          "score": out[keep]})
    PRED.mkdir(parents=True, exist_ok=True)
    tag = f"TABM_{seed}__{window}"
    frame.to_parquet(PRED / f"{tag}.parquet", index=False)
    summary = metrics.summarize_rankic(metrics.daily_rankic(
        frame.merge(data[["date", "code", "future_return"]], on=["date", "code"],
                    how="left", validate="one_to_one")))
    return {"recipe": "TABM", "seed": seed, "window": window, "k": k,
            "best_epoch": int(best_epoch), "best_valid_ic": float(best_ic),
            "epochs_run": len(history), "n_input_dims": int(x_all.shape[1]),
            "mean_rankic": summary["mean_rankic"], "n_days": summary["n_days"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    data = s4_mlp.panel()
    log_path = FUS / "d2_tabm.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"records": []}
    for window in args.windows.split(","):
        started = time.time()
        record = train(data, window, args.seed, k=args.k)
        record["seconds"] = round(time.time() - started, 1)
        record["commit"] = "28e47ae301c92ec37787dde1ce923a0793f405b4"
        report["records"] = [r for r in report["records"]
                             if not (r["window"] == window and r["seed"] == args.seed)]
        report["records"].append(record)
        log_path.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
