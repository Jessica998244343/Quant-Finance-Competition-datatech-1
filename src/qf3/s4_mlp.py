"""D2 §9: ResMLP N0 (scaled raw + G1 ranks) and N1 (G1 ranks only) experts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cv, metrics, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
PRED = FUS / "predictions_nn"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
IDX = 20261005


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(self.norm(x))))
        return x + self.fc2(h)


class ResMLP(nn.Module):
    def __init__(self, d_in: int, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([ResBlock(width, dropout) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)).squeeze(-1)


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def build_design(frame: pd.DataFrame, mode: str) -> "Design":
    design = Design(mode).fit(frame)
    return design


class Design:
    """Median/IQR scaling + clip, optional G1 daily-rank block, missing masks."""

    def __init__(self, mode: str):
        self.mode = mode                       # "N0" (raw + ranks) or "N1" (ranks only)
        self.median: dict[str, float] = {}
        self.iqr: dict[str, float] = {}
        self.features = FEATURES

    def fit(self, frame: pd.DataFrame) -> "Design":
        for column in self.features:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.median[column] = float(values.median()) if values.notna().any() else 0.0
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
            self.iqr[column] = iqr if iqr > 0 else 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(
            lambda s: ((s.rank(method="average") - (s.notna().sum() + 1) / 2.0)
                       / s.notna().sum()) if s.notna().any() else s)
        for column in self.features:
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = values.isna().to_numpy(dtype=float)
            if column in G1_COLUMNS and self.mode == "N1":
                blocks.append(np.nan_to_num(ranks[column].to_numpy(dtype=float)).reshape(-1, 1))
            else:
                filled = values.fillna(self.median[column]).to_numpy(dtype=float)
                scaled = np.clip((filled - self.median[column]) / self.iqr[column], -5, 5)
                blocks.append(scaled.reshape(-1, 1))
            blocks.append(mask.reshape(-1, 1))
        if self.mode == "N0":
            blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
        return np.concatenate(blocks, axis=1).astype(np.float32)


def day_batches(data: pd.DataFrame, design: Design, mask: np.ndarray,
                batch_days: int = 8):
    x = design.transform(data)
    dates = cv.to_int_date(data["date"])
    ordered = []
    for date, index in pd.Series(np.nonzero(mask)[0]).groupby(dates[mask]):
        ordered.append(index.to_numpy())
    return x, dates, ordered


def train(data: pd.DataFrame, window: str, mode: str, seed: int,
          width: int = 128, dropout: float = 0.1, weight_decay: float = 1e-3,
          epochs: int = 100, patience: int = 12, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = cv.fold_split(data, window)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    design = build_design(data.loc[internal], mode)
    x_all = design.transform(data)
    dates = cv.to_int_date(data["date"])
    tensor_x = torch.from_numpy(x_all)
    tensor_z = torch.from_numpy(np.nan_to_num(z))

    def day_groups(mask: np.ndarray) -> list[np.ndarray]:
        index = np.nonzero(mask)[0]
        return [g.to_numpy() for _, g in
                pd.Series(index).groupby(dates[index])]

    train_days = day_groups(internal)
    valid_days = day_groups(early)
    model = ResMLP(x_all.shape[1], width, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def predict_days(groups: list[np.ndarray]) -> np.ndarray:
        model.eval()
        out = np.zeros(len(x_all))
        with torch.no_grad():
            for group in groups:
                out[group] = model(tensor_x[group]).numpy()
        return out

    def day_ic(groups: list[np.ndarray], preds: np.ndarray) -> float:
        values = []
        for group in groups:
            y = raw_returns[group]
            p = preds[group]
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
    best_ic, best_epoch, best_state = -np.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_days))
        losses = []
        for start in range(0, len(order), 8):
            batch = [train_days[i] for i in order[start:start + 8]]
            optimizer.zero_grad()
            losses_day = []
            for group in batch:
                preds = model(tensor_x[group])
                losses_day.append(F.mse_loss(preds, tensor_z[group]))
            loss = torch.stack(losses_day).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        preds = predict_days(valid_days)
        ic = day_ic(valid_days, preds)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    frames = []
    outer_days = day_groups(outer)
    total_epochs = best_epoch
    for _ in range(total_epochs):
        model.train()
        order = rng.permutation(len(outer_days))
        for start in range(0, len(order), 8):
            batch = [outer_days[i] for i in order[start:start + 8]]
            optimizer.zero_grad()
            losses_day = [F.mse_loss(model(tensor_x[g]), tensor_z[g]) for g in batch]
            loss = torch.stack(losses_day).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    predict_days_out = predict_days(day_groups(predict))
    keep = np.nonzero(predict)[0]
    frame = pd.DataFrame({"date": dates[keep].astype("int64"),
                          "code": data.loc[keep, "code"].to_numpy(),
                          "score": predict_days_out[keep]})
    PRED.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_{seed}__{window}"
    frame.to_parquet(PRED / f"{tag}.parquet", index=False)
    torch.save(model.state_dict(), PRED / f"{tag}.pt")
    pd.DataFrame(history).to_parquet(PRED / f"{tag}__curve.parquet", index=False)
    summary = metrics.summarize_rankic(metrics.daily_rankic(
        frame.merge(data[["date", "code", "future_return"]], on=["date", "code"],
                    how="left", validate="one_to_one")))
    return {"recipe": mode, "seed": seed, "window": window,
            "best_epoch": int(best_epoch), "best_valid_ic": float(best_ic),
            "epochs_run": len(history), "n_input_dims": int(x_all.shape[1]),
            "mean_rankic": summary["mean_rankic"], "n_days": summary["n_days"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", default="N0,N1")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--seed", type=int, default=IDX)
    args = parser.parse_args()
    FUS.mkdir(parents=True, exist_ok=True)
    data = panel()
    log_path = FUS / "d2_mlp.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"records": []}
    for mode in args.recipes.split(","):
        for window in args.windows.split(","):
            started = time.time()
            record = train(data, window, mode, args.seed)
            record["seconds"] = round(time.time() - started, 1)
            report["records"] = [r for r in report["records"]
                                 if not (r["recipe"] == mode and r["window"] == window
                                         and r["seed"] == args.seed)]
            report["records"].append(record)
            log_path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
