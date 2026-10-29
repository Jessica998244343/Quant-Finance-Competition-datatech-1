"""Torch-free MLP ensemble (scikit-learn) as a drop-in replacement for the TabM member."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from . import cv, metrics, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
PRED = FUS / "predictions_skmlp"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)


class Design:
    """Same N0 representation as the torch model, without any torch import."""

    def __init__(self):
        self.median = {}
        self.iqr = {}
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
            filled = values.fillna(self.median[column]).to_numpy(dtype=float)
            scaled = np.clip((filled - self.median[column]) / self.iqr[column], -5, 5)
            blocks.append(scaled.reshape(-1, 1))
            blocks.append(mask.reshape(-1, 1))
        blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
        return np.concatenate(blocks, axis=1).astype(np.float32)


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def train(data: pd.DataFrame, window: str, seed: int, hidden=(128, 128),
          epochs: int = 30, learning_rate: float = 1e-3,
          batch_size: int = 2400) -> dict:
    split = cv.fold_split(data, window)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    dates = cv.to_int_date(data["date"])
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    design = Design().fit(data.loc[internal])
    x_all = design.transform(data)

    def day_ic(mask: np.ndarray, preds: np.ndarray) -> float:
        values = []
        index = np.nonzero(mask)[0]
        for _, group in pd.Series(index).groupby(dates[index]):
            idx = group.to_numpy()
            y, p = raw_returns[idx], preds[idx]
            keep = np.isfinite(y)
            if keep.sum() < 2:
                continue
            ry = pd.Series(y[keep]).rank(method="average")
            rp = pd.Series(p[keep]).rank(method="average")
            if ry.nunique() == 1 or rp.nunique() == 1:
                continue
            values.append(float(np.corrcoef(ry, rp)[0, 1]))
        return float(np.mean(values)) if values else float("nan")

    best_ic, best_state, history = -np.inf, None, []
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         solver="adam", learning_rate_init=learning_rate,
                         batch_size=batch_size, max_iter=1, warm_start=True,
                         shuffle=True, random_state=seed, alpha=1e-4,
                         early_stopping=False, tol=0.0)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.fit(x_all[internal], z[internal])
        ic = day_ic(early, model.predict(x_all))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic = ic
            best_state = (np.array(model.coefs_[0], copy=True),)
            best_epoch = epoch
            best_params = ([c.copy() for c in model.coefs_],
                           [b.copy() for b in model.intercepts_])
    seconds = time.time() - started
    model.coefs_, model.intercepts_ = best_params
    preds = model.predict(x_all)
    keep = np.nonzero(predict)[0]
    frame = pd.DataFrame({"date": dates[keep].astype("int64"),
                          "code": data.loc[keep, "code"].to_numpy(),
                          "score": preds[keep]})
    PRED.mkdir(parents=True, exist_ok=True)
    tag = f"SKMLP_{seed}__{window}"
    frame.to_parquet(PRED / f"{tag}.parquet", index=False)
    pd.DataFrame(history).to_parquet(PRED / f"{tag}__curve.parquet", index=False)
    weights = {f"coef_{i}": c for i, c in enumerate(model.coefs_)}
    weights.update({f"intercept_{i}": b for i, b in enumerate(model.intercepts_)})
    np.savez(PRED / f"{tag}.npz", **weights)
    summary = metrics.summarize_rankic(metrics.daily_rankic(
        frame.merge(data[["date", "code", "future_return"]], on=["date", "code"],
                    how="left", validate="one_to_one")))
    return {"recipe": "SKMLP", "seed": seed, "window": window,
            "best_epoch": int(best_epoch), "best_valid_ic": float(best_ic),
            "seconds": round(seconds, 1), "hidden": list(hidden),
            "epochs_run": len(history),
            "mean_rankic": summary["mean_rankic"], "n_days": summary["n_days"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV19")
    parser.add_argument("--seeds", default="20261005")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    data = panel()
    log_path = FUS / "d3_skmlp.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"records": []}
    for window in args.windows.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            record = train(data, window, seed, epochs=args.epochs)
            report["records"] = [r for r in report["records"]
                                 if not (r["window"] == window and r["seed"] == seed)]
            report["records"].append(record)
            log_path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
