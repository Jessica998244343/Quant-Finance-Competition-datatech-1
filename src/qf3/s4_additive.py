"""D2 §8: additive experts A0 (ridge) and A1 (per-column cubic splines + ridge)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from . import cv, metrics, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

FUS = Path("artifacts/stage3_fusion")
PRED = FUS / "predictions_add"
LAMBDAS = [0.01, 0.1, 1.0]
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)


def panel() -> pd.DataFrame:
    frame = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


class AdditiveDesign:
    """Fold-local median/IQR scaling, clip, masks and optional per-column splines."""

    def __init__(self, spline: bool):
        self.spline = spline
        self.columns: list[str] = []
        self.median: dict[str, float] = {}
        self.iqr: dict[str, float] = {}
        self.constant: dict[str, bool] = {}
        self.transformers: dict[str, SplineTransformer] = {}

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "AdditiveDesign":
        self.columns = list(columns)
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
            self.median[column] = median
            self.iqr[column] = iqr if iqr > 0 else 1.0
            self.constant[column] = iqr <= 0
            if self.spline and iqr > 0:
                filled = values.fillna(median).to_numpy(dtype=float).reshape(-1, 1)
                transformer = SplineTransformer(n_knots=5, degree=3, knots="quantile",
                                                include_bias=False,
                                                extrapolation="constant")
                transformer.fit(filled)
                self.transformers[column] = transformer
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = values.isna().to_numpy(dtype=float)
            filled = values.fillna(self.median[column]).to_numpy(dtype=float)
            scaled = np.clip((filled - self.median[column]) / self.iqr[column], -5, 5)
            if self.spline and not self.constant[column]:
                basis = self.transformers[column].transform(filled.reshape(-1, 1))
                blocks.append(basis)
            else:
                blocks.append(scaled.reshape(-1, 1))
            blocks.append(mask.reshape(-1, 1))
        return np.concatenate(blocks, axis=1)


def evaluate(frame: pd.DataFrame, panel_frame: pd.DataFrame) -> float:
    merged = frame.merge(panel_frame[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def train(data: pd.DataFrame, window: str, spline: bool, threads: int = 2) -> dict:
    split = cv.fold_split(data, window)
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]
    design = AdditiveDesign(spline).fit(data.loc[internal], FEATURES)
    x_internal = design.transform(data.loc[internal])
    x_early = design.transform(data.loc[early])
    x_outer = design.transform(data.loc[outer])
    x_predict = design.transform(data.loc[predict])
    w_internal = s2r_train.normalise_within(raw_w, internal)[internal]
    w_outer = s2r_train.normalise_within(raw_w, outer)[outer]
    sum_w_internal = float(w_internal.sum())
    dates = cv.to_int_date(data["date"])

    best_lambda, best_ic = None, -np.inf
    for lam in LAMBDAS:
        model = Ridge(alpha=lam * sum_w_internal, fit_intercept=True)
        model.fit(x_internal, z[internal], sample_weight=w_internal)
        preds = model.predict(x_early)
        ic = cv.daily_spearman(dates[early], z[early], preds)
        if np.isfinite(ic) and ic > best_ic:
            best_lambda, best_ic = lam, ic

    started = time.time()
    sum_w_outer = float(w_outer.sum())
    model = Ridge(alpha=best_lambda * sum_w_outer, fit_intercept=True)
    model.fit(x_outer, z[outer], sample_weight=w_outer)
    seconds = time.time() - started
    preds = model.predict(x_predict)
    frame = pd.DataFrame({"date": dates[predict].astype("int64"),
                          "code": data.loc[predict, "code"].to_numpy(),
                          "score": preds})
    PRED.mkdir(parents=True, exist_ok=True)
    tag = f"{'A1' if spline else 'A0'}_{window}"
    frame.to_parquet(PRED / f"{tag}.parquet", index=False)
    summary = metrics.summarize_rankic(metrics.daily_rankic(
        frame.merge(data[["date", "code", "future_return"]], on=["date", "code"],
                    how="left", validate="one_to_one")))
    return {"recipe": tag, "window": window, "lambda": best_lambda,
            "inner_ic_at_lambda": best_ic, "n_features": int(x_internal.shape[1]),
            "refit_seconds": round(seconds, 1),
            "mean_rankic": summary["mean_rankic"], "n_days": summary["n_days"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", default="A0,A1")
    parser.add_argument("--windows", default="DEV19,DEV20")
    args = parser.parse_args()
    FUS.mkdir(parents=True, exist_ok=True)
    data = panel()
    log_path = FUS / "d2_additive.json"
    report = json.loads(log_path.read_text()) if log_path.exists() else {"records": []}
    for name in args.recipes.split(","):
        for window in args.windows.split(","):
            record = train(data, window, spline=(name == "A1"))
            report["records"] = [r for r in report["records"]
                                 if not (r["recipe"] == f"{name}_{window}")]
            report["records"].append(record)
            log_path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps(record, default=str), flush=True)


if __name__ == "__main__":
    main()
