"""V1 §7: corrected TabM fold training (no warm-start) with immediate artifact save."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import cv, metrics, s2r_train
from .constants import F0_COLUMNS, G1_COLUMNS

S4 = Path("artifacts/stage4_verified_ic")
OUT = S4 / "v1_tabm"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
TABM = dict(k=8, width=128, blocks=2, dropout=0.1)
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 100, 12, 1e-3, 1e-3, 1.0, 8


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Design:
    """N0 representation; statistics are fitted on the fitting sample only."""

    def __init__(self):
        self.median: dict[str, float] = {}
        self.iqr: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "Design":
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.median[column] = float(values.median()) if values.notna().any() else 0.0
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
            self.iqr[column] = iqr if iqr > 0 else 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        counts = frame.groupby("date")[list(G1_COLUMNS)].transform(
            lambda s: s.notna().sum())
        ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(
            lambda s: s.rank(method="average"))
        ranks = (ranks - (counts + 1.0) / 2.0) / counts
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = values.isna().to_numpy(dtype=float)
            filled = values.fillna(self.median[column]).to_numpy(dtype=float)
            scaled = np.clip((filled - self.median[column]) / self.iqr[column], -5, 5)
            blocks.append(scaled.reshape(-1, 1))
            blocks.append(mask.reshape(-1, 1))
        blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def to_json(self) -> dict:
        return {"mode": "N0", "features": FEATURES,
                "median": self.median, "iqr": self.iqr}


def _new_model(d_in: int, seed: int):
    """Fresh model + optimiser + RNG (never resumed from inner weights)."""
    import sys
    sys.path.insert(0, "src")
    from .s4_tabm import TabMWrapper
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabMWrapper(d_in, **TABM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    return model, optimizer, np.random.default_rng(seed)


def _day_groups(dates: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    index = np.nonzero(mask)[0]
    return [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]


def _daily_ic(dates, raw_returns, groups, preds) -> float:
    values = []
    for group in groups:
        y, p = raw_returns[group], preds[group]
        keep = np.isfinite(y)
        if keep.sum() < 2:
            continue
        ry = pd.Series(y[keep]).rank(method="average")
        rp = pd.Series(p[keep]).rank(method="average")
        if ry.nunique() == 1 or rp.nunique() == 1:
            continue
        values.append(float(np.corrcoef(ry, rp)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def _predict(model, x, groups) -> np.ndarray:
    model.eval()
    out = np.zeros(len(x))
    with torch.no_grad():
        for group in groups:
            out[group] = model(torch.from_numpy(x[group])).mean(dim=1).numpy()
    return out


def _train_epochs(model, optimizer, rng, x, z, tensor_x, tensor_z, groups,
                  epochs: int) -> None:
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(groups))
        for start in range(0, len(order), BATCH_DAYS):
            batch = [groups[i] for i in order[start:start + BATCH_DAYS]]
            optimizer.zero_grad()
            per_member = []
            for group in batch:
                preds = model(tensor_x[group])
                target = tensor_z[group].unsqueeze(1).expand_as(preds)
                per_member.append(F.mse_loss(preds, target, reduction="none").mean(dim=0))
            torch.stack(per_member).mean(dim=0).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()


def train_fold(data: pd.DataFrame, window: str, seed: int) -> dict:
    start = time.time()
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    label_end = cv.to_int_date(data["label_end_date"])
    z, raw_w, finite = s2r_train.rank_target_unnormalised(data)
    raw_returns = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    # ---- inner stage: preprocessing fitted on inner fit sample only --------
    inner_design = Design().fit(data.loc[internal])
    x_inner = inner_design.transform(data)
    tensor_x = torch.from_numpy(x_inner)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    inner_groups = _day_groups(dates, internal)
    valid_groups = _day_groups(dates, early)
    model, optimizer, rng = _new_model(x_inner.shape[1], seed)
    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _train_epochs(model, optimizer, rng, x_inner, z, tensor_x, tensor_z,
                      inner_groups, 1)
        ic = _daily_ic(dates, raw_returns, valid_groups, _predict(model, x_inner, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    # ---- outer stage: fresh model/optimiser/RNG, outer-fitted preprocessing --
    outer_design = Design().fit(data.loc[outer])
    x_outer = outer_design.transform(data)
    outer_tensor_x = torch.from_numpy(x_outer)
    outer_groups = _day_groups(dates, outer)
    model2, optimizer2, rng2 = _new_model(x_outer.shape[1], seed)
    _train_epochs(model2, optimizer2, rng2, x_outer, z, outer_tensor_x, tensor_z,
                  outer_groups, best_epoch)
    predict_groups = _day_groups(dates, predict)
    preds_all = _predict(model2, x_outer, predict_groups)
    keep = np.nonzero(predict)[0]
    predictions = pd.DataFrame({"date": dates[keep].astype("int64"),
                                "code": data.loc[keep, "code"].to_numpy(),
                                "score": preds_all[keep]})
    seconds = time.time() - start

    # ---- immediate save (fail loudly if anything is missing) --------------
    tag = f"tabm_pc_{seed}__{window}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in model2.state_dict().items()},
              str(out / "tabm.safetensors"))
    (out / "design.json").write_text(json.dumps(outer_design.to_json(), indent=2))
    pd.DataFrame(history).to_parquet(out / "inner_curve.parquet", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    config = {"d_in": int(x_outer.shape[1]), **TABM}
    (out / "tabm_config.json").write_text(json.dumps(config, indent=2))
    meta = {
        "recipe_id": f"TABM_CORRECTED_{window}_{seed}",
        "window": window, "seed": seed,
        "outer_protocol": {
            "reinit_model": True, "reinit_optimizer": True, "reinit_rng": True,
            "refit_preprocessing_on_outer_fit_set": True,
            "selected_epochs_from_inner_only": int(best_epoch),
            "inner_fit_mask": "label_end_date < es_start and label_end_date <= train_end",
            "outer_fit_mask": "label_end_date <= train_end",
        },
        "train_cutoff": int(split["train_end"]),
        "es_start": int(split["es_start"]),
        "max_inner_label_end": int(np.nanmax(label_end[internal])),
        "max_early_label_end": int(np.nanmax(label_end[early])),
        "max_outer_label_end": int(np.nanmax(label_end[outer])),
        "n_internal": int(internal.sum()), "n_early": int(early.sum()),
        "best_epoch": int(best_epoch), "best_inner_ic": float(best_ic),
        "final_inner_ic": float(history[-1]["valid_daily_ic"]),
        "inner_epochs_run": len(history),
        "train_threads": 2, "backend": "torch CPU 2.14.1",
        "seconds": round(seconds, 1),
    }
    (out / "atom_manifest.json").write_text(json.dumps(meta, indent=2))
    required = ["tabm.safetensors", "design.json", "inner_curve.parquet",
                "predictions.parquet", "tabm_config.json", "atom_manifest.json"]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"V1 artifact save incomplete: {missing}")
    summary = metrics.summarize_rankic(metrics.daily_rankic(
        predictions.merge(data[["date", "code", "future_return"]],
                          on=["date", "code"], how="left", validate="one_to_one")))
    meta["outer_mean_rankic"] = summary["mean_rankic"]
    meta["outer_n_days"] = summary["n_days"]
    meta["hashes"] = {"model": _sha(out / "tabm.safetensors"),
                      "design": _sha(out / "design.json"),
                      "trainer_source": _sha(Path(__file__))}
    (out / "atom_manifest.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="DEV19")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(S4.parent / "stage2_ic_ir" / "panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    meta = train_fold(data, args.window, args.seed)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
