"""§4.6: corrected ResMLP N0 control with full artifacts and anchor marginal IC.

Same corrected protocol as the TabM folds: inner epoch selection -> fresh
model/optimiser/RNG + preprocessing refit on the outer fit set -> train the
selected epoch from scratch.  Marginal IC is measured against the frozen V1
anchor by adding the member at alpha in {0.05, 0.10, 0.25, 0.50}.
"""

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

from . import cv, s2r_train
from .s4_mlp import ResMLP, build_design

S4 = Path("artifacts/stage4_verified_ic")
OUT = S4 / "v1_resmlp"
SEED = 20261005
WINDOWS = ("DEV19", "DEV20")
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 100, 12, 1e-3, 1e-3, 1.0, 8
TABM_THREADS = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _groups(dates: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    index = np.nonzero(mask)[0]
    return [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]


def _daily_ic(raw: np.ndarray, groups, preds: np.ndarray) -> float:
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


def _new(d_in: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResMLP(d_in)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    return model, optimizer, np.random.default_rng(seed)


def _run(model, optimizer, rng, tensor_x, tensor_z, day_groups) -> None:
    model.train()
    order = rng.permutation(len(day_groups))
    for start in range(0, len(order), BATCH_DAYS):
        batch = [day_groups[i] for i in order[start:start + BATCH_DAYS]]
        optimizer.zero_grad()
        losses = [F.mse_loss(model(tensor_x[group]), tensor_z[group]) for group in batch]
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()


def _predict(model, x: np.ndarray, groups) -> np.ndarray:
    model.eval()
    out = np.zeros(len(x))
    with torch.no_grad():
        for group in groups:
            out[group] = model(torch.from_numpy(x[group])).numpy()
    return out


def train_fold(data: pd.DataFrame, window: str, seed: int = SEED) -> dict:
    torch.set_num_threads(TABM_THREADS)
    start = time.time()
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    inner_design = build_design(data.loc[internal], "N0")
    x_inner = inner_design.transform(data)
    tensor_x, tensor_z = torch.from_numpy(x_inner), torch.from_numpy(np.nan_to_num(z))
    inner_groups, valid_groups = _groups(dates, internal), _groups(dates, early)
    model, optimizer, rng = _new(x_inner.shape[1], seed)
    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _run(model, optimizer, rng, tensor_x, tensor_z, inner_groups)
        ic = _daily_ic(raw, valid_groups, _predict(model, x_inner, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    outer_design = build_design(data.loc[outer], "N0")
    x_outer = outer_design.transform(data)
    outer_tensor_x = torch.from_numpy(x_outer)
    outer_groups = _groups(dates, outer)
    model2, optimizer2, rng2 = _new(x_outer.shape[1], seed)
    for _ in range(best_epoch):
        _run(model2, optimizer2, rng2, outer_tensor_x, tensor_z, outer_groups)
    preds = _predict(model2, x_outer, _groups(dates, predict))
    keep = np.nonzero(predict)[0]
    predictions = pd.DataFrame({"date": dates[keep].astype("int64"),
                                "code": data.loc[keep, "code"].to_numpy(),
                                "score": preds[keep]})
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"resmlp_n0_{seed}__{window}"
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in model2.state_dict().items()},
              str(OUT / f"{tag}.safetensors"))
    (OUT / f"{tag}_design.json").write_text(json.dumps(
        {"mode": outer_design.mode, "median": outer_design.median,
         "iqr": outer_design.iqr}, indent=2))
    pd.DataFrame(history).to_parquet(OUT / f"{tag}_curve.parquet", index=False)
    predictions.to_parquet(OUT / f"{tag}_predictions.parquet", index=False)
    meta = {"recipe_id": tag, "window": window, "seed": seed,
            "outer_protocol": {"reinit_model": True, "reinit_optimizer": True,
                               "reinit_rng": True, "refit_preprocessing_on_outer_fit_set": True,
                               "selected_epochs_from_inner_only": int(best_epoch)},
            "train_cutoff": int(split["train_end"]), "best_epoch": int(best_epoch),
            "best_inner_ic": float(best_ic), "inner_epochs_run": len(history),
            "train_threads": TABM_THREADS, "seconds": round(time.time() - start, 1)}
    (OUT / f"{tag}_manifest.json").write_text(json.dumps(meta, indent=2))
    return meta | {"predictions": predictions, "tag": tag}


def marginal_ic(window: str, predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    anchor = pd.read_parquet(S4 / f"v1_anchor__{window}.parquet")
    merged = anchor.merge(predictions, on=["date", "code"], suffixes=("_anchor", "_new"),
                          validate="one_to_one").merge(
        labels, on=["date", "code"], how="left", validate="one_to_one")

    def ics_of(frame: pd.DataFrame, column: str) -> list[float]:
        values = []
        for _, block in frame.groupby("date"):
            keep = np.isfinite(block["future_return"])
            if keep.sum() < 2:
                continue
            y = block.loc[keep, "future_return"].rank(method="average")
            p = block.loc[keep, column].rank(method="average")
            if y.nunique() == 1 or p.nunique() == 1:
                continue
            values.append(float(np.corrcoef(y, p)[0, 1]))
        return values

    out = {}
    for alpha in ALPHAS:
        pieces = []
        for date, block in merged.groupby("date", sort=True):
            a = block["score_anchor"].to_numpy()
            n = pd.Series(block["score_new"].to_numpy()).rank(method="average")
            n = ((n - (len(n) + 1) / 2.0) / len(n)).to_numpy()
            value = (1.0 - alpha) * a + alpha * n
            rank = pd.Series(value).rank(method="average")
            rank = ((rank - (len(rank) + 1) / 2.0) / len(rank)).to_numpy()
            pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                        "score": rank,
                                        "future_return": block["future_return"].to_numpy()}))
        fused = pd.concat(pieces, ignore_index=True)
        out[str(alpha)] = float(np.mean(ics_of(fused, "score")))
    out["anchor_rankic"] = out["0.0"]
    out["resmlp_standalone_rankic"] = float(np.mean(ics_of(merged, "score_new")))
    out["delta_vs_anchor"] = {key: out[key] - out["anchor_rankic"]
                              for key in out if key not in
                              ("anchor_rankic", "resmlp_standalone_rankic",
                               "delta_vs_anchor")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--out", default=str(S4 / "v1_resmlp_marginal.json"))
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    labels = data[["date", "code", "future_return"]]
    report = {"seed": SEED, "windows": {}}
    for window in args.windows.split(","):
        meta = train_fold(data, window)
        result = marginal_ic(window, meta["predictions"], labels)
        report["windows"][window] = {"best_epoch": meta["best_epoch"],
                                     "best_inner_ic": meta["best_inner_ic"],
                                     "seconds": meta["seconds"], **result}
        print(window, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in result.items() if k != "delta_vs_anchor"}),
              flush=True)
    report["note"] = "corrected protocol; single seed per the review's fallback rule"
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
