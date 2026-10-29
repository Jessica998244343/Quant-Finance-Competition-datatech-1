"""§10.2: two single-variable TabM comparisons on the corrected protocol.

  dropout  : 0.1 -> 0.2  (everything else unchanged)
  loss     : MSE -> mean_member[MSE(p,z) + 0.1 * (1 - Pearson(p,z))]
             (per internal member per day; variance <= 1e-8 => term 0, registered)

Epoch selection always uses the original H1 daily Spearman.  The corrected
protocol is preserved: inner selection, then a fresh model/optimiser/RNG with
preprocessing refit on the outer fit set.
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
from .constants import F0_COLUMNS, G1_COLUMNS

V2 = Path("artifacts/stage4_verified_ic/v2")
V1T = Path("artifacts/stage4_verified_ic/v1_tabm")
BASE_FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 100, 12, 1e-3, 1e-3, 1.0, 8
THREADS = 2
CORR_EPS = 1e-8


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Design:
    """N0 representation; statistics fitted on the fitting sample only."""

    def __init__(self):
        self.median, self.iqr = {}, {}

    def fit(self, frame: pd.DataFrame) -> "Design":
        for column in BASE_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.median[column] = float(values.median()) if values.notna().any() else 0.0
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
            self.iqr[column] = iqr if iqr > 0 else 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        counts = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.notna().sum())
        ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.rank(method="average"))
        ranks = (ranks - (counts + 1.0) / 2.0) / counts
        for column in BASE_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = values.isna().to_numpy(dtype=float)
            filled = values.fillna(self.median[column]).to_numpy(dtype=float)
            scaled = np.clip((filled - self.median[column]) / self.iqr[column], -5, 5)
            blocks.append(scaled.reshape(-1, 1))
            blocks.append(mask.reshape(-1, 1))
        blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def to_json(self) -> dict:
        return {"mode": "N0", "features": BASE_FEATURES,
                "median": self.median, "iqr": self.iqr}


def _new(d_in: int, seed: int, dropout: float):
    from .s4_tabm_wrapper_src import TabMWrapper

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabMWrapper(d_in, k=8, width=128, blocks=2, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    return model, optimizer, np.random.default_rng(seed)


def _groups(dates: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    index = np.nonzero(mask)[0]
    return [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]


def _daily_ic(raw, groups, preds) -> float:
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


def _predict(model, x, groups) -> np.ndarray:
    model.eval()
    out = np.zeros(len(x))
    with torch.no_grad():
        for group in groups:
            out[group] = model(torch.from_numpy(x[group])).mean(dim=1).numpy()
    return out


def _day_loss(preds: torch.Tensor, target: torch.Tensor, loss_mode: str) -> torch.Tensor:
    mse = F.mse_loss(preds, target, reduction="none").mean(dim=0)      # per member
    if loss_mode == "mse":
        return mse
    centred_p = preds - preds.mean(dim=0, keepdim=True)
    centred_z = target - target.mean(dim=0, keepdim=True)
    covariance = (centred_p * centred_z).mean(dim=0)
    variance = centred_p.pow(2).mean(dim=0) * centred_z.pow(2).mean(dim=0)
    corr = torch.where(variance <= CORR_EPS, torch.zeros_like(variance),
                       covariance / torch.sqrt(variance.clamp_min(1e-30)))
    return mse + 0.1 * (1.0 - corr)


def _run(model, optimizer, rng, tensor_x, tensor_z, groups, epochs, loss_mode) -> None:
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(groups))
        for start in range(0, len(order), BATCH_DAYS):
            batch = [groups[i] for i in order[start:start + BATCH_DAYS]]
            optimizer.zero_grad()
            parts = []
            for group in batch:
                preds = model(tensor_x[group])
                target = tensor_z[group].unsqueeze(1).expand_as(preds)
                parts.append(_day_loss(preds, target, loss_mode))
            torch.stack(parts).mean(dim=0).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()


def train_fold(data: pd.DataFrame, window: str, seed: int, dropout: float,
               loss_mode: str) -> dict:
    torch.set_num_threads(THREADS)
    start = time.time()
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    inner_design = Design().fit(data.loc[internal])
    x_inner = inner_design.transform(data)
    tensor_x, tensor_z = torch.from_numpy(x_inner), torch.from_numpy(np.nan_to_num(z))
    inner_groups, valid_groups = _groups(dates, internal), _groups(dates, early)
    model, optimizer, rng = _new(x_inner.shape[1], seed, dropout)
    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _run(model, optimizer, rng, tensor_x, tensor_z, inner_groups, 1, loss_mode)
        ic = _daily_ic(raw, valid_groups, _predict(model, x_inner, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    outer_design = Design().fit(data.loc[outer])
    x_outer = outer_design.transform(data)
    outer_groups = _groups(dates, outer)
    model2, optimizer2, rng2 = _new(x_outer.shape[1], seed, dropout)
    _run(model2, optimizer2, rng2, torch.from_numpy(x_outer), tensor_z,
         outer_groups, best_epoch, loss_mode)
    preds = _predict(model2, x_outer, _groups(dates, predict))
    keep = np.nonzero(predict)[0]
    predictions = pd.DataFrame({"date": dates[keep].astype("int64"),
                                "code": data.loc[keep, "code"].to_numpy(),
                                "score": preds[keep]})
    tag = f"tabm_{loss_mode}_d{dropout}_{seed}__{window}"
    out = V2 / tag
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in model2.state_dict().items()},
              str(out / "tabm.safetensors"))
    (out / "design.json").write_text(json.dumps(outer_design.to_json(), indent=2))
    pd.DataFrame(history).to_parquet(out / "inner_curve.parquet", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    meta = {"tag": tag, "window": window, "seed": seed, "dropout": dropout,
            "loss_mode": loss_mode, "best_epoch": int(best_epoch),
            "best_inner_ic": float(best_ic), "inner_epochs_run": len(history),
            "train_cutoff": int(split["train_end"]), "train_threads": THREADS,
            "seconds": round(time.time() - start, 1),
            "protocol": "inner select -> fresh model/optimiser/RNG -> outer refit preprocessing"}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    return meta | {"predictions": predictions}


def daily_ic(frame: pd.DataFrame) -> float:
    values = []
    for _, block in frame.groupby("date"):
        keep = np.isfinite(block["future_return"]) & np.isfinite(block["score"])
        if keep.sum() < 2:
            continue
        y = block.loc[keep, "future_return"].rank(method="average")
        p = block.loc[keep, "score"].rank(method="average")
        if y.nunique() == 1 or p.nunique() == 1:
            continue
        values.append(float(np.corrcoef(y, p)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true",
                        help="verify dropout0.1/mse reproduces the shipped CONF21 npz")
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--seeds", default="",
                        help="comma-separated seed list override (single-variable gate extension)")
    parser.add_argument("--configs", default="mse:0.1,mse:0.2,mse_corr:0.1",
                        help="comma-separated loss_mode:dropout list")
    parser.add_argument("--out", default=str(V2 / "v2_tabm_variants.json"))
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    labels = data[["date", "code", "future_return"]]
    V2.mkdir(parents=True, exist_ok=True)

    if args.control:
        from .s6_v1_package import export_tabm_npz  # noqa: F401  (same mapping)
        meta = train_fold(data, "CONF21", args.seed, 0.1, "mse")
        shipped = Path("releases/stage4_verified_ic/build/V1_pkg/model") / f"tabm_seed{args.seed}.npz"
        source = V1T / f"tabm_pc_{args.seed}__CONF21" / "tabm.safetensors"
        from safetensors.torch import load_file
        sd = load_file(str(V2 / meta["tag"] / "tabm.safetensors"))
        arrays = {"W0": sd["backbone.blocks.0.0.weight"], "r0": sd["backbone.blocks.0.0.r"],
                  "s0": sd["backbone.blocks.0.0.s"], "b0": sd["backbone.blocks.0.0.bias"],
                  "W1": sd["backbone.blocks.1.0.weight"], "r1": sd["backbone.blocks.1.0.r"],
                  "s1": sd["backbone.blocks.1.0.s"], "b1": sd["backbone.blocks.1.0.bias"],
                  "Wh": sd["head.weight"], "bh": sd["head.bias"],
                  "k": np.array([8], dtype=np.float32)}
        target = V2 / f"control_seed{args.seed}.npz"
        to_array = lambda v: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        np.savez(target, **{k: np.ascontiguousarray(to_array(v), dtype=np.float32)
                            for k, v in arrays.items()})
        report = {"control_tag": meta["tag"], "best_epoch": meta["best_epoch"],
                  "npz_bytes_identical": _sha(target) == _sha(shipped),
                  "source_safetensors": str(source)}
        report["passed"] = bool(report["npz_bytes_identical"])
        (V2 / "v2_tabm_control.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed])
    configs = [tuple(item.split(":")) for item in args.configs.split(",")]
    report = {"seeds": seeds, "configs": args.configs, "records": []}
    for window in args.windows.split(","):
        anchor = pd.read_parquet(
            Path("artifacts/stage4_verified_ic") / f"v1_anchor__{window}.parquet")
        for seed in seeds:
          for loss_mode, dropout in configs:
            dropout = float(dropout)
            meta = train_fold(data, window, seed, dropout, loss_mode)
            frame = meta["predictions"].merge(labels, on=["date", "code"], how="left",
                                              validate="one_to_one")
            fused = anchor.rename(columns={"score": "anchor"}).merge(
                meta["predictions"].rename(columns={"score": "new"}),
                on=["date", "code"], validate="one_to_one").merge(
                labels, on=["date", "code"], how="left", validate="one_to_one")
            pieces = []
            for _, block in fused.groupby("date", sort=True):
                def centred(values):
                    ranks = pd.Series(values).rank(method="average")
                    return ((ranks - (len(ranks) + 1) / 2.0) / len(ranks)).to_numpy()
                value = 0.9 * block["anchor"].to_numpy() + 0.1 * centred(block["new"].to_numpy())
                pieces.append(pd.DataFrame({"date": block["date"].to_numpy(),
                                            "code": block["code"].to_numpy(),
                                            "score": centred(value),
                                            "future_return": block["future_return"].to_numpy()}))
            record = {"window": window, "seed": seed, "loss_mode": loss_mode,
                      "dropout": dropout,
                      "best_epoch": meta["best_epoch"], "seconds": meta["seconds"],
                      "standalone_rankic": daily_ic(frame),
                      "anchor_plus_alpha0.10_rankic": daily_ic(pd.concat(pieces, ignore_index=True))}
            report["records"].append(record)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in record.items()}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
