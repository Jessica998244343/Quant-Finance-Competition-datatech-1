"""Package-internal TabM trainer (V1 protocol); training dependency only."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from alpha_constants import F0_COLUMNS, G1_COLUMNS
from alpha_train_model import TabMWrapper

FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
TABM = dict(k=8, width=128, blocks=2, dropout=0.1)
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 100, 12, 1e-3, 1e-3, 1.0, 8


class Design:
    def __init__(self):
        self.median, self.iqr = {}, {}

    def fit(self, frame):
        for column in FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            self.median[column] = float(values.median()) if values.notna().any() else 0.0
            q75, q25 = values.quantile(0.75), values.quantile(0.25)
            iqr = float(q75 - q25) if pd.notna(q75) and pd.notna(q25) else 0.0
            self.iqr[column] = iqr if iqr > 0 else 1.0
        return self

    def transform(self, frame):
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

    def to_json(self):
        return {"mode": "N0", "features": FEATURES,
                "median": self.median, "iqr": self.iqr}


def _groups(dates, mask):
    index = np.nonzero(mask)[0]
    return [g.to_numpy() for _, g in pd.Series(index).groupby(dates[index])]


def _daily_ic(raw, groups, preds):
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


def _new(d_in, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabMWrapper(d_in, **TABM)
    return model, torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD), \
        np.random.default_rng(seed)


def _predict(model, x, groups):
    model.eval()
    out = np.zeros(len(x))
    with torch.no_grad():
        for group in groups:
            out[group] = model(torch.from_numpy(x[group])).mean(dim=1).numpy()
    return out


def _run(model, optimizer, rng, tensor_x, tensor_z, groups, n):
    for _ in range(n):
        model.train()
        order = rng.permutation(len(groups))
        for start in range(0, len(order), BATCH_DAYS):
            batch = [groups[i] for i in order[start:start + BATCH_DAYS]]
            optimizer.zero_grad()
            parts = []
            for group in batch:
                preds = model(tensor_x[group])
                target = tensor_z[group].unsqueeze(1).expand_as(preds)
                parts.append(F.mse_loss(preds, target, reduction="none").mean(dim=0))
            torch.stack(parts).mean(dim=0).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()


def train_fold(data, split, seed, label_col="future_return"):
    dates = pd.to_numeric(data["date"], errors="coerce").astype("float64").to_numpy()
    label = pd.to_numeric(data[label_col], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(data["date"]).rank(method="average")
    counts = label.where(finite).groupby(data["date"]).transform("count")
    z = ((ranks - (counts + 1.0) / 2.0) / counts).to_numpy(dtype=float)
    raw = label.to_numpy(dtype=float)
    internal = split["internal"] & finite
    early = split["early_stop"] & finite
    outer = split["outer_train"] & finite
    predict = split["predict"]

    inner = Design().fit(data.loc[internal])
    x_in = inner.transform(data)
    tx, tz = torch.from_numpy(x_in), torch.from_numpy(np.nan_to_num(z))
    model, optimizer, rng = _new(x_in.shape[1], seed)
    history, best, best_epoch = [], -np.inf, 0
    valid_groups = _groups(dates, early)
    for epoch in range(1, MAX_EPOCHS + 1):
        _run(model, optimizer, rng, tx, tz, _groups(dates, internal), 1)
        ic = _daily_ic(raw, valid_groups, _predict(model, x_in, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best:
            best, best_epoch = ic, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    outer_design = Design().fit(data.loc[outer])
    x_out = outer_design.transform(data)
    model2, optimizer2, rng2 = _new(x_out.shape[1], seed)
    _run(model2, optimizer2, rng2, torch.from_numpy(x_out), tz,
         _groups(dates, outer), best_epoch)
    preds = _predict(model2, x_out, _groups(dates, predict))
    keep = np.nonzero(predict)[0]
    return {"model": model2, "design": outer_design.to_json(),
            "history": pd.DataFrame(history),
            "predictions": pd.DataFrame({"date": dates[keep].astype("int64"),
                                         "code": data.loc[keep, "code"].to_numpy(),
                                         "score": preds[keep]}),
            "best_epoch": int(best_epoch), "best_inner_ic": float(best)}
