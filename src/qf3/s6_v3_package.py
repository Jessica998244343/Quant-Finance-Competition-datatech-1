"""V3 §C: assemble the full-data (label_end <= 2022-12-30) three-member package.

The package carries nine atoms - CatBoost C1 x3, TabM x3 and TCN x3 - fused
with the frozen flat graph ``{c1: 0.375, tabm: 0.375, tcn: 0.25}``.  Inference
is pure NumPy for TabM and TCN and pure CatBoost for C1, so ``requirements.txt``
stays torch-free (train.py needs the extra torch stack, listed separately).

Nothing here trains: the builder reads the already-saved full-fit atoms under
``artifacts/stage4_verified_ic/v2/full_2022`` and refuses to finish if any
weight cannot be bound to its saved design / preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

S4 = Path("artifacts/stage4_verified_ic")
V2 = S4 / "v2"
FULL = V2 / "full_2022"
FINAL_BUILD = Path("releases/stage3_fusion/build/FINAL_v1")
V1_BUILD = Path("releases/stage4_verified_ic/build/V1_pkg")
BUILD = Path("releases/stage4_verified_ic/build/V3_full_pkg")
SEEDS = (20261005, 20261017, 20261029)
WEIGHTS = {"c1": 0.375, "tabm": 0.375, "tcn": 0.25}
FUSION_GRAPH = "atom raw -> per-seed daily centred rank -> seed mean -> family weight -> final rank"
TRAIN_LABEL_CUTOFF = 20221230


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# static inference sources
# --------------------------------------------------------------------------
TCN_STATIC = '''"""Static NumPy forward for the frozen TCN weights (no torch, explicit ops)."""

import numpy as np


class TcnStatic:
    """W20 TCN (8 channels, 20 steps) evaluated with plain NumPy FP32.

    Explicit architecture copied from the frozen weights, no dynamic dispatch:
    ``Linear(static) -> ReLU``; a causal 1x1 input conv; two residual blocks of
    two causal k=3 dilated convs (dilations 1, 2) with ReLU (dropout off at
    eval); the last time position is concatenated with the static projection and
    mapped by a 64-wide MLP to one output.
    """

    def __init__(self, a: dict):
        self.static_W, self.static_b = a["static_W"], a["static_b"]
        self.input_W, self.input_b = a["input_W"], a["input_b"]
        self.blocks = [
            (a[f"b{i}_c1w"], a[f"b{i}_c1b"], a[f"b{i}_c2w"], a[f"b{i}_c2b"], d)
            for i, d in enumerate((1, 2))]
        self.fuse0_W, self.fuse0_b = a["fuse0_W"], a["fuse0_b"]
        self.fuse2_W, self.fuse2_b = a["fuse2_W"], a["fuse2_b"]

    @classmethod
    def from_npz(cls, path) -> "TcnStatic":
        required = ["static_W", "static_b", "input_W", "input_b",
                    "fuse0_W", "fuse0_b", "fuse2_W", "fuse2_b"]
        for i in range(2):
            required += [f"b{i}_c1w", f"b{i}_c1b", f"b{i}_c2w", f"b{i}_c2b"]
        with np.load(path, allow_pickle=False) as data:
            missing = [n for n in required if n not in data.files]
            extra = [n for n in data.files if n not in required]
            if missing or extra:
                raise ValueError(f"tcn npz key mismatch: missing={missing} extra={extra}")
            for name in required:
                value = data[name]
                if value.dtype != np.float32:
                    raise ValueError(f"{name}: dtype {value.dtype} != float32")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name}: contains non-finite values")
            arrays = {name: data[name] for name in required}
        return cls(arrays)

    @staticmethod
    def _pad_conv(x, W, b, dilation):
        _, _, kernel = W.shape
        pad = (kernel - 1) * dilation
        xpad = np.pad(x, ((0, 0), (0, 0), (pad, 0))).astype(np.float32)
        steps = x.shape[2]
        out = np.zeros((x.shape[0], W.shape[0], steps), dtype=np.float32)
        for k in range(kernel):
            window = xpad[:, :, k * dilation: k * dilation + steps]
            out += np.einsum("bct,oc->bot", window, W[:, :, k])
        return out + b[None, :, None]

    def score(self, static_x: np.ndarray, sequence: np.ndarray) -> np.ndarray:
        static = np.maximum(static_x @ self.static_W.T + self.static_b, 0.0)
        h = np.einsum("bct,oc->bot", sequence.transpose(0, 2, 1).astype(np.float32),
                      self.input_W[:, :, 0]) + self.input_b[None, :, None]
        for c1w, c1b, c2w, c2b, dilation in self.blocks:
            residual = h
            h = np.maximum(self._pad_conv(h, c1w, c1b, dilation), 0.0)
            h = np.maximum(self._pad_conv(h, c2w, c2b, dilation), 0.0) + residual
        z = np.concatenate([static, h[:, :, -1]], axis=1)
        z = np.maximum(z @ self.fuse0_W.T + self.fuse0_b, 0.0)
        return (z @ self.fuse2_W.T + self.fuse2_b).ravel()
'''

TCN_SEQ = '''"""W20 sequence construction for the packaged TCN (matches training exactly)."""

import numpy as np
import pandas as pd

STEPS = 20
PRICE_POINTS = 21
CHANNELS = 8


def build_sequence(market: pd.DataFrame, barra: pd.DataFrame, day_int: int):
    """Return (codes, (n, 20, 8) float32) for the legal pool on ``day_int``.

    Mirrors ``s6_v2_tcn.build_sequences``: 21 price points -> 20 one-day returns
    over a single trading calendar restricted to the same-day Barra membership.
    """
    membership = barra.loc[:, ["日期", "代码"]]
    hist = market.loc[market["日期"] <= day_int].merge(
        membership, on=["日期", "代码"], how="inner", sort=False)
    calendar = sorted(pd.unique(hist["日期"]).tolist())
    if not calendar:
        raise ValueError("no market history available for the requested day")
    window = calendar[-PRICE_POINTS:]
    # Deterministic short-history handling: missing leading days become all-NaN
    # columns (standardised to 0, validity channel 0), never a silent gain.
    window = [None] * (PRICE_POINTS - len(window)) + window
    codes = barra.loc[barra["日期"] == day_int, "代码"].drop_duplicates().tolist()
    panels = {}
    for column in ("open", "high", "low", "close", "volume", "amount", "is_suspend"):
        sub = hist.loc[:, ["代码", "日期", column]].drop_duplicates(["代码", "日期"])
        panels[column] = sub.pivot(index="代码", columns="日期", values=column)
    prices = np.zeros((len(codes), PRICE_POINTS, 7), dtype=np.float64)
    for index, column in enumerate(
            ("open", "high", "low", "close", "volume", "amount", "is_suspend")):
        prices[:, :, index] = panels[column].reindex(
            index=codes, columns=window).to_numpy(dtype=float)
    open_, high, low = prices[:, :, 0], prices[:, :, 1], prices[:, :, 2]
    close = prices[:, :, 3]
    volume, amount, suspend = prices[:, :, 4], prices[:, :, 5], prices[:, :, 6]
    out = np.zeros((len(codes), STEPS, CHANNELS), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        previous, current = close[:, :-1], close[:, 1:]
        out[:, :, 0] = np.log(np.where((previous > 0) & (current > 0),
                                       current / previous, np.nan))
        cur_open = open_[:, 1:]
        out[:, :, 1] = np.log(np.where((cur_open > 0) & (current > 0),
                                       current / cur_open, np.nan))
        out[:, :, 2] = ((high[:, 1:] - low[:, 1:])
                        / np.where(current > 0, current, np.nan))
        span = high[:, 1:] - low[:, 1:]
        out[:, :, 3] = ((current - low[:, 1:]) / np.where(span > 0, span, np.nan))
        out[:, :, 4] = np.log1p(np.where(volume[:, 1:] > 0, volume[:, 1:], np.nan))
        out[:, :, 5] = np.log1p(np.where(amount[:, 1:] > 0, amount[:, 1:], np.nan))
        out[:, :, 6] = np.where(np.isfinite(suspend[:, 1:]), suspend[:, 1:], 1.0)
        out[:, :, 7] = np.isfinite(current).astype(np.float32)
    return codes, out


def standardise(sequence: np.ndarray, channel_stats: dict) -> np.ndarray:
    """Fold-frozen channel scaling: clip to [-5, 5] then NaN -> 0 (channels 0-5)."""
    rows = np.array(sequence, dtype=np.float32, copy=True)
    for key, (median, iqr) in channel_stats.items():
        channel = int(key)
        rows[:, :, channel] = np.clip((rows[:, :, channel] - median) / iqr, -5, 5)
    return np.nan_to_num(rows, nan=0.0)
'''

TCN_TRAIN = '''"""Package-internal W20 TCN trainer (REV1 protocol); training dependency only."""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import alpha_dataio as dataio
from alpha_constants import BARRA_TABLE, MARKET_TABLE
from alpha_train_tabm import Design, _groups

STEPS = 20
PRICE_POINTS = 21
CHANNELS = 8
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 80, 10, 1e-3, 1e-3, 1.0, 4
THREADS = 2


def rank_target(frame, label_col="future_return"):
    label = pd.to_numeric(frame[label_col], errors="coerce")
    finite = label.notna().to_numpy()
    ranks = label.groupby(frame["date"]).rank(method="average")
    counts = label.where(label.notna()).groupby(frame["date"]).transform("count")
    z = (ranks - (counts + 1.0) / 2.0) / counts
    return z.to_numpy(dtype=float), (1.0 / counts).to_numpy(dtype=float), finite


def build_sequences(data_dir="."):
    tables = dataio.load_all(data_dir)
    membership = tables[BARRA_TABLE].loc[:, ["\\u65e5\\u671f", "\\u4ee3\\u7801"]]
    market = tables[MARKET_TABLE].merge(membership, on=["\\u65e5\\u671f", "\\u4ee3\\u7801"],
                                        how="inner", sort=False)
    barra = tables[BARRA_TABLE]
    calendar = sorted(pd.unique(market["\\u65e5\\u671f"]).tolist())
    position = {day: index for index, day in enumerate(calendar)}
    panels = {}
    for column in ("open", "high", "low", "close", "volume", "amount", "is_suspend"):
        sub = market.loc[:, ["\\u4ee3\\u7801", "\\u65e5\\u671f", column]].drop_duplicates(
            ["\\u4ee3\\u7801", "\\u65e5\\u671f"])
        panels[column] = sub.pivot(index="\\u4ee3\\u7801", columns="\\u65e5\\u671f", values=column)
    sequences, keys = {}, {}
    for day in calendar:
        index = position[day]
        if index < STEPS:
            continue
        codes = barra.loc[barra["\\u65e5\\u671f"] == day, "\\u4ee3\\u7801"].drop_duplicates().tolist()
        window = calendar[index - STEPS:index + 1]
        prices = np.zeros((len(codes), PRICE_POINTS, 7), dtype=np.float64)
        for column_index, column in enumerate(
                ("open", "high", "low", "close", "volume", "amount", "is_suspend")):
            prices[:, :, column_index] = panels[column].reindex(
                index=codes, columns=window).to_numpy(dtype=float)
        open_, high, low = prices[:, :, 0], prices[:, :, 1], prices[:, :, 2]
        close = prices[:, :, 3]
        volume, amount, suspend = prices[:, :, 4], prices[:, :, 5], prices[:, :, 6]
        out = np.zeros((len(codes), STEPS, CHANNELS), dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            previous, current = close[:, :-1], close[:, 1:]
            out[:, :, 0] = np.log(np.where((previous > 0) & (current > 0),
                                           current / previous, np.nan))
            cur_open = open_[:, 1:]
            out[:, :, 1] = np.log(np.where((cur_open > 0) & (current > 0),
                                           current / cur_open, np.nan))
            out[:, :, 2] = ((high[:, 1:] - low[:, 1:])
                            / np.where(current > 0, current, np.nan))
            span = high[:, 1:] - low[:, 1:]
            out[:, :, 3] = ((current - low[:, 1:]) / np.where(span > 0, span, np.nan))
            out[:, :, 4] = np.log1p(np.where(volume[:, 1:] > 0, volume[:, 1:], np.nan))
            out[:, :, 5] = np.log1p(np.where(amount[:, 1:] > 0, amount[:, 1:], np.nan))
            out[:, :, 6] = np.where(np.isfinite(suspend[:, 1:]), suspend[:, 1:], 1.0)
            out[:, :, 7] = np.isfinite(current).astype(np.float32)
        sequences[day] = out
        keys[day] = codes
    return sequences, keys


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.dilation = dilation

    def forward(self, x):
        h = F.pad(x, (2 * self.dilation, 0))
        h = self.drop(F.relu(self.conv1(h)))
        h = F.pad(h, (2 * self.dilation, 0))
        h = self.drop(F.relu(self.conv2(h)))
        return h + x


class TCNModel(nn.Module):
    def __init__(self, d_static, channels=32, hidden=64, dropout=0.1, use_time=True):
        super().__init__()
        self.use_time = use_time
        self.static = nn.Sequential(nn.Linear(d_static, channels), nn.ReLU())
        self.input = nn.Conv1d(CHANNELS, channels, 1)
        self.blocks = nn.ModuleList([ResidualBlock(channels, dilation, dropout)
                                     for dilation in (1, 2)])
        self.fuse = nn.Sequential(nn.Linear(2 * channels, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, static_x, sequence):
        static = self.static(static_x)
        h = self.input(sequence.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.fuse(torch.cat([static, h[:, :, -1]], dim=1)).squeeze(-1)


def _channel_stats(data, sequence_map, mask):
    stats = {}
    days = data.loc[mask, "date"].unique()
    for channel in range(CHANNELS - 2):
        values = np.concatenate([sequence_map[day][:, :, channel].ravel()
                                 for day in days if day in sequence_map])
        values = values[np.isfinite(values)]
        median = float(np.median(values)) if values.size else 0.0
        q75, q25 = (np.percentile(values, 75), np.percentile(values, 25)) \\
            if values.size else (0.0, 0.0)
        iqr = float(q75 - q25)
        stats[channel] = [median, iqr if iqr > 0 else 1.0]
    return stats


def _sequence_tensor(data, sequence_map, codes_by_day, stats):
    rows = np.zeros((len(data), STEPS, CHANNELS), dtype=np.float32)
    for day, block in data.groupby("date", sort=False):
        if day not in sequence_map:
            continue
        lookup = {code: index for index, code in enumerate(codes_by_day[day])}
        for position, code in zip(block.index.to_numpy(), block["code"].to_numpy()):
            index = lookup.get(code)
            if index is not None:
                rows[position] = sequence_map[day][index]
    tensor = torch.from_numpy(rows)
    for channel, (median, iqr) in stats.items():
        tensor[:, :, channel] = torch.clamp((tensor[:, :, channel] - median) / iqr, -5, 5)
    return torch.nan_to_num(tensor, nan=0.0)


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


def train_fold(data, sequence_map, codes_by_day, split, seed):
    torch.set_num_threads(THREADS)
    started = time.time()
    dates = pd.to_numeric(data["date"], errors="coerce").astype("float64").to_numpy()
    z, _, finite = rank_target(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    internal, early = split["internal"] & finite, split["early_stop"] & finite
    outer = split["outer_train"] & finite
    inner_design = Design().fit(data.loc[internal])
    static_inner = inner_design.transform(data)
    stats = _channel_stats(data, sequence_map, internal)
    sequence_tensor = _sequence_tensor(data, sequence_map, codes_by_day, stats)
    static_tensor = torch.from_numpy(static_inner)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    torch.manual_seed(seed)
    model = TCNModel(static_inner.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    inner_groups, valid_groups = _groups(dates, internal), _groups(dates, early)

    def predict_on(groups):
        model.eval()
        out = np.zeros(len(static_tensor))
        with torch.no_grad():
            for group in groups:
                out[group] = model(static_tensor[group], sequence_tensor[group]).numpy()
        return out

    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(inner_groups))
        for step in range(0, len(order), BATCH_DAYS):
            batch = [inner_groups[i] for i in order[step:step + BATCH_DAYS]]
            optimizer.zero_grad()
            losses = [F.mse_loss(model(static_tensor[g], sequence_tensor[g]), tensor_z[g])
                      for g in batch]
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        ic = _daily_ic(raw, valid_groups, predict_on(valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    outer_design = Design().fit(data.loc[outer])
    static_outer = outer_design.transform(data)
    outer_stats = _channel_stats(data, sequence_map, outer)
    sequence_tensor = _sequence_tensor(data, sequence_map, codes_by_day, outer_stats)
    outer_tensor = torch.from_numpy(static_outer)
    torch.manual_seed(seed)
    model2 = TCNModel(static_outer.shape[1])
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=WD)
    rng2 = np.random.default_rng(seed)
    outer_groups = _groups(dates, outer)
    for _ in range(best_epoch):
        model2.train()
        order = rng2.permutation(len(outer_groups))
        for step in range(0, len(order), BATCH_DAYS):
            batch = [outer_groups[i] for i in order[step:step + BATCH_DAYS]]
            optimizer2.zero_grad()
            losses = [F.mse_loss(model2(outer_tensor[g], sequence_tensor[g]), tensor_z[g])
                      for g in batch]
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), CLIP)
            optimizer2.step()
    return {"model": model2, "design": outer_design.to_json(), "channel_stats": outer_stats,
            "best_epoch": int(best_epoch), "best_inner_ic": float(best_ic),
            "history": pd.DataFrame(history), "seconds": round(time.time() - started, 1)}
'''

MODELS = '''"""V3 scorer: 0.375 C1 + 0.375 TabM + 0.25 TCN on the frozen flat graph."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_tabminput, alpha_tabm_static, alpha_tcn_seq, alpha_tcn_static


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def _seed(name: str) -> int:
    match = re.search(r"seed(\\d+)", name)
    if match is None:
        raise ValueError(f"member name without a seed identity: {name}")
    return int(match.group(1))


class Scorer:
    """Frozen-graph scorer.  Contract: exactly one complete trading day per call."""

    def __init__(self, model_dir: str):
        from catboost import CatBoostRegressor

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.weights = dict(meta["weights"])
        self.fusion_graph = meta.get("fusion_graph")
        self.c1_seeds = [_seed(n) for n in meta["members"]["c1"]]
        self.tabm_seeds = [_seed(n) for n in meta["members"]["tabm"]]
        self.tcn_seeds = [_seed(n) for n in meta["members"]["tcn"]]
        if not (self.c1_seeds == self.tabm_seeds == self.tcn_seeds):
            raise ValueError("family seed identities differ")
        self.cat = []
        for name in meta["members"]["c1"]:
            model = CatBoostRegressor()
            model.load_model(str(self.dir / name))
            self.cat.append(model)
        self.tabm_design = json.loads((self.dir / "tabm_design.json").read_text())
        self.tcn_design = json.loads((self.dir / "tcn_design.json").read_text())
        self.tcn_channel_stats = json.loads((self.dir / "tcn_channel_stats.json").read_text())
        self.tabm = [alpha_tabm_static.TabmStatic.from_npz(self.dir / name)
                     for name in meta["members"]["tabm"]]
        self.tcn = [alpha_tcn_static.TcnStatic.from_npz(self.dir / name)
                    for name in meta["members"]["tcn"]]

    def _one_day(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "date" in frame.columns and frame["date"].nunique(dropna=False) > 1:
            raise ValueError("Scorer contract: exactly one trading day per call")
        return frame

    def member_raw(self, frame: pd.DataFrame, sequence: np.ndarray) -> dict:
        frame = self._one_day(frame)
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        scores = {}
        for seed, model in zip(self.c1_seeds, self.cat):
            scores[f"c1_{seed}"] = np.asarray(model.predict(matrix), dtype=float).ravel()
        tabm_input = alpha_tabminput.transform(frame, self.tabm_design)
        for seed, model in zip(self.tabm_seeds, self.tabm):
            scores[f"tabm_{seed}"] = np.asarray(model.score(tabm_input), dtype=float).ravel()
        tcn_input = alpha_tabminput.transform(frame, self.tcn_design)
        rows = alpha_tcn_seq.standardise(sequence, self.tcn_channel_stats)
        for seed, model in zip(self.tcn_seeds, self.tcn):
            scores[f"tcn_{seed}"] = np.asarray(model.score(tcn_input, rows), dtype=float).ravel()
        return scores

    def atom_scores(self, frame: pd.DataFrame, sequence: np.ndarray) -> dict:
        raw = self.member_raw(frame, sequence)
        out = {}
        for family, seeds in (("c1", self.c1_seeds), ("tabm", self.tabm_seeds),
                              ("tcn", self.tcn_seeds)):
            value = np.zeros(len(frame))
            for seed in seeds:
                value += _rank(raw[f"{family}_{seed}"]) / len(seeds)
            out[family] = value
        return out

    def score(self, frame: pd.DataFrame, sequence: np.ndarray) -> np.ndarray:
        atoms = self.atom_scores(frame, sequence)
        fused = sum(self.weights[name] * atoms[name] for name in ("c1", "tabm", "tcn"))
        return _rank(fused)
'''

INFERENCE = '''"""Official inference entry point (V3 full-data research package)."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_features, alpha_tcn_seq
from .alpha_models import Scorer


class InferenceModel:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.scorer = Scorer(model_dir)
        self.features = self.scorer.features

    def predict(self, as_of_date: str, data: dict) -> pd.DataFrame:
        day = int(as_of_date.replace("-", ""))
        market, factor, barra = data["MarketData"], data["Factor33_winsor"], data["Barra_factor"]
        frame = alpha_features.build_day_features(market, factor, barra, day)
        universe = barra.loc[barra["\\u65e5\\u671f"] == day, "\\u4ee3\\u7801"].drop_duplicates().tolist()
        frame = frame.set_index("code").reindex(universe).reset_index()
        codes, sequence = alpha_tcn_seq.build_sequence(market, barra, day)
        if list(codes) != list(frame["code"]):
            raise ValueError("sequence universe differs from the feature universe")
        score = np.asarray(self.scorer.score(frame, sequence), dtype=float).ravel()
        if score.shape[0] != len(frame) or not np.isfinite(score).all():
            raise ValueError("scorer produced invalid scores")
        return pd.DataFrame({"date": as_of_date, "code": frame["code"], "score": score})
'''


# --------------------------------------------------------------------------
# weight export
# --------------------------------------------------------------------------
def export_tabm_npz(seed: int, out: Path) -> dict:
    from safetensors.torch import load_file

    src = FULL / f"tabm_seed{seed}.safetensors"
    sd = load_file(str(src))
    arrays = {
        "W0": sd["backbone.blocks.0.0.weight"].numpy(),
        "r0": sd["backbone.blocks.0.0.r"].numpy(),
        "s0": sd["backbone.blocks.0.0.s"].numpy(),
        "b0": sd["backbone.blocks.0.0.bias"].numpy(),
        "W1": sd["backbone.blocks.1.0.weight"].numpy(),
        "r1": sd["backbone.blocks.1.0.r"].numpy(),
        "s1": sd["backbone.blocks.1.0.s"].numpy(),
        "b1": sd["backbone.blocks.1.0.bias"].numpy(),
        "Wh": sd["head.weight"].numpy(),
        "bh": sd["head.bias"].numpy(),
        "k": np.array([8], dtype=np.float32),
    }
    np.savez(out, **{name: np.ascontiguousarray(v, dtype=np.float32)
                     for name, v in arrays.items()})
    return {"source": str(src), "source_sha": _sha(src), "npz_sha": _sha(out)}


def export_tcn_npz(seed: int, out: Path) -> dict:
    from safetensors.torch import load_file

    src = V2 / f"tcn_rev1_tcn_s{seed}_FULL22" / "model.safetensors"
    sd = load_file(str(src))
    arrays = {
        "static_W": sd["static.0.weight"].numpy(),
        "static_b": sd["static.0.bias"].numpy(),
        "input_W": sd["input.weight"].numpy(),
        "input_b": sd["input.bias"].numpy(),
        "fuse0_W": sd["fuse.0.weight"].numpy(),
        "fuse0_b": sd["fuse.0.bias"].numpy(),
        "fuse2_W": sd["fuse.2.weight"].numpy(),
        "fuse2_b": sd["fuse.2.bias"].numpy(),
    }
    for i in range(2):
        arrays[f"b{i}_c1w"] = sd[f"blocks.{i}.conv1.weight"].numpy()
        arrays[f"b{i}_c1b"] = sd[f"blocks.{i}.conv1.bias"].numpy()
        arrays[f"b{i}_c2w"] = sd[f"blocks.{i}.conv2.weight"].numpy()
        arrays[f"b{i}_c2b"] = sd[f"blocks.{i}.conv2.bias"].numpy()
    np.savez(out, **{name: np.ascontiguousarray(v, dtype=np.float32)
                     for name, v in arrays.items()})
    return {"source": str(src), "source_sha": _sha(src), "npz_sha": _sha(out)}


def _read_catboost_trees(path: Path) -> int:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(path))
    return int(model.tree_count_)


def build() -> dict:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "model").mkdir(parents=True)
    (BUILD / "__init__.py").write_text('"""V3 full-data research package."""\n')
    import sys
    sys.path.insert(0, "src")
    from . import s4_package
    mapping = {"constants.py": "alpha_constants.py", "featurelib.py": "alpha_features.py",
               "dataio.py": "alpha_dataio.py", "labels.py": "alpha_labels.py",
               "panel.py": "alpha_panel.py", "cv.py": "alpha_cv.py"}
    for source, target in mapping.items():
        text = s4_package.rewrite((Path("src/qf3") / source).read_text())
        (BUILD / target).write_text(text)
    shutil.copy(FINAL_BUILD / "alpha_tabminput.py", BUILD / "alpha_tabminput.py")
    shutil.copy(V1_BUILD / "alpha_tabm_static.py", BUILD / "alpha_tabm_static.py")
    (BUILD / "alpha_tcn_static.py").write_text(TCN_STATIC)
    (BUILD / "alpha_tcn_seq.py").write_text(TCN_SEQ)
    (BUILD / "alpha_tcn_train.py").write_text(TCN_TRAIN)
    (BUILD / "alpha_models.py").write_text(MODELS)
    (BUILD / "inference.py").write_text(INFERENCE)

    members = {"c1": [], "tabm": [], "tcn": []}
    hashes, binding = {}, {"c1": {}, "tabm": {}, "tcn": {}}
    for seed in SEEDS:
        cat_src = FULL / f"catboost_c1_seed{seed}.cbm"
        cat_dst = BUILD / "model" / f"catboost_c1_seed{seed}.cbm"
        shutil.copy(cat_src, cat_dst)
        members["c1"].append(cat_dst.name)
        hashes[cat_dst.name] = _sha(cat_dst)
        binding["c1"][cat_dst.name] = {"trees": _read_catboost_trees(cat_dst),
                                       "source": str(cat_src), "source_sha": _sha(cat_src)}
        info = export_tabm_npz(seed, BUILD / "model" / f"tabm_seed{seed}.npz")
        members["tabm"].append(f"tabm_seed{seed}.npz")
        hashes[f"tabm_seed{seed}.npz"] = info["npz_sha"]
        binding["tabm"][f"tabm_seed{seed}.npz"] = info
        info = export_tcn_npz(seed, BUILD / "model" / f"tcn_seed{seed}.npz")
        members["tcn"].append(f"tcn_seed{seed}.npz")
        hashes[f"tcn_seed{seed}.npz"] = info["npz_sha"]
        binding["tcn"][f"tcn_seed{seed}.npz"] = info

    designs = []
    for seed in SEEDS:
        designs.append((FULL / f"design_seed{seed}.json").read_text())
    if len(set(designs)) != 1:
        raise RuntimeError("TabM designs differ across seeds")
    (BUILD / "model" / "tabm_design.json").write_text(designs[0])

    tcn_pre = [json.loads((V2 / f"tcn_rev1_tcn_s{seed}_FULL22" / "preprocessing.json").read_text())
               for seed in SEEDS]
    design_keys = ("design_median", "design_iqr")
    for key in design_keys:
        if len({json.dumps(item[key], sort_keys=True) for item in tcn_pre}) != 1:
            raise RuntimeError(f"TCN {key} differs across seeds")
    if len({json.dumps(item["channel_stats"], sort_keys=True) for item in tcn_pre}) != 1:
        raise RuntimeError("TCN channel stats differ across seeds")
    tabm_design = json.loads(designs[0])
    if any(abs(tcn_pre[0]["design_median"][k] - tabm_design["median"][k]) > 1e-9
           or abs(tcn_pre[0]["design_iqr"][k] - tabm_design["iqr"][k]) > 1e-9
           for k in tabm_design["features"]):
        raise RuntimeError("TCN static design != TabM design")
    (BUILD / "model" / "tcn_design.json").write_text(json.dumps(
        {"mode": "N0", "features": tabm_design["features"],
         "median": tcn_pre[0]["design_median"], "iqr": tcn_pre[0]["design_iqr"]}, indent=2))
    (BUILD / "model" / "tcn_channel_stats.json").write_text(
        json.dumps(tcn_pre[0]["channel_stats"], indent=2))

    from .constants import F0_COLUMNS, G1_COLUMNS
    meta = {"version": "V3_full_pkg", "train_label_cutoff": TRAIN_LABEL_CUTOFF,
            "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": WEIGHTS, "members": members, "fusion_graph": FUSION_GRAPH,
            "train_cutoff_note": "outer fit uses label_end_date <= 2022-12-30",
            "runtime": "NumPy inference for TabM/TCN; no torch required at run time",
            "member_hashes": hashes, "atom_binding": binding}
    (BUILD / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    import numpy, pandas, pyarrow, catboost, torch, safetensors
    (BUILD / "requirements.txt").write_text(
        f"numpy=={numpy.__version__}\npandas=={pandas.__version__}\n"
        f"pyarrow=={pyarrow.__version__}\ncatboost=={catboost.__version__}\n")
    (BUILD / "train_requirements.txt").write_text(
        (BUILD / "requirements.txt").read_text()
        + f"torch=={torch.__version__}\nsafetensors=={safetensors.__version__}\n"
        "rtdl_num_embeddings==0.0.12\n")
    shutil.copy(FINAL_BUILD / "LICENSE.tabm", BUILD / "LICENSE.tabm")
    shutil.copy(V1_BUILD / "alpha_train_model.py", BUILD / "alpha_train_model.py")
    shutil.copy(V1_BUILD / "alpha_train_tabm.py", BUILD / "alpha_train_tabm.py")
    shutil.copy(Path("src/qf3/s6_v3_train_src.py"), BUILD / "train.py")
    (BUILD / "TABM_SOURCE.md").write_text(
        "# TabM architecture provenance\n\n"
        "`alpha_train_model.py` reproduces only the components used by this model\n"
        "from the Yandex TabM reference implementation (Apache-2.0, LICENSE.tabm):\n"
        "- repository: https://github.com/yandex-research/tabm\n"
        "- file: `tabm.py`, `__version__ = '0.0.3'`, vendored commit "
        "28e47ae301c92ec37787dde1ce923a0793f405b4\n\n"
        "The activation is a fixed `nn.ReLU`; there is no dynamic dispatch.\n")
    return {"build": str(BUILD), "members": {k: len(v) for k, v in members.items()}}


def zip_package(zip_path: Path) -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("submission") / path.relative_to(BUILD))
    with zipfile.ZipFile(zip_path) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
        names = sorted(archive.namelist())
    return {"zip": str(zip_path), "bytes": zip_path.stat().st_size,
            "uncompressed_bytes": uncompressed, "files": len(names),
            "sha256": _sha(zip_path)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
