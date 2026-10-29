"""§12 (REV1): TCN W20 with a same-budget static control.

Fixes required by the review:
  * ``torch.manual_seed(seed)`` runs **before** every model construction, so the
    initialisation is controlled (a self-check proves it);
  * the sequence uses **21 price points -> 20 one-day returns**, so no step has an
    artificial NaN return; presence mask and channel validity are explicit;
  * every run saves model weights, preprocessing/channel statistics, the inner
    curve, per-day predictions and a manifest;
  * the member scan uses the canonical flat graph (no intermediate family rank).
"""

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

from . import cv, dataio, s2r_train
from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE
from .s6_fusion import as_family, family_q, fuse, push_down_alpha
from .s6_v2_tabm_variants import Design

V2 = Path("artifacts/stage4_verified_ic/v2")
S4 = Path("artifacts/stage4_verified_ic")
STEPS = 20            # one-day return steps
PRICE_POINTS = 21     # STEPS + 1
CHANNELS = 8
MAX_EPOCHS, PATIENCE, LR, WD, CLIP, BATCH_DAYS = 80, 10, 1e-3, 1e-3, 1.0, 4
THREADS = 2
SEED = 20261005
ALLOW_EXISTING = False      # new runs refuse to overwrite by default (review §4.4)
WINDOWS = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_sequences(data_dir: str = "Source/Data") -> dict:
    """{date: (codes, 20, 8)} with 21 price points per window (20 valid returns)."""
    tables = dataio.load_all(data_dir)
    membership = tables[BARRA_TABLE].loc[:, ["日期", "代码"]]
    market = tables[MARKET_TABLE].merge(membership, on=["日期", "代码"], how="inner",
                                        sort=False)
    barra = tables[BARRA_TABLE]
    calendar = sorted(pd.unique(market["日期"]).tolist())
    position = {day: index for index, day in enumerate(calendar)}
    panels = {}
    for column in ("open", "high", "low", "close", "volume", "amount", "is_suspend"):
        sub = market.loc[:, ["代码", "日期", column]].drop_duplicates(["代码", "日期"])
        panels[column] = sub.pivot(index="代码", columns="日期", values=column)
    sequences, keys = {}, {}
    for day in calendar:
        index = position[day]
        if index < STEPS:
            continue
        codes = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
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
            previous = close[:, :-1]
            current = close[:, 1:]
            out[:, :, 0] = np.log(np.where((previous > 0) & (current > 0),
                                           current / previous, np.nan))
            cur_open = open_[:, 1:]
            out[:, :, 1] = np.log(np.where((cur_open > 0) & (current > 0),
                                           current / cur_open, np.nan))
            out[:, :, 2] = ((high[:, 1:] - low[:, 1:])
                            / np.where(current > 0, current, np.nan))
            span = high[:, 1:] - low[:, 1:]
            out[:, :, 3] = ((current - low[:, 1:])
                            / np.where(span > 0, span, np.nan))
            out[:, :, 4] = np.log1p(np.where(volume[:, 1:] > 0, volume[:, 1:], np.nan))
            out[:, :, 5] = np.log1p(np.where(amount[:, 1:] > 0, amount[:, 1:], np.nan))
            out[:, :, 6] = np.where(np.isfinite(suspend[:, 1:]), suspend[:, 1:], 1.0)
            out[:, :, 7] = np.isfinite(current).astype(np.float32)
        sequences[day] = out
        keys[day] = codes
    return {"sequences": sequences, "keys": keys}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
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
    def __init__(self, d_static: int, channels: int = 32, hidden: int = 64,
                 dropout: float = 0.1, use_time: bool = True):
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
        if self.use_time:
            h = self.input(sequence.transpose(1, 2))
            for block in self.blocks:
                h = block(h)
            time = h[:, :, -1]
        else:
            time = torch.zeros_like(static)
        return self.fuse(torch.cat([static, time], dim=1)).squeeze(-1)


def init_is_controlled(seed: int) -> bool:
    """Prove the seed reset (not the ambient RNG) determines the initial weights."""
    torch.manual_seed(12345)
    torch.randn(7)
    torch.manual_seed(seed)
    first = TCNModel(141)
    torch.manual_seed(999)
    torch.randn(13)
    torch.manual_seed(seed)
    second = TCNModel(141)
    a = dict(first.state_dict())
    b = dict(second.state_dict())
    return all(torch.equal(a[key], b[key]) for key in a)


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


def frame_ic(frame: pd.DataFrame) -> float:
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


def _channel_stats(data, sequence_map, mask) -> dict:
    stats = {}
    days = data.loc[mask, "date"].unique()
    for channel in range(CHANNELS - 2):
        values = np.concatenate([sequence_map[day][:, :, channel].ravel()
                                 for day in days if day in sequence_map])
        values = values[np.isfinite(values)]
        median = float(np.median(values)) if values.size else 0.0
        q75, q25 = (np.percentile(values, 75), np.percentile(values, 25)) \
            if values.size else (0.0, 0.0)
        iqr = float(q75 - q25)
        stats[channel] = [median, iqr if iqr > 0 else 1.0]
    return stats


def _sequence_tensor(data, sequence_map, codes_by_day, stats, shuffle_time) -> torch.Tensor:
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
    tensor = torch.nan_to_num(tensor, nan=0.0)
    if shuffle_time:
        generator = torch.Generator().manual_seed(20261006)
        tensor = tensor[:, torch.randperm(STEPS, generator=generator), :]
    return tensor


def train_fold(data, sequence_map, codes_by_day, window, seed, use_time,
               shuffle_time=False, evaluate=True, split=None) -> dict:
    torch.set_num_threads(THREADS)
    start = time.time()
    split = split if split is not None else cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    internal, early = split["internal"] & finite, split["early_stop"] & finite
    outer, predict = split["outer_train"] & finite, split["predict"]
    inner_design = Design().fit(data.loc[internal])
    static_inner = inner_design.transform(data)
    stats = _channel_stats(data, sequence_map, internal)
    sequence_tensor = _sequence_tensor(data, sequence_map, codes_by_day, stats, shuffle_time)
    static_tensor = torch.from_numpy(static_inner)
    tensor_z = torch.from_numpy(np.nan_to_num(z))
    torch.manual_seed(seed)                     # BEFORE construction (review §4.1)
    model = TCNModel(static_inner.shape[1], use_time=use_time)
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
    sequence_tensor = _sequence_tensor(data, sequence_map, codes_by_day, outer_stats,
                                       shuffle_time)
    outer_tensor = torch.from_numpy(static_outer)
    torch.manual_seed(seed)                     # fresh model / optimiser / RNG
    model2 = TCNModel(static_outer.shape[1], use_time=use_time)
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
    model2.eval()
    if evaluate:
        result = np.zeros(len(static_tensor))
        with torch.no_grad():
            for group in _groups(dates, predict):
                result[group] = model2(outer_tensor[group], sequence_tensor[group]).numpy()
        keep = np.nonzero(predict)[0]
        predictions = pd.DataFrame({"date": dates[keep].astype("int64"),
                                    "code": data.loc[keep, "code"].to_numpy(),
                                    "score": result[keep]})
    else:
        predictions = None
    label = "tcn" if use_time and not shuffle_time else (
        "static" if not use_time else "shuffle")
    tag = f"tcn_rev1_{label}_s{seed}_{window}"
    from .s6_io import new_run_dir
    out = new_run_dir(V2 / tag, ALLOW_EXISTING)
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous() for k, v in model2.state_dict().items()},
              str(out / "model.safetensors"))
    pd.DataFrame(history).to_parquet(out / "inner_curve.parquet", index=False)
    if predictions is not None:
        predictions.to_parquet(out / "predictions.parquet", index=False)
    (out / "preprocessing.json").write_text(json.dumps(
        {"channel_stats": {str(k): v for k, v in outer_stats.items()},
         "design_mode": "N0",
         "design_median": outer_design.median, "design_iqr": outer_design.iqr}, indent=2))
    manifest = {"tag": tag, "window": window, "seed": seed, "use_time": use_time,
                "shuffle_time": shuffle_time, "steps": STEPS,
                "price_points": PRICE_POINTS, "channels": CHANNELS,
                "best_epoch": int(best_epoch), "best_inner_ic": float(best_ic),
                "inner_epochs_run": len(history), "train_cutoff": int(split["train_end"]),
                "train_threads": THREADS, "seconds": round(time.time() - start, 1),
                "seed_set_before_construction": True,
                "outer_metrics_computed": bool(evaluate)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["hashes"] = {"model": _sha(out / "model.safetensors")}
    if predictions is not None:
        manifest["hashes"]["predictions"] = _sha(out / "predictions.parquet")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest | {"predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--labels", default="tcn,static,shuffle")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(V2 / "v2_rev1_tcn.json"))
    parser.add_argument("--deploy", action="store_true",
                        help="train cutoff-2021 models only (RES22 window, no 2022 evaluation)")
    parser.add_argument("--halfyear", action="store_true",
                        help="train the TCN on the three H2 starting points (single seed)")
    parser.add_argument("--hy-windows", default="2019H2,2020H2,2021H2")
    parser.add_argument("--allow-existing", action="store_true",
                        help="reuse an existing run directory (off by default)")
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    if args.allow_existing:
        globals()["ALLOW_EXISTING"] = True
    labels = data[["date", "code", "future_return"]]
    built = build_sequences()
    sequence_map, codes_by_day = built["sequences"], built["keys"]
    if args.halfyear:
        from .s6_v2_deploy_models import HALF_YEARS, custom_split

        report = {"seed": args.seed, "windows": {}, "alphas": list(ALPHAS)}
        for window, spec in HALF_YEARS.items():
            if window not in args.hy_windows.split(","):
                continue
            split = custom_split(data, spec["train_end"], spec["predict_start"],
                                 spec["predict_end"])
            manifest = train_fold(data, sequence_map, codes_by_day, window, args.seed,
                                  True, False, evaluate=True, split=split)
            report["windows"][window] = {
                "train_label_cutoff": spec["train_end"],
                "best_epoch": int(manifest["best_epoch"]),
                "seconds": manifest["seconds"],
                "manifest": str(V2 / manifest["tag"] / "manifest.json")}
            print("halfyear TCN", window, "best_epoch", manifest["best_epoch"], flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    if args.deploy:
        report = {"seed": args.seed, "deploy": [], "window": "RES22",
                  "train_label_cutoff": 20211231, "outer_metrics_computed": False}
        for seed in (20261005, 20261017, 20261029):
            manifest = train_fold(data, sequence_map, codes_by_day, "RES22", seed,
                                  True, False, evaluate=False)
            report["deploy"].append({k: v for k, v in manifest.items() if k != "predictions"})
            print("deploy TCN seed", seed, "best_epoch", manifest["best_epoch"], flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    report = {"seed": args.seed, "steps": STEPS, "price_points": PRICE_POINTS,
              "channels": CHANNELS, "init_controlled": init_is_controlled(args.seed),
              "windows": {}}
    variants = {"tcn": (True, False), "static": (False, False), "shuffle": (True, True)}
    for window in args.windows.split(","):
        parents = {"c1": as_family(pd.read_parquet(
            S4 / f"v1_c1_seedmean__{window}.parquet"), column="c1", label=f"c1_{window}"),
            "tabm": as_family(pd.read_parquet(
                S4 / f"v1_tabm_seedmean__{window}.parquet"), column="tabm",
                label=f"tabm_{window}")}
        entry = {}
        for label in args.labels.split(","):
            use_time, shuffle = variants[label]
            manifest = train_fold(data, sequence_map, codes_by_day, window, args.seed,
                                  use_time, shuffle)
            frame = manifest["predictions"].merge(labels, on=["date", "code"], how="left",
                                                  validate="one_to_one")
            new_q = family_q([manifest["predictions"]], "score")
            components = dict(parents)
            components[label] = new_q
            curve = {}
            for alpha in ALPHAS:
                weights = push_down_alpha({"c1": 0.5, "tabm": 0.5}, label, alpha)
                fused = fuse(components, weights).merge(
                    labels, on=["date", "code"], how="left", validate="one_to_one")
                curve[str(alpha)] = frame_ic(fused)
            entry[label] = {"standalone_rankic": frame_ic(frame),
                            "best_epoch": manifest["best_epoch"],
                            "seconds": manifest["seconds"], "ic_by_alpha": curve,
                            "delta_vs_parent": {k: curve[k] - curve["0.0"] for k in curve},
                            "manifest": str(V2 / manifest["tag"] / "manifest.json")}
            print(window, label, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                             for k, v in entry[label].items()
                                             if k not in ("ic_by_alpha", "manifest")}),
                  flush=True)
        report["windows"][window] = entry
    # DEPRECATED (review §4.4): this trainer writes per-run evidence only.  The
    # cross-window DEV aggregation used to mix a three-entry window list with a
    # two-entry weight list; the supported summary entry is now
    #   python -m qf3.s6_v2_completion --command family --windows DEV19,DEV20[,CONF21]
    report["aggregation"] = "moved to qf3.s6_v2_completion (this file only trains/saves)"
    if len(windows) > 1:
        print("NOTE: cross-window aggregation moved to `python -m qf3.s6_v2_completion`",
              flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"init_controlled": report["init_controlled"],
                      "windows": list(report["windows"]),
                      "aggregation": report["aggregation"]}, indent=2))


if __name__ == "__main__":
    main()
