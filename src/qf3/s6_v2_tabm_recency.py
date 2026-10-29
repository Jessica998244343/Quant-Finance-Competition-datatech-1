"""§10.3 (triggered): TabM trained on the most recent 504 legal trading days.

Same architecture, objective, optimiser and epoch-selection protocol as the
expanding control; only the training window changes.  Evaluated against the
paired expanding control and, under the canonical flat graph, as an extra member
of the frozen anchor.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cv, s2r_train
from .s6_fusion import as_family, daily_centred_rank, family_q, fuse, push_down_alpha
from .s6_v2_tabm_variants import Design, _groups, _new, _predict, _run

V2 = Path("artifacts/stage4_verified_ic/v2")
S4 = Path("artifacts/stage4_verified_ic")
RECENT_DAYS = 504
EARLY_STOP_DAYS = 126
SEED = 20261005
ALPHAS = (0.0, 0.05, 0.10, 0.25, 0.50)
WINDOWS = {"DEV19": (20181231, 2019), "DEV20": (20191231, 2020)}


def recency_split(frame: pd.DataFrame, train_end: int, year: int,
                  days: int = RECENT_DAYS) -> dict:
    dates = cv.to_int_date(frame["date"])
    label_end = cv.to_int_date(frame["label_end_date"])
    eligible = label_end <= train_end
    train_dates = np.sort(pd.unique(dates[eligible]))
    recent_start = int(train_dates[-days]) if len(train_dates) > days else int(train_dates[0])
    outer = eligible & (dates >= recent_start)
    recent_dates = np.sort(pd.unique(dates[outer]))
    es_days = min(EARLY_STOP_DAYS, max(1, len(recent_dates) // 4))
    es_start = int(recent_dates[-es_days])
    return {"train_end": train_end, "recent_start": recent_start, "es_start": es_start,
            "outer_train": outer, "internal": outer & (label_end < es_start),
            "early_stop": outer & (dates >= es_start),
            "predict": (dates >= year * 10000 + 101) & (dates <= year * 10000 + 1231)}


def train(data: pd.DataFrame, split: dict, seed: int) -> dict:
    torch.set_num_threads(2)
    start = time.time()
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    raw = pd.to_numeric(data["future_return"], errors="coerce").to_numpy(float)
    internal, early = split["internal"] & finite, split["early_stop"] & finite
    outer, predict = split["outer_train"] & finite, split["predict"]
    inner_design = Design().fit(data.loc[internal])
    x_inner = inner_design.transform(data)
    tensor_x, tensor_z = torch.from_numpy(x_inner), torch.from_numpy(np.nan_to_num(z))
    inner_groups, valid_groups = _groups(dates, internal), _groups(dates, early)
    model, optimizer, rng = _new(x_inner.shape[1], seed, 0.1)
    history, best_ic, best_epoch = [], -np.inf, 0
    for epoch in range(1, 101):
        _run(model, optimizer, rng, tensor_x, tensor_z, inner_groups, 1, "mse")
        ic = _daily_ic(raw, valid_groups, _predict(model, x_inner, valid_groups))
        history.append({"epoch": epoch, "valid_daily_ic": ic})
        if np.isfinite(ic) and ic > best_ic:
            best_ic, best_epoch = ic, epoch
        if epoch - best_epoch >= 12:
            break
    outer_design = Design().fit(data.loc[outer])
    x_outer = outer_design.transform(data)
    model2, optimizer2, rng2 = _new(x_outer.shape[1], seed, 0.1)
    _run(model2, optimizer2, rng2, torch.from_numpy(x_outer), tensor_z,
         _groups(dates, outer), best_epoch, "mse")
    preds = _predict(model2, x_outer, _groups(dates, predict))
    keep = np.nonzero(predict)[0]
    frame = pd.DataFrame({"date": dates[keep].astype("int64"),
                          "code": data.loc[keep, "code"].to_numpy(), "score": preds[keep]})
    return {"best_epoch": int(best_epoch), "best_inner_ic": float(best_ic),
            "recent_days": int(split["outer_train"].sum()),
            "recent_signal_days": int(pd.unique(dates[split["outer_train"]]).size),
            "seconds": round(time.time() - start, 1), "predictions": frame}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(V2 / "v2_rev1_tabm_recency.json"))
    args = parser.parse_args()
    data = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    labels = data[["date", "code", "future_return"]]
    report = {"recent_days": RECENT_DAYS, "seed": SEED, "alphas": list(ALPHAS),
              "windows": {}}
    for window, (train_end, year) in WINDOWS.items():
        split = recency_split(data, train_end, year)
        result = train(data, split, SEED)
        tag = f"tabm_recent{RECENT_DAYS}_{SEED}__{window}"
        out = V2 / tag
        out.mkdir(parents=True, exist_ok=True)
        result["predictions"].to_parquet(out / "predictions.parquet", index=False)
        (out / "manifest.json").write_text(json.dumps(
            {k: v for k, v in result.items() if k != "predictions"}, indent=2))
        control = pd.read_parquet(
            S4 / "v1_tabm" / f"tabm_pc_{SEED}__{window}" / "predictions.parquet")
        control_ic = frame_ic(control.merge(labels, on=["date", "code"], how="left",
                                            validate="one_to_one"))
        recent_ic = frame_ic(result["predictions"].merge(
            labels, on=["date", "code"], how="left", validate="one_to_one"))
        parents = {"c1": as_family(pd.read_parquet(
            S4 / f"v1_c1_seedmean__{window}.parquet"), column="c1", label=f"c1_{window}"),
            "tabm": as_family(pd.read_parquet(
                S4 / f"v1_tabm_seedmean__{window}.parquet"), column="tabm",
                label=f"tabm_{window}")}
        new_q = family_q([result["predictions"]], "score")
        components = dict(parents)
        components["recent"] = new_q
        curve = {}
        for alpha in ALPHAS:
            weights = push_down_alpha({"c1": 0.5, "tabm": 0.5}, "recent", alpha)
            fused = fuse(components, weights).merge(
                labels, on=["date", "code"], how="left", validate="one_to_one")
            curve[str(alpha)] = frame_ic(fused)
        report["windows"][window] = {
            "recent_signal_days": result["recent_signal_days"],
            "best_epoch": result["best_epoch"], "best_inner_ic": result["best_inner_ic"],
            "seconds": result["seconds"],
            "control_rankic": control_ic, "recent_rankic": recent_ic,
            "delta_vs_control": recent_ic - control_ic,
            "ic_by_alpha": curve,
            "delta_vs_parent": {key: curve[key] - curve["0.0"] for key in curve},
            "artifact": str(out)}
        print(window, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in report["windows"][window].items()
                                  if k != "ic_by_alpha"}), flush=True)
    dev = {key: float(np.average([report["windows"][w]["delta_vs_parent"][key]
                                  for w in WINDOWS], weights=[244, 243]))
           for key in report["windows"]["DEV19"]["delta_vs_parent"]}
    report["dev_merged_delta_vs_parent"] = dev
    report["dev_merged_delta_vs_control"] = float(np.average(
        [report["windows"][w]["delta_vs_control"] for w in WINDOWS], weights=[244, 243]))
    report["best_alpha"] = max(dev, key=dev.get)
    report["best_dev_delta"] = dev[report["best_alpha"]]
    report["both_dev_nonneg"] = all(report["windows"][w]["delta_vs_control"] >= 0
                                    for w in WINDOWS)
    report["decision"] = ("extend to more seeds and the 2021 check"
                          if report["best_dev_delta"] >= 0.0005 and report["both_dev_nonneg"]
                          else "no extension: below the DEV gate")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    main()
