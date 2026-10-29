"""V1 anchor: seed means, C1 control check, P0/P1 selection-rule comparison, IC table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, metrics, s2r_train, s3_ranks, s6_nn_v1 as S

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
WINDOWS = ("DEV19", "DEV20", "CONF21")
YEAR = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}
SEEDS = S.SEEDS
WEIGHTS = {"c1": 0.5, "tabm": 0.5}
P1_TOL = 0.0005


def seed_mean_tabm(window: str) -> pd.DataFrame:
    merged = None
    for index, seed in enumerate(SEEDS):
        frame = pd.read_parquet(
            S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "predictions.parquet")
        frame = frame.rename(columns={"score": f"s{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(SEEDS)):
            value += s3_ranks.rank_within_day(
                block[f"s{index}"].reset_index(drop=True)).to_numpy() / len(SEEDS)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "tabm": value}))
    return pd.concat(pieces, ignore_index=True)


def c1_seed_mean(window: str) -> pd.DataFrame:
    merged = None
    for index, seed in enumerate(SEEDS):
        frame = pd.read_parquet(FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet")
        frame = frame.rename(columns={"score": f"c{index}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        value = np.zeros(len(block))
        for index in range(len(SEEDS)):
            value += s3_ranks.rank_within_day(
                block[f"c{index}"].reset_index(drop=True)).to_numpy() / len(SEEDS)
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "c1": value}))
    return pd.concat(pieces, ignore_index=True)


def ic(frame: pd.DataFrame, panel: pd.DataFrame, col: str = "score") -> float:
    sub = frame[["date", "code", col]].rename(columns={col: "score"})
    merged = sub.merge(panel[["date", "code", "future_return"]], on=["date", "code"],
                       how="left", validate="one_to_one")
    return metrics.summarize_rankic(metrics.daily_rankic(merged))["mean_rankic"]


def p1_epoch(window: str, seed: int) -> int:
    curve = pd.read_parquet(S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}"
                            / "inner_curve.parquet")
    values = curve["valid_daily_ic"].to_numpy(dtype=float)
    best = np.nanmax(values)
    near = np.nonzero(values >= best - P1_TOL)[0]
    return int(curve["epoch"].to_numpy()[int(near[0])])


def refit_at(data: pd.DataFrame, window: str, epochs: int, seed: int) -> dict:
    """Outer-only refit at a preselected epoch (P1); same reinit protocol as P0."""
    import torch  # noqa: F401  (backend evidence)
    split = cv.fold_split(data, window)
    dates = cv.to_int_date(data["date"])
    z, _, finite = s2r_train.rank_target_unnormalised(data)
    outer = split["outer_train"] & finite
    design = S.Design().fit(data.loc[outer])
    x = design.transform(data)
    tensor_x = __import__("torch").from_numpy(x)
    tensor_z = __import__("torch").from_numpy(np.nan_to_num(z))
    model, optimizer, rng = S._new_model(x.shape[1], seed)
    S._train_epochs(model, optimizer, rng, x, z, tensor_x, tensor_z,
                    S._day_groups(dates, outer), epochs)
    predict = split["predict"]
    preds = S._predict(model, x, S._day_groups(dates, predict))
    keep = np.nonzero(predict)[0]
    frame = pd.DataFrame({"date": dates[keep].astype("int64"),
                          "code": data.loc[keep, "code"].to_numpy(),
                          "score": preds[keep]})
    return {"score": frame}


def main() -> None:
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    data = panel.sort_values(["date", "code"]).reset_index(drop=True)
    report = {"tabm_seed_means": {}, "anchor": {}, "p1": {}, "c1_control": {}}
    for window in WINDOWS:
        tabm = seed_mean_tabm(window)
        c1 = c1_seed_mean(window)
        merged = tabm.merge(c1, on=["date", "code"], validate="one_to_one")
        pieces = []
        for date, block in merged.groupby("date", sort=True):
            u = (WEIGHTS["c1"] * block["c1"].reset_index(drop=True)
                 + WEIGHTS["tabm"] * block["tabm"].reset_index(drop=True))
            pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                        "score": s3_ranks.rank_within_day(u).to_numpy()}))
        anchor = pd.concat(pieces, ignore_index=True)
        anchor.to_parquet(S4 / f"v1_anchor__{window}.parquet", index=False)
        tabm.to_parquet(S4 / f"v1_tabm_seedmean__{window}.parquet", index=False)
        c1.to_parquet(S4 / f"v1_c1_seedmean__{window}.parquet", index=False)
        report["anchor"][window] = {
            "tabm_seedmean": ic(tabm, panel, "tabm"),
            "c1_seedmean": ic(c1, panel, "c1"),
            "anchor_0.5_0.5": ic(anchor, panel),
            "year": YEAR[window],
        }
        # C1 control identity check: three seed files must exist and differ
        hashes = []
        for seed in SEEDS:
            path = FUS / "models_cat" / f"C1_{seed}__{window}.cbm"
            hashes.append((seed, path.exists(),
                           __import__("hashlib").sha256(path.read_bytes()).hexdigest()[:12]))
        report["c1_control"][window] = {"seeds": hashes,
                                        "all_present": all(h[1] for h in hashes),
                                        "distinct": len({h[2] for h in hashes}) == len(SEEDS)}
    # P0 vs P1 on the main seed / three windows
    for window in WINDOWS:
        p0_path = S4 / "v1_tabm" / f"tabm_pc_20261005__{window}"
        p0 = pd.read_parquet(p0_path / "predictions.parquet")
        epoch = p1_epoch(window, 20261005)
        p0_epoch = int(json.loads((p0_path / "atom_manifest.json").read_text())["best_epoch"])
        p1 = refit_at(data, window, epoch, 20261005)["score"]
        panel_ic = panel.loc[panel["date"] // 10000 == YEAR[window]]
        report["p1"][window] = {"p0_epoch": p0_epoch,
                                "p1_epoch": epoch,
                                "p0_ic": ic(p0, panel), "p1_ic": ic(p1, panel),
                                "delta": ic(p1, panel) - ic(p0, panel)}
        print(f"P1 {window}: epoch {epoch} -> IC {report['p1'][window]['p1_ic']:.6f} "
              f"(P0 {report['p1'][window]['p0_ic']:.6f}, Δ{report['p1'][window]['delta']:+.6f})",
              flush=True)
    (S4 / "v1_anchor_analysis.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "c1_control"}, indent=2, default=str))


if __name__ == "__main__":
    main()
