"""S1 evaluation: seed ensemble, reference comparison, stability control, backtests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_runner, cv, metrics, s3_ranks, s3_s1, s2r_train, trainers

S3 = Path("artifacts/stage3_ic_first")
PRED = S3 / "predictions"
MODELS = S3 / "models"
BT = S3 / "backtests"
WIN = ("DEV19", "DEV20", "CONF21")
SEEDS = (20261005, 20261017, 20261029)


def seed_ensemble() -> dict:
    """Per-seed per-day fused ranks are already single models; average the three."""
    out = {}
    for window in WIN:
        frames = [pd.read_parquet(PRED / f"L3_L0_{seed}__{window}.parquet")
                  for seed in SEEDS]
        merged = frames[0].rename(columns={"score": "s0"})
        for index, frame in enumerate(frames[1:], start=1):
            merged = merged.merge(frame.rename(columns={"score": f"s{index}"}),
                                  on=["date", "code"], validate="one_to_one")
        pieces = []
        for date, block in merged.groupby("date", sort=True):
            value = np.zeros(len(block))
            for index in range(len(SEEDS)):
                value += s3_ranks.rank_within_day(
                    block[f"s{index}"].reset_index(drop=True)).to_numpy() / len(SEEDS)
            pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                        "score": value}))
        ensemble = pd.concat(pieces, ignore_index=True)
        ensemble.to_parquet(PRED / f"MAIN_SEED_ENSEMBLE__{window}.parquet", index=False)
        out[window] = ensemble
    return out


def ic_table(panel: pd.DataFrame, sources: dict) -> dict:
    out = {}
    for label, mapping in sources.items():
        for windows, tag in ((("DEV19", "DEV20"), "dev"), (("CONF21",), "conf21")):
            if any(not Path(mapping[w]).exists() for w in windows):
                continue
            frames = [pd.read_parquet(mapping[w]) for w in windows]
            merged = pd.concat(frames, ignore_index=True).merge(
                panel[["date", "code", "future_return"]], on=["date", "code"],
                how="left", validate="one_to_one")
            summary = metrics.summarize_rankic(metrics.daily_rankic(merged))
            out[f"{label}_{tag}"] = summary["mean_rankic"]
    return out


def stability_control(panel: pd.DataFrame, window: str, name: str = "L3_stability") -> dict:
    """Two consecutive 63-day inner validation blocks; median round rule."""
    split = cv.fold_split(panel, window)
    dates = cv.to_int_date(panel["date"])
    finite = panel["future_return"].notna().to_numpy()
    train_dates = np.sort(pd.unique(dates[split["outer_train"] & finite]))
    if len(train_dates) < 200:
        return {}
    seg2_end = len(train_dates)
    seg2_start = seg2_end - 63
    seg1_start = seg2_start - 63
    segments = [(train_dates[seg1_start], train_dates[seg2_start - 1]),
                (train_dates[seg2_start], train_dates[seg2_end - 1])]
    overrides = dict(trainers.CAPACITY_GRID["P5"])
    overrides.update(s3_s1.CAPACITY["L3"])
    rounds = []
    for index, (start, end) in enumerate(segments):
        custom = {
            "train_end": split["train_end"],
            "es_start": int(start),
            "inner_end": int(end),
            "internal": (dates < start) & finite,
            "early_stop": (dates >= start) & (dates <= end) & finite,
            "outer_train": split["outer_train"],
            "predict": split["predict"],
        }
        result = s2r_train.train_lgbm_fold_local(panel, s3_s1.BASE, custom,
                                                 overrides, seed=20261005, threads=8)
        rounds.append(int(result["rounds"]))
    median_rounds = int(np.median(rounds))
    # refit on the outer window with the median round count
    dates_all = cv.to_int_date(panel["date"])
    z, raw_w, finite_all = s2r_train.rank_target_unnormalised(panel)
    x = panel[s3_s1.BASE].to_numpy(dtype=np.float32)
    outer = split["outer_train"] & finite_all
    params = trainers.lgb_params(overrides, 20261005, 8)
    params.pop("_num_rounds")
    import lightgbm as lgb
    dataset = lgb.Dataset(x[outer], label=z[outer], weight=raw_w[outer] / np.nanmean(raw_w[outer]),
                          feature_name=s3_s1.BASE)
    booster = lgb.train(params, dataset, num_boost_round=median_rounds)
    predict = split["predict"]
    preds = booster.predict(x[predict], num_iteration=median_rounds)
    frame = pd.DataFrame({"date": dates_all[predict].astype("int64"),
                          "code": panel.loc[predict, "code"].to_numpy(),
                          "score": preds})
    frame.to_parquet(PRED / f"{name}__{window}.parquet", index=False)
    booster.save_model(str(MODELS / f"{name}__{window}.txt"))
    merged = frame.merge(panel[["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    summary = metrics.summarize_rankic(metrics.daily_rankic(merged))
    record = {"window": window, "segment_rounds": rounds,
              "median_rounds": median_rounds,
              "mean_rankic": summary["mean_rankic"]}
    s3_s1.log({"stage": "S1-stability", **record})
    return record


def run_backtest(name: str, frames: list[pd.DataFrame], start: str, end: str) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["date", "code"]).any():
        raise ValueError("duplicate scores")
    (out_dir / "config.json").write_text(json.dumps(
        {"start": start, "end": end, "rows": int(len(scores)),
         "data_view": "artifacts/beginning/data_zero_st"}, indent=2))
    inference = backtest_runner.score_lookup_inference(scores)
    m, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=start, end_date=end,
        inference=inference, output_dir=None, async_inference=False)
    backtest_runner.save_run(out_dir, m, tables, account)
    (out_dir / "run_summary.json").write_text(json.dumps(
        {"run": name, "metrics": m}, indent=2, default=str))
    print(json.dumps({"run": name, "IC": m["mean_rankic"], "IR": m["information_ratio"],
                      "excess": m["annualized_excess_return"]}, default=str))
    return m


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(S3 / "panel_ext.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    ensemble = seed_ensemble()
    sources = {
        "R_SINGLE_L0P5": {w: Path("artifacts/stage2_ic_ir/a_revision1/predictions") /
                             f"ref_corrected_lgbm__{w}.parquet" for w in WIN},
        "R_WORK_cat0.5": {w: Path("artifacts/stage2_ic_ir/b/predictions") /
                             f"ref_corrected_cat0.5__{w}.parquet" for w in WIN},
        "R_DEPLOY": {w: Path("artifacts/stage2_ic_ir/a_revision1/predictions") /
                        f"origmodel_realinput__{w}.parquet" for w in WIN},
        "MAIN_SINGLE_L3": {w: PRED / f"L3_L0_20261005__{w}.parquet" for w in WIN},
        "MAIN_SEED_ENSEMBLE": {w: PRED / f"MAIN_SEED_ENSEMBLE__{w}.parquet" for w in WIN},
        "L1_ranked": {w: PRED / f"L1_p5_20261005__{w}.parquet" for w in WIN},
        "G_sep_L3": {w: PRED / f"G_sep_L0_20261005__{w}.parquet" for w in WIN},
        "G_path_L3": {w: PRED / f"G_path_L0_20261005__{w}.parquet" for w in WIN},
    }
    report = {"ic": ic_table(panel, sources), "stability": {}}
    for window in ("DEV19", "DEV20"):
        report["stability"][window] = stability_control(panel, window)
    report["backtests"] = {}
    report["backtests"]["MAIN_SINGLE_L3"] = run_backtest(
        "s3_bt_main_single_dev", [pd.read_parquet(PRED / f"L3_L0_20261005__{w}.parquet")
                                  for w in ("DEV19", "DEV20")],
        "2019-01-02", "2020-12-31")
    report["backtests"]["MAIN_SINGLE_L3_conf21"] = run_backtest(
        "s3_bt_main_single_conf21", [pd.read_parquet(PRED / "L3_L0_20261005__CONF21.parquet")],
        "2021-01-04", "2021-12-31")
    report["backtests"]["MAIN_SEED_ENSEMBLE"] = run_backtest(
        "s3_bt_main_seed_ensemble_dev",
        [ensemble[w] for w in ("DEV19", "DEV20")], "2019-01-02", "2020-12-31")
    report["backtests"]["MAIN_SEED_ENSEMBLE_conf21"] = run_backtest(
        "s3_bt_main_seed_ensemble_conf21", [ensemble["CONF21"]],
        "2021-01-04", "2021-12-31")
    (S3 / "s1_evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["ic"], indent=2))
    print(json.dumps(report["stability"], indent=2))


if __name__ == "__main__":
    main()
