"""Stage-B shared helpers: member loading, blending, IC evaluation, backtest runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest_runner, metrics, s2r_blend

B = Path("artifacts/stage2_ic_ir/b")
PRED = B / "predictions"
BT = B / "backtests"
REV = Path("artifacts/stage2_ic_ir/a_revision1")
REGISTRY = B / "experiments.jsonl"
WINDOWS = ("DEV19", "DEV20", "CONF21")

# member score sources
SOURCES = {
    "ref_corrected": {
        "lgbm_p5": REV / "predictions" / "ref_corrected_lgbm__{window}.parquet",
        "catboost": REV / "predictions" / "ref_corrected_catboost__{window}.parquet",
    },
    "bridge_realinput": {
        "lgbm_p5": Path("artifacts/stage2_ic_ir/predictions") /
                    "origmodel_realinput_members__{window}.parquet",
        "catboost": Path("artifacts/stage2_ic_ir/predictions") /
                    "origmodel_realinput_members__{window}.parquet",
    },
}


def load_members(source: str, window: str) -> pd.DataFrame:
    if source == "ref_corrected":
        lgbm = pd.read_parquet(str(SOURCES[source]["lgbm_p5"]).format(window=window))
        cat = pd.read_parquet(str(SOURCES[source]["catboost"]).format(window=window))
        frame = lgbm.merge(cat, on=["date", "code"], suffixes=("_lgbm", "_cat"),
                           validate="one_to_one")
        return frame.rename(columns={"score_lgbm": "lgbm_p5", "score_cat": "catboost"})
    frame = pd.read_parquet(str(SOURCES[source]["lgbm_p5"]).format(window=window))
    out = frame.loc[:, ["date", "code", "score_lgbm_p5", "score_catboost"]].rename(
        columns={"score_lgbm_p5": "lgbm_p5", "score_catboost": "catboost"})
    out["date"] = pd.to_numeric(out["date"]).astype("int64")
    return out


def blend_weight(source: str, window: str, cat_weight: float) -> pd.DataFrame:
    members = load_members(source, window)
    return s2r_blend.blend_frame(members, ["lgbm_p5", "catboost"],
                                 [1.0 - cat_weight, cat_weight])


def blend_seed_average(source: str, window: str, seed_frames: list[pd.DataFrame],
                       base_weight: float = 1.0) -> pd.DataFrame:
    """Average several member sets' per-day centred ranks."""
    combined = None
    for frame in seed_frames:
        ranked = s2r_blend.blend_frame(frame, ["lgbm_p5", "catboost"],
                                       [1.0 - base_weight, base_weight])
        ranked = ranked.rename(columns={"score": "score_i"})
        combined = ranked if combined is None else combined.merge(
            ranked, on=["date", "code"], validate="one_to_one")
    columns = [c for c in combined.columns if c.startswith("score_i")]
    combined["score"] = combined[columns].mean(axis=1)
    return combined.loc[:, ["date", "code", "score"]]


def ic_summary(preds: pd.DataFrame, panel: pd.DataFrame) -> dict:
    merged = preds.merge(panel.loc[:, ["date", "code", "future_return"]],
                         on=["date", "code"], how="left", validate="one_to_one")
    rankic = metrics.daily_rankic(merged)
    summary = metrics.summarize_rankic(rankic)
    return summary, rankic


def run_backtest(name: str, score_frame: pd.DataFrame, start: str, end: str) -> dict:
    out_dir = BT / name
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    scores = score_frame.loc[:, ["date", "code", "score"]].copy()
    if scores.duplicated(["date", "code"]).any():
        raise ValueError("duplicate scores")
    (out_dir / "config.json").write_text(json.dumps(
        {"start": start, "end": end, "rows": int(len(scores)),
         "data_view": "artifacts/beginning/data_zero_st",
         "contract": "configs/beginning/evaluation_contract.json"}, indent=2))
    inference = backtest_runner.score_lookup_inference(scores)
    started = time.time()
    m, tables, account = backtest_runner.run_backtest(
        data_dir="artifacts/beginning/data_zero_st", start_date=start, end_date=end,
        inference=inference, output_dir=None, async_inference=False)
    elapsed = time.time() - started
    backtest_runner.save_run(out_dir, m, tables, account)
    (out_dir / "run_summary.json").write_text(json.dumps(
        {"run": name, "seconds": round(elapsed, 1), "metrics": m}, indent=2, default=str))
    return m


def log(record: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
