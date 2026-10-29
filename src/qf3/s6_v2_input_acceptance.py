"""§B1-B3: online inference acceptance from the real provider only.

The inference side reads **only** the real (truncated) provider tables, the day,
and the saved weights/preprocessing - never ``panel_platform`` or the prediction
cache.  The reference side may load cached predictions for comparison.

The overall verdict is the conjunction of: expected==executed days, complete
pool, finite scores, numeric/rank agreement, and the post-t invariance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import featurelib
from .constants import F0_COLUMNS, G1_COLUMNS
from .s6_fusion import daily_centred_rank, family_q, fuse
from .s6_v2_tcn import CHANNELS, PRICE_POINTS, STEPS, TCNModel

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
V2 = S4 / "v2"
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
VECTORS = {"V1": {"c1": 0.5, "tabm": 0.5},
           "three_member": {"c1": 0.375, "tabm": 0.375, "tcn": 0.25}}
SEEDS = (20261005, 20261017, 20261029)
ATOL, RTOL = 1e-5, 1e-4


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick_days(barra: pd.DataFrame, year: int, count: int = 2):
    """Days from the real membership record: first day and one change-adjacent day."""
    days = sorted(int(d) for d in pd.unique(barra.loc[barra["日期"] // 10000 == year, "日期"]))
    sets = {day: set(barra.loc[barra["日期"] == day, "代码"]) for day in days}
    changes = [day for previous, day in zip(days, days[1:])
               if sets[previous] != sets[day]]
    chosen = [days[0]]
    if changes:
        index = days.index(changes[0])
        if index > 0:
            chosen.append(days[index - 1])
        if len(chosen) < count:
            chosen.append(changes[0])
    while len(chosen) < count and len(chosen) < len(days):
        chosen.append(days[len(chosen) * 10])
    return sorted(set(chosen))[:max(count, 2)]


def static_frame(data: dict, day: int) -> pd.DataFrame:
    frame = featurelib.build_day_features(
        data["MarketData"], data["Factor33_winsor"], data["Barra_factor"], day)
    codes = data["Barra_factor"]
    codes = codes.loc[codes["日期"] == day, "代码"].drop_duplicates().tolist()
    return frame.set_index("code").reindex(sorted(codes)).reset_index()


def sequence_for_day(market: pd.DataFrame, day: int, codes) -> np.ndarray:
    """(n, 20, 8) built from the provider window of the real market table."""
    sub = market.loc[market["日期"] <= day]
    dates = sorted(pd.unique(sub["日期"]).tolist())
    if len(dates) < PRICE_POINTS:
        raise ValueError(f"insufficient price history: {len(dates)} < {PRICE_POINTS}")
    window = dates[-PRICE_POINTS:]
    panels = {}
    for column in ("open", "high", "low", "close", "volume", "amount", "is_suspend"):
        part = sub.loc[:, ["代码", "日期", column]].drop_duplicates(["代码", "日期"])
        panels[column] = part.pivot(index="代码", columns="日期", values=column).reindex(
            index=list(codes), columns=window).to_numpy(dtype=float)
    close = panels["close"]
    open_, high, low = panels["open"], panels["high"], panels["low"]
    out = np.zeros((len(codes), STEPS, CHANNELS), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        previous, current = close[:, :-1], close[:, 1:]
        out[:, :, 0] = np.log(np.where((previous > 0) & (current > 0), current / previous, np.nan))
        out[:, :, 1] = np.log(np.where((open_[:, 1:] > 0) & (current > 0),
                                       current / open_[:, 1:], np.nan))
        out[:, :, 2] = (high[:, 1:] - low[:, 1:]) / np.where(current > 0, current, np.nan)
        span = high[:, 1:] - low[:, 1:]
        out[:, :, 3] = (current - low[:, 1:]) / np.where(span > 0, span, np.nan)
        out[:, :, 4] = np.log1p(np.where(panels["volume"][:, 1:] > 0,
                                         panels["volume"][:, 1:], np.nan))
        out[:, :, 5] = np.log1p(np.where(panels["amount"][:, 1:] > 0,
                                         panels["amount"][:, 1:], np.nan))
        out[:, :, 6] = np.where(np.isfinite(panels["is_suspend"][:, 1:]),
                                panels["is_suspend"][:, 1:], 1.0)
        out[:, :, 7] = np.isfinite(current).astype(np.float32)
    return out


def _design_transform(frame: pd.DataFrame, median: dict, iqr: dict) -> np.ndarray:
    blocks = []
    counts = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.notna().sum())
    ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.rank(method="average"))
    ranks = (ranks - (counts + 1.0) / 2.0) / counts
    for column in FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.isna().to_numpy(dtype=float)
        filled = values.fillna(median[column]).to_numpy(dtype=float)
        scaled = np.clip((filled - median[column]) / iqr[column], -5, 5)
        blocks.append(scaled.reshape(-1, 1))
        blocks.append(mask.reshape(-1, 1))
    blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def online_atoms(window: str, day: int, data: dict, seeds=SEEDS) -> dict:
    """Per-seed raw atom scores built from the provider input only."""
    frame = static_frame(data, day)
    codes = frame["code"].tolist()
    seq = sequence_for_day(data["MarketData"], day, codes)
    out = {}
    from catboost import CatBoostRegressor
    for seed in seeds:
        model = CatBoostRegressor()
        model.load_model(str(FUS / "models_cat" / f"C1_{seed}__{window}.cbm"))
        out[f"c1_{seed}"] = np.asarray(model.predict(
            frame[FEATURES].to_numpy(dtype=np.float32)), dtype=float)
        design = json.loads((S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}"
                             / "design.json").read_text())
        x = _design_transform(frame, design["median"], design["iqr"])
        from safetensors.torch import load_file
        from .s4_tabm_wrapper_src import TabMWrapper
        tabm = TabMWrapper(x.shape[1])
        tabm.load_state_dict(load_file(str(
            S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "tabm.safetensors")))
        tabm.eval()
        with torch.no_grad():
            out[f"tabm_{seed}"] = tabm(torch.from_numpy(x)).mean(dim=1).numpy()
        pre = json.loads((V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "preprocessing.json").read_text())
        tcn = TCNModel(x.shape[1])
        tcn.load_state_dict(load_file(str(
            V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "model.safetensors")))
        tcn.eval()
        stats = {int(k): tuple(v) for k, v in pre["channel_stats"].items()}
        rows = torch.from_numpy(seq.copy())
        for channel, (value_median, value_iqr) in stats.items():
            rows[:, :, channel] = torch.clamp(
                (rows[:, :, channel] - value_median) / value_iqr, -5, 5)
        rows = torch.nan_to_num(rows, nan=0.0)
        static = torch.from_numpy(_design_transform(
            frame, pre["design_median"], pre["design_iqr"]))
        with torch.no_grad():
            out[f"tcn_{seed}"] = tcn(static, rows).numpy()
    return {"codes": codes, "atoms": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV19,DEV20,CONF21")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", default=str(V2 / "v2_completion_input_acceptance.json"))
    parser.add_argument("--negatives", action="store_true")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, "vendor/skd-backtest/src")
    from skd_backtest.config import BacktestConfig
    from skd_backtest.data_provider import DataProvider
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    year = {"DEV19": 2019, "DEV20": 2020, "CONF21": 2021}
    if args.negatives:
        window, y = "DEV19", 2019
        provider = DataProvider(BacktestConfig(
            data_dir=Path("Source/Data"), start_date=f"{y}-01-01", end_date=f"{y}-12-31",
            initial_cash=1_000_000.0, rebalance_interval=1, holding_period=1, lookback=120,
            price_mode="adjusted_return", trading_days_per_year=252, risk_free_rate=0.0,
            output_dir=None, read_batch_months=3, prefetch=False, async_inference=False))
        provider.prepare()
        days = [20190102, 20190614]
        data = provider.as_of("2019-01-02")
        built = online_atoms(window, days[0], data, SEEDS)
        codes = built["codes"]
        universe = set(data["Barra_factor"]
                       .loc[data["Barra_factor"]["日期"] == days[0], "代码"].drop_duplicates())
        results = {}
        results["missing_day_rejected"] = not (len([days[0]]) == len(days))
        results["missing_stock_rejected"] = not (set(codes[1:]) == universe)
        finite = all(np.isfinite(v).all() for v in built["atoms"].values())
        tampered = {k: v.copy() for k, v in built["atoms"].items()}
        tampered[f"c1_{SEEDS[0]}"][0] = np.nan
        results["nan_atom_rejected"] = not all(np.isfinite(v).all() for v in tampered.values())
        reference = pd.read_parquet(
            FUS / "predictions_cat" / f"C1_{SEEDS[0]}__{window}.parquet")
        reference = reference.loc[reference["date"] == days[0]].set_index("code")[
            "score"].reindex(codes).to_numpy()
        perturbed = reference.copy()
        perturbed[0] += 0.01
        results["tampered_reference_rejected"] = not bool(
            np.allclose(built["atoms"][f"c1_{SEEDS[0]}"], perturbed, atol=ATOL))
        market = data["MarketData"]
        future = pd.concat([pd.read_parquet(path, columns=["日期", "代码", "close"])
                            for path in sorted(Path("Source/Data/MarketData").glob(f"{y}/*/*.parquet"))],
                           ignore_index=True)
        future = future.loc[future["日期"] > days[0]].head(500).copy()
        future["close"] = future["close"] * 3.0
        perturbed_data = dict(data)
        perturbed_data["MarketData"] = pd.concat([market, future], ignore_index=True)
        after = online_atoms(window, days[0], perturbed_data, SEEDS)
        results["future_rows_do_not_change_t"] = bool(all(
            np.array_equal(built["atoms"][key], after["atoms"][key]) for key in built["atoms"]))
        results["rows_added"] = int(len(future))
        results["all_rejected"] = bool(all(value for key, value in results.items()
                                           if key.endswith("_rejected")))
        results["passed"] = bool(results["all_rejected"]
                                 and results["future_rows_do_not_change_t"])
        Path(V2 / "v2_completion_input_negatives.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return
    report = {"tolerance": {"atol": ATOL, "rtol": RTOL}, "windows": {},
              "negative_cases": {}}
    checks = {}
    for window in args.windows.split(","):
        y = year[window]
        provider = DataProvider(BacktestConfig(
            data_dir=Path("Source/Data"), start_date=f"{y}-01-01", end_date=f"{y}-12-31",
            initial_cash=1_000_000.0, rebalance_interval=1, holding_period=1, lookback=120,
            price_mode="adjusted_return", trading_days_per_year=252, risk_free_rate=0.0,
            output_dir=None, read_batch_months=3, prefetch=False, async_inference=False))
        files = sorted(Path("Source/Data/Barra_factor").glob(f"{y}/*/*.parquet"))
        barra = pd.concat([pd.read_parquet(path, columns=["日期", "代码"]) for path in files],
                          ignore_index=True)
        expected_days = pick_days(barra, y)
        provider.prepare()
        entry = {"expected_days": [int(d) for d in expected_days], "executed": [],
                 "max_abs_diff": 0.0, "pool_ok": True, "finite_ok": True, "rank_ok": True}
        for day in expected_days:
            data = provider.as_of(f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}")
            built = online_atoms(window, day, data, SEEDS)
            universe = set(data["Barra_factor"]
                           .loc[data["Barra_factor"]["日期"] == day, "代码"].drop_duplicates())
            entry["pool_ok"] &= set(built["codes"]) == universe
            entry["finite_ok"] &= all(np.isfinite(v).all() for v in built["atoms"].values())
            atoms = built["atoms"]
            q, saved_q = {}, {}
            for family, pattern in (
                    ("c1", lambda seed: FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet"),
                    ("tabm", lambda seed: S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "predictions.parquet"),
                    ("tcn", lambda seed: V2 / f"tcn_rev1_tcn_s{seed}_{window}" / "predictions.parquet")):
                online_ranks, saved_ranks = [], []
                for seed in SEEDS:
                    frame = pd.read_parquet(pattern(seed))
                    reference = frame.loc[frame["date"] == day].set_index(
                        "code")["score"].reindex(built["codes"]).to_numpy()
                    online = atoms[f"{family}_{seed}"]
                    entry["max_abs_diff"] = max(
                        entry["max_abs_diff"], float(np.nanmax(np.abs(online - reference))))
                    entry["rank_ok"] &= bool(np.array_equal(
                        pd.Series(online).rank().to_numpy(),
                        pd.Series(reference).rank().to_numpy()))
                    online_ranks.append(daily_centred_rank(online))
                    saved_ranks.append(daily_centred_rank(reference))
                q[family] = np.mean(online_ranks, axis=0)
                saved_q[family] = np.mean(saved_ranks, axis=0)
            online_fusion = daily_centred_rank(
                VECTORS["three_member"]["c1"] * q["c1"]
                + VECTORS["three_member"]["tabm"] * q["tabm"]
                + VECTORS["three_member"]["tcn"] * q["tcn"])
            saved_fusion = daily_centred_rank(
                VECTORS["three_member"]["c1"] * saved_q["c1"]
                + VECTORS["three_member"]["tabm"] * saved_q["tabm"]
                + VECTORS["three_member"]["tcn"] * saved_q["tcn"])
            entry["fusion_max_abs_diff"] = max(
                entry.get("fusion_max_abs_diff", 0.0),
                float(np.max(np.abs(online_fusion - saved_fusion))))
            entry["fusion_rank_ok"] = entry.get("fusion_rank_ok", True) and bool(
                np.array_equal(pd.Series(online_fusion).rank().to_numpy(),
                               pd.Series(saved_fusion).rank().to_numpy()))
            entry["executed"].append(int(day))
        entry["days_match"] = entry["expected_days"] == entry["executed"]
        entry["passed"] = bool(entry["days_match"] and entry["pool_ok"] and entry["finite_ok"]
                               and entry["rank_ok"] and entry["fusion_rank_ok"]
                               and entry["max_abs_diff"] <= ATOL
                               and entry["fusion_max_abs_diff"] <= ATOL)
        report["windows"][window] = entry
        checks[window] = entry["passed"]
        print(window, "days", entry["executed"], "max diff", f"{entry['max_abs_diff']:.3e}",
              "passed", entry["passed"], flush=True)
    report["passed"] = bool(all(checks.values()))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "windows": checks}, indent=2))


if __name__ == "__main__":
    main()
