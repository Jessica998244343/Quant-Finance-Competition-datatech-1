"""V1 §7.2 step 5: year fixture through the real package entry vs cached atoms."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio, featurelib, metrics, s3_ranks

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
S3 = Path("artifacts/stage3_ic_first")
PKG_SRC = Path("releases/stage3_fusion/build/TREE_v1")
FIX = S4 / "v1_fixtures"
SEEDS = (20261005, 20261017, 20261029)
WEIGHTS = {"c1": 0.5, "tabm": 0.5}

SCORER = '''"""V1 year fixture scorer: C1 seed-mean + TabM, frozen graph."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


class Scorer:
    def __init__(self, model_dir: str):
        import torch
        from catboost import CatBoostRegressor
        from safetensors.torch import load_file

        from . import alpha_tabminput, alpha_tabm_wrapper

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.weights = dict(meta["weights"])
        self.cat = []
        for name in meta["members"]["c1"]:
            m = CatBoostRegressor()
            m.load_model(str(self.dir / name))
            self.cat.append(m)
        config = json.loads((self.dir / "tabm_config.json").read_text())
        self.tabm = alpha_tabm_wrapper.TabMWrapper(**config)
        self.tabm.load_state_dict(load_file(str(self.dir / "tabm.safetensors")))
        self.tabm.eval()
        self.design = json.loads((self.dir / "tabm_design.json").read_text())
        self._tabminput = alpha_tabminput
        self._torch = torch

    def atom_scores(self, frame: pd.DataFrame) -> dict:
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        c1 = np.zeros(len(frame))
        for model in self.cat:
            c1 += _rank(model.predict(matrix)) / len(self.cat)
        with self._torch.no_grad():
            x = self._torch.from_numpy(self._tabminput.transform(frame, self.design))
            tabm_raw = self.tabm(x).mean(dim=1).numpy()
        # 冻结图：每个家族先取当日秩，再按家族权重相加
        return {"c1": c1, "tabm": _rank(tabm_raw), "tabm_raw": tabm_raw}

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        atoms = self.atom_scores(frame)
        fused = self.weights["c1"] * atoms["c1"] + self.weights["tabm"] * atoms["tabm"]
        return _rank(fused)
'''


def build_fixture(window: str, year: int) -> Path:
    """Assemble a package-shaped year fixture (not a submission) from year atoms."""
    root = FIX / window / "submission"
    if root.exists():
        shutil.rmtree(root)
    (root / "model").mkdir(parents=True)
    fallback = Path("releases/stage3_fusion/build/FINAL_v1")
    for name in ("__init__.py", "alpha_constants.py", "alpha_features.py",
                 "alpha_tabminput.py", "alpha_tabm_wrapper.py", "tabm_official.py",
                 "LICENSE.tabm", "alpha_dataio.py", "alpha_labels.py", "alpha_panel.py",
                 "alpha_cv.py"):
        source = (PKG_SRC / name) if (PKG_SRC / name).exists() else (fallback / name)
        if not source.exists():
            raise FileNotFoundError(f"missing package module {name}")
        shutil.copy(source, root / name)
    (root / "alpha_models.py").write_text(SCORER)
    inference = (PKG_SRC / "inference.py").read_text().replace("TREE_v1", f"V1_FIXTURE_{window}")
    inference = inference.replace("from .alpha_models import Scorer",
                                  "from .alpha_models import Scorer")
    (root / "inference.py").write_text(inference)
    members = {"c1": [], "tabm": "tabm.safetensors"}
    for seed in SEEDS:
        src = FUS / "models_cat" / f"C1_{seed}__{window}.cbm"
        if not src.exists():
            raise FileNotFoundError(f"missing year C1 atom: {src}")
        shutil.copy(src, root / "model" / src.name)
        members["c1"].append(src.name)
    tabm_dir = S4 / "v1_tabm" / f"tabm_pc_{SEEDS[0]}__{window}"
    for name in ("tabm.safetensors", "tabm_config.json", "design.json"):
        shutil.copy(tabm_dir / name, root / "model" / name)
    shutil.copy(tabm_dir / "design.json", root / "model" / "tabm_design.json")
    from .constants import F0_COLUMNS, G1_COLUMNS
    meta = {"version": f"V1_FIXTURE_{window}", "year": year,
            "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": WEIGHTS, "members": members,
            "fusion_graph": "atom rank -> seed mean -> family weight -> final rank",
            "note": "year fixture for entry verification only; not a submission package"}
    (root / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    return root


def load_fixture(root: Path):
    for name in list(sys.modules):
        if name.startswith("_v1_fix"):
            del sys.modules[name]
    pkg = "_v1_fix"
    spec = importlib.util.spec_from_file_location(
        pkg, root / "inference.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def reference_fusion(window: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Cached atoms -> frozen-graph fusion (the reference the fixture must reproduce)."""
    cat = None
    for seed in SEEDS:
        frame = pd.read_parquet(FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet")
        frame = frame.rename(columns={"score": f"c{seed}"})
        cat = frame if cat is None else cat.merge(frame, on=["date", "code"],
                                                  validate="one_to_one")
    tabm = pd.read_parquet(S4 / "v1_tabm" / f"tabm_pc_{SEEDS[0]}__{window}"
                           / "predictions.parquet").rename(columns={"score": "tabm"})
    merged = cat.merge(tabm, on=["date", "code"], validate="one_to_one")
    pieces = []
    for date, block in merged.groupby("date", sort=True):
        q_c1 = np.zeros(len(block))
        for seed in SEEDS:
            q_c1 += s3_ranks.rank_within_day(
                block[f"c{seed}"].reset_index(drop=True)).to_numpy() / len(SEEDS)
        q_tabm = s3_ranks.rank_within_day(block["tabm"].reset_index(drop=True)).to_numpy()
        u = WEIGHTS["c1"] * q_c1 + WEIGHTS["tabm"] * q_tabm
        pieces.append(pd.DataFrame({"date": date, "code": block["code"].to_numpy(),
                                    "score": s3_ranks.rank_within_day(
                                        pd.Series(u)).to_numpy()}))
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="DEV19")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--representative-days", default="")
    parser.add_argument("--full-year", action="store_true", default=True)
    args = parser.parse_args()
    root = build_fixture(args.window, args.year)
    module = load_fixture(root)
    model = module.InferenceModel(str(root / "model"))
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    tables = dataio.load_all("Source/Data")
    calendar = sorted(pd.unique(tables["MarketData"]["日期"]).tolist())

    # 真实 provider：与平台 DataProvider 相同的每日成分过滤语义
    from skd_backtest.config import BacktestConfig
    from skd_backtest.data_provider import DataProvider
    year_days = [d for d in calendar if d // 10000 == args.year]
    start = f"{args.year}-01-01"
    provider = DataProvider(BacktestConfig(
        data_dir=Path("Source/Data"), start_date=start,
        end_date=f"{args.year}-12-31", initial_cash=1_000_000.0,
        rebalance_interval=1, holding_period=1, lookback=120,
        price_mode="adjusted_return", trading_days_per_year=252,
        risk_free_rate=0.0, output_dir=None, read_batch_months=3,
        prefetch=False, async_inference=False))
    provider.prepare()

    def window_data(day: int) -> dict:
        return provider.as_of(f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}")

    reference = reference_fusion(args.window, panel)
    report = {"window": args.window, "fixture": str(root), "checks": {}}

    # 参考：代表性日与全年合并为升序序列，只前向查询一次
    panel_days = sorted(int(d) for d in
                        panel.loc[panel["date"] // 10000 == args.year, "date"].unique())
    if args.representative_days:
        days = [int(d) for d in args.representative_days.split(",")]
    else:  # 该年首个、中位、末个交易日（均来自当年合法信号日）
        days = [panel_days[0], panel_days[len(panel_days) // 2], panel_days[-1]]
    query_days = sorted(set(days) | set(panel_days)) if args.full_year else sorted(set(days))
    outs, universes = {}, {}
    for day in query_days:
        as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
        data = window_data(day)
        outs[day] = model.predict(as_of, data)
        barra = data["Barra_factor"]
        universes[day] = set(barra.loc[barra["日期"] == day, "代码"].drop_duplicates())
    rep = {}
    for day in days:
        out = outs[day]
        ref = reference.loc[reference["date"] == day].set_index("code")
        merged = out.set_index("code").join(ref, how="outer", rsuffix="_ref",
                                            lsuffix="_fix")
        rep[f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"] = {
            "pool_exact": bool(set(out["code"]) == universes[day]),
            "rows": int(len(out)),
            "finite": bool(np.isfinite(out["score"]).all()),
            "max_abs_score_diff": float(np.nanmax(np.abs(
                merged["score_fix"] - merged["score_ref"]))),
            "rank_identical": bool(np.array_equal(
                out.set_index("code")["score"].rank(method="average").reindex(
                    ref.index).to_numpy(),
                ref["score"].rank(method="average").to_numpy())),
        }
    report["checks"]["representative_days"] = rep

    if args.full_year:
        rows = []
        for day in panel_days:
            out = outs[day]
            ref = reference.loc[reference["date"] == day]
            merged = out.merge(ref, on="code", suffixes=("_fix", "_ref"))
            rows.append({
                "date": day,
                "max_abs_diff": float(np.max(np.abs(merged["score_fix"]
                                                    - merged["score_ref"]))),
                "rank_identical": bool(np.array_equal(
                    merged["score_fix"].rank(method="average").to_numpy(),
                    merged["score_ref"].rank(method="average").to_numpy())),
            })
        frame = pd.DataFrame(rows)
        report["checks"]["full_year"] = {
            "days": int(len(frame)),
            "max_abs_diff_overall": float(frame["max_abs_diff"].max()),
            "days_all_rank_identical": bool(frame["rank_identical"].all()),
        }

    # cached vs fixture daily-IC comparison, with a real future-perturbation test
    day = query_days[-1]
    as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
    fresh = DataProvider(BacktestConfig(
        data_dir=Path("Source/Data"), start_date=start, end_date=as_of,
        initial_cash=1_000_000.0, rebalance_interval=1, holding_period=1,
        lookback=120, price_mode="adjusted_return", trading_days_per_year=252,
        risk_free_rate=0.0, output_dir=None, read_batch_months=3,
        prefetch=False, async_inference=False))
    fresh.prepare()
    data = fresh.as_of(as_of)
    base = model.predict(as_of, data)
    perturbed = {k: v.copy() for k, v in data.items()}
    future = tables["MarketData"].loc[tables["MarketData"]["日期"] > day].head(500).copy()
    future["close"] = future["close"] * 3.0
    perturbed["MarketData"] = pd.concat([perturbed["MarketData"], future],
                                        ignore_index=True)
    after = model.predict(as_of, perturbed)
    report["checks"]["true_future_perturbation"] = {
        "identical": bool(np.array_equal(base["score"].to_numpy(),
                                         after["score"].to_numpy())),
        "rows_added": int(len(future)),
    }
    (S4 / f"v1_fixture_{args.window}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
