"""§4.1: six-member NumPy annual entry check (C1 x3 + TabM x3) per window.

Builds a package-shaped fixture from the year's three-seed weights, loads the
*shipped* package scorer against it, drives it with the real ``DataProvider``
over the whole year and compares, per atom and per day, against the saved annual
predictions and the frozen anchor.  Expected dates come from the independent
`panel_platform` calendar, not from the predictions being checked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
BUILD = Path("releases/stage4_verified_ic/build/V1_pkg")
FIXTURES = Path("/tmp/v1fixtures")
SEEDS = (20261005, 20261017, 20261029)
WEIGHTS = {"c1": 0.5, "tabm": 0.5}
FUSION_GRAPH = "atom rank -> seed mean -> family weight -> final rank"
# Pre-registered tolerance for the TabM atom raw values: the shipped NumPy static
# forward vs the saved PyTorch forward (V0 measured <=1.19e-07; V0 atol=1e-5).
TABM_ATOL = 1e-5
WINDOWS = {
    "DEV19": {"train_end": 20181231, "year": 2019},
    "DEV20": {"train_end": 20191231, "year": 2020},
    "CONF21": {"train_end": 20201231, "year": 2021},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_npz(window: str, seed: int, out: Path) -> None:
    from safetensors.torch import load_file

    source = S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "tabm.safetensors"
    sd = load_file(str(source))
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
    np.savez(out, **{name: np.ascontiguousarray(value, dtype=np.float32)
                     for name, value in arrays.items()})


def build_fixture(window: str) -> Path:
    from .constants import F0_COLUMNS, G1_COLUMNS

    root = FIXTURES / window
    if root.exists():
        shutil.rmtree(root)
    (root / "model").mkdir(parents=True)
    (root / "__init__.py").write_text('"""annual fixture (not a submission)"""\n')
    for name in ("alpha_models.py", "alpha_tabminput.py", "alpha_tabm_static.py",
                 "alpha_features.py", "alpha_constants.py", "inference.py"):
        shutil.copy(BUILD / name, root / name)
    members = {"c1": [], "tabm": []}
    for seed in SEEDS:
        cat = root / "model" / f"catboost_c1_seed{seed}.cbm"
        shutil.copy(FUS / "models_cat" / f"C1_{seed}__{window}.cbm", cat)
        members["c1"].append(cat.name)
        npz = root / "model" / f"tabm_seed{seed}.npz"
        export_npz(window, seed, npz)
        members["tabm"].append(npz.name)
    tabm_dir = S4 / "v1_tabm" / f"tabm_pc_{SEEDS[0]}__{window}"
    shutil.copy(tabm_dir / "design.json", root / "model" / "tabm_design.json")
    shutil.copy(tabm_dir / "tabm_config.json", root / "model" / "tabm_config.json")
    spec = WINDOWS[window]
    meta = {"version": f"V1_ANNUAL_FIXTURE_{window}",
            "train_label_cutoff": spec["train_end"],
            "validation_year": spec["year"],
            "preprocessing_fit_cutoff": spec["train_end"],
            "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": WEIGHTS, "members": members, "fusion_graph": FUSION_GRAPH,
            "note": "annual entry fixture; not a submission package"}
    (root / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    return root


def load_scorer(root: Path, alias: str):
    for name in list(sys.modules):
        if name.startswith(alias):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        alias, root / "inference.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return (importlib.import_module(f"{alias}.alpha_models").Scorer,
            importlib.import_module(f"{alias}.alpha_features"))


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
    parser.add_argument("--windows", default="DEV19,DEV20")
    parser.add_argument("--out-dir", default="artifacts/stage4_verified_ic")
    args = parser.parse_args()
    sys.path.insert(0, "vendor/skd-backtest/src")
    from skd_backtest.config import BacktestConfig
    from skd_backtest.data_provider import DataProvider
    from . import dataio  # noqa: F401  (kept for parity with the fixture toolchain)

    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    labels = panel[["date", "code", "future_return"]]
    out_dir = Path(args.out_dir)
    for window in args.windows.split(","):
        spec = WINDOWS[window]
        year = spec["year"]
        root = build_fixture(window)
        Scorer, alpha_features = load_scorer(root, f"v1_fix_{window.lower()}")
        scorer = Scorer(str(root / "model"))
        provider = DataProvider(BacktestConfig(
            data_dir=Path("Source/Data"), start_date=f"{year}-01-01",
            end_date=f"{year}-12-31", initial_cash=1_000_000.0,
            rebalance_interval=1, holding_period=1, lookback=120,
            price_mode="adjusted_return", trading_days_per_year=252,
            risk_free_rate=0.0, output_dir=None, read_batch_months=3,
            prefetch=False, async_inference=False))
        provider.prepare()
        expected_days = sorted(int(day) for day in
                               panel.loc[panel["date"] // 10000 == year, "date"].unique())
        cached_c1 = {seed: pd.read_parquet(
            FUS / "predictions_cat" / f"C1_{seed}__{window}.parquet")
            for seed in SEEDS}
        cached_tabm = {seed: pd.read_parquet(
            S4 / "v1_tabm" / f"tabm_pc_{seed}__{window}" / "predictions.parquet")
            for seed in SEEDS}
        anchor = pd.read_parquet(S4 / f"v1_anchor__{window}.parquet")

        raw, atoms, fused = [], [], []
        pool_ok = dup_ok = finite_ok = True
        atom_diff = {f"{family}_{seed}": 0.0 for family in ("c1", "tabm") for seed in SEEDS}
        for day in expected_days:
            as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
            data = provider.as_of(as_of)
            barra = data["Barra_factor"]
            universe = set(barra.loc[barra["日期"] == day, "代码"].drop_duplicates())
            frame = alpha_features.build_day_features(
                data["MarketData"], data["Factor33_winsor"], data["Barra_factor"], day)
            frame = frame.set_index("code").reindex(sorted(universe)).reset_index()
            members = scorer.member_raw(frame)
            pool_ok &= set(frame["code"]) == universe
            dup_ok &= not frame["code"].duplicated().any()
            finite_ok &= all(np.isfinite(value).all() for value in members.values())
            for family in ("c1", "tabm"):
                for seed in SEEDS:
                    key = f"{family}_{seed}"
                    if family == "c1":
                        cached = cached_c1[seed]
                    else:
                        cached = cached_tabm[seed]
                    reference = cached.loc[cached["date"] == day].set_index("code")["score"]
                    value = pd.Series(members[key], index=frame["code"].to_numpy())
                    aligned = value.reindex(reference.index).to_numpy()
                    atom_diff[key] = max(atom_diff[key], float(
                        np.max(np.abs(aligned - reference.to_numpy()))))
            day_atoms = scorer.atom_scores(frame)
            day_score = scorer.score(frame)
            atoms.append(pd.DataFrame({"date": day, "code": frame["code"],
                                       "c1": day_atoms["c1"], "tabm": day_atoms["tabm"]}))
            fused.append(pd.DataFrame({"date": day, "code": frame["code"], "score": day_score}))

        fusion = pd.concat(fused, ignore_index=True)
        merged = fusion.merge(anchor.rename(columns={"score": "anchor"}),
                              on=["date", "code"], how="outer", validate="one_to_one")
        keys_match = bool(len(merged) == len(anchor) == len(fusion)
                          and merged["score"].notna().all() and merged["anchor"].notna().all())
        max_diff = float(np.nanmax(np.abs(merged["score"] - merged["anchor"]))) \
            if keys_match else None
        ranks_ok = bool(all(np.array_equal(
            block["score"].rank().to_numpy(),
            block["anchor"].rank().to_numpy())
            for _, block in merged.groupby("date")))
        ics = fusion.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
        report = {
            "window": window, "validation_year": year,
            "train_label_cutoff": spec["train_end"],
            "expected_days": len(expected_days),
            "anchor_days": int(anchor["date"].nunique()),
            "days_match_independent_calendar": bool(
                set(expected_days) == set(int(day) for day in anchor["date"].unique())),
            "pool_exact_all_days": bool(pool_ok), "no_duplicates_all_days": bool(dup_ok),
            "finite_atoms_all_days": bool(finite_ok), "atom_keys_match": keys_match,
            "max_abs_atom_diff_vs_saved": atom_diff,
            "max_abs_fused_diff_vs_anchor": max_diff,
            "rank_identical_all_days": ranks_ok,
            "rankic_fused": daily_ic(ics),
            "fixture": str(root),
        }
        report["passed"] = bool(
            report["days_match_independent_calendar"] and pool_ok and dup_ok and finite_ok
            and keys_match and max_diff == 0.0 and ranks_ok
            and all(value == 0.0 for key, value in atom_diff.items()
                    if key.startswith("c1_"))
            and all(value <= TABM_ATOL for key, value in atom_diff.items()
                    if key.startswith("tabm_")))
        (out_dir / f"v1_annual_fixture_{window}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
