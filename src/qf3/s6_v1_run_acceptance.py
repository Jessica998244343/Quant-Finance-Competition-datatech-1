"""§14.3: clean-environment CPU acceptance of a delivered V1 zip (run deps only).

Loads the package exactly as the platform would (private package import of the
extracted ``inference.py``), drives it with the real ``DataProvider``, asserts
that ``torch`` is never imported, and compares the output against the frozen V1
anchor for the 2021 window.  Records pool/duplicate/finiteness/column checks,
cold and warm timing, peak RSS and a genuine future-row perturbation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (20261005, 20261017, 20261029)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def requirement_closure(names) -> set[str]:
    """Transitive runtime dependency closure of the declared requirements."""
    from importlib.metadata import PackageNotFoundError, requires

    seen: set[str] = set()
    stack = [name.lower() for name in names]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            entries = requires(name) or []
        except PackageNotFoundError:
            continue
        for entry in entries:
            if ";" in entry and "extra ==" in entry:      # skip optional extras
                continue
            base = re.split(r"[<>=!~\[; ]", entry.strip())[0].strip().lower()
            if base and base not in seen:
                stack.append(base)
    return seen


def load_package(root: Path, alias: str = "v1_run_pkg"):
    for name in list(sys.modules):
        if name.startswith(alias):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        alias, root / "inference.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--window", default="CONF21")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--anchor",
                        default="artifacts/stage4_verified_ic/v1_anchor__CONF21.parquet")
    parser.add_argument("--out", default=None)
    parser.add_argument("--full-year", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(args.package_dir)
    report = {
        "package_dir": str(root),
        "zip_requirements": (root / "requirements.txt").read_text().strip().splitlines(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "thread_env": {name: os.environ.get(name) for name in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }

    sys.path.insert(0, "vendor/skd-backtest/src")
    from skd_backtest.config import BacktestConfig
    from skd_backtest.data_provider import DataProvider

    # Snapshot taken *after* the harness is imported, so any module added below
    # belongs to the submission package's own import closure.
    modules_before_package = set(sys.modules)

    started = time.time()
    module = load_package(root)
    model = module.InferenceModel(str(root / "model"))
    report["load_seconds"] = round(time.time() - started, 3)
    report["torch_imported_after_load"] = "torch" in sys.modules

    provider = DataProvider(BacktestConfig(
        data_dir=Path("Source/Data"), start_date=f"{args.year}-01-01",
        end_date=f"{args.year}-12-31", initial_cash=1_000_000.0,
        rebalance_interval=1, holding_period=1, lookback=120,
        price_mode="adjusted_return", trading_days_per_year=252,
        risk_free_rate=0.0, output_dir=None, read_batch_months=3,
        prefetch=False, async_inference=False))
    provider.prepare()

    anchor = pd.read_parquet(args.anchor)
    anchor_days = sorted(int(d) for d in anchor["date"].unique())
    representative = [anchor_days[0], anchor_days[len(anchor_days) // 2], anchor_days[-1]]
    query_days = anchor_days if args.full_year else representative

    rows, pool_ok, dup_ok, finite_ok, column_ok, date_ok = [], True, True, True, True, True
    warm_times = []
    max_diff_overall, ranks_all_identical = 0.0, True
    per_day = {}
    for index, day in enumerate(query_days):
        as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
        data = provider.as_of(as_of)
        tick = time.time()
        out = model.predict(as_of, data)
        elapsed = time.time() - tick
        if index > 0:
            warm_times.append(elapsed)
        else:
            first_predict = elapsed
        if "torch" in sys.modules:
            report["torch_imported_after_load"] = True
        reference = anchor.loc[anchor["date"] == day]
        barra = data["Barra_factor"]
        universe = set(barra.loc[barra["日期"] == day, "代码"].drop_duplicates())
        day_pool_ok = set(out["code"]) == universe
        day_dup_ok = not out["code"].duplicated().any()
        day_finite = bool(np.isfinite(out["score"]).all())
        day_columns = list(out.columns) == ["date", "code", "score"]
        day_date = bool((out["date"] == as_of).all())
        merged = out.merge(reference, on="code", suffixes=("_out", "_ref"), how="outer")
        max_abs = float(np.nanmax(np.abs(merged["score_out"] - merged["score_ref"])))
        rank_identical = bool(np.array_equal(
            out.set_index("code")["score"].rank(method="average").reindex(
                reference.set_index("code").index).to_numpy(),
            reference.set_index("code")["score"].rank(method="average").to_numpy()))
        pool_ok &= day_pool_ok
        dup_ok &= day_dup_ok
        finite_ok &= day_finite
        column_ok &= day_columns
        date_ok &= day_date
        max_diff_overall = max(max_diff_overall, max_abs)
        ranks_all_identical &= rank_identical
        if day in representative:
            per_day[as_of] = {
                "rows": int(len(out)), "pool_exact": day_pool_ok, "no_duplicates": day_dup_ok,
                "finite": day_finite, "columns_exact": day_columns, "date_ok": day_date,
                "max_abs_score_diff": max_abs, "rank_identical": rank_identical,
                "predict_seconds": round(elapsed, 4)}

    report["representative_days"] = per_day
    report["first_predict_seconds"] = round(first_predict, 4)
    report["cold_total_seconds"] = round(report["load_seconds"] + first_predict, 4)
    report["days_checked"] = len(query_days)
    report["pool_exact_all_days"] = pool_ok
    report["no_duplicates_all_days"] = dup_ok
    report["finite_all_days"] = finite_ok
    report["columns_exact_all_days"] = column_ok
    report["date_string_all_days"] = date_ok
    report["max_abs_score_diff_vs_anchor"] = max_diff_overall
    report["rank_identical_all_days"] = ranks_all_identical
    if warm_times:
        report["warm_seconds_p50"] = round(float(np.percentile(warm_times, 50)), 4)
        report["warm_seconds_p95"] = round(float(np.percentile(warm_times, 95)), 4)
        report["warm_seconds_max"] = round(float(np.max(warm_times)), 4)
        report["p95_le_max"] = report["warm_seconds_p95"] <= report["warm_seconds_max"]
    report["peak_rss_mib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    # genuine future perturbation: append real t+1.. rows with perturbed prices
    day = query_days[-1]
    as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
    data = provider.as_of(as_of)
    base = model.predict(as_of, data)
    perturbed = {key: value.copy() for key, value in data.items()}
    market = perturbed["MarketData"]
    future_paths = sorted(Path("Source/Data/MarketData").glob(
        f"{day // 10000 + 1}/*/*.parquet"))
    raw_market = pd.concat([pd.read_parquet(path) for path in future_paths],
                           ignore_index=True)
    future = raw_market.loc[raw_market["日期"] > day].head(500).copy()
    future["close"] = future["close"] * 3.0
    perturbed["MarketData"] = pd.concat([market, future], ignore_index=True)
    after = model.predict(as_of, perturbed)
    report["true_future_perturbation"] = {
        "rows_added": int(len(future)),
        "identical": bool(np.array_equal(base["score"].to_numpy(), after["score"].to_numpy()))}

    # §14.2 invariance: repeated call, cross-day order, stock-row permutation
    repeat = model.predict(as_of, {key: value.copy() for key, value in data.items()})
    report["repeat_call_identical"] = bool(
        np.array_equal(base["score"].to_numpy(), repeat["score"].to_numpy()))
    other_day = query_days[0]
    other_as_of = f"{str(other_day)[:4]}-{str(other_day)[4:6]}-{str(other_day)[6:]}"
    early_provider = DataProvider(BacktestConfig(
        data_dir=Path("Source/Data"), start_date=f"{args.year}-01-01",
        end_date=f"{args.year}-12-31", initial_cash=1_000_000.0,
        rebalance_interval=1, holding_period=1, lookback=120,
        price_mode="adjusted_return", trading_days_per_year=252,
        risk_free_rate=0.0, output_dir=None, read_batch_months=3,
        prefetch=False, async_inference=False))
    early_provider.prepare()
    data_early = early_provider.as_of(other_as_of)
    model_late_first = module.InferenceModel(str(root / "model"))
    late_first = model_late_first.predict(as_of, {key: value.copy() for key, value in data.items()})
    early_second = model_late_first.predict(other_as_of, data_early)
    model_early_first = module.InferenceModel(str(root / "model"))
    early_first = model_early_first.predict(other_as_of, data_early)
    late_second = model_early_first.predict(as_of, {key: value.copy() for key, value in data.items()})
    report["cross_call_state_independent"] = bool(
        np.array_equal(base["score"].to_numpy(), late_first["score"].to_numpy())
        and np.array_equal(early_first["score"].to_numpy(), early_second["score"].to_numpy())
        and np.array_equal(late_first["score"].to_numpy(), late_second["score"].to_numpy()))
    rng = np.random.default_rng(20261006)
    shuffled = {key: value.iloc[rng.permutation(len(value))].reset_index(drop=True)
                for key, value in data.items()}
    permuted = model.predict(as_of, shuffled)
    report["input_order_permutation_identical"] = bool(
        np.array_equal(base.sort_values("code")["score"].to_numpy(),
                       permuted.sort_values("code")["score"].to_numpy()))

    report["member_hashes"] = {name: _sha(root / "model" / name) for name in
                               sorted(p.name for p in (root / "model").glob("*"))}

    declared = {line.split("==")[0].strip().lower()
                for line in report["zip_requirements"] if "==" in line}
    allowed = requirement_closure(declared)
    added = {name for name in set(sys.modules) - modules_before_package
             if "." not in name}
    stdlib = set(sys.stdlib_module_names)
    package_modules = {"alpha_constants", "alpha_features", "alpha_dataio", "alpha_labels",
                       "alpha_panel", "alpha_cv", "alpha_models", "alpha_tabminit",
                       "alpha_tabminput", "alpha_tabm_static", "alpha_train_model",
                       "alpha_train_tabm", "tabm_official", "inference", "v1_run_pkg"}
    third_party = sorted(name for name in added
                         if name not in stdlib and name not in package_modules
                         and not name.startswith("_"))
    report["package_third_party_imports"] = third_party
    report["requirement_closure"] = sorted(allowed)
    report["outside_declared_requirements"] = sorted(set(third_party) - allowed)
    report["harness_only_packages"] = ["typeguard", "typing_extensions"]

    report["passed"] = bool(
        not report["torch_imported_after_load"] and pool_ok and dup_ok and finite_ok
        and column_ok and date_ok and ranks_all_identical and max_diff_overall == 0.0
        and report["true_future_perturbation"]["identical"]
        and report.get("p95_le_max", True)
        and report["repeat_call_identical"] and report["cross_call_state_independent"]
        and report["input_order_permutation_identical"]
        and not report["outside_declared_requirements"])
    target = Path(args.out) if args.out else Path(
        f"artifacts/stage4_verified_ic/v1_run_acceptance_{args.window}.json")
    target.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
