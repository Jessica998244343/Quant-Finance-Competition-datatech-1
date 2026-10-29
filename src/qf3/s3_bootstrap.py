"""S0: correct paired block bootstrap over daily return paths (never NAV levels)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

S3 = Path("artifacts/stage3_ic_first")


def load_returns(run_dir: Path) -> pd.DataFrame:
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    equity = equity.loc[:, ["date", "portfolio_return", "benchmark_return",
                            "active_return"]].copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity["portfolio_return"] = pd.to_numeric(equity["portfolio_return"], errors="coerce")
    equity["benchmark_return"] = pd.to_numeric(equity["benchmark_return"], errors="coerce")
    return equity.sort_values("date").reset_index(drop=True)


def annualise(returns: np.ndarray) -> float:
    """prod(1+r) wealth, then the platform's annualisation with days = len."""
    days = len(returns)
    if days == 0:
        return float("nan")
    wealth = float(np.prod(1.0 + returns))
    if wealth <= 0:
        return float("nan")
    return wealth ** (252.0 / days) - 1.0


def metrics_from(portfolio: np.ndarray, benchmark: np.ndarray) -> tuple[float, float]:
    excess = annualise(portfolio) - annualise(benchmark)
    active = portfolio - benchmark
    if len(active) < 2:
        return excess, float("nan")
    te = float(np.std(active, ddof=1) * np.sqrt(252))
    ir = excess / te if te > 0 else float("nan")
    return excess, ir


def paired_bootstrap(candidate_dir: Path, reference_dir: Path, block: int = 20,
                     n_boot: int = 2000, seed: int = 20261005) -> dict:
    cand = load_returns(candidate_dir)
    ref = load_returns(reference_dir)
    merged = cand.merge(ref, on="date", suffixes=("_c", "_r"), validate="one_to_one")
    n = len(merged)
    if n < 2:
        return {"n_days": n}
    p_c = merged["portfolio_return_c"].to_numpy(dtype=float)
    p_r = merged["portfolio_return_r"].to_numpy(dtype=float)
    b_c = merged["benchmark_return_c"].to_numpy(dtype=float)
    b_r = merged["benchmark_return_r"].to_numpy(dtype=float)
    if not np.allclose(b_c, b_r, atol=1e-12):
        raise ValueError("benchmark series differ between runs; cannot pair")

    excess_c, ir_c = metrics_from(p_c, b_c)
    excess_r, ir_r = metrics_from(p_r, b_c)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n_blocks) * block
    offsets = np.arange(block)
    deltas_ir, deltas_excess = [], []
    for _ in range(n_boot):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        idx = (starts[picks][:, None] + offsets[None, :]).ravel()
        idx = idx[idx < n]
        ec, ic = metrics_from(p_c[idx], b_c[idx])
        er, ir_ref = metrics_from(p_r[idx], b_c[idx])
        if np.isfinite(ic) and np.isfinite(ir_ref):
            deltas_ir.append(ic - ir_ref)
        if np.isfinite(ec) and np.isfinite(er):
            deltas_excess.append(ec - er)
    return {
        "n_days": int(n), "block": block, "n_boot": n_boot,
        "candidate": {"excess": excess_c, "ir": ir_c},
        "reference": {"excess": excess_r, "ir": ir_r},
        "delta_ir_point": ir_c - ir_r,
        "delta_excess_point": excess_c - excess_r,
        "delta_ir_ci90": [float(np.quantile(deltas_ir, 0.05)),
                          float(np.quantile(deltas_ir, 0.95))],
        "delta_ir_ci95": [float(np.quantile(deltas_ir, 0.025)),
                          float(np.quantile(deltas_ir, 0.975))],
        "delta_excess_ci95": [float(np.quantile(deltas_excess, 0.025)),
                              float(np.quantile(deltas_excess, 0.975))],
        "share_delta_ir_positive": float(np.mean(np.array(deltas_ir) > 0)),
    }


def validate() -> dict:
    """Known-series check: repeated blocks must compound by count."""
    rng = np.random.default_rng(7)
    returns = 0.001 + 0.01 * rng.standard_normal(100)
    block, n_boot, seed = 5, 200, 3
    n = len(returns)
    rng2 = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n_blocks) * block
    offsets = np.arange(block)
    wealth = []
    for _ in range(n_boot):
        picks = rng2.integers(0, n_blocks, size=n_blocks)
        idx = (starts[picks][:, None] + offsets[None, :]).ravel()
        idx = idx[idx < n]
        wealth.append(float(np.prod(1.0 + returns[idx])))

    const = np.full(50, 0.002)
    const_wealth = []
    rng3 = np.random.default_rng(11)
    nb = int(np.ceil(len(const) / block))
    const_starts = np.arange(nb) * block
    for _ in range(50):
        picks = rng3.integers(0, nb, size=nb)
        idx = (const_starts[picks][:, None] + offsets[None, :]).ravel()
        idx = idx[idx < len(const)]
        const_wealth.append(float(np.prod(1.0 + const[idx])))
    return {
        "resampled_wealth_mean": float(np.mean(wealth)),
        "full_sample_wealth": float(np.prod(1.0 + returns)),
        "constant_series_wealth_unique": sorted({round(w, 12) for w in const_wealth}),
        "expected_constant_wealth": round((1.002) ** len(const), 12),
        "constant_compounds_by_count": bool(
            max(const_wealth) - min(const_wealth) < 1e-12),
    }


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    out = {"validation": validate(), "comparisons": {}}
    pairs = {
        "dev_seed_ensemble_vs_R_WORK": ("s3_bt_seed_ensemble_dev19_20",
                                        Path("artifacts/stage2_ic_ir/b/backtests/"
                                             "b1_bt_cat0.5_dev19_20")),
        "conf21_seed_ensemble_vs_R_WORK": ("s3_bt_seed_ensemble_conf21",
                                           Path("artifacts/stage2_ic_ir/b/backtests/"
                                                "b1_bt_cat0.5_conf21")),
    }
    for name, (run, reference) in pairs.items():
        cand_dir = S3 / "backtests" / run
        if not cand_dir.exists():
            continue
        out["comparisons"][name] = {
            "block20": paired_bootstrap(cand_dir, reference, 20),
            "block10": paired_bootstrap(cand_dir, reference, 10),
            "block40": paired_bootstrap(cand_dir, reference, 40),
        }
    (S3 / "s0_bootstrap.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["validation"], indent=2))
    for name, value in out["comparisons"].items():
        b = value["block20"]
        print(name, "dIR=%.3f" % b["delta_ir_point"],
              "95%CI", [round(x, 3) for x in b["delta_ir_ci95"]],
              "dExcess=%.4f" % b["delta_excess_point"])


if __name__ == "__main__":
    main()
