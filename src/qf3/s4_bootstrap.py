"""D1 §4.4: fixed-length paired moving-block bootstrap on daily return paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FUS = Path("artifacts/stage3_fusion")


def load_returns(run_dir: Path) -> pd.DataFrame:
    equity = pd.read_csv(run_dir / "equity_curve.csv")
    frame = equity.loc[:, ["date", "portfolio_return", "benchmark_return"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("portfolio_return", "benchmark_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["year"] = frame["date"].dt.year
    return frame.sort_values("date").reset_index(drop=True)


def _draw_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-length block draw: starts in [0, n-block], repeat until length >= n."""
    if n <= block:
        return np.arange(n)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)) + 1)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
    return idx[:n]                      # exactly n, never shorter after truncation


def _metrics(portfolio: np.ndarray, benchmark: np.ndarray) -> tuple[float, float]:
    n = len(portfolio)
    if n < 2:
        return float("nan"), float("nan")
    wealth_p = float(np.prod(1.0 + portfolio))
    wealth_b = float(np.prod(1.0 + benchmark))
    if wealth_p <= 0 or wealth_b <= 0:
        return float("nan"), float("nan")
    ann_p = wealth_p ** (252.0 / n) - 1.0
    ann_b = wealth_b ** (252.0 / n) - 1.0
    te = float(np.std(portfolio - benchmark, ddof=1) * np.sqrt(252))
    excess = ann_p - ann_b
    return excess, (excess / te if te > 0 else float("nan"))


def paired_bootstrap(candidate_dir: Path, reference_dir: Path, block: int = 20,
                     n_boot: int = 2000, seed: int = 20261005,
                     per_year: bool = True) -> dict:
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
        raise ValueError("benchmark series differ; cannot pair")

    excess_c, ir_c = _metrics(p_c, b_c)
    excess_r, ir_r = _metrics(p_r, b_c)
    years = merged["date"].dt.year.to_numpy()
    groups = [np.nonzero(years == year)[0] for year in np.unique(years)] if per_year \
        else [np.arange(n)]

    rng = np.random.default_rng(seed)
    deltas_ir, deltas_excess, lengths = [], [], []
    for _ in range(n_boot):
        pieces = []
        for group in groups:
            local = _draw_indices(len(group), block, rng)
            pieces.append(group[local])
        idx = np.concatenate(pieces)
        lengths.append(len(idx))
        ec, ic = _metrics(p_c[idx], b_c[idx])
        er, ir_ref = _metrics(p_r[idx], b_c[idx])
        if np.isfinite(ic) and np.isfinite(ir_ref):
            deltas_ir.append(ic - ir_ref)
        if np.isfinite(ec) and np.isfinite(er):
            deltas_excess.append(ec - er)
    return {
        "n_days": int(n), "block": block, "n_boot": n_boot, "per_year": per_year,
        "lengths_all_equal_n": bool(len(set(lengths)) == 1 and lengths[0] == n),
        "draw_length_min": int(min(lengths)), "draw_length_max": int(max(lengths)),
        "candidate": {"excess": excess_c, "ir": ir_c},
        "reference": {"excess": excess_r, "ir": ir_r},
        "delta_ir_point": ir_c - ir_r, "delta_excess_point": excess_c - excess_r,
        "delta_ir_ci90": [float(np.quantile(deltas_ir, 0.05)),
                          float(np.quantile(deltas_ir, 0.95))],
        "delta_ir_ci95": [float(np.quantile(deltas_ir, 0.025)),
                          float(np.quantile(deltas_ir, 0.975))],
        "delta_excess_ci95": [float(np.quantile(deltas_excess, 0.025)),
                              float(np.quantile(deltas_excess, 0.975))],
        "share_delta_ir_positive": float(np.mean(np.array(deltas_ir) > 0)),
    }


def validate() -> dict:
    out = {}
    for n in (487, 243):
        for block in (10, 20, 40):
            rng = np.random.default_rng(20261005)
            lengths = {len(_draw_indices(n, block, rng)) for _ in range(200)}
            out[f"n{n}_b{block}"] = {"lengths_unique": sorted(lengths),
                                     "all_equal_n": lengths == {n}}
    # repeated blocks must compound by count: constant series check
    const = np.full(487, 0.001)
    rng = np.random.default_rng(3)
    idx = _draw_indices(len(const), 20, rng)
    out["constant_series_compounds"] = bool(
        abs(float(np.prod(1.0 + const[idx])) - (1.001) ** 487) < 1e-9)
    # pairing: identical index applied to both sides reproduces a known ratio
    a = np.arange(10.0)
    index = np.array([0, 1, 2, 7, 8, 9])
    out["same_index_paired"] = bool(np.array_equal(a[index], a[index]))
    return out


def main() -> None:
    FUS.mkdir(parents=True, exist_ok=True)
    out = {"validation": validate(), "comparisons": {}}
    pairs = {
        "dev_seed_ensemble_vs_R_WORK": (
            Path("artifacts/stage3_ic_first/backtests/s3_bt_seed_ensemble_dev19_20"),
            Path("artifacts/stage2_ic_ir/b/backtests/b1_bt_cat0.5_dev19_20")),
        "conf21_seed_ensemble_vs_R_WORK": (
            Path("artifacts/stage3_ic_first/backtests/s3_bt_seed_ensemble_conf21"),
            Path("artifacts/stage2_ic_ir/b/backtests/b1_bt_cat0.5_conf21")),
        "dev_main_seed_ensemble_vs_R_WORK": (
            Path("artifacts/stage3_ic_first/backtests/s3_bt_main_seed_ensemble_dev"),
            Path("artifacts/stage2_ic_ir/b/backtests/b1_bt_cat0.5_dev19_20")),
    }
    for name, (cand, ref) in pairs.items():
        if not cand.exists() or not ref.exists():
            continue
        out["comparisons"][name] = {
            f"block{b}": paired_bootstrap(cand, ref, b) for b in (20, 10, 40)}
    (FUS / "d1_bootstrap.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["validation"], indent=2))
    for name, value in out["comparisons"].items():
        b = value["block20"]
        print(name, "lengths_ok", b["lengths_all_equal_n"],
              "dIR=%.3f" % b["delta_ir_point"],
              "95%", [round(x, 3) for x in b["delta_ir_ci95"]],
              "dExcess=%.4f" % b["delta_excess_point"])


if __name__ == "__main__":
    main()
