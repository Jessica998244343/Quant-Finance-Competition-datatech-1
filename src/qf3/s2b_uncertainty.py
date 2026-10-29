"""B: paired block-bootstrap uncertainty for IC and net-active-return (IR)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, s2b_common


def _equity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "equity_curve.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def paired_ir_bootstrap(a_dir: Path, b_dir: Path, block: int = 20,
                        n_boot: int = 2000, seed: int = 20261005) -> dict:
    a = _equity(a_dir)
    b = _equity(b_dir)
    merged = a[["date", "portfolio_nav", "benchmark_nav", "active_return"]].merge(
        b[["date", "portfolio_nav", "benchmark_nav", "active_return"]],
        on="date", suffixes=("_a", "_b"), validate="one_to_one")
    n = len(merged)
    if n < 2:
        return {}

    def metrics_from(block_frame: pd.DataFrame, side: str) -> tuple[float, float]:
        nav = block_frame[f"portfolio_nav_{side}"].to_numpy(dtype=float)
        bnav = block_frame[f"benchmark_nav_{side}"].to_numpy(dtype=float)
        active = block_frame[f"active_return_{side}"].to_numpy(dtype=float)
        days = len(block_frame)
        if nav[0] <= 0 or bnav[0] <= 0 or days < 2:
            return np.nan, np.nan
        ann_p = (nav[-1] / nav[0]) ** (252.0 / days) - 1.0
        ann_b = (bnav[-1] / bnav[0]) ** (252.0 / days) - 1.0
        te = float(np.nanstd(active, ddof=1) * np.sqrt(252))
        return float(ann_p - ann_b), (float((ann_p - ann_b) / te) if te else np.nan)

    excess_a, ir_a = metrics_from(merged, "a")
    excess_b, ir_b = metrics_from(merged, "b")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n_blocks) * block
    deltas_ir, deltas_excess = [], []
    for _ in range(n_boot):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        idx = (starts[picks][:, None] + np.arange(block)[None, :]).ravel()
        idx = np.sort(idx[idx < n])
        sample = merged.iloc[idx]
        ea, ia = metrics_from(sample, "a")
        eb, ib = metrics_from(sample, "b")
        if np.isfinite(ia) and np.isfinite(ib):
            deltas_ir.append(ia - ib)
        if np.isfinite(ea) and np.isfinite(eb):
            deltas_excess.append(ea - eb)
    return {
        "n_days": int(n),
        "excess_a": excess_a, "ir_a": ir_a,
        "excess_b": excess_b, "ir_b": ir_b,
        "delta_ir_point": (ir_a - ir_b) if np.isfinite(ir_a) and np.isfinite(ir_b) else None,
        "delta_excess_point": (excess_a - excess_b)
        if np.isfinite(excess_a) and np.isfinite(excess_b) else None,
        "delta_ir_ci90": [float(np.quantile(deltas_ir, 0.05)),
                          float(np.quantile(deltas_ir, 0.95))] if deltas_ir else None,
        "delta_ir_ci95": [float(np.quantile(deltas_ir, 0.025)),
                          float(np.quantile(deltas_ir, 0.975))] if deltas_ir else None,
        "delta_excess_ci95": [float(np.quantile(deltas_excess, 0.025)),
                              float(np.quantile(deltas_excess, 0.975))]
        if deltas_excess else None,
        "share_delta_ir_positive": float(np.mean(np.array(deltas_ir) > 0))
        if deltas_ir else None,
    }


def ic_bootstrap(a_rankic: Path, b_rankic: Path, block: int = 20,
                 n_boot: int = 2000) -> dict:
    a = pd.read_parquet(a_rankic)
    b = pd.read_parquet(b_rankic)
    return metrics.paired_ic_difference(a, b, block=block, n_boot=n_boot)


def main() -> None:
    reference = s2b_common.BT / "b1_bt_cat0.5_dev19_20"
    seed = s2b_common.BT / "b2_seedavg"
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")

    def rankic(tag: str, windows=("DEV19", "DEV20")) -> pd.DataFrame:
        frames = [pd.read_parquet(s2b_common.PRED / f"{tag}__{w}.parquet")
                  for w in windows]
        _, rk = s2b_common.ic_summary(pd.concat(frames, ignore_index=True), panel)
        return rk

    ref_dev = rankic("ref_corrected_cat0.5")
    seed_dev = rankic("b2_seedavg")
    ref_conf = rankic("ref_corrected_cat0.5", ("CONF21",))
    seed_conf = rankic("b2_seedavg", ("CONF21",))
    ic = {
        "dev_block20": metrics.paired_ic_difference(seed_dev, ref_dev, block=20),
        "dev_block10": metrics.paired_ic_difference(seed_dev, ref_dev, block=10),
        "dev_block40": metrics.paired_ic_difference(seed_dev, ref_dev, block=40),
        "conf21_block20": metrics.paired_ic_difference(seed_conf, ref_conf, block=20),
    }
    out = {
        "dev": {
            "seed_average_vs_reference": paired_ir_bootstrap(seed, reference),
            "seed_average_vs_reference_block10": paired_ir_bootstrap(
                seed, reference, block=10),
            "seed_average_vs_reference_block40": paired_ir_bootstrap(
                seed, reference, block=40),
        },
        "conf21": {
            "seed_average_vs_reference": paired_ir_bootstrap(
                s2b_common.BT / "b2_seedavg_conf21",
                s2b_common.BT / "b1_bt_cat0.5_conf21"),
        },
        "ic": ic,
    }
    s2b_common.write_json(s2b_common.B / "b_uncertainty.json", out)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
