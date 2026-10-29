"""Materialise auxiliary data views (never touching the read-only source)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

MIRROR_DATASETS = (
    "MarketData", "MarketDataRawOpen", "Barra_factor",
    "HS300_index", "HS300_weight", "HS300_industry", "Factor33_winsor",
)


def build_zero_st_view(source_dir, target_dir) -> Path:
    """Mirror the data directory with ``Factor33_winsor.is_st`` set to 0.

    Only the execution auxiliary field changes; all other bytes are reused via
    symlinks so the original read-only data is never modified.
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    factor_src = source / "Factor33_winsor"
    factor_dst = target / "Factor33_winsor"
    for path in sorted(factor_src.glob("*/*/*.parquet")):
        rel = path.relative_to(factor_src)
        out = factor_dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pd.read_parquet(path)
        if "is_st" not in table.columns:
            raise ValueError(f"{path} has no is_st column")
        table["is_st"] = 0
        table.to_parquet(out, index=False)

    for name in MIRROR_DATASETS:
        if name == "Factor33_winsor":
            continue
        src = source / name
        dst = target / name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src, target_is_directory=True)
    return target
