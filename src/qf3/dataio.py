"""Reading the official monthly parquet tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE

SOURCE_NAMES = (FACTOR_TABLE, BARRA_TABLE, MARKET_TABLE)


def _files(data_dir: Path, name: str, start_year: int, end_year: int) -> list[Path]:
    root = Path(data_dir) / name
    files = []
    for year in range(start_year, end_year + 1):
        files.extend(sorted(root.glob(f"{year}/*/*.parquet")))
    return files


def load_range(data_dir, name: str, start_year: int, end_year: int,
               columns=None) -> pd.DataFrame:
    files = _files(Path(data_dir), name, start_year, end_year)
    if not files:
        raise FileNotFoundError(f"no files for {name} in {start_year}-{end_year}")
    parts = [pd.read_parquet(path, columns=columns) for path in files]
    table = pd.concat(parts, ignore_index=True)
    keys = [name for name in ("日期", "代码") if name in table.columns]
    if keys:
        table = table.sort_values(keys, kind="stable")
    return table.reset_index(drop=True)


def load_all(data_dir, start_year: int = 2016, end_year: int = 2022) -> dict[str, pd.DataFrame]:
    return {name: load_range(data_dir, name, start_year, end_year)
            for name in SOURCE_NAMES}


def calendar(tables: dict[str, pd.DataFrame]) -> list[int]:
    return sorted(pd.unique(tables[MARKET_TABLE]["日期"]).tolist())
