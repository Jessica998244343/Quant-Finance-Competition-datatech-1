"""Build (or reuse) the cached feature panel and label table."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import panel

CACHE_DIR = Path("artifacts/beginning/cache")


def build(data_dir: str = "Source/Data", cache_dir: Path = CACHE_DIR,
          force: bool = False) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel_path = cache_dir / "panel.parquet"
    meta_path = cache_dir / "panel_meta.json"
    if panel_path.exists() and not force:
        print(f"reuse {panel_path}")
        return panel_path, meta_path

    started = time.time()
    features, labels = panel.build_panel(data_dir)
    merged = panel.merge_labels(features, labels)
    merged.to_parquet(panel_path, index=False)
    labels.to_parquet(cache_dir / "labels.parquet", index=False)
    meta = {
        "rows": int(len(merged)),
        "universe_rows": int(len(labels)),
        "dates": [int(merged["date"].min()), int(merged["date"].max())],
        "n_dates": int(merged["date"].nunique()),
        "label_coverage": float(merged["future_return"].notna().mean()),
        "build_seconds": round(time.time() - started, 1),
        "data_dir": str(data_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return panel_path, meta_path


def load(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir) / "panel.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Source/Data")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.data_dir, Path(args.cache_dir), args.force)
