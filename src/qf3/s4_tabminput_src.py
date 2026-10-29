"""TabM input: fold-frozen median/IQR scaling, clip and missing masks (N0)."""

import numpy as np
import pandas as pd

try:
    from .alpha_constants import F0_COLUMNS, G1_COLUMNS  # noqa: F401
except ImportError:
    from alpha_constants import F0_COLUMNS, G1_COLUMNS  # noqa: F401

MODE = "N0"


def transform(frame: pd.DataFrame, design: dict) -> np.ndarray:
    """Reproduce the training-time N0 representation from frozen statistics."""
    median = design["median"]
    iqr = design["iqr"]
    features = design["features"]
    blocks = []
    ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(
        lambda s: ((s.rank(method="average") - (s.notna().sum() + 1) / 2.0)
                   / s.notna().sum()) if s.notna().any() else s)
    for column in features:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.isna().to_numpy(dtype=float)
        filled = values.fillna(median[column]).to_numpy(dtype=float)
        scaled = np.clip((filled - median[column]) / iqr[column], -5, 5)
        blocks.append(scaled.reshape(-1, 1))
        blocks.append(mask.reshape(-1, 1))
    blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
    return np.concatenate(blocks, axis=1).astype(np.float32)
