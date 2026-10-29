"""Official inference entry point: InferenceModel (skd-backtest contract)."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_features
from .alpha_models import Scorer


class InferenceModel:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.scorer = Scorer(model_dir)
        self.features = self.scorer.features

    def predict(self, as_of_date: str, data: dict) -> pd.DataFrame:
        day = int(as_of_date.replace("-", ""))
        market = data["MarketData"]
        factor = data["Factor33_winsor"]
        barra = data["Barra_factor"]
        frame = alpha_features.build_day_features(market, factor, barra, day)
        universe = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
        frame = frame.set_index("code").reindex(universe).reset_index()
        score = np.asarray(self.scorer.score(frame), dtype=float).ravel()
        if score.shape[0] != len(frame):
            raise ValueError("scorer returned an unexpected number of scores")
        score = np.where(np.isfinite(score), score, 0.0)
        return pd.DataFrame({"date": as_of_date, "code": frame["code"],
                             "score": score}).drop_duplicates("code").reset_index(drop=True)
