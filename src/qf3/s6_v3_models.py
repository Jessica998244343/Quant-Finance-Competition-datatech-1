"""V3 §A1/A3: parameterised loader for the cut-off-2021 confirmation models.

Loads the nine candidate atoms (C1 x3, TabM x3, TCN x3) and the three L5
controls from the frozen manifest paths, runs them on a **single complete day**
and applies the frozen flat graph.  The only difference from the annual fixtures
is the model path and the legal training cutoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import F0_COLUMNS, G1_COLUMNS
from .s6_fusion import daily_centred_rank
from .s6_v2_tcn import CHANNELS, STEPS, TCNModel
from .s6_v2_tabm_variants import Design as TabmDesign

V2 = Path("artifacts/stage4_verified_ic/v2")
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SEEDS = (20261005, 20261017, 20261029)
TREE_WEIGHTS = {"l5": 0.5, "c1": 0.5}


class ConfirmationModels:
    """Model set with explicit paths; all inference is CPU-only."""

    def __init__(self, c1_path, tabm_dir, tcn_dir, l5_path, seeds=SEEDS):
        self.c1_path = c1_path
        self.tabm_dir = tabm_dir
        self.tcn_dir = tcn_dir
        self.l5_path = l5_path
        self.seeds = tuple(seeds)
        self.cat, self.tabm, self.tcn, self.l5 = {}, {}, {}, {}
        from catboost import CatBoostRegressor
        from safetensors.torch import load_file

        torch.set_num_threads(2)
        for seed in self.seeds:
            model = CatBoostRegressor()
            model.load_model(str(self.c1_path(seed)))
            self.cat[seed] = model
            tabm_dir = Path(self.tabm_dir(seed))
            design_file = (tabm_dir / "design.json") if (
                tabm_dir / "design.json").exists() else (
                tabm_dir / f"design_seed{seed}.json")
            design = json.loads(design_file.read_text())
            from .s4_tabm_wrapper_src import TabMWrapper
            net = TabMWrapper(2 * len(FEATURES) + len(G1_COLUMNS))
            weight = (tabm_dir / "tabm.safetensors") if (
                tabm_dir / "tabm.safetensors").exists() else (
                tabm_dir / f"tabm_seed{seed}.safetensors")
            net.load_state_dict(load_file(str(weight)))
            net.eval()
            self.tabm[seed] = (net, design)
            tcn_dir = Path(self.tcn_dir(seed))
            pre = json.loads((tcn_dir / "preprocessing.json").read_text())
            tcn = TCNModel(2 * len(FEATURES) + len(G1_COLUMNS))
            tcn.load_state_dict(load_file(str(tcn_dir / "model.safetensors")))
            tcn.eval()
            self.tcn[seed] = (tcn, pre)
            import lightgbm as lgb
            self.l5[seed] = lgb.Booster(model_file=str(self.l5_path(seed)))


    # ---- single-day atoms -------------------------------------------------
    def atoms(self, frame: pd.DataFrame, sequence: np.ndarray) -> dict:
        matrix = frame[FEATURES].to_numpy(dtype=np.float32)
        out = {}
        for seed in self.seeds:
            out[f"c1_{seed}"] = np.asarray(self.cat[seed].predict(matrix), dtype=float).ravel()
            out[f"l5_{seed}"] = np.asarray(self.l5[seed].predict(matrix), dtype=float).ravel()
            net, design = self.tabm[seed]
            inner = TabmDesign().fit(frame)
            inner.median, inner.iqr = design["median"], design["iqr"]
            x = inner.transform(frame)
            with torch.no_grad():
                out[f"tabm_{seed}"] = net(torch.from_numpy(x)).mean(dim=1).numpy()
            tcn, pre = self.tcn[seed]
            rows = torch.from_numpy(np.asarray(sequence, dtype=np.float32).copy())
            for channel, (median, iqr) in {int(k): tuple(v)
                                           for k, v in pre["channel_stats"].items()}.items():
                rows[:, :, channel] = torch.clamp((rows[:, :, channel] - median) / iqr, -5, 5)
            rows = torch.nan_to_num(rows, nan=0.0)
            static = TabmDesign().fit(frame)
            static.median, static.iqr = pre["design_median"], pre["design_iqr"]
            with torch.no_grad():
                out[f"tcn_{seed}"] = tcn(
                    torch.from_numpy(static.transform(frame)), rows).numpy()
        return out

    # ---- frozen graph -----------------------------------------------------
    def family(self, frame: pd.DataFrame, sequence: np.ndarray) -> dict:
        atoms = self.atoms(frame, sequence)
        families = {}
        for family in ("c1", "tabm", "tcn", "l5"):
            value = np.zeros(len(frame))
            for seed in self.seeds:
                value += daily_centred_rank(atoms[f"{family}_{seed}"]) / len(self.seeds)
            families[family] = value
        return families

    def fusion(self, families: dict, weights: dict) -> np.ndarray:
        if abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("weights must sum to 1")
        value = np.zeros(len(next(iter(families.values()))))
        for name, weight in weights.items():
            value += weight * families[name]
        return daily_centred_rank(value)

    def tree_control(self, families: dict) -> np.ndarray:
        return self.fusion(families, TREE_WEIGHTS)


def deploy_models(seeds=SEEDS) -> ConfirmationModels:
    """The frozen cut-off-2021 confirmation set (label_end_date <= 20211231)."""
    deploy = V2 / "deploy_2021cutoff"
    return ConfirmationModels(
        c1_path=lambda seed: deploy / f"catboost_c1_seed{seed}.cbm",
        tabm_dir=lambda seed: deploy,
        tcn_dir=lambda seed: V2 / f"tcn_rev1_tcn_s{seed}_RES22",
        l5_path=lambda seed: deploy / f"l5_seed{seed}.txt", seeds=seeds)


def annual_models(window: str, seeds=SEEDS) -> ConfirmationModels:
    """The already validated annual research models (for loader equivalence)."""
    tabm_root = Path("artifacts/stage4_verified_ic/v1_tabm")
    return ConfirmationModels(
        c1_path=lambda seed: Path("artifacts/stage3_fusion/models_cat")
        / f"C1_{seed}__{window}.cbm",
        tabm_dir=lambda seed: tabm_root / f"tabm_pc_{seed}__{window}",
        tcn_dir=lambda seed: V2 / f"tcn_rev1_tcn_s{seed}_{window}",
        l5_path=lambda seed: Path("artifacts/stage3_fusion/alt_models")
        / f"l5_seed{seed}.txt", seeds=seeds)
