"""Assemble TREE_v1: torch-free package = 0.5 x L5 (3 seeds) + 0.5 x CatBoost C1 (3 seeds)."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy, pandas, pyarrow, lightgbm, catboost

FUS = Path("artifacts/stage3_fusion")
ALT = FUS / "alt_models"
BUILD = Path("releases/stage3_fusion/build/TREE_v1")
SRC = Path("src/qf3")
SEEDS = (20261005, 20261017, 20261029)

SCORER = '''"""Tree-only scorer: 0.5 x L5 seed-mean + 0.5 x CatBoost C1 seed-mean."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _centred_rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


class Scorer:
    def __init__(self, model_dir: str):
        import lightgbm as lgb
        from catboost import CatBoostRegressor

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.weights = dict(meta["weights"])
        self.lgbm = [lgb.Booster(model_file=str(self.dir / name))
                     for name in meta["members"]["l5"]]
        self.cat = []
        for name in meta["members"]["c1"]:
            model = CatBoostRegressor()
            model.load_model(str(self.dir / name))
            self.cat.append(model)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        l5 = np.zeros(len(frame))
        for booster in self.lgbm:
            l5 += _centred_rank(booster.predict(matrix)) / len(self.lgbm)
        c1 = np.zeros(len(frame))
        for model in self.cat:
            c1 += _centred_rank(model.predict(matrix)) / len(self.cat)
        return _centred_rank(self.weights["l5"] * l5 + self.weights["c1"] * c1)
'''


def build() -> dict:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "model").mkdir(parents=True)
    (BUILD / "__init__.py").write_text('"""TREE_v1 submission package."""\n')
    from . import s4_package
    mapping = {"constants.py": "alpha_constants.py", "featurelib.py": "alpha_features.py",
               "dataio.py": "alpha_dataio.py", "labels.py": "alpha_labels.py",
               "panel.py": "alpha_panel.py", "cv.py": "alpha_cv.py"}
    for source, target in mapping.items():
        (BUILD / target).write_text(s4_package.rewrite((SRC / source).read_text()))
    (BUILD / "alpha_models.py").write_text(SCORER)
    inference = (SRC / "s4_alt_build.py").read_text()
    import re
    block = inference.split("INFERENCE = '''", 1)[1].split("'''", 1)[0]
    (BUILD / "inference.py").write_text(block.replace("ALT_v1", "TREE_v1"))
    (BUILD / "train.py").write_text((SRC / "s4_tree_train_src.py").read_text())
    members = {"l5": [f"l5_seed{s}.txt" for s in SEEDS],
               "c1": [f"c1_seed{s}.cbm" for s in SEEDS]}
    for group in members.values():
        for name in group:
            shutil.copy(ALT / name, BUILD / "model" / name)
    from .constants import F0_COLUMNS, G1_COLUMNS
    meta = {"version": "TREE_v1", "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": {"l5": 0.5, "c1": 0.5}, "members": members,
            "fusion": "per-member seed-mean of daily centred ranks, then 0.5/0.5, output re-ranked",
            "trained_on": "official 2016-2022 legal mature H1 labels",
            "rounds": {"l5_rounds": 311, "catboost_c1_trees": 875},
            "note": "tree-only, CPU-native, no torch, no network, no cross-call state"}
    (BUILD / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    (BUILD / "requirements.txt").write_text(
        f"numpy=={numpy.__version__}\npandas=={pandas.__version__}\n"
        f"pyarrow=={pyarrow.__version__}\nlightgbm=={lightgbm.__version__}\n"
        f"catboost=={catboost.__version__}\n")
    return {"build_dir": str(BUILD)}


def zip_package(zip_path: Path) -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("submission") / path.relative_to(BUILD))
    with zipfile.ZipFile(zip_path) as archive:
        uncompressed = sum(info.file_size for info in archive.infolist())
    return {"zip": str(zip_path), "bytes": zip_path.stat().st_size,
            "uncompressed_bytes": uncompressed,
            "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(build())
    print(zip_package(Path("releases/stage3_fusion/TREE_v1.zip")))
