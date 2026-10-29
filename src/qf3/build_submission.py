"""Assemble a self-contained, official-template-compliant submission package."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

SRC = Path("src/qf3")

# package module name -> (source file, new module name)
MODULE_MAP = {
    "constants.py": "alpha_constants.py",
    "featurelib.py": "alpha_features.py",
    "dataio.py": "alpha_dataio.py",
    "labels.py": "alpha_labels.py",
    "panel.py": "alpha_panel.py",
    "cv.py": "alpha_cv.py",
    "trainers.py": "alpha_trainers.py",
}


def _rewrite(text: str) -> str:
    # The platform loads `inference.py` as a private package
    # (`spec_from_file_location(..., submodule_search_locations=[dir])`), so all
    # intra-package helper imports must stay relative.
    text = text.replace(
        "from . import dataio, featurelib",
        "from . import alpha_dataio as dataio, alpha_features as featurelib")
    text = text.replace("from . import labels as labels_mod",
                        "from . import alpha_labels as labels_mod")
    text = text.replace("from . import cv\n", "from . import alpha_cv as cv\n")
    text = text.replace("from . import cv", "from . import alpha_cv as cv")
    text = re.sub(r"from \.(?!alpha_)(\w+) import",
                  lambda m: f"from .alpha_{m.group(1)} import", text)
    return text


def copy_modules(target: Path) -> list[str]:
    written = []
    for source, dest in MODULE_MAP.items():
        text = (SRC / source).read_text()
        text = _rewrite(text)
        (target / dest).write_text(text)
        written.append(dest)
    return written


INFERENCE_TEMPLATE = '''"""Official inference entry point: `InferenceModel` (skd-backtest contract)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_features
from .alpha_model import Scorer


class InferenceModel:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.scorer = Scorer(self.model_dir, meta)

    def predict(self, as_of_date: str, data: dict) -> pd.DataFrame:
        day = int(as_of_date.replace("-", ""))
        market = data["MarketData"]
        factor = data["Factor33_winsor"]
        barra = data["Barra_factor"]

        frame = alpha_features.build_day_features(market, factor, barra, day)
        universe = (barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist())
        frame = frame.set_index("code").reindex(universe).reset_index()
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        score = np.asarray(self.scorer.score(matrix), dtype=float).ravel()
        if score.shape[0] != len(frame):
            raise ValueError("model returned an unexpected number of scores")
        score = np.where(np.isfinite(score), score, 0.0)
        out = pd.DataFrame({"date": as_of_date, "code": frame["code"], "score": score})
        return out.drop_duplicates("code").reset_index(drop=True)
'''


def write_inference(target: Path) -> None:
    (target / "inference.py").write_text(INFERENCE_TEMPLATE)


REQUIREMENTS = """numpy=={numpy}
pandas=={pandas}
pyarrow=={pyarrow}
scipy=={scipy}
scikit-learn=={sklearn}
lightgbm=={lightgbm}
catboost=={catboost}
"""


MODEL_TEMPLATE = '''"""Native model loading and daily centred-rank blending (package-internal)."""

from pathlib import Path

import numpy as np
import pandas as pd


def _centered_rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


class Scorer:
    def __init__(self, model_dir, meta: dict):
        self.meta = meta
        self.members = []
        for member in meta["members"]:
            path = Path(model_dir) / member["file"]
            if member["type"] == "lgbm":
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(path))
            elif member["type"] == "catboost":
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(str(path))
            else:
                raise ValueError(member["type"])
            self.members.append((member, model))
        self.total_weight = float(sum(m["weight"] for m, _ in self.members))

    def score(self, matrix: np.ndarray) -> np.ndarray:
        out = np.zeros(matrix.shape[0], dtype=float)
        for member, model in self.members:
            preds = np.asarray(model.predict(matrix), dtype=float).ravel()
            out += (float(member["weight"]) / self.total_weight) * _centered_rank(preds)
        return out
'''


TRAIN_TEMPLATE = '''"""Reproduce `model/` from the public 2016-2022 data (official train.py).

Run standalone:  python train.py --data-dir <official data dir> --out model
The helper modules import each other relatively (the platform loads this
directory as a private package), so train.py registers the directory as a
package before importing them.
"""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = "_alpha_submission"


def register_package() -> None:
    if PKG in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PKG, HERE / "__init__.py", submodule_search_locations=[str(HERE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)


def build_training_panel(data_dir: str) -> pd.DataFrame:
    panel_mod = importlib.import_module(PKG + ".alpha_panel")
    features, labels = panel_mod.build_panel(data_dir)
    return panel_mod.merge_labels(features, labels)


def train_member(member: dict, panel: pd.DataFrame, seed: int, threads: int):
    trainers = importlib.import_module(PKG + ".alpha_trainers")
    cv_mod = importlib.import_module(PKG + ".alpha_cv")
    features = member["features"]
    params = member.get("params", {})
    rounds = int(member["rounds"])
    y, w, finite = cv_mod.make_target(panel)
    x = panel[features].to_numpy(dtype=np.float32)
    if member["type"] == "lgbm":
        import lightgbm as lgb
        base = (dict(trainers.B0_PARAMS) if member.get("capacity") == "B0"
                else dict(trainers.CAPACITY_GRID[member["capacity"]]))
        base.update(params)
        settings = trainers.lgb_params(base, seed=seed, threads=threads, num_rounds=rounds)
        settings.pop("_num_rounds")
        dataset = lgb.Dataset(x[finite], label=y[finite], weight=w[finite],
                              feature_name=features)
        return lgb.train(settings, dataset, num_boost_round=rounds)
    if member["type"] == "catboost":
        from catboost import CatBoostRegressor, Pool
        settings = {
            "loss_function": "RMSE", "iterations": rounds, "learning_rate": 0.03,
            "border_count": 128, "random_strength": 1.0, "bootstrap_type": "Bernoulli",
            "subsample": 0.8, "thread_count": threads, "random_seed": seed,
            "nan_mode": "Min", "verbose": False, "allow_writing_files": False,
        }
        settings.update(params)
        model = CatBoostRegressor(**settings)
        model.fit(Pool(x[finite], label=y[finite], weight=w[finite],
                       feature_names=features))
        return model
    raise ValueError(member["type"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="model")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    register_package()
    recipe = json.loads((HERE / "recipe.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = build_training_panel(args.data_dir)
    members_meta = []
    for member in recipe["members"]:
        print(f"training {member['name']} ...", flush=True)
        model = train_member(member, panel, recipe["seed"], args.threads)
        filename = f"{member['name']}.{'txt' if member['type'] == 'lgbm' else 'cbm'}"
        path = out / filename
        if member["type"] == "lgbm":
            model.save_model(str(path))
        else:
            model.save_model(str(path), format="cbm")
        members_meta.append({**member, "file": filename})
    meta = {
        "version": recipe["version"],
        "features": recipe["features"],
        "members": members_meta,
        "seed": recipe["seed"],
        "label": "O2O adjusted open H=1, daily cross-sectional average-tied rank",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done", meta["version"])


if __name__ == "__main__":
    main()
'''


def write_requirements(target: Path, versions: dict) -> None:
    (target / "requirements.txt").write_text(REQUIREMENTS.format(**versions))


def write_model_module(target: Path) -> None:
    (target / "alpha_model.py").write_text(MODEL_TEMPLATE)


def write_train(target: Path) -> None:
    (target / "train.py").write_text(TRAIN_TEMPLATE)


def assemble(target: Path) -> dict:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model").mkdir(exist_ok=True)
    (target / "__init__.py").write_text('"""Self-contained submission package."""\n')
    modules = copy_modules(target)
    write_inference(target)
    write_model_module(target)
    write_train(target)
    return {"target": str(target), "modules": modules}
