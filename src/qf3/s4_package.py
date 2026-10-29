"""D3 §13.3: assemble the FINAL_v1 CPU submission package (static, self-contained)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

FUS = Path("artifacts/stage3_fusion")
FINAL = FUS / "final_models"
BUILD = Path("releases/stage3_fusion/build/FINAL_v1")
SRC = Path("src/qf3")
VENDOR_TABM = Path("vendor/tabm/tabm.py")

COPY_MODULES = ("constants.py", "featurelib.py", "dataio.py", "labels.py",
                "panel.py", "cv.py")
MODULE_NAMES = {
    "constants.py": "alpha_constants.py", "featurelib.py": "alpha_features.py",
    "dataio.py": "alpha_dataio.py", "labels.py": "alpha_labels.py",
    "panel.py": "alpha_panel.py", "cv.py": "alpha_cv.py",
}


def rewrite(text: str) -> str:
    text = text.replace("from . import dataio, featurelib",
                        "from . import alpha_dataio as dataio, alpha_features as featurelib")
    text = text.replace("from . import labels as labels_mod",
                        "from . import alpha_labels as labels_mod")
    text = text.replace("from . import cv\n", "from . import alpha_cv as cv\n")
    text = text.replace("from . import cv", "from . import alpha_cv as cv")
    text = re.sub(r"from \.(?!alpha_)(\w+) import",
                  lambda m: f"from .alpha_{m.group(1)} import", text)
    # make every intra-package import work both as a package and as a script
    def dual_block(match):
        names = match.group(2)
        indented = "\n".join("    " + line.strip() for line in names.strip().splitlines())
        return (f"try:\n    from .{match.group(1)} import (\n{indented}\n    )\n"
                f"except ImportError:\n    from {match.group(1)} import (\n"
                f"{indented}\n    )")

    def dual_line(match):
        names = match.group(2)
        return (f"try:\n    from .{match.group(1)} import {names}\n"
                f"except ImportError:\n    from {match.group(1)} import {names}")
    text = re.sub(r"from \.(alpha_\w+) import \(([^)]*)\)", dual_block, text)
    text = re.sub(r"from \.(alpha_\w+) import (?!\()([^\n]+)", dual_line, text)
    # "from . import X as y" forms need the same dual treatment
    pairs = [
        ("from . import alpha_dataio as dataio, alpha_features as featurelib",
         "import alpha_dataio as dataio\n    import alpha_features as featurelib"),
        ("from . import alpha_labels as labels_mod",
         "import alpha_labels as labels_mod"),
        ("from . import alpha_cv as cv", "import alpha_cv as cv"),
    ]
    for relative, absolute in pairs:
        if relative in text:
            body = "\n".join("    " + line.strip() for line in absolute.splitlines())
            text = text.replace(
                relative,
                f"try:\n    {relative}\nexcept ImportError:\n{body}")
    return text


MODEL_CODE = '''"""Load the two frozen members and produce the fused daily score."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _centered_rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


class Scorer:
    """0.5 x CatBoost C1 + 0.5 x TabM, ranks computed inside the daily pool."""

    def __init__(self, model_dir: str):
        import torch
        from catboost import CatBoostRegressor
        from safetensors.torch import load_file

        from . import alpha_tabm_wrapper

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.weights = dict(meta["weights"])
        self.cat = CatBoostRegressor()
        self.cat.load_model(str(self.dir / "catboost_c1.cbm"))
        config = json.loads((self.dir / "tabm_config.json").read_text())
        self.tabm = alpha_tabm_wrapper.TabMWrapper(**config)
        state = load_file(str(self.dir / "tabm.safetensors"))
        self.tabm.load_state_dict(state)
        self.tabm.eval()
        self.design = json.loads((self.dir / "tabm_design.json").read_text())

    def _tabm_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        from . import alpha_tabminput
        return alpha_tabminput.transform(frame, self.design)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        matrix = frame[self.features].to_numpy(dtype=np.float32)
        cat = np.asarray(self.cat.predict(matrix), dtype=float).ravel()
        tabm_matrix = torch.from_numpy(self._tabm_matrix(frame))
        with torch.no_grad():
            tabm = self.tabm(tabm_matrix).mean(dim=1).numpy()
        return (self.weights["catboost_c1"] * _centered_rank(cat)
                + self.weights["tabm"] * _centered_rank(tabm))
'''


def write_module(name: str, text: str) -> None:
    (BUILD / name).write_text(text)


def build() -> dict:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "model").mkdir(parents=True)
    (BUILD / "__init__.py").write_text('"""FINAL_v1 submission package."""\n')
    for source, target in MODULE_NAMES.items():
        write_module(target, rewrite((SRC / source).read_text()))
    shutil.copy(VENDOR_TABM, BUILD / "tabm_official.py")
    shutil.copy(Path("vendor/tabm/LICENSE"), BUILD / "LICENSE.tabm")
    (BUILD / "alpha_tabm_wrapper.py").write_text(
        (SRC / "s4_tabm_wrapper_src.py").read_text())
    (BUILD / "alpha_tabminput.py").write_text(
        (SRC / "s4_tabminput_src.py").read_text())
    (BUILD / "alpha_models.py").write_text(MODEL_CODE)
    (BUILD / "inference.py").write_text((SRC / "s4_inference_src.py").read_text())
    (BUILD / "train.py").write_text((SRC / "s4_train_src.py").read_text())
    # model artifacts
    shutil.copy(FINAL / "catboost_c1.cbm", BUILD / "model" / "catboost_c1.cbm")
    shutil.copy(FINAL / "tabm_design.json", BUILD / "model" / "tabm_design.json")
    import torch
    blob = torch.load(FINAL / "tabm.pt", map_location="cpu", weights_only=False)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in blob["state_dict"].items()},
              str(BUILD / "model" / "tabm.safetensors"))
    config = {"d_in": int(blob["d_in"]), "k": int(blob["k"]),
              "width": int(blob["width"]), "blocks": int(blob["blocks"]),
              "dropout": float(blob["dropout"])}
    (BUILD / "model" / "tabm_config.json").write_text(json.dumps(config, indent=2))
    from .constants import F0_COLUMNS, G1_COLUMNS
    meta = {"version": "FINAL_v1", "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": {"catboost_c1": 0.5, "tabm": 0.5},
            "fusion": "daily centred rank of each member, then 0.5/0.5, output re-ranked",
            "trained_on": "official 2016-2022 legal mature H1 labels (466,033 rows)",
            "rounds": {"catboost_c1_trees": 875, "tabm_epochs": 13},
            "note": "no network, no in-package fit, no cross-call state"}
    (BUILD / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    import torch
    import numpy, pandas, pyarrow, catboost, safetensors
    (BUILD / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        f"numpy=={numpy.__version__}\npandas=={pandas.__version__}\n"
        f"pyarrow=={pyarrow.__version__}\ncatboost=={catboost.__version__}\n"
        "torch==2.14.1+cpu\n"
        f"safetensors=={safetensors.__version__}\n"
        "rtdl_num_embeddings==0.0.12\n")
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
