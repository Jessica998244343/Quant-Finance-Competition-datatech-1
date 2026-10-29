"""V1 §4: assemble the research package (NumPy TabM inference, no torch at run time).

The builder writes the corrected model identity (training label cutoff vs
validation year kept separate), binds every packaged weight to its source atom
record, and refuses to finish if the machine manifest check fails.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
FUS = Path("artifacts/stage3_fusion")
FINAL_BUILD = Path("releases/stage3_fusion/build/FINAL_v1")
BUILD = Path("releases/stage4_verified_ic/build/V1_pkg")
SEEDS = (20261005, 20261017, 20261029)
WINDOW = "CONF21"
WEIGHTS = {"c1": 0.5, "tabm": 0.5}
FUSION_GRAPH = "atom rank -> seed mean -> family weight -> final rank"
TRAIN_LABEL_CUTOFF = 20201231       # CONF21 window: outer fit uses label_end <= 2020-12-31
VALIDATION_YEAR = 2021              # the year the atoms predict

STATIC = '''"""Static NumPy forward for the frozen TabM weights (no torch, explicit ops)."""

import numpy as np


class TabmStatic:
    """TabM (BatchEnsemble MLP, k members) evaluated with plain NumPy FP32.

    Explicit architecture copied from the frozen weights, not a generic factory:
    2 blocks of (LinearBatchEnsemble -> ReLU -> Dropout off at eval) followed by
    a LinearEnsemble head; member outputs are averaged.
    """

    def __init__(self, arrays: dict):
        self.W0, self.r0, self.s0, self.b0 = (arrays["W0"], arrays["r0"],
                                              arrays["s0"], arrays["b0"])
        self.W1, self.r1, self.s1, self.b1 = (arrays["W1"], arrays["r1"],
                                              arrays["s1"], arrays["b1"])
        self.Wh, self.bh = arrays["Wh"], arrays["bh"]
        self.k = int(arrays["k"][0])

    @classmethod
    def from_npz(cls, path) -> "TabmStatic":
        """Strict loader: full key set, dtype, shape, finiteness and k consistency."""
        required = ("W0", "r0", "s0", "b0", "W1", "r1", "s1", "b1", "Wh", "bh", "k")
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in required if name not in data.files]
            extra = [name for name in data.files if name not in required]
            if missing or extra:
                raise ValueError(f"npz key mismatch: missing={missing} extra={extra}")
            for name in required:
                value = data[name]
                if value.dtype != np.float32:
                    raise ValueError(f"{name}: dtype {value.dtype} != float32")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name}: contains non-finite values")
            arrays = {name: data[name] for name in required}
        k_value = arrays["k"]
        if k_value.shape != (1,) or not float(k_value[0]).is_integer():
            raise ValueError(f"k must be a 1-element integer-valued array, got {k_value}")
        k = int(k_value[0])
        d_in = arrays["W0"].shape[1]
        d_block = arrays["W0"].shape[0]
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        if d_in < 1 or d_block < 1:
            raise ValueError(f"d_in/d_block must be positive, got {d_in}/{d_block}")
        expected = {
            "W0": (d_block, d_in), "r0": (k, d_in), "s0": (k, d_block),
            "b0": (k, d_block), "W1": (d_block, d_block), "r1": (k, d_block),
            "s1": (k, d_block), "b1": (k, d_block), "Wh": (k, d_block, 1),
            "bh": (k, 1),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name}: shape {arrays[name].shape} != {shape}")
        return cls(arrays)

    def layer_outputs(self, x: np.ndarray) -> dict:
        h = np.asarray(x, dtype=np.float32)[:, None, :] * self.r0[None, :, :]
        h = h @ self.W0.T
        h = h * self.s0[None, :, :] + self.b0[None, :, :]
        h = np.maximum(h, 0.0)
        block0 = h
        h = h * self.r1[None, :, :]
        h = h @ self.W1.T
        h = h * self.s1[None, :, :] + self.b1[None, :, :]
        h = np.maximum(h, 0.0)
        block1 = h
        out = np.einsum("nki,kio->nko", h, self.Wh) + self.bh[None, :, :]
        return {"block0": block0, "block1": block1, "head": out[:, :, 0]}

    def member_outputs(self, x: np.ndarray) -> np.ndarray:
        return self.layer_outputs(x)["head"]

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.member_outputs(x).mean(axis=1)
'''

MODELS = '''"""V1 scorer: w_c1 x CatBoost C1 (3 seeds) + w_tabm x TabM (3 seeds), frozen graph."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_tabminput, alpha_tabm_static


def _rank(values) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average")
    return ((ranks - (len(series) + 1) / 2.0) / len(series)).to_numpy()


def _seed(name: str) -> int:
    match = re.search(r"seed(\\d+)", name)
    if match is None:
        raise ValueError(f"member name without a seed identity: {name}")
    return int(match.group(1))


class Scorer:
    """Frozen-graph scorer.  Contract: exactly one complete trading day per call."""

    def __init__(self, model_dir: str):
        from catboost import CatBoostRegressor

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.features = list(meta["features"])
        self.weights = dict(meta["weights"])
        self.fusion_graph = meta.get("fusion_graph")
        self.c1_seeds = [_seed(name) for name in meta["members"]["c1"]]
        self.tabm_seeds = [_seed(name) for name in meta["members"]["tabm"]]
        if self.c1_seeds != self.tabm_seeds:
            raise ValueError("C1 and TabM seed identities differ")
        self.cat = []
        for name in meta["members"]["c1"]:
            model = CatBoostRegressor()
            model.load_model(str(self.dir / name))
            self.cat.append(model)
        with open(self.dir / "tabm_design.json") as handle:
            self.design = json.load(handle)
        self.tabm = [alpha_tabm_static.TabmStatic.from_npz(self.dir / name)
                     for name in meta["members"]["tabm"]]

    def _one_day(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "date" in frame.columns and frame["date"].nunique(dropna=False) > 1:
            raise ValueError("Scorer contract: exactly one trading day per call")
        return frame

    def member_raw(self, frame: pd.DataFrame) -> dict:
        """Raw scores of all six atoms, before any ranking."""
        frame = self._one_day(frame)
        matrix = frame[self.features].to_numpy(dtype=np.float32)
        scores = {}
        for seed, model in zip(self.c1_seeds, self.cat):
            scores[f"c1_{seed}"] = np.asarray(model.predict(matrix), dtype=float).ravel()
        transformed = alpha_tabminput.transform(frame, self.design)
        for seed, model in zip(self.tabm_seeds, self.tabm):
            scores[f"tabm_{seed}"] = np.asarray(
                model.score(transformed), dtype=float).ravel()
        return scores

    def atom_scores(self, frame: pd.DataFrame) -> dict:
        raw = self.member_raw(frame)
        out = {}
        for family, seeds in (("c1", self.c1_seeds), ("tabm", self.tabm_seeds)):
            value = np.zeros(len(frame))
            for seed in seeds:
                value += _rank(raw[f"{family}_{seed}"]) / len(seeds)
            out[family] = value
        return out

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        atoms = self.atom_scores(frame)
        fused = self.weights["c1"] * atoms["c1"] + self.weights["tabm"] * atoms["tabm"]
        return _rank(fused)
'''

INFERENCE = '''"""Official inference entry point (V1 research package)."""

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
        frame = alpha_features.build_day_features(
            data["MarketData"], data["Factor33_winsor"], data["Barra_factor"], day)
        barra = data["Barra_factor"]
        universe = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
        frame = frame.set_index("code").reindex(universe).reset_index()
        score = np.asarray(self.scorer.score(frame), dtype=float).ravel()
        if score.shape[0] != len(frame) or not np.isfinite(score).all():
            raise ValueError("scorer produced invalid scores")
        return pd.DataFrame({"date": as_of_date, "code": frame["code"], "score": score})
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_tabm_npz(seed: int, out: Path) -> dict:
    from safetensors.torch import load_file

    src = S4 / "v1_tabm" / f"tabm_pc_{seed}__{WINDOW}" / "tabm.safetensors"
    sd = load_file(str(src))
    arrays = {
        "W0": sd["backbone.blocks.0.0.weight"].numpy(),
        "r0": sd["backbone.blocks.0.0.r"].numpy(),
        "s0": sd["backbone.blocks.0.0.s"].numpy(),
        "b0": sd["backbone.blocks.0.0.bias"].numpy(),
        "W1": sd["backbone.blocks.1.0.weight"].numpy(),
        "r1": sd["backbone.blocks.1.0.r"].numpy(),
        "s1": sd["backbone.blocks.1.0.s"].numpy(),
        "b1": sd["backbone.blocks.1.0.bias"].numpy(),
        "Wh": sd["head.weight"].numpy(),
        "bh": sd["head.bias"].numpy(),
        "k": np.array([8], dtype=np.float32),
    }
    np.savez(out, **{name: np.ascontiguousarray(value, dtype=np.float32)
                     for name, value in arrays.items()})
    return {"source": str(src), "source_sha": _sha(src), "npz_sha": _sha(out)}


def _bind_atoms(build: Path, members: dict) -> dict:
    """Bind every packaged weight to the atom record that produced it."""
    from catboost import CatBoostRegressor
    from .constants import F0_COLUMNS, G1_COLUMNS

    records = json.loads((FUS / "d1_catboost.json").read_text())["records"]
    lookup = {(record["recipe"], record["window"], int(record["seed"])): record
              for record in records}
    binding = {"c1": {}, "tabm": {}}
    for name in members["c1"]:
        seed = int(name.split("seed")[1].split(".")[0])
        record = lookup[("C1", WINDOW, seed)]
        model = CatBoostRegressor()
        model.load_model(str(build / "model" / name))
        if int(model.tree_count_) != int(record["selected_trees"]):
            raise SystemExit(f"C1 seed {seed}: packaged trees != atom record")
        binding["c1"][name] = {
            "selected_trees": int(record["selected_trees"]),
            "best_checkpoint": int(record["best_checkpoint"]),
            "best_inner_ic": float(record["best_inner_ic"]),
            "source_record": "artifacts/stage3_fusion/d1_catboost.json"}
    for name in members["tabm"]:
        seed = int(name.split("seed")[1].split(".")[0])
        manifest_path = S4 / "v1_tabm" / f"tabm_pc_{seed}__{WINDOW}" / "atom_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        safetensors = S4 / "v1_tabm" / f"tabm_pc_{seed}__{WINDOW}" / "tabm.safetensors"
        if _sha(safetensors) != manifest["hashes"]["model"]:
            raise SystemExit(f"TabM seed {seed}: safetensors hash != atom manifest")
        binding["tabm"][name] = {
            "source_atom_manifest": str(manifest_path),
            "source_atom_manifest_sha": _sha(manifest_path),
            "source_safetensors_sha": _sha(safetensors),
            "best_epoch": int(manifest["best_epoch"]),
            "train_cutoff": int(manifest["train_cutoff"])}
    binding["features_contract"] = list(F0_COLUMNS) + list(G1_COLUMNS)
    return binding


def build() -> dict:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "model").mkdir(parents=True)
    (BUILD / "__init__.py").write_text('"""V1 research package."""\n')
    import sys
    sys.path.insert(0, "src")
    from . import s4_package
    mapping = {"constants.py": "alpha_constants.py", "featurelib.py": "alpha_features.py",
               "dataio.py": "alpha_dataio.py", "labels.py": "alpha_labels.py",
               "panel.py": "alpha_panel.py", "cv.py": "alpha_cv.py"}
    for source, target in mapping.items():
        text = s4_package.rewrite((Path("src/qf3") / source).read_text())
        (BUILD / target).write_text(text)
    shutil.copy(FINAL_BUILD / "alpha_tabminput.py", BUILD / "alpha_tabminput.py")
    (BUILD / "alpha_tabm_static.py").write_text(STATIC)
    (BUILD / "alpha_models.py").write_text(MODELS)
    (BUILD / "inference.py").write_text(INFERENCE)
    shutil.copy(Path("src/qf3/s6_v1_train_src.py"), BUILD / "train.py")
    shutil.copy(Path("src/qf3/s6_v1_train_tabm_src.py"), BUILD / "alpha_train_tabm.py")
    shutil.copy(Path("src/qf3/s4_tabm_wrapper_src.py"), BUILD / "alpha_train_model.py")
    shutil.copy(FINAL_BUILD / "LICENSE.tabm", BUILD / "LICENSE.tabm")
    (BUILD / "TABM_SOURCE.md").write_text(
        "# TabM architecture provenance\n\n"
        "`alpha_train_model.py` reproduces only the components used by this model\n"
        "from the Yandex TabM reference implementation:\n\n"
        "- repository: https://github.com/yandex-research/tabm\n"
        "- file: `tabm.py`, `__version__ = '0.0.3'`\n"
        "- vendored commit: 28e47ae301c92ec37787dde1ce923a0793f405b4\n"
        "- license: Apache-2.0 (see `LICENSE.tabm`)\n\n"
        "The activation is a fixed `nn.ReLU`; there is no dynamic dispatch such as\n"
        "`getattr(nn, activation)`.  Equivalence with the reference implementation\n"
        "(parameter layout, initialization, forward, gradients, optimiser step) is\n"
        "recorded in the project artifact `v1_tabm_equiv.json`.\n")

    members = {"c1": [], "tabm": []}
    hashes = {}
    for seed in SEEDS:
        cat_src = FUS / "models_cat" / f"C1_{seed}__{WINDOW}.cbm"
        cat_dst = BUILD / "model" / f"catboost_c1_seed{seed}.cbm"
        shutil.copy(cat_src, cat_dst)
        members["c1"].append(cat_dst.name)
        hashes[cat_dst.name] = _sha(cat_dst)
        npz_dst = BUILD / "model" / f"tabm_seed{seed}.npz"
        info = export_tabm_npz(seed, npz_dst)
        members["tabm"].append(npz_dst.name)
        hashes[npz_dst.name] = info["npz_sha"]
    designs = {}
    for seed in SEEDS:
        path = S4 / "v1_tabm" / f"tabm_pc_{seed}__{WINDOW}" / "design.json"
        designs[seed] = _sha(path)
    if len(set(designs.values())) != 1:
        raise RuntimeError(f"TabM designs differ across seeds: {designs}")
    shutil.copy(S4 / "v1_tabm" / f"tabm_pc_{SEEDS[0]}__{WINDOW}" / "design.json",
                BUILD / "model" / "tabm_design.json")
    shutil.copy(S4 / "v1_tabm" / f"tabm_pc_{SEEDS[0]}__{WINDOW}" / "tabm_config.json",
                BUILD / "model" / "tabm_config.json")
    from .constants import F0_COLUMNS, G1_COLUMNS
    meta = {"version": "V1_pkg",
            "train_label_cutoff": TRAIN_LABEL_CUTOFF,
            "validation_year": VALIDATION_YEAR,
            "preprocessing_fit_cutoff": TRAIN_LABEL_CUTOFF,
            "features": list(F0_COLUMNS) + list(G1_COLUMNS),
            "weights": WEIGHTS, "members": members,
            "fusion_graph": FUSION_GRAPH,
            "train_cutoff_note": "outer fit uses label_end_date <= 2020-12-31",
            "runtime": "NumPy inference for TabM; no torch required at run time",
            "member_hashes": hashes, "design_hash": designs[SEEDS[0]],
            "source_atoms": _bind_atoms(BUILD, members)}
    (BUILD / "model" / "meta.json").write_text(json.dumps(meta, indent=2))
    import numpy, pandas, pyarrow, catboost, torch, safetensors
    (BUILD / "requirements.txt").write_text(
        f"numpy=={numpy.__version__}\npandas=={pandas.__version__}\n"
        f"pyarrow=={pyarrow.__version__}\ncatboost=={catboost.__version__}\n")
    (BUILD / "train_requirements.txt").write_text(
        (BUILD / "requirements.txt").read_text()
        + f"torch=={torch.__version__}\nsafetensors=={safetensors.__version__}\n"
        "rtdl_num_embeddings==0.0.12\n")
    from . import s6_v1_manifest_check
    result = s6_v1_manifest_check.check_package(
        meta, BUILD / "model",
        expected_seeds=SEEDS, expected_cutoff=TRAIN_LABEL_CUTOFF,
        expected_year=VALIDATION_YEAR, expected_graph=FUSION_GRAPH)
    if not all(result.values()):
        raise SystemExit(f"package manifest check failed: {result}")
    return {"build": str(BUILD), "members": {k: len(v) for k, v in members.items()},
            "design_identical_across_seeds": True, "manifest_checks": result}


def zip_package(zip_path: Path) -> dict:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("submission") / path.relative_to(BUILD))
    with zipfile.ZipFile(zip_path) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
        names = sorted(archive.namelist())
    return {"zip": str(zip_path), "bytes": zip_path.stat().st_size,
            "uncompressed_bytes": uncompressed, "files": len(names),
            "sha256": _sha(zip_path)}


if __name__ == "__main__":
    print(build())
    print(zip_package(Path("releases/stage4_verified_ic/V1_pkg.zip")))
