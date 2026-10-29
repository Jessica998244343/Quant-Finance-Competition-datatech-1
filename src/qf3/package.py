"""Train the frozen final models and assemble the release packages."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_submission, trainers
from .constants import F0_COLUMNS, G1_COLUMNS

FEATURES_F0 = list(F0_COLUMNS)
FEATURES_F0G1 = list(F0_COLUMNS) + list(G1_COLUMNS)

RECIPES = {
    "baseline_v1": {
        "version": "baseline_v1",
        "seed": 20261005,
        "features": FEATURES_F0,
        "members": [
            {"name": "lgbm_b0", "type": "lgbm", "capacity": "B0", "params": {},
             "rounds": 270, "weight": 1.0, "features": FEATURES_F0},
        ],
    },
    "strong_baseline_v1": {
        "version": "strong_baseline_v1",
        "seed": 20261005,
        "features": FEATURES_F0G1,
        "members": [
            {"name": "lgbm_p5", "type": "lgbm", "capacity": "P5", "params": {},
             "rounds": 418, "weight": 0.75, "features": FEATURES_F0G1},
            {"name": "catboost_d8_l2_15", "type": "catboost", "capacity": None,
             "params": {"depth": 8, "l2_leaf_reg": 15.0}, "rounds": 310,
             "weight": 0.25, "features": FEATURES_F0G1},
        ],
    },
}


def train_member(member: dict, panel: pd.DataFrame, seed: int, threads: int = 8):
    features = member["features"]
    finite = panel["future_return"].notna().to_numpy()
    if member["type"] == "lgbm":
        import lightgbm as lgb
        from .cv import make_target
        y, w, _ = make_target(panel)
        x = panel[features].to_numpy(dtype=np.float32)
        base = dict(trainers.B0_PARAMS) if member.get("capacity") == "B0" else \
            dict(trainers.CAPACITY_GRID[member["capacity"]])
        base.update(member.get("params", {}))
        params = trainers.lgb_params(base, seed=seed, threads=threads)
        rounds = int(member["rounds"])
        params.pop("_num_rounds")
        dataset = lgb.Dataset(x[finite], label=y[finite], weight=w[finite],
                              feature_name=features)
        return lgb.train(params, dataset, num_boost_round=rounds)
    if member["type"] == "catboost":
        from catboost import CatBoostRegressor, Pool
        from .cv import make_target
        y, w, _ = make_target(panel)
        x = panel[features].to_numpy(dtype=np.float32)
        settings = {
            "loss_function": "RMSE", "iterations": int(member["rounds"]),
            "learning_rate": 0.03, "border_count": 128, "random_strength": 1.0,
            "bootstrap_type": "Bernoulli", "subsample": 0.8, "thread_count": threads,
            "random_seed": seed, "nan_mode": "Min", "verbose": False,
            "allow_writing_files": False,
        }
        settings.update(member.get("params", {}))
        model = CatBoostRegressor(**settings)
        model.fit(Pool(x[finite], label=y[finite], weight=w[finite],
                       feature_names=features))
        return model
    raise ValueError(member["type"])


def build_package(recipe_name: str, panel: pd.DataFrame, out_dir: Path,
                  threads: int = 8) -> dict:
    recipe = RECIPES[recipe_name]
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    info = build_submission.assemble(out_dir)
    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=2))
    import numpy, pandas, pyarrow, scipy, sklearn, lightgbm, catboost
    versions = {"numpy": numpy.__version__, "pandas": pandas.__version__,
                "pyarrow": pyarrow.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                "catboost": catboost.__version__}
    build_submission.write_requirements(out_dir, versions)
    model_dir = out_dir / "model"
    members_meta = []
    for member in recipe["members"]:
        print(f"  training {member['name']} ...", flush=True)
        model = train_member(member, panel, recipe["seed"], threads)
        filename = f"{member['name']}.{'txt' if member['type'] == 'lgbm' else 'cbm'}"
        path = model_dir / filename
        if member["type"] == "lgbm":
            model.save_model(str(path))
        else:
            model.save_model(str(path), format="cbm")
        members_meta.append({k: v for k, v in member.items() if k != "features"}
                            | {"file": filename})
    meta = {
        "version": recipe["version"],
        "features": recipe["features"],
        "members": members_meta,
        "seed": recipe["seed"],
        "trained_on": "2016-2022 official data, all valid labeled samples",
        "label": "O2O adjusted open H=1, daily cross-sectional average-tied rank",
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    info["model_files"] = sorted(p.name for p in model_dir.iterdir())
    return info


def zip_package(package_dir: Path, zip_path: Path) -> dict:
    package_dir = Path(package_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path("submission") / path.relative_to(package_dir))
    import hashlib
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return {"zip": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": digest}


def main() -> None:
    panel = pd.read_parquet("artifacts/beginning/cache/panel.parquet")
    releases = Path("releases")
    reports = {}
    for recipe_name, target in (("baseline_v1", "releases/build/baseline_v1"),
                                ("strong_baseline_v1", "releases/build/strong_baseline_v1")):
        print(f"building {recipe_name}", flush=True)
        info = build_package(recipe_name, panel, Path(target))
        zip_info = zip_package(Path(target), releases / f"{recipe_name}.zip")
        reports[recipe_name] = {**info, **zip_info}
        print(json.dumps({**info, **zip_info}, indent=2))
    (releases / "build_report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
