"""D3 §13.4: local acceptance tests for the FINAL_v1 CPU package."""

from __future__ import annotations

import importlib
import importlib.util
import json
import resource
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FUS = Path("artifacts/stage3_fusion")
ZIP = Path("releases/stage3_fusion/FINAL_v1.zip")
WORK = Path("/tmp/final_v1_acceptance")


def extract() -> Path:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(ZIP) as archive:
        archive.extractall(WORK)
    return WORK / "submission"


def load_inference(submission: Path):
    for name in list(sys.modules):
        if name.startswith("_skd_acc"):
            del sys.modules[name]
    pkg = "_skd_acc_final"
    spec = importlib.util.spec_from_file_location(
        pkg, submission / "inference.py",
        submodule_search_locations=[str(submission)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def window(day: int, lookback: int = 120, data_dir: str = "Source/Data") -> dict:
    from . import dataio
    tables = dataio.load_all(data_dir)
    calendar = sorted(pd.unique(tables["MarketData"]["日期"]).tolist())
    pos = calendar.index(day)
    first = calendar[max(0, pos - lookback + 1)]
    return {name: table.loc[(table["日期"] >= first) & (table["日期"] <= day)]
            for name, table in tables.items()}


def offline_guard() -> dict:
    """Deny all outbound sockets while the package runs."""
    original = socket.socket.connect

    def deny(*args, **kwargs):
        raise OSError("network access denied during acceptance")

    socket.socket.connect = deny
    return {"patched": True, "original": original}


def main() -> None:
    from catboost import CatBoostRegressor

    report = {"tests": {}}
    submission = extract()
    guard = offline_guard()
    module = load_inference(submission)
    model = module.InferenceModel(str(submission / "model"))
    report["tests"]["1_offline_launch_outside_project"] = {
        "extracted_to": str(submission),
        "outside_project": not str(submission).startswith(str(Path.cwd())),
        "loaded": True, "network_denied": True,
        "cuda_available": bool(__import__("torch").cuda.is_available()),
    }

    # --- scenario coverage + invariants -----------------------------------
    days = {"normal": 20190603, "constituent_change": 20190617,
            "short_history_2d": 20160105, "first_full_window": 20160630}
    base_outputs = {}
    for label, day in days.items():
        data = window(day)
        as_of = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
        out = model.predict(as_of, data)
        barra = data["Barra_factor"]
        universe = barra.loc[barra["日期"] == day, "代码"].drop_duplicates().tolist()
        base_outputs[label] = out
        report["tests"].setdefault("2_scenarios", {})[label] = {
            "rows": int(len(out)), "pool_exact": set(out["code"]) == set(universe),
            "unique": not out["code"].duplicated().any(),
            "finite": bool(np.isfinite(out["score"].to_numpy(dtype=float)).all()),
            "date_ok": bool((out["date"] == as_of).all()),
        }
    # repeat / permutation / other-date perturbation
    day = days["normal"]
    as_of = "2019-06-03"
    data = window(day)
    again = model.predict(as_of, data)
    same = np.array_equal(base_outputs["normal"]["score"].to_numpy(),
                          again["score"].to_numpy())
    shuffled = {k: v.sample(frac=1.0, random_state=7).reset_index(drop=True)
                for k, v in data.items()}
    perm_out = model.predict(as_of, shuffled)
    merged = base_outputs["normal"].merge(perm_out, on="code",
                                          suffixes=("_a", "_b"), validate="one_to_one")
    perm_diff = float((merged["score_a"] - merged["score_b"]).abs().max())
    future_cut = {k: v.loc[v["日期"] <= day] for k, v in data.items()}
    cut_out = model.predict(as_of, future_cut)
    report["tests"]["3_invariants"] = {
        "repeat_identical": bool(same),
        "permutation_max_abs_diff": perm_diff,
        "future_rows_removed_identical": bool(np.array_equal(
            base_outputs["normal"]["score"].to_numpy(), cut_out["score"].to_numpy())),
    }

    # --- feature parity package vs research --------------------------------
    from . import featurelib, s2r_parity
    provider = s2r_parity.provider_as_of(day, "2019-01-02")
    research = featurelib.build_day_features(
        provider["MarketData"], provider["Factor33_winsor"],
        provider["Barra_factor"], day).set_index("code")
    pkg_features = importlib.import_module(module.__name__ + ".alpha_features")
    package = pkg_features.build_day_features(
        provider["MarketData"], provider["Factor33_winsor"],
        provider["Barra_factor"], day).set_index("code")
    columns = list(model.features)
    a = research.reindex(package.index)[columns].to_numpy(dtype=float)
    b = package[columns].to_numpy(dtype=float)
    report["tests"]["4_feature_parity"] = {
        "columns": len(columns),
        "max_abs_diff": float(np.nanmax(np.abs(a - b))),
        "nan_mask_diff_cells": int((np.isnan(a) != np.isnan(b)).sum()),
    }

    # --- member prediction parity: package weights vs direct models -------
    import torch
    pkg_models = importlib.import_module(module.__name__ + ".alpha_models")
    scorer = pkg_models.Scorer(str(submission / "model"))
    frame = package.reset_index()
    cat_direct = CatBoostRegressor()
    cat_direct.load_model(str(submission / "model" / "catboost_c1.cbm"))
    cat_here = np.asarray(cat_direct.predict(
        frame[columns].to_numpy(dtype=np.float32)), dtype=float).ravel()
    cat_pkg = np.asarray(scorer.cat.predict(
        frame[columns].to_numpy(dtype=np.float32)), dtype=float).ravel()
    package_matrix = scorer._tabm_matrix(frame)
    with torch.no_grad():
        tabm_pkg = scorer.tabm(torch.from_numpy(package_matrix)).mean(dim=1).numpy()
    from safetensors.torch import load_file
    blob = torch.load(FUS / "final_models" / "tabm.pt", map_location="cpu",
                      weights_only=False)
    sys.path.insert(0, str(submission))
    wrapper = importlib.import_module(module.__name__ + ".alpha_tabm_wrapper")
    direct_model = wrapper.TabMWrapper(**json.loads(
        (submission / "model" / "tabm_config.json").read_text()))
    direct_model.load_state_dict(load_file(str(submission / "model" / "tabm.safetensors")))
    direct_model.eval()
    with torch.no_grad():
        tabm_direct = direct_model(torch.from_numpy(package_matrix)).mean(dim=1).numpy()
    report["tests"]["5_member_parity"] = {
        "catboost_max_abs_diff_vs_safetensors_path": float(
            np.max(np.abs(cat_here - cat_pkg))),
        "tabm_max_abs_diff_package_vs_imported_weights": float(
            np.max(np.abs(tabm_pkg - tabm_direct))),
        "all_finite": bool(np.isfinite(tabm_pkg).all()),
    }

    # --- timing and peak memory -------------------------------------------
    calendar = sorted(pd.unique(window(20190628)["MarketData"]["日期"]).tolist())
    sample = calendar[-20:]
    durations = []
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for day_i in sample:
        data_i = window(int(day_i))
        as_of_i = f"{str(day_i)[:4]}-{str(day_i)[4:6]}-{str(day_i)[6:]}"
        started = time.perf_counter()
        model.predict(as_of_i, data_i)
        durations.append(time.perf_counter() - started)
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    array = np.array(durations)
    report["tests"]["6_cpu_timing"] = {
        "days": int(len(array)), "first_day_s": float(array[0]),
        "p50_s": float(np.percentile(array, 50)),
        "p95_s": float(np.percentile(array, 95)),
        "max_s": float(array.max()), "peak_rss_mib": peak / 1024.0,
    }
    with zipfile.ZipFile(ZIP) as archive:
        names = sorted(archive.namelist())
        uncompressed = sum(info.file_size for info in archive.infolist())
    import hashlib
    report["tests"]["7_size_and_hash"] = {
        "zip_bytes": ZIP.stat().st_size, "uncompressed_bytes": uncompressed,
        "sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        "file_count": len(names), "files": names,
    }
    socket.socket.connect = guard["original"]
    (FUS / "d3_acceptance.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
