"""Re-run the packaged train.py and compare against the shipped model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import acceptance
from .constants import BARRA_TABLE, FACTOR_TABLE, MARKET_TABLE

OUT = Path("artifacts/beginning/cpu_acceptance")


def compare_models(submission: Path, repro_dir: Path, data_dir: str,
                   dates=(20160630, 20190603, 20210315)) -> dict:
    import importlib
    module = acceptance.load_inference(submission)
    scorer_mod = importlib.import_module(module.__name__ + ".alpha_model")
    meta = json.loads((submission / "model" / "meta.json").read_text())
    shipped = scorer_mod.Scorer(submission / "model", meta)
    retrained = scorer_mod.Scorer(repro_dir, meta)
    tables = acceptance.dataio.load_all(data_dir)
    from . import featurelib
    diffs = {}
    for day in dates:
        window = acceptance.window_for(tables, day)
        frame = featurelib.build_day_features(window[MARKET_TABLE], window[FACTOR_TABLE],
                                              window[BARRA_TABLE], day)
        x = frame[meta["features"]].to_numpy(dtype=np.float32)
        a = shipped.score(x)
        b = retrained.score(x)
        diffs[str(day)] = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    return diffs


def main() -> None:
    data_dir = "Source/Data"
    zip_path = Path("releases/strong_baseline_v1.zip")
    work = OUT / "repro"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work)
    submission = work / "submission"
    repro_dir = (OUT / "repro_model_strong").resolve()
    result = subprocess.run(
        [sys.executable, "train.py", "--data-dir", str(Path(data_dir).resolve()),
         "--out", str(repro_dir), "--threads", "8"],
        cwd=submission, capture_output=True, text=True)
    report = {"returncode": result.returncode, "stdout_tail": result.stdout[-2000:],
              "stderr_tail": result.stderr[-2000:]}
    if result.returncode == 0:
        report["max_abs_prediction_diff"] = compare_models(submission, repro_dir, data_dir)
        report["files"] = sorted(p.name for p in repro_dir.iterdir())
    (OUT / "reproduction_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
