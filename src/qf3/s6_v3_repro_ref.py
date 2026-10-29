"""V3 final-compliance P0-1 step 1/2: record the read-only reference evidence.

Binds the original submission ZIP (SHA-256 + CRC), the public data inputs and
their hashes, and the interpreter/dependency/thread/machine facts used for the
independent ``train.py`` reproduction run.  Nothing here trains or modifies the
submitted package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(root: Path) -> dict:
    return {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": _sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def _data_files(data_dir: Path) -> dict:
    files = sorted(p for p in data_dir.rglob("*.parquet"))
    return {"count": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "sha256_by_relpath": {str(p.relative_to(data_dir)): _sha(p) for p in files}}


def _environment() -> dict:
    import numpy
    import pandas
    import pyarrow
    import catboost
    import torch
    import safetensors
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                     "pyarrow": pyarrow.__version__, "catboost": catboost.__version__,
                     "torch": torch.__version__, "safetensors": safetensors.__version__},
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--extract-to", required=True)
    parser.add_argument("--ref-copy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        crc_error = archive.testzip()
        names = sorted(archive.namelist())

    extract_to = Path(args.extract_to)
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_to)
    submission = extract_to / "submission"
    if not (submission / "inference.py").is_file() or not (submission / "model").is_dir():
        raise SystemExit("extracted package is missing inference.py or model/")

    ref_copy = Path(args.ref_copy)
    ref_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, ref_copy)
    Path(ref_copy).chmod(0o444)

    report = {
        "original_zip": {"path": str(zip_path), "bytes": zip_path.stat().st_size,
                         "sha256": _sha(zip_path), "crc_error": crc_error,
                         "entries": len(names)},
        "reference_copy": {"path": str(ref_copy.resolve()), "sha256": _sha(ref_copy),
                           "same_as_original": _sha(ref_copy) == _sha(zip_path)},
        "extract_dir": str(extract_to.resolve()),
        "package_files": _package_files(submission),
        "data_inputs": {"dir": str(Path(args.data_dir).resolve()), **_data_files(Path(args.data_dir))},
        "environment": _environment(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"zip_sha256": report["original_zip"]["sha256"],
                      "crc_error": crc_error, "package_files": len(report["package_files"]),
                      "data_files": report["data_inputs"]["count"],
                      "python": report["environment"]["versions"]}, indent=2))


if __name__ == "__main__":
    main()
