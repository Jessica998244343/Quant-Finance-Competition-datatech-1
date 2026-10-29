"""V0 §5.2: explicit static NumPy forward for the frozen TabM weights + equivalence proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
SOURCE = Path("releases/stage3_fusion/build/FINAL_v1/model")


@dataclass
class TabmStatic:
    """TabM (BatchEnsemble MLP, k members) evaluated with plain NumPy FP32.

    Architecture copied explicitly from the frozen weights, not from a generic
    factory: 2 blocks of (LinearBatchEnsemble -> ReLU -> Dropout(eval: off))
    followed by a LinearEnsemble head; member outputs are averaged.
    """

    W0: np.ndarray      # (d_block, d_in)
    r0: np.ndarray      # (k, d_in)
    s0: np.ndarray      # (k, d_block)
    b0: np.ndarray      # (k, d_block)
    W1: np.ndarray      # (d_block, d_block)
    r1: np.ndarray      # (k, d_block)
    s1: np.ndarray      # (k, d_block)
    b1: np.ndarray      # (k, d_block)
    Wh: np.ndarray      # (k, d_block, 1)
    bh: np.ndarray      # (k, 1)
    k: int

    @classmethod
    def from_npz(cls, path: Path) -> "TabmStatic":
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
        return cls(k=k, **{name: arrays[name] for name in (
            "W0", "r0", "s0", "b0", "W1", "r1", "s1", "b1", "Wh", "bh")})

    def layer_outputs(self, x: np.ndarray) -> dict:
        """Return every intermediate tensor so equivalence can be checked layer-wise."""
        h = np.asarray(x, dtype=np.float32)[:, None, :] * self.r0[None, :, :]
        h = h @ self.W0.T
        h = h * self.s0[None, :, :] + self.b0[None, :, :]
        h = np.maximum(h, 0.0)                     # ReLU
        block0 = h
        h = h * self.r1[None, :, :]
        h = h @ self.W1.T
        h = h * self.s1[None, :, :] + self.b1[None, :, :]
        h = np.maximum(h, 0.0)                     # ReLU
        block1 = h
        out = np.einsum("nki,kio->nko", h, self.Wh) + self.bh[None, :, :]
        return {"block0": block0, "block1": block1, "head": out[:, :, 0]}

    def member_outputs(self, x: np.ndarray) -> np.ndarray:
        """Return per-member raw scores, shape (n, k)."""
        return self.layer_outputs(x)["head"]

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.member_outputs(x).mean(axis=1)


def export_from_safetensors(safetensors_path: Path, out_npz: Path,
                            out_schema: Path) -> dict:
    from safetensors.torch import load_file

    sd = load_file(str(safetensors_path))
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
    }
    k = int(arrays["r0"].shape[0])
    arrays["k"] = np.array([k], dtype=np.float32)
    arrays = {name: np.ascontiguousarray(value, dtype=np.float32)
              for name, value in arrays.items()}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)
    schema = {
        "format": "npz", "allow_pickle": False, "dtype": "float32",
        "architecture": "TabM BatchEnsemble: 2 x (LinearBatchEnsemble, ReLU, "
                        "Dropout off at eval) + LinearEnsemble head, member mean",
        "source_safetensors": str(safetensors_path),
        "source_safetensors_sha256": hashlib.sha256(
            Path(safetensors_path).read_bytes()).hexdigest(),
        "npz_sha256": hashlib.sha256(Path(out_npz).read_bytes()).hexdigest(),
        "exporter_source": str(Path(__file__)),
        "exporter_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "source_note": "frozen FINAL_v1 full-data TabM weights (Yandex TabM, Apache-2.0)",
        "params": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
        "forward_ops": {
            "linear_batch_ensemble": "h = ((x * r) @ W.T) * s + b",
            "activation": "ReLU",
            "head": "einsum('nki,kio->nko', h, Wh) + bh",
            "aggregation": "mean over k members",
        },
        "no_dynamic_dispatch": True,
    }
    out_schema.write_text(json.dumps(schema, indent=2))
    return schema


def equivalence(safetensors_path: Path, npz_path: Path, cases: dict) -> dict:
    """Compare the static NumPy forward against the PyTorch reference."""
    import torch
    import sys
    sys.path.insert(0, "src")
    from qf3.s4_tabm import TabMWrapper

    static = TabmStatic.from_npz(npz_path)
    config = json.loads((SOURCE / "tabm_config.json").read_text())
    model = TabMWrapper(**config)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(safetensors_path)))
    model.eval()

    report = {}
    for name, matrix in cases.items():
        x = np.ascontiguousarray(matrix, dtype=np.float32)
        with torch.no_grad():
            ref_members = model(torch.from_numpy(x)).numpy()
            ref = ref_members.mean(axis=1)
        got_members = static.member_outputs(x)
        got = got_members.mean(axis=1)
        err = np.abs(got - ref)
        denom = np.maximum(np.abs(ref), 1e-12)
        report[name] = {
            "rows": int(x.shape[0]), "members": int(got_members.shape[1]),
            "max_abs_err": float(err.max()),
            "p99_abs_err": float(np.quantile(err, 0.99)),
            "max_rel_err": float((err / denom).max()),
            "member_max_abs_err": float(np.abs(got_members - ref_members).max()),
            "rank_identical": bool(np.array_equal(
                pd.Series(got).rank(method="average").to_numpy(),
                pd.Series(ref).rank(method="average").to_numpy())),
            "all_finite": bool(np.isfinite(got).all()),
        }
    return report
