"""V0 review command (R4/R5): reproducible TabM export equivalence with hard verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path("artifacts/stage4_verified_ic")
PKG = Path("releases/stage3_fusion/build/FINAL_v1")
SRC_SAFETENSORS = PKG / "model" / "tabm.safetensors"
DESIGN = PKG / "model" / "tabm_design.json"
NPZ = S4 / "tabm_static_fp32.npz"
SEED = 20261005
ATOL, RTOL = 1e-5, 1e-4
CASE_DAYS = (20190603, 20190604, 20191202)
MISSING_FRACTION = 0.2
MISSING_COLUMNS = ("ret_1", "intraday", "turn")

# 必需场景集合（冻结）：缺一项、空集或替换都视为验收失败
REQUIRED_CASES = (
    "real_20190603", "real_20190604", "real_20191202",
    "raw_nan_through_preprocessing", "transform_perturb_zeroed",
    "extreme_clip", "constant_rows", "single_row", "small_pool_37",
)
RANK_CHECKED_CASES = ("real_20190603", "real_20190604", "real_20191202", "small_pool_37")
LAYERS = ("block0", "block1", "head", "mean")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_transform():
    """Load the frozen preprocessing exactly as the package ships it."""
    import importlib.util
    name = "_v0_pkg"
    spec = importlib.util.spec_from_file_location(
        name, PKG / "alpha_tabminput.py", submodule_search_locations=[str(PKG)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_cases() -> tuple[dict, dict]:
    """Fixed case set: real days, raw-NaN through frozen preprocessing, degenerate pools."""
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    design = json.loads(DESIGN.read_text())
    transform = _package_transform().transform
    rng = np.random.default_rng(SEED)
    cases, meta = {}, {}

    def real_frame(day: int) -> pd.DataFrame:
        return panel.loc[panel["date"] == day].sort_values("code").reset_index(drop=True)

    for day in CASE_DAYS:
        frame = real_frame(day)
        matrix = transform(frame, design).astype(np.float32)
        cases[f"real_{day}"] = matrix
        meta[f"real_{day}"] = {"source": f"panel date {day}", "rows": int(len(frame)),
                               "note": "frozen preprocessing, real features"}

    # raw features set to NaN, then run the FROZEN preprocessing (真 NaN 场景)
    frame = real_frame(CASE_DAYS[0])
    damaged = frame.copy()
    for column in MISSING_COLUMNS:
        damaged.loc[rng.random(len(damaged)) < MISSING_FRACTION, column] = np.nan
    cases["raw_nan_through_preprocessing"] = transform(damaged, design).astype(np.float32)
    meta["raw_nan_through_preprocessing"] = {
        "source": f"panel date {CASE_DAYS[0]}",
        "naN_columns": list(MISSING_COLUMNS), "fraction": MISSING_FRACTION,
        "rows": int(len(damaged)), "note": "真实 NaN 经过冻结 median 填补 + mask"}

    base = cases[f"real_{CASE_DAYS[0]}"]
    cases["transform_perturb_zeroed"] = np.where(
        rng.random(base.shape) < MISSING_FRACTION, 0.0, base).astype(np.float32)
    meta["transform_perturb_zeroed"] = {"source": "post-transform zeroing",
                                        "note": "数值扰动，非真实缺失"}
    cases["extreme_clip"] = np.clip(base * 50.0, -5.0, 5.0).astype(np.float32)
    meta["extreme_clip"] = {"source": "base x50 clipped to [-5,5]"}
    cases["constant_rows"] = np.tile(base[:1], (len(base), 1)).astype(np.float32)
    meta["constant_rows"] = {"source": "single row replicated"}
    cases["single_row"] = base[:1].copy()
    cases["small_pool_37"] = base[:37].copy()
    return cases, meta


def torch_reference(cases: dict) -> dict:
    """Per-layer reference from the real nn.Module via forward hooks (not a re-derivation)."""
    import torch
    sys.path.insert(0, "src")
    from safetensors.torch import load_file
    from qf3.s4_tabm import TabMWrapper

    config = json.loads((PKG / "model" / "tabm_config.json").read_text())
    model = TabMWrapper(**config)
    model.load_state_dict(load_file(str(SRC_SAFETENSORS)))
    model.eval()
    captured: dict = {}
    handles = [
        model.backbone.blocks[0].register_forward_hook(
            lambda m, i, o: captured.__setitem__("block0", o.detach().numpy())),
        model.backbone.blocks[1].register_forward_hook(
            lambda m, i, o: captured.__setitem__("block1", o.detach().numpy())),
    ]
    out = {}
    with torch.no_grad():
        for name, matrix in cases.items():
            captured.clear()
            members = model(torch.from_numpy(matrix)).numpy()
            out[name] = {"block0": captured["block0"], "block1": captured["block1"],
                         "head": members, "mean": members.mean(axis=1)}
    for handle in handles:
        handle.remove()
    return out


def _failures(msg: str) -> dict:
    return {"passed": False, "errors": [msg]}


def compare(cases: dict, meta: dict, *, enforce_required: bool = True) -> dict:
    """Hard verdict. With ``enforce_required`` the case set must equal REQUIRED_CASES
    exactly; empty/missing/substituted case sets can never pass."""
    from qf3.s5_tabm_export import TabmStatic

    errors: list[str] = []
    if enforce_required:
        missing = [name for name in REQUIRED_CASES if name not in cases]
        extra = [name for name in cases if name not in REQUIRED_CASES]
        if missing:
            errors.append(f"missing required cases: {missing}")
        if extra:
            errors.append(f"unexpected cases: {extra}")
        if not cases:
            errors.append("empty case set")

    static = TabmStatic.from_npz(NPZ)
    if errors:
        return {"criterion": f"|a-b| <= {ATOL} + {RTOL}*|ref|", "cases": {},
                "tolerance_failures": 0, "coverage_ok": False,
                "shape_failures": 0, "finite_failures": 0, "rank_failures": 0,
                "errors": errors, "passed": False}

    # 输入形状先于任何前向调用校验，避免把非法输入交给参考实现
    for name, matrix in cases.items():
        if matrix.ndim != 2 or matrix.size == 0 or matrix.shape[1] != static.W0.shape[1]:
            errors.append(
                f"{name}: invalid input shape {matrix.shape}; expected (n, {static.W0.shape[1]})")
    if errors:
        return {"criterion": f"|a-b| <= {ATOL} + {RTOL}*|ref|", "cases": {},
                "tolerance_failures": 0, "coverage_ok": True, "n_cases": len(cases),
                "shape_failures": len(errors), "finite_failures": 0, "rank_failures": 0,
                "errors": errors, "passed": False}

    reference = torch_reference(cases)
    results, failures = {}, 0
    shape_failures = finite_failures = rank_failures = 0
    for name, matrix in cases.items():
        if matrix.size == 0 or matrix.ndim != 2:
            errors.append(f"{name}: invalid input shape {matrix.shape}")
            shape_failures += 1
            continue
        got = static.layer_outputs(matrix)
        ref = reference[name]
        record = {"meta": meta.get(name, {}), "rows": int(matrix.shape[0]),
                  "input_shape": list(matrix.shape), "shape_ok": True, "finite": True}
        # 形状必须先于数值比较，禁止广播把 shape 错误掩盖成通过
        for layer in LAYERS:
            a = got["head"].mean(axis=1) if layer == "mean" else got[layer]
            b = ref["mean"] if layer == "mean" else ref[layer]
            if a.shape != b.shape:
                record["shape_ok"] = False
                shape_failures += 1
                errors.append(f"{name}/{layer}: shape {a.shape} != {b.shape}")
        if not record["shape_ok"]:
            results[name] = record
            continue
        if not (np.isfinite(got["head"]).all() and np.isfinite(ref["head"]).all()):
            record["finite"] = False
            finite_failures += 1
            errors.append(f"{name}: non-finite values produced")
        for layer in LAYERS:
            if layer == "mean":
                a = got["head"].mean(axis=1)
                b = ref["mean"]
            elif layer == "head":
                a = got["head"]
                b = ref["head"]
            else:
                a, b = got[layer], ref[layer]
            err = np.abs(a - b)
            tol = ATOL + RTOL * np.abs(b)
            ok = bool((err <= tol).all())
            failures += 0 if ok else 1
            record[layer] = {
                "max_abs_err": float(err.max()),
                "p99_abs_err": float(np.quantile(err, 0.99)),
                "max_rel_err": float((err / np.maximum(np.abs(b), 1e-12)).max()),
                "combined_tolerance_pass": ok,
                "n_over_tol": int((err > tol).sum()),
            }
        if name.startswith(("real_", "small_pool")):
            rank_ok = bool(np.array_equal(
                pd.Series(got["head"].mean(axis=1)).rank(method="average").to_numpy(),
                pd.Series(ref["mean"]).rank(method="average").to_numpy()))
            record["rank_identical_mean"] = rank_ok
            if name in RANK_CHECKED_CASES and not rank_ok:
                rank_failures += 1
                errors.append(f"{name}: daily rank differs")
        record["all_finite"] = bool(np.isfinite(got["head"]).all())
        results[name] = record
    passed = (failures == 0 and not errors and shape_failures == 0
              and finite_failures == 0 and rank_failures == 0)
    return {"criterion": f"|a-b| <= {ATOL} + {RTOL}*|ref|",
            "cases": results, "coverage_ok": True, "n_cases": len(cases),
            "required_cases": list(REQUIRED_CASES),
            "tolerance_failures": failures, "shape_failures": shape_failures,
            "finite_failures": finite_failures, "rank_failures": rank_failures,
            "errors": errors, "passed": passed}


def negative_fixtures() -> dict:
    """Prove the verifier cannot pass on empty/missing/short/broken input."""
    cases, meta = build_cases()
    out = {}

    def verdict(label, c, m):
        """An exception during verification is itself a rejection."""
        try:
            return compare(c, m, enforce_required=True)["passed"]
        except Exception as exc:  # noqa: BLE001
            return "rejected_by_exception:" + type(exc).__name__

    out["empty_case_set"] = verdict("empty", {}, {})
    subset = {k: v for k, v in cases.items() if k != "small_pool_37"}
    out["missing_one_case"] = verdict("missing", subset, meta)
    out["substituted_case"] = verdict(
        "substituted",
        {**{k: v for k, v in cases.items() if k != "single_row"},
         "made_up_case": cases["single_row"]}, meta)
    zero_width = {name: (value[:, :0] if name == "real_20190603" else value)
                  for name, value in cases.items()}
    out["zero_width_input"] = verdict("zero_width", zero_width, meta)
    out["all_required_ok"] = verdict("positive", cases, meta)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(S4 / "v0_rev1_acceptance.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out} (pass --force to replace)")
    cases, meta = build_cases()
    report = compare(cases, meta)
    report["negative_fixtures"] = negative_fixtures()
    if report["negative_fixtures"]["all_required_ok"] is not True:
        report["passed"] = False
        report["errors"].append("positive case set did not pass")
    for name, passed in report["negative_fixtures"].items():
        if name == "all_required_ok":
            if passed is not True:
                report["passed"] = False
            continue
        if passed is not False:
            report["passed"] = False
            report["errors"].append(f"negative fixture {name} was not rejected")
    report["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "torch": __import__("torch").__version__,
        "seed": SEED, "cuda_available": bool(__import__("torch").cuda.is_available()),
        "acceptance_script": "src/qf3/s5_v0_acceptance.py",
    }
    report["hashes"] = {
        "source_safetensors": _sha(SRC_SAFETENSORS),
        "npz": _sha(NPZ),
        "exporter_source": _sha(Path("src/qf3/s5_tabm_export.py")),
        "design": _sha(DESIGN),
        "acceptance_script": _sha(Path("src/qf3/s5_v0_acceptance.py")),
        "preprocessing_code": _sha(PKG / "alpha_tabminput.py"),
        "model_config": _sha(PKG / "model" / "tabm_config.json"),
        "reference_forward_code": _sha(Path("src/qf3/s4_tabm.py")),
        "input_panel": _sha(Path("artifacts/stage2_ic_ir/panel_platform.parquet")),
    }
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    for name, record in report["cases"].items():
        print(f"  {name:32s} mean max_abs {record['mean']['max_abs_err']:.2e} "
              f"tol_pass {record['mean']['combined_tolerance_pass']} "
              f"rank_ok {record.get('rank_identical_mean', 'n/a')}")
    assert report["passed"], "V0 acceptance failed: tolerance violations found"


if __name__ == "__main__":
    main()
