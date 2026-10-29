"""REV1: key-safe allocation comparison between backtest runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataio
from .constants import BARRA_TABLE

REV = Path("artifacts/stage2_ic_ir/a_revision1")


def _load(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / f"{name}.csv")


def allocation_delta(new_dir: Path, ref_dir: Path, label: str,
                     data_dir: str = "Source/Data") -> dict:
    left = _load(new_dir, "target_weights")
    right = _load(ref_dir, "target_weights")
    for frame in (left, right):
        frame["signal_date"] = frame["signal_date"].astype(str)
        frame["code"] = frame["code"].astype(str)
    merged = left.merge(right, on=["signal_date", "code"], how="outer",
                        suffixes=("_new", "_ref"), indicator=True)
    both = merged.loc[merged["_merge"] == "both"]
    only_new = merged.loc[merged["_merge"] == "left_only", ["signal_date", "code"]]
    only_ref = merged.loc[merged["_merge"] == "right_only", ["signal_date", "code"]]
    dw = (both["target_weight_new"] - both["target_weight_ref"]).to_numpy()

    # industry / style active-exposure comparison on common dates
    industry = pd.read_csv(Path(data_dir) / "HS300_industry" / "2019" /
                           "hs300_industry_2019.csv", encoding="gbk")
    industry.columns = ["date", "code", "name", "industry_name", "industry_code"]
    barra = dataio.load_range(data_dir, BARRA_TABLE, 2019, 2020)
    style_cols = [c for c in barra.columns if c not in ("日期", "代码", "名称")]

    exposure_rows = []
    for date, block in both.groupby("signal_date"):
        day_int = int(date.replace("-", ""))
        active_new = (block["target_weight_new"] - block["benchmark_weight_new"]).to_numpy()
        active_ref = (block["target_weight_ref"] - block["benchmark_weight_ref"]).to_numpy()
        ind = industry.loc[industry["date"] == day_int, ["code", "industry_code"]]
        joined = block[["code"]].merge(ind, on="code", how="left")
        if joined["industry_code"].notna().all():
            g = joined.groupby("industry_code").indices
            ind_new = max(abs(active_new[g[k]].sum()) for k in g)
            ind_ref = max(abs(active_ref[g[k]].sum()) for k in g)
        else:
            ind_new = ind_ref = np.nan
        styles = barra.loc[barra["日期"] == day_int].set_index("代码")
        styles = styles.reindex(block["code"]).loc[:, style_cols].apply(
            pd.to_numeric, errors="coerce")
        X = styles.to_numpy(dtype=float)
        style_new = np.nanmax(np.abs(X.T @ active_new)) if X.size else np.nan
        style_ref = np.nanmax(np.abs(X.T @ active_ref)) if X.size else np.nan
        exposure_rows.append({"signal_date": date,
                              "industry_new": ind_new, "industry_ref": ind_ref,
                              "style_new": style_new, "style_ref": style_ref})
    exposure = pd.DataFrame(exposure_rows)

    orders_new = _load(new_dir, "orders")
    orders_ref = _load(ref_dir, "orders")
    rejected_new = int((orders_new["status"] == "REJECTED").sum())
    rejected_ref = int((orders_ref["status"] == "REJECTED").sum())
    return {
        "label": label,
        "rows_new": int(len(left)), "rows_ref": int(len(right)),
        "common_keys": int(len(both)),
        "only_in_new": {"n": int(len(only_new)),
                        "sample": only_new.head(5).to_dict("records")},
        "only_in_ref": {"n": int(len(only_ref)),
                        "sample": only_ref.head(5).to_dict("records")},
        "max_abs_target_weight_delta": float(np.max(np.abs(dw))),
        "n_common_gt_1e-6": int((np.abs(dw) > 1e-6).sum()),
        "max_abs_benchmark_weight_delta": float(np.max(np.abs(
            both["benchmark_weight_new"] - both["benchmark_weight_ref"]))),
        "active_weight_max_new": float(np.max(np.abs(both["target_weight_new"]
                                                     - both["benchmark_weight_new"]))),
        "active_weight_max_ref": float(np.max(np.abs(both["target_weight_ref"]
                                                     - both["benchmark_weight_ref"]))),
        "industry_exposure_max_abs_diff": float(
            np.nanmax(np.abs(exposure["industry_new"] - exposure["industry_ref"])))
        if len(exposure) else None,
        "style_exposure_max_abs_diff": float(
            np.nanmax(np.abs(exposure["style_new"] - exposure["style_ref"])))
        if len(exposure) else None,
        "rejected_new": rejected_new, "rejected_ref": rejected_ref,
        "rejected_relative_change": abs(rejected_new - rejected_ref) / max(1, rejected_ref),
    }


def main() -> None:
    pairs = [
        ("rev1_realinput_vs_legacy_ref", REV / "backtests" / "rev1_bt_realinput_dev19_20",
         Path("artifacts/beginning/backtests/bt_SSTAR_dev19_20")),
        ("rev1_aligned_vs_legacy_ref", REV / "backtests" / "rev1_bt_aligned_dev19_20",
         Path("artifacts/beginning/backtests/bt_SSTAR_dev19_20")),
        ("rev1_realinput_vs_a_old_buggy",
         REV / "backtests" / "rev1_bt_realinput_dev19_20",
         Path("artifacts/stage2_ic_ir/backtests/bt_realinput_dev19_20")),
    ]
    out = {}
    for label, new_dir, ref_dir in pairs:
        if not new_dir.exists() or not ref_dir.exists():
            continue
        out[label] = allocation_delta(new_dir, ref_dir, label)
    (REV / "allocation_deltas.json").write_text(json.dumps(out, indent=2, default=str))
    for label, value in out.items():
        print(label)
        for key in ("common_keys", "only_in_new", "only_in_ref",
                    "max_abs_target_weight_delta", "n_common_gt_1e-6",
                    "active_weight_max_new", "active_weight_max_ref",
                    "industry_exposure_max_abs_diff", "style_exposure_max_abs_diff",
                    "rejected_new", "rejected_ref", "rejected_relative_change"):
            value_print = value[key]
            if isinstance(value_print, dict):
                value_print = value_print.get("n")
            print(f"   {key}: {value_print}")


if __name__ == "__main__":
    main()
