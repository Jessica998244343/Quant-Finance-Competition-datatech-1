"""§B5: rebuild the online TCN inputs from the real tables and compare raw outputs.

Uses the saved run preprocessing (design median/IQR and channel statistics) and
the saved model; inputs are rebuilt from the three real tables with the same
membership filter, never from the prediction cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import F0_COLUMNS, G1_COLUMNS
from .s6_v2_tcn import CHANNELS, STEPS, TCNModel, build_sequences

V2 = Path("artifacts/stage4_verified_ic/v2")
FEATURES = list(F0_COLUMNS) + list(G1_COLUMNS)
SAMPLE = {"DEV19": (20190102, 20190628, 20191202),
          "DEV20": (20200102, 20200630, 20201202),
          "CONF21": (20210104, 20210603, 20211202)}


def _static(frame: pd.DataFrame, median: dict, iqr: dict) -> np.ndarray:
    blocks = []
    counts = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.notna().sum())
    ranks = frame.groupby("date")[list(G1_COLUMNS)].transform(lambda s: s.rank(method="average"))
    ranks = (ranks - (counts + 1.0) / 2.0) / counts
    for column in FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.isna().to_numpy(dtype=float)
        filled = values.fillna(median[column]).to_numpy(dtype=float)
        scaled = np.clip((filled - median[column]) / iqr[column], -5, 5)
        blocks.append(scaled.reshape(-1, 1))
        blocks.append(mask.reshape(-1, 1))
    blocks.append(np.nan_to_num(ranks.to_numpy(dtype=float)))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--out", default=str(V2 / "v2_completion_tcn_input_check.json"))
    args = parser.parse_args()
    panel = pd.read_parquet("artifacts/stage2_ic_ir/panel_platform.parquet")
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    built = build_sequences()
    sequence_map, codes_by_day = built["sequences"], built["keys"]
    torch.set_num_threads(2)
    report = {"seed": args.seed, "days": {}, "tolerance": {"atol": 1e-5, "rtol": 1e-4}}
    worst = 0.0
    for window, days in SAMPLE.items():
        run = V2 / f"tcn_rev1_tcn_s{args.seed}_{window}"
        pre = json.loads((run / "preprocessing.json").read_text())
        median, iqr = pre["design_median"], pre["design_iqr"]
        stats = {int(k): tuple(v) for k, v in pre["channel_stats"].items()}
        from safetensors.torch import load_file
        model = TCNModel(len(FEATURES) * 2 + len(G1_COLUMNS))
        model.load_state_dict(load_file(str(run / "model.safetensors")))
        model.eval()
        saved = pd.read_parquet(run / "predictions.parquet")
        for day in days:
            if day not in sequence_map:
                continue
            block = panel.loc[panel["date"] == day].reset_index(drop=True)
            codes = codes_by_day[day]
            lookup = {code: index for index, code in enumerate(codes)}
            keep = block["code"].map(lookup).notna().to_numpy()
            block = block.loc[keep].reset_index(drop=True)
            rows = np.stack([sequence_map[day][lookup[code]] for code in block["code"]])
            rows = torch.from_numpy(rows.astype(np.float32))
            for channel, (value_median, value_iqr) in stats.items():
                rows[:, :, channel] = torch.clamp(
                    (rows[:, :, channel] - value_median) / value_iqr, -5, 5)
            rows = torch.nan_to_num(rows, nan=0.0)
            static = torch.from_numpy(_static(block, median, iqr))
            with torch.no_grad():
                online = model(static, rows).numpy()
            reference = saved.loc[saved["date"] == day].set_index("code")["score"]
            aligned = pd.Series(online, index=block["code"]).reindex(reference.index).to_numpy()
            diff = np.abs(aligned - reference.to_numpy())
            worst = max(worst, float(np.nanmax(diff)))
            report["days"][f"{window}:{day}"] = {
                "rows": int(len(reference)), "max_abs_diff": float(np.nanmax(diff)),
                "rank_identical": bool(np.array_equal(
                    pd.Series(aligned).rank().to_numpy(),
                    pd.Series(reference.to_numpy()).rank().to_numpy()))}
            print(window, day, "rows", len(reference),
                  "max diff", f"{float(np.nanmax(diff)):.3e}", flush=True)
    report["worst_abs_diff"] = worst
    report["passed"] = bool(worst <= 1e-5)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"worst": worst, "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
