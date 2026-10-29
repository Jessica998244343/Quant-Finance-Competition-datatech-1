"""B3 backtests: magnitude-target experts alone and mixed with the B1 reference."""

from __future__ import annotations

import argparse

import pandas as pd

from . import s2b_common, s2r_blend


def build(tag: str, alpha: float | None, windows) -> pd.DataFrame:
    frames = []
    for window in windows:
        expert = pd.read_parquet(s2b_common.PRED / f"{tag}__{window}.parquet")
        if alpha is None:
            frames.append(expert)
            continue
        ref = pd.read_parquet(s2b_common.PRED /
                              f"ref_corrected_cat0.5__{window}.parquet")
        both = ref.rename(columns={"score": "reference"}).merge(
            expert.rename(columns={"score": "expert"}), on=["date", "code"],
            validate="one_to_one")
        blended = s2r_blend.blend_frame(both, ["reference", "expert"],
                                        [1 - alpha, alpha])
        blended.to_parquet(s2b_common.PRED /
                           f"{tag}_a{alpha:g}__{window}.parquet", index=False)
        frames.append(blended)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="b3_lam0.1:dev19_20,b3_lam0.25:dev19_20")
    args = parser.parse_args()
    rows = []
    for spec in args.specs.split(","):
        name, period = spec.split(":")
        tag, alpha = (name.split("_a") + [None])[:2] if "_a" in name else (name, None)
        alpha = float(alpha) if alpha else None
        windows = ["DEV19", "DEV20"] if period == "dev19_20" else ["CONF21"]
        start, end = ("2019-01-02", "2020-12-31") if period == "dev19_20" \
            else ("2021-01-04", "2021-12-31")
        frame = build(tag, alpha, windows)
        metrics_out = s2b_common.run_backtest(f"b3_bt_{name}_{period}", frame, start, end)
        row = {"run": f"b3_bt_{name}_{period}", "mean_rankic": metrics_out["mean_rankic"],
               "annualized_excess_return": metrics_out["annualized_excess_return"],
               "tracking_error": metrics_out["tracking_error"],
               "information_ratio": metrics_out["information_ratio"],
               "turnover": metrics_out["turnover"]}
        rows.append(row)
        s2b_common.log({"stage": "B3-backtest", **row})
        print(row, flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
