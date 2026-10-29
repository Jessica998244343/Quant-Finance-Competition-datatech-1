"""Thin wrapper around the locked skd-backtest engine with the frozen contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skd_backtest import BacktestEngine, CostConfig, FeeScheduleEntry, OptimizerConfig

from .constants import EVALUATION_CONTRACT


def optimizer_config() -> OptimizerConfig:
    opt = EVALUATION_CONTRACT["optimizer"]
    return OptimizerConfig(
        method=opt["method"],
        long_only=opt["long_only"],
        fully_invested=opt["fully_invested"],
        single_name_weight_limit=opt["single_name_weight_limit"],
        active_weight_limit=opt["active_weight_limit"],
        industry_exposure_limit=opt["industry_exposure_limit"],
        barra_style_exposure_limit=opt["barra_style_exposure_limit"],
        turnover_limit=opt["turnover_limit"],
    )


def cost_config(zero_cost: bool = False, multiplier: float = 1.0) -> CostConfig:
    """``multiplier`` scales every explicit rate/minimum together (scenario runs)."""
    if zero_cost:
        return CostConfig()
    costs = EVALUATION_CONTRACT["costs"]
    schedule = tuple(
        FeeScheduleEntry(entry["effective_date"],
                         entry["stamp_tax_rate"] * multiplier,
                         entry["transfer_fee_rate"] * multiplier)
        for entry in costs["fee_schedule"]
    )
    return CostConfig(
        commission_rate=costs["commission_rate"] * multiplier,
        minimum_commission=costs["minimum_commission"] * multiplier,
        slippage=costs["slippage"] * multiplier,
        fee_schedule=schedule,
    )


def make_engine(
    *,
    data_dir,
    start_date: str,
    end_date: str,
    inference=None,
    submission_dir=None,
    output_dir=None,
    async_inference: bool = False,
    prefetch: bool = True,
    zero_cost: bool = False,
    cost_multiplier: float = 1.0,
    seed: int = 20261005,
) -> BacktestEngine:
    contract = EVALUATION_CONTRACT
    if (inference is None) == (submission_dir is None):
        raise ValueError("provide exactly one of inference / submission_dir")
    return BacktestEngine(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        inference=inference,
        submission_dir=submission_dir,
        initial_cash=contract["initial_cash"],
        rebalance_interval=contract["rebalance_interval"],
        holding_period=contract["holding_period"],
        lookback=contract["lookback"],
        read_batch_months=contract["read_batch_months"],
        prefetch=prefetch,
        async_inference=async_inference,
        price_mode=contract["price_mode"],
        label_price_basis=contract["label_price_basis"],
        trading_days_per_year=contract["trading_days_per_year"],
        risk_free_rate=contract["risk_free_rate"],
        optimizer_config=optimizer_config(),
        cost_config=cost_config(zero_cost, cost_multiplier),
        output_dir=output_dir,
        friendly_output=False,
        random_seed=seed,
    )


def run_backtest(**kwargs) -> tuple[dict, dict, dict]:
    engine = make_engine(**kwargs)
    metrics = engine.run()
    return metrics, engine.tables, engine.account


def score_lookup_inference(scores: pd.DataFrame):
    """Inference callable that replays precomputed scores (offline OOF runs)."""
    table = scores.copy()
    by_date = {int(d): g.loc[:, ["code", "score"]] for d, g in table.groupby("date")}

    def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        day = int(as_of_date.replace("-", ""))
        if day not in by_date:
            raise KeyError(f"no precomputed scores for {as_of_date}")
        block = by_date[day].copy()
        block.insert(0, "date", as_of_date)
        return block

    return inference


def barra_column_inference(column: str = "动量"):
    """Fixed, training-free engineering signal (D1 smoke only)."""

    def inference(as_of_date: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        day = int(as_of_date.replace("-", ""))
        barra = data["Barra_factor"]
        today = barra.loc[barra["日期"] == day, ["代码", column]]
        return pd.DataFrame({
            "date": as_of_date,
            "code": today["代码"].to_numpy(),
            "score": pd.to_numeric(today[column], errors="coerce").to_numpy(),
        })

    return inference


def save_run(result_dir, metrics: dict, tables: dict, account: dict) -> None:
    import json

    out = Path(result_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "account.json").write_text(json.dumps(account, indent=2, default=str))
