"""Frozen names and contracts for the phase-1 research pipeline."""

from __future__ import annotations

# --- source tables -------------------------------------------------------
FACTOR_TABLE = "Factor33_winsor"
BARRA_TABLE = "Barra_factor"
MARKET_TABLE = "MarketData"

# 33 factor columns of Factor33_winsor (everything except 日期/代码/名称).
F33_FACTOR_COLUMNS = (
    "turn", "change_ratio", "daily_return", "momentum_5", "reversal_5",
    "volatility_5", "sma_20", "ema_20", "macd_diff_12_26_9",
    "macd_dea_12_26_9", "macd_hist_12_26_9", "rsi_12", "kdj_k_9_3_3",
    "kdj_d_9_3_3", "bias_20", "cci_14", "atr_14", "total_market_cap",
    "float_market_cap", "pe_ttm", "pb", "ps_ttm", "roe_avg_ttm",
    "roa_avg_ttm", "gross_profit_rate_ttm", "net_profit_rate_ttm",
    "debt_to_asset_lf", "current_ratio_lf", "beta_000300SH_22", "list_days",
    "is_suspend", "is_st", "is_new",
)

# Columns that never enter the model (identifier / banned label).
BANNED_COLUMNS = ("名称", "is_st")

BARRA_STYLE_COLUMNS = (
    "市值", "贝塔", "动量", "残差波动", "非线性市值",
    "账面市值比", "流动性", "盈利", "成长", "杠杆",
)

# 0/1 status columns kept on their raw scale (no cross-sectional rank).
STATUS_COLUMNS = ("is_suspend", "is_new")

# --- feature sets --------------------------------------------------------
F0_FACTOR_COLUMNS = tuple(c for c in F33_FACTOR_COLUMNS if c not in BANNED_COLUMNS)
BARRA_FEATURE_NAMES = (
    "barra_size", "barra_beta", "barra_momentum", "barra_residvol",
    "barra_nonlinear_size", "barra_bp", "barra_liquidity", "barra_earnings",
    "barra_growth", "barra_leverage",
)
F0_COLUMNS = F0_FACTOR_COLUMNS + BARRA_FEATURE_NAMES

G1_COLUMNS = (
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "ret_60", "ret_119",
    "rv_5", "rv_20", "rv_60", "relma_5", "relma_20", "relma_60",
    "hl_range", "close_pos", "overnight", "intraday",
    "overnight_mean_5", "intraday_mean_5",
)
G2_COLUMNS = (
    "amt_abn", "vol_abn", "turn_abn", "std_shock", "dir_x_amt",
    "on_intra_diff", "on_intra_diff_z",
)
G3_COLUMNS = (
    "g3_roe_minus_pb", "g3_roa_minus_debt",
    "g3_gross_minus_net", "g3_rev5_x_roe",
)
FEATURE_COLUMNS = F0_COLUMNS + G1_COLUMNS + G2_COLUMNS + G3_COLUMNS

GROUP_COLUMNS = {
    "F0": F0_COLUMNS,
    "G1": G1_COLUMNS,
    "G2": G2_COLUMNS,
    "G3": G3_COLUMNS,
}

# --- evaluation contract (local proxy, not official) ---------------------
EVALUATION_CONTRACT = {
    "contract_version": "beginning-v1",
    "rebalance_interval": 1,
    "holding_period": 1,
    "lookback": 120,
    "price_mode": "adjusted_return",
    "label_price_basis": "adjusted_open",
    "initial_cash": 1_000_000.0,
    "trading_days_per_year": 252,
    "risk_free_rate": 0.0,
    "optimizer": {
        "method": "barra",
        "long_only": True,
        "fully_invested": True,
        "single_name_weight_limit": 0.10,
        "active_weight_limit": 0.01,
        "industry_exposure_limit": 0.02,
        "barra_style_exposure_limit": 0.02,
        "turnover_limit": 1.0,
    },
    "costs": {
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "slippage": 0.0005,
        "fee_schedule": [
            {
                "effective_date": "2016-01-01",
                "stamp_tax_rate": 0.001,
                "transfer_fee_rate": 0.00002,
            }
        ],
    },
    "read_batch_months": 3,
    "execution_st_policy": "zero_proxy",
    "seeds": {"main": 20261005, "review": [20261017, 20261029]},
    "note": "local proxy contract; not the official hidden configuration",
}

# Forward-year expanding windows (outer cutoffs are the last trading day).
OUTER_WINDOWS = {
    "DEV19": {"train_end": "2018-12-31", "predict_year": 2019},
    "DEV20": {"train_end": "2019-12-31", "predict_year": 2020},
    "CONF21": {"train_end": "2020-12-31", "predict_year": 2021},
    "RES22": {"train_end": "2021-12-31", "predict_year": 2022},
}

EARLY_STOP_DAYS = 126
