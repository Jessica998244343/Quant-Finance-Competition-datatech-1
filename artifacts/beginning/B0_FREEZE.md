# B0 冻结记录

B0 是本地强基线的比较参照，冻结后不再覆盖。

## 身份

- 模型：LightGBM 4.7.0 `gbdt` / `regression`（L2），目标为每日截面平均并列秩 `z=(rank-(n+1)/2)/n`。
- 特征集 F0（42 列）：Factor33_winsor 去掉 `名称`/`is_st` 后的 32 列，加 10 列 Barra 风格（原值减 0.5）。连续列用当日合法池截面秩中心化，`is_suspend`/`is_new` 保留 0/1 原值。
- 样本权重：日期内股票等权 `1/n_t`，再整体归一到均值 1；不使用时间衰减。
- 线程 8，`deterministic=True`，`force_row_wise=True`，主种子 20261005。

## 参数

```json
{
  "boosting": "gbdt",
  "objective": "regression",
  "learning_rate": 0.03,
  "num_leaves": 31,
  "max_depth": -1,
  "min_data_in_leaf": 200,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.8,
  "bagging_freq": 1,
  "lambda_l1": 0.0,
  "lambda_l2": 5.0,
  "max_bin": 127,
  "seed": 20261005,
  "num_threads": 8
}
```

## 训练协议

按年向前 expanding：外层训练只看 `label_end_date <= 外层截止日` 的样本；外层训练末端 126 个交易日为早停区间，内部拟合样本的 `label_end_date` 严格早于早停区间起始日；早停指标为每日平均 Spearman（
`higher_is_better=True`，默认 metric 关闭）；找到轮数后用全部外层合格样本按固定轮数重新拟合，再预测外层年份。

## 开发折结果（冻结时）

| 窗口 | 外层截止 | 早停起始 | 外层样本 | 内部拟合 | 早停样本 | 最佳轮数 | 早停日 Spearman | Mean RankIC | RankICIR | 正 IC 比例 |
|---|---|---|---|---|---|---|---|---|---|---|
| DEV19 | 20181231 | 20180626 | 174978 | 137649 | 36755 | 266 | 0.0452 | 0.05178 | 0.410 | 0.652 |
| DEV20 | 20191231 | 20190627 | 247938 | 209636 | 37703 | 273 | 0.0373 | 0.04387 | 0.355 | 0.642 |

两个开发折的早停轮数（266/273）都远低于 3000 上限，因此不触发扩到 5000 的条件。

## 特征列名（F0，42 列）

```
turn, change_ratio, daily_return, momentum_5, reversal_5, volatility_5, sma_20, ema_20, macd_diff_12_26_9, macd_dea_12_26_9, macd_hist_12_26_9, rsi_12, kdj_k_9_3_3, kdj_d_9_3_3, bias_20, cci_14, atr_14, total_market_cap, float_market_cap, pe_ttm, pb, ps_ttm, roe_avg_ttm, roa_avg_ttm, gross_profit_rate_ttm, net_profit_rate_ttm, debt_to_asset_lf, current_ratio_lf, beta_000300SH_22, list_days, is_suspend, is_new, barra_size, barra_beta, barra_momentum, barra_residvol, barra_nonlinear_size, barra_bp, barra_liquidity, barra_earnings, barra_growth, barra_leverage
```

## 预测与回测

- 预测：`artifacts/beginning/predictions/B0__DEV19.parquet`、`B0__DEV20.parquet`。
- 回测：见 `artifacts/beginning/backtests/` 下 B0 对应运行目录。
- CONF21 与全量回退包在冻结后补齐。