# B0 冻结基线

- 研究折预测：`predictions_DEV19.parquet`、`predictions_DEV20.parquet`、`predictions_CONF21.parquet`。
- 逐日 RankIC：`rankic_*.parquet`。
- 汇总：`metrics.csv`；配置：`config.json`；冻结记录：`../B0_FREEZE.md`。
- 全量回退包：`releases/baseline_v1.zip`（`.venv` 解压后独立跑 `train.py` 可逐位复现）。
