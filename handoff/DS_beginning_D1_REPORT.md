# D1 报告：环境、本地回测契约与功能核验

日期：2026-10-05（Asia/Shanghai）。执行范围：`Prompt_for_DS_beginning.md` 第一节。
本报告按用户指示在完成自测后直接进入第二节，不做中途交付；内容为第一节的完整证据。

## 1. 做了什么

1. 在项目根建立独立 Python 3.11 环境 `.venv`（Python 3.11.17，由 `miniconda3/envs/quant` 的
   3.11.17 创建），锁定 `requirements.lock`；研究依赖与提交依赖分开说明。
2. 从 GitHub 克隆官方回测平台到 `vendor/skd-backtest`，锁定 commit
   `ec5d35872fb3dec58d385039c8c18787460c18a9`（tag `v1.0.1`，版本 1.0.1），未做任何改动。
3. 阅读并交叉核对官方三份 PDF、数据 README、平台 `docs/usage.md`、`docs/interfaces.md` 与
   平台源码（`label_provider.py`、`prediction_evaluator.py`、`config.py`、`data_provider.py`）。
4. 固定本地代理评价契约（`configs/beginning/evaluation_contract.{json,toml}`），
   H=1、O2O（`adjusted_open`）、`lookback=120`、Barra 组合约束、非零费用。
5. 建立 is_st 零值数据视图 `artifacts/beginning/data_zero_st/`：仅把 `Factor33_winsor.is_st`
   改 0（84 个月度文件重写），其余目录用符号链接复用只读原始数据，未修改 `Source/Data`。
   用户已确认主办方最终评测把 is_st 全部置零，因此主参考契约采用 `zero_proxy`。
6. 用固定、无训练的工程信号（当日 Barra `动量` 百分位）跑 2019-01-02 至 2019-02-15
   共 27 个交易日的真实回放，并对标签、Spearman、净值/IR 做独立复算。
7. 写 `artifacts/beginning/d1/rules_matrix.md`，把官方条款、用户确认、本地代理假设与未决事项分列。

## 2. 实际结果与误差

核心一致性检查（`artifacts/beginning/d1/smoke/smoke_report.json`）全部通过：

| 检查 | 结果 |
|---|---|
| 独立复算 O2O 标签与平台 `predictions.future_return` 逐值一致 | true（`atol=1e-10`） |
| 标签缺失掩码与平台一致 | true |
| 独立复算每日 Spearman 与平台 `rankic` 表逐日一致 | true |
| 由 `equity_curve` 独立复算年化收益、跟踪误差、IR 与平台 `metrics` 一致 | true（IR 1.117501423172709） |
| 输出股票池 = 当日 Barra 成分（每日 300 只） | true |
| 无重复 code / 分数全部有限 | true |
| 费用确实进入账户（`transaction_cost` 与逐笔 `total_cost` 之和一致） | 19206.18 元 |
| 目标权重完整、逐日 300 只 | true |
| 约束实际生效 | 单股 ≤0.0771（限 0.10）、主动 ≤0.0100000000000001（限 0.01）、日换手 ≤0.9769（限 1.0）、行业暴露 ≤0.0200000000000016、风格暴露 ≤0.0200000000000065 |
| 真实成交与目标权重不同 | true（目标 300 只，实际持仓 83–98 只；拒单含 `SUSPENDED` 与 `INSUFFICIENT_CASH`） |

工程信号本身的表现（仅用于证明通路，不是基线）：
Mean RankIC 0.0241、RankICIR 0.125、组合 IR 1.1175、双边换手合计 10.33、失败订单 28。

耗时：整个冒烟回测 7.4 秒（27 个交易日，含数据读取与独立复算）。峰值内存远低于预留的 3 GiB。

## 3. 复现命令

```bash
cd /home/jessica998244343/DATA/Quant-Finance3
source .venv/bin/activate            # 或直接使用 .venv/bin/python
python -m pip install -r requirements.lock
python -m pip install -e vendor/skd-backtest
python -m qf3.d1_smoke               # 需要 PYTHONPATH=src
```

环境重建（本机）：`~/miniconda3/envs/quant/bin/python -m venv .venv` 后按 `requirements.lock` 安装。

## 4. 环境事实

- CPU：16 逻辑核 / 8 物理核；内存 15.0 GiB（总 16393072640 B，可用约 13.4 GiB）。
- GPU 探针全部失败并已记录（`artifacts/beginning/d1/gpu_probe.log`）：
  NVML 被操作系统拒绝；LightGBM `device=gpu` 报 “No OpenCL device found”；
  CatBoost `task_type=GPU` 报 CUDA error 35（驱动版本不足）。全流程按 CPU 前进，未尝试修驱动。
- 线程策略：模型 8 线程、BLAS/OMP 限 4，合计不超过 12。
- 主要包版本：numpy 2.4.6、pandas 3.0.6、pyarrow 25.0.1、scipy 1.17.1、scikit-learn 1.9.1、
  LightGBM 4.7.0、CatBoost 1.2.10、skd-backtest 1.0.1。

## 5. 哪些配置非官方

组合约束数值（单股 10%、主动 1%、行业/风格 2%、双边换手 1.0）、费率（佣金 0.0003、最低 5 元、
滑点 0.0005、印花税 0.001、过户费 0.00002）、`read_batch_months=3`、开发窗口划分与晋级阈值，
都是本项目选定的本地代理，不是主办方公布的隐藏配置。详见 `rules_matrix.md`。

## 6. 问题及其影响

1. 官方隐藏评测的组合约束与费率未公开，本地 IR 只能用于同口径比较，不能换算成官方分数。
2. `skd-backtest` 的 `raw_price` 模式缺少完整真实价格与限价数据，故采用 `adjusted_return`；
   这是平台文档允许的推荐路径。
3. is_st 零值视图与源值视图在本次冒烟窗口完全一致（见第二节敏感性），说明该字段在本地执行层
   未产生额外约束，但两种口径仍分列记录，不混用。
4. 平台涨跌停价由 `MarketDataRawOpen` 与 is_st 推算，属平台注释中声明的“临时推算路径”，
   不等于交易所原始限价；报告中不宣称已重建真实限价。

## 7. 第二节预计资源

特征面板一次构建（1584 个信号日 × 300 只 = 475200 行、72 列）约 145 秒；单折 LightGBM 训练
17–30 秒、CatBoost 8–11 秒；2 年连续回测约 43 秒、单年约 23 秒。全流程在 16 GiB 内存内完成，
无需 GPU。

## 8. 完成条件核对

- Python 3.11 环境可重建：是（`.venv` + `requirements.lock`）。
- 平台可本地运行：是（真实回放产出七张审计表）。
- H=1 与数据接口已核对：是（与平台逐值对齐）。
- 非零费用与约束已验证：是。
- is_st 与其他代理假设明确：是（`rules_matrix.md`、`zero_proxy` 主契约）。
- 关键阻塞未被默认通过：是（GPU 不可用已记录并改走 CPU，未掩盖）。

按用户指示，本报告完成后未停止，直接进入第二节。
