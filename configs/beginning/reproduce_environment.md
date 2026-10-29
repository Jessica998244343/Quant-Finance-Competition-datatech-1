# 环境重建说明（第一节）

## 研究环境（本地训练 + 回测）

2026-10-05 配置复核修订：以下使用已有锁文件恢复版本，不执行无版本约束的库升级。
已有工作环境不需要重装；只在重建环境时执行安装命令。

```bash
cd /home/jessica998244343/DATA/Quant-Finance3
~/miniconda3/envs/quant/bin/python -m venv .venv      # Python 3.11.17
.venv/bin/python -m pip install -r requirements.lock
# vendor/skd-backtest 须预先准备为下文所列 commit；绑定本地受审查源码。
.venv/bin/python -m pip install -e vendor/skd-backtest
.venv/bin/python -m pip check
source scripts/activate.sh
```

`requirements.lock` 是 `pip freeze` 快照。其中 `skd-backtest` 行实际是固定 commit 的
editable Git URL，不是相对路径。新工作区先准备相同 commit 的 vendor 源码，再执行上述安装。
已有 vendor 检查 HEAD 与工作区，不要为重建说明覆盖本地改动：

```bash
git clone https://github.com/HenryZ16/skd-backtest.git vendor/skd-backtest
git -C vendor/skd-backtest checkout ec5d35872fb3dec58d385039c8c18787460c18a9
git -C vendor/skd-backtest status --short
```

## 回测入口和配置身份

- Python API 使用 `src/qf3/constants.py::EVALUATION_CONTRACT`；
  `configs/beginning/evaluation_contract.json` 是同内容的项目契约，不是平台 CLI 的配置结构。
- CLI 使用 `configs/beginning/evaluation_contract.toml`，由平台 `load_config` 读取。
  其中路径相对 TOML 所在目录，因此数据路径为 `../../artifacts/beginning/data_zero_st`。
- TOML 是一个指定日期的示例配置。实际研究另建唯一 run ID 和输出目录，不覆盖历史回测。
- 平台优化器使用 SciPy `linprog`，当前不需要额外安装 cvxpy、osqp 或其他求解器包。
- 使用 `source scripts/activate.sh` 后再调用研究模块，以设置 `.venv`、`PYTHONPATH` 与默认线程变量；
  单独运行 `.venv/bin/python` 不会自动设置 `PYTHONPATH` 或线程变量。

## 提交包环境（评测机 CPU、离线）

评测环境按包内 `submission/requirements.txt` 预装依赖，只包含
numpy / pandas / pyarrow / scipy / scikit-learn / lightgbm / catboost，全部锁定到本机版本。
提交包不依赖 `skd-backtest`、不依赖 GPU、不联网。

## 线程与内存

- 模型参数线程 8；激活脚本为 `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`
  提供默认值 4，保留调用者已有设置。线程池可能嵌套，不能把 8+4 当作系统线程数的硬上限；
  单个重训练任务运行，记录实际线程与峰值内存，避免多个线程池过量并发。
- 特征面板 475200 × 76 以 parquet 缓存（约 124 MB）；单折训练峰值 RSS < 2 GiB。
