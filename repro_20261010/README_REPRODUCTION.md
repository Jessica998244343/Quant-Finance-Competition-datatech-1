# V3_full_pkg 复现说明（提交包 train.py 独立复跑）

本说明用于从公开数据独立重现 `V3_full_pkg` 的 9 个原子模型，供人工审查使用。
它不改动原提交包，也不产出新的提交版本。

## 1. 参照身份（只读）

| 项 | 值 |
|---|---|
| 原包 | `releases/stage4_verified_ic/V3_full_pkg.zip` |
| SHA-256 | `8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153` |
| 只读副本 | `repro_20261010/V3_full_pkg_reference.zip`（同 SHA-256，chmod 444） |
| 成员 | CatBoost C1 ×3（seed 20261005/20261017/20261029）+ TabM ×3 + TCN ×3 |
| 融合 | `0.375·C1 + 0.375·TabM + 0.25·TCN`，扁平图（原子 raw → 逐 seed 日中心秩 → seed 均值 → 家族加权 → 最终日秩） |
| 训练截止 | `label_end_date <= 20221230`（外层成熟标签；前 119 个交易日为 warm-up） |

参照证据：`repro_20261010/reference.json`（ZIP CRC/SHA、34 个包内文件哈希、336 个
`Source/Data` 公开数据文件哈希、解释器与依赖版本、线程环境）。

## 2. 训练环境准备（与推理环境分开）

推理只依赖 `requirements.txt`（numpy / pandas / pyarrow / catboost）；训练另需
`train_requirements.txt` 中的 `torch==2.14.1+cpu`、`safetensors==0.8.0`、
`rtdl_num_embeddings==0.0.12`。`torch==2.14.1+cpu` 需要 **PyTorch CPU 轮子索引**，
否则 PyPI 上的同名版本会拉入整条 CUDA 栈：

```bash
# Python 3.11（提交规则要求的解释器版本）
python3.11 -m venv <env>
<env>/bin/python -m pip install -r <submission>/train_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
# 校验
<env>/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.14.1+cpu False
```

本次实际安装日志：`repro_20261010/install_train_env311.log`（29 个包，含 `torch-2.14.1+cpu`）。
安装阶段需要网络；**训练与推理阶段不联网**（包内无联网代码，运行时不下载任何内容）。

## 3. 执行训练（原样运行包内脚本）

```bash
# cwd 必须是一个稳定目录（不要放在稍后会被删除的临时解压目录里）
cd <stable_dir>
PYTHONPATH=<submission> OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  <env>/bin/python <submission>/train.py \
    --data-dir <abs path to official Source/Data> \
    --out <fresh output dir> \
    --threads 6
```

- `--out` 必须指向全新目录，**不得**指向已提交的 `model/` 或构建目录。
- CatBoost 线程固定 6（`--threads 6`）；NN 线程在包内固定为 2
  （`alpha_train_tabm.TABM_THREADS=2`、`alpha_tcn_train.THREADS=2`）。
- 脚本按 seed 依次训练 C1 → TabM → TCN；每个成员训练完成即落盘。
- 完整 stdout/stderr 与退出码留档：`repro_20261010/train_stdout.log`。

## 4. 产出

`--out` 目录内：

- `catboost_c1_seed{20261005,20261017,20261029}.cbm` + 各 `__curve.parquet`（内层选轮曲线）；
- `tabm_seed{...}.npz`（静态 NumPy 导出）+ `design_seed*.json` 合并出的 `tabm_design.json`；
- `tcn_seed{...}.npz`（静态 NumPy 导出）+ `tcn_design.json` + `tcn_channel_stats.json`；
- `meta.json`（成员、权重、融合图、训练截止、成员哈希、C1 树数）。

## 5. 随机性与确定性说明

- 每个成员训练前显式 `torch.manual_seed(seed)` 与 `numpy.random.seed(seed)`；批内顺序由
  `numpy.random.default_rng(seed)` 生成，未使用全局时间/系统熵。
- 树模型随机种子 = 成员 seed；CatBoost 同线程数确定。
- NN 线程数固定为 2；改变线程数不保证位一致，因此复现须保持 2。
- 因此不同机器上同输入、同线程应逐位或接近逐位一致；允许的数值差异按检查器口径
  `atol=1e-5, rtol=1e-4` 逐元素判定，并**分别**报告日秩与最终分数是否完全一致。

## 6. 复核（强检查器）

```bash
PYTHONPATH=src python -m qf3.s6_v3_repro_check \
  --reference-model <submission>/model \
  --repro <fresh output dir> \
  --submission <submission> \
  --zip-sha256 8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153 \
  --negative-controls \
  --out artifacts/stage4_verified_ic/v3_train_repro.json
# 退出码 0 = 通过；任何失败都非零退出
```

检查内容：成员集合/数量/meta 语义门 → 树结构（树数、叶值、分裂边界）→ NN 逐张量逐元素容差
→ 预处理与融合权重 → 连续信号段 + 跨年/短历史/成分变更日的端到端 9 原子 raw、逐 seed 日秩、
家族均值、最终分数与本地 IC 差 → 5 类负例必须被拒。详见
[`src/qf3/s6_v3_repro_check.py`](../src/qf3/s6_v3_repro_check.py)。

## 7. 限制

- 本复现只检验"提交包能否从公开数据重建"，不改变原包、不用于选型、不预测私榜。
- 训练期内的 IC 只用于复现比对，**不构成样本外表现**。
- 修订包是否制作/上传由用户决定，执行 Agent 不擅自更改或重新上传原包。
