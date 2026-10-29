# V3_full_pkg 全量复现与恢复说明

2026-10-12，全量九成员复跑及严格验收完成。最终结果见`FINAL_REPRO_RECEIPT.json`和`../handoff/REVIEW_V3_FULL_REPRO_COMPLETED_20261012.md`。

本目录是审查与复现资料，不是新的提交包。原提交为`../releases/stage4_verified_ic/V3_full_pkg.zip`，SHA256：`8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153`。

## 环境

Python3.11.17；numpy2.4.6、pandas3.0.6、pyarrow25.0.1、catboost1.2.10、torch2.14.1+cpu、safetensors0.8.0、rtdl_num_embeddings0.0.12。
本次使用已安装`../repro_20261010/train_env311`，未新装依赖。旧实际安装日志与完整原包/数据清单在`../repro_20261010/reference.json`及该目录安装日志中。

若需要在新环境运行原包，先用Python3.11创建环境，按原包`train_requirements.txt`安装训练依赖。CPU torch版本需要PyTorch CPU索引：

```bash
<python3.11> -m venv <env>
<env>/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r <submission>/train_requirements.txt
```

安装阶段与离线训练/推理阶段分开；新机器上须重新记录环境，不应把当前机器的恢复身份直接搬过去继续训练。

## 原包整体入口

这是主办方可直接复跑的原始入口。输出指向全新目录，不能覆盖原模型：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 <env>/bin/python <submission>/train.py --data-dir <official-data-dir> --out <fresh-model-dir> --threads 6
```

## 本次已执行的可恢复审查运行

从工作区根目录运行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 repro_20261010/train_env311/bin/python -B -u scripts/v3_repro_resume.py >> repro_20261012/train.log 2>&1
```

数据、原包、代码、版本、线程身份绑定到`identity.json`。完整缓存从公开原始文件生成。完成成员核验SHA后跳过；NN按最新有效epoch恢复，保留两代检查点；锁防重复进程。不要编辑已冻结的运行器或原包来让身份检查通过。

实际第二个TabM从内层8轮恢复，最终所有参数仍逐字节一致。恢复辅助调度改变了执行分段，保持原数据、目标、架构、初始化、每个更新步骤、选轮和outer重训；小规模原函数对照、真实恢复曲线和完整参数比对分别留证。

## 最终检查

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src repro_20261010/train_env311/bin/python -B -m qf3.s6_v3_repro_check --reference-model repro_20261012/original/submission/model --repro repro_20261012/model --submission repro_20261012/original/submission --panel repro_20261012/cache/raw_rebuilt_panel.parquet --zip-sha256 8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153 --negative-controls --out artifacts/stage4_verified_ic/v3_train_repro.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 repro_20261010/train_env311/bin/python -B scripts/v3_repro_receipt.py
```

最终严格检查退出码0、passed=true；24唯一日期所有raw/家族/最终score差0。六NN参数文件SHA相同；C1二进制GUID/完成时间变化，但除这两项外的整个模型JSON相同，详见`catboost_canonical_equality.json`。

最终诊断IC为训练内24日均值0.19002665054319479，不能当样本外或公榜分数。五类负例全部被拒。`final_gate_before_date_dedup.json`、`receipt_before_date_dedup.json`只保留为去重修正的历史，不作为最终验收。

`cache_identity_adoption.json`记录训练前只修正结果JSON通道键类型、缓存构建/训练函数未变的身份更新。`runner_initial.py`与`identity_initial.json`是历史记录。正式模型身份以`identity.json`、`progress.json`和最终收据为准。

原模型没有修改，无需重新提交。最终主办方人工审查规则仍以官方材料为准。
