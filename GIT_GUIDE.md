# 本项目的 Git 管理约定

Git 管理源码、脚本、配置、依赖锁定文件、官方规则 PDF、交接文档及轻量审计元数据。原始数据、虚拟环境、缓存、模型权重、提交压缩包和迁移备份留在本地，不进入 Git。忽略规则不会删除任何现有文件。

`artifacts/`、`releases/` 和 `repro_*/` 仅选择性保留报告与 JSON 元数据；这些元数据可能引用未纳入 Git 的权重、数据和历史输出。Git 仓库本身不是完整实验备份，迁移时需要另行保管这些文件。

当前最终提交包为 `releases/stage4_verified_ic/V3_full_pkg.zip`，SHA256：

```text
8cb919e477504fa53813ed7218fda122107c4b86de4ae9beb29509958ae66153
```

全量复现验收入口为 `repro_20261012/README_REPRODUCTION.md`，完成凭据为该目录的 `FINAL_REPRO_RECEIPT.json`。

## 第三方源码

第三方目录保持各自独立仓库，不作为嵌套 gitlink 或子模块加入本仓库。本次检查两个目录均无未提交变更：

| 目录 | 来源 | 锁定提交 |
| --- | --- | --- |
| `vendor/skd-backtest` | https://github.com/HenryZ16/skd-backtest.git | `ec5d35872fb3dec58d385039c8c18787460c18a9` |
| `vendor/tabm` | https://github.com/yandex-research/tabm.git | `28e47ae301c92ec37787dde1ce923a0793f405b4` |

新机器需从来源克隆并 checkout 到上述提交，再按环境复现文档准备依赖与数据。如果以后修改第三方源码，须另外保存补丁或将修改纳入对应仓库，父仓库不会自动记录。

## 日常操作

首次提交前配置真实身份，避免使用虚构作者：

```bash
git config --local user.name "你的名字"
git config --local user.email "你的邮箱"
git status --short
git add .
git diff --cached --stat
git commit -m "Initialize project source and audit documentation"
```

此后按完整、可审查的小改动提交；提交前检查暂存差异。未配置远程仓库，也不会自动推送。竞赛材料和成果如需远程备份，应使用获得授权的私有仓库，避免公开竞赛数据。
