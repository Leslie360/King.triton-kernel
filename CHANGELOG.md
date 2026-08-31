# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-08-31

初版开源发布。

### 核心洞见

- **奖励"执行正确"，编译只是硬门槛——编译率 ≠ 正确率。** 本项目训练方法的核心主张：reward 信号是"跑对了没"，不是"编译过了没"。编译失败为硬惩罚（-1.0 量级），正确性权重 0.4、性能 speedup 0.3、编译 0.3，配多轮修复（默认 5 轮）与 verifier 自适应终止。
- 该主张已是领域共识（Dr.Kernel / daVinci / DRTriton / CUDA Agent 等均以真实 GPU 执行验证为核心奖励信号），本项目在**评测口径诚实化**上做差异化工程。

### 评估环境（kernelgym/）

- 独立 GPU 分布式评估环境：仅依赖 torch/triton/redis/fastapi/httpx，**无 verl/ray/vllm 耦合**，可独立启动判分 server。
- subprocess worker 池隔离 kernel 执行：CUDA 错误 / 非法内存访问被隔离，worker 不崩溃。
- 真实执行验证：CUDA 事件计时 + correctness 校验（rtol/atol 容差）+ 多后端支持。

### 口径方法论

- **reference 分母固定**（`InMemoryReferenceCache`，key=uuid+ref_hash+is_valid）：判分不重跑 reference 分母，消除 run 级分母抖动。
- 权威口径 = **per-problem best-of-history**（每题任一样本任一轮最优，跨全部轮次）。
- **口径审计纪律**：对外数字必须附口径（指标/样本数/轮次/refcache 状态/脚本 SHA），禁止口径混用。详见 `docs/eval_calibre.md`。

### 发布清扫

- kernelgym/、drkernel/、evals/ 均为**真实代码副本**（非软链、非共享盘）。
- 已排除：私有数据集、训练 checkpoint、内部日志、verl_patch 内部 overlay、内部路径/主机名。
- 训练栈（verl 集成）依赖上游 verl，未随仓库发布，需自行安装对齐版本。
- 详细清扫清单见 `docs/RELEASE_CLEANLIST.md`。

### 工程化治理

- 新增 GitHub Actions CI（`.github/workflows/ci.yml`）：`lint`（ruff check + ruff format --check）与 `test`（CPU-only 纯逻辑测试，跳过 gpu/redis/slow）。
- 新增 issue / PR 模板、CONTRIBUTING.md、SECURITY.md、CODE_OF_CONDUCT.md。
- 数字口径：评测表 TBD（v2 口径 reference_cache=ON 重测中），见 README。

[0.1.0]: https://github.com/<owner>/King.triton-kernel/releases/tag/v0.1.0
