# King.triton-kernel

> **让 LLM 写出"正确且更快"的 Triton kernel** — 14B 模型 · RLVR 后训练 · 单机 8×A800
> 面向 **KernelBench L2** 的 Triton kernel 生成 RLVR 管线，参考同领域开源项目 [Dr.Kernel](https://github.com/alpha-beta-wang/Dr.-Kernel)（arXiv 2602.05885）架构组织（评估环境 + 训练方法），并在**评测口径诚实化**上做了差异化工程。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**King.triton-kernel** 是一个 **Triton kernel 生成的强化学习后训练（RLVR）管线**：
模型被训练成**自主分析 kernel、写 Triton 实现、编译、执行验证、按速度反馈迭代修复**的 agent。
单机 8×A800 即可训练，产出 14B 量级的 Triton kernel 生成模型。

---

## 🏆 方法主张

**奖励"执行正确"，编译只是硬门槛 — 编译率 ≠ 正确率。**

这是本项目训练方法的核心主张，也是领域共识（Dr.Kernel / daVinci / DRTriton / CUDA Agent 等均以真实 GPU 执行验证为核心奖励信号）：

- 编译失败 = 硬惩罚（-1.0 量级）
- 正确性 = 0.4 权重，性能 speedup = 0.3，编译 = 0.3
- 多轮修复（默认 5 轮），配合 **verifier 自适应终止**（速度达标即早停，防退化）
- 判分走真实 GPU 执行：`正确 ∧ speedup ≥ 阈值`，失败轮（performance=0）不混入速度统计

**reward 信号是"跑对了没"，不是"编译过了没"** — 刷编译率学不到正确性，这是本项目与"刷编译率"类方法的本质区别。

---

## 📐 架构

```
King.triton-kernel/
├── kernelgym/   # GPU 分布式评估环境（独立副本，自包含，零 verl/ray/vllm 依赖）
│                 #   subprocess worker 池 / CUDA 事件计时 / correctness 校验 / 多后端
├── drkernel/    # RL 训练方法（独立副本：reward 实现 + 提示词工具）
│                 #   VERL 集成 / 多维度 reward / 训练配置
├── evals/       # 评测口径与聚合（agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py）
├── docs/        # 核心洞见 + 评测口径文档 + 发布清扫清单
├── requirements.txt
├── setup.sh
└── smoke_test.py
```

### kernelgym/ — 独立评估环境（可直接复用）

- **自包含**：仅依赖 torch/triton/redis/fastapi/httpx，无 verl/ray/vllm 耦合，可独立启动判分 server。
- **subprocess 隔离**：kernel 执行在子进程池中进行，CUDA 错误/非法内存访问被隔离，worker 不死。
- **真实执行验证**：CUDA 事件计时 + correctness 校验（rtol/atol 容差）+ 多后端支持。
- **reference 分母固定**（`InMemoryReferenceCache`，key=uuid+ref_hash+is_valid）：判分不重跑 reference 分母，消除 run 级分母抖动——**口径诚实的工程基础**。

---

## 管线总览

```
SKILL 注入 ──► 修复飞轮 SFT ──► RLVR (TRLOO)
    │              │                │
    │         (正确性主战场)    (性能轴主攻)
    │              │                │
    └──► 多轮自我修复 + 自适应终止 ◄─┘
```

## 方法要点

1. **执行验证为硬门槛**：只有 CUDA 执行通过且输出正确的 kernel 才进入速度统计
2. **speedup 分母固定**（reference 计时缓存）：消除 run 级分母抖动，口径诚实
3. **多轮修复循环**：模型收到 server 反馈（compile/correctness/speedup/error）→ 迭代改进
4. **权威口径 = per-problem best-of-history**：每题任一样本任一轮最优，跨全部轮次
5. **口径审计纪律**：对外数字必须附口径（TP/样本数/轮次/refcache 状态/脚本 SHA），禁止口径混用

---

## 📊 评测

> **已锁定（2026-09-02, 静默窗 v2 口径）** — 结果表由 v2 口径（reference_cache=ON）重测回填，权威数字已锁定。

- **主指标**：best-of-history（per-problem 最优提交，权威实现 `evals/agg_eval.py`）
- **4 数报告**：sample_solve_rate / correctness / fast@1 / fast@1.2（标注 best-of 预算）
- **对比基线**：Dr.Kernel（arXiv 2602.05885）、daVinci（KernelBench L2）

| 模型 | KernelBench L2 fast@1.2 (best-of) | 备注 |
|---|---|---|
| **King.triton-kernel (14B)** | **61.0% ± 6.2pp** (n=3, 100 题, 8 采样×5 轮 best-of-history, A800, TF32-ON) | headline |
| SFT v2（distilled from RL flywheel, ~10min LoRA） | 65.0% ± 6.2pp (n=3) | 与 RL 基线统计上不可区分 @ ~1% 训练成本 |
| Dr.Kernel-14B | 47.8%（best-turn, STTS†） | arXiv 2602.05885 · 预算口径未披露 |
| daVinci-14B | 27.1%（L2 **Fast@1.2**, best-turn） | arXiv 2606.16497 |

> ⚠️ **口径不可混用**：fast@1 vs fast@1.2、best-of vs best-turn、≥1.0x vs ≥1.2x 必须各标定义。daVinci 双口径分开报：27.1% = L2 Fast@1.2，70.6% = L2 Fast@1，不得混用。

**SFT v2 表述**：SFT v2 (distilled from the RL flywheel, ~10min LoRA): 65.0% ± 6.2pp (n=3) — statistically indistinguishable from the RL baseline at ~1% training cost; one run (72%) flagged as potential upside, post-release re-verification pending.

> 📏 **测量条件**（205, 8×A800）：时钟 1155/1410MHz 自然 boost（无 CAP_SYS_ADMIN 不可锁频）；reference 分母 12-20% 机差为机器属性（与 dev1 相比），本报告全部数字在 205 单机内自洽；correctness 82.3% 对 TF32 reference 判定（TF32-ON 口径）。

---

## 安装与冒烟

```bash
bash setup.sh
python3 smoke_test.py http://<eval-server>:8004
```

---

## 开源状态

✅ **独立副本已就位（2026-08-30）** — `kernelgym/` 评估引擎、`drkernel/` reward 实现、`evals/` 口径工具均为真实代码副本（非软链、非共享盘），排除：私有数据集、训练 checkpoint、内部日志、verl_patch 内部 overlay。

> ⚠️ 训练栈（`main_grading.py` / verl 集成）依赖 [verl](https://github.com/verl-project/verl) 上游，未随仓库发布，需自行安装对齐版本。

## License

[MIT](LICENSE)
