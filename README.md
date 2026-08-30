# King.triton-kernel

> **让 LLM 写出"正确且更快"的 Triton kernel** — 14B 模型 · RLVR 后训练 · 单机 8×A800
> 参考同领域开源项目 [Dr.Kernel](https://github.com/alpha-beta-wang/Dr.-Kernel) 的架构组织（KernelGYM 评估环境 + drkernel 训练方法）。

[![arXiv](https://img.shields.io/badge/arXiv-待发布-blue)](https://github.com/alpha-beta-wang/Dr.-Kernel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**King.triton-kernel** 是一个 **Triton kernel 生成的强化学习后训练（RLVR）管线**：
模型被训练成**自主分析 kernel、写 Triton 实现、编译、执行验证、按速度反馈迭代修复**的 agent。

## 🏆 核心洞见（项目最有价值的部分）

**奖励"执行正确"，编译只是硬门槛 — 编译率 ≠ 正确率。**

- 编译失败 = 硬惩罚（-1.0 量级）
- 正确性 = 0.4 权重，性能 speedup = 0.3，编译 = 0.3
- 多轮修复（默认 5 轮），配合 **verifier 自适应终止**（速度达标即早停，防退化）
- 判分走真实 GPU 执行：`正确 ∧ speedup ≥ 阈值`，失败轮（performance=0）不混入速度统计

这是本项目与"刷编译率"类方法的本质区别：**reward 信号是"跑对了没"，不是"编译过了没"**。

## 架构

```
King.triton-kernel/
├── kernelgym/   # GPU 分布式评估环境（独立副本，自包含）
│                 #   subprocess worker 池 / CUDA 事件计时 / correctness 校验 / 多后端
├── drkernel/    # RL 训练方法（独立副本：reward 实现 + 提示词工具）
│                 #   VERL 集成 / 多维度 reward / 训练配置
├── evals/       # 评测口径与聚合（agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py）
├── docs/        # 核心洞见 + 评测口径文档
├── scripts/     # 训练/评测/部署脚本
├── assets/      # 图表
├── requirements.txt
├── setup.sh
├── smoke_test.py
└── README.md
```

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

## 评测

- **主指标**：best-of-history（per-problem 最优提交）
- **4 数报告**：sample_solve_rate / correctness / fast@1 / fast@1.2（标注 best-of 预算）
- 对比基线：Dr.Kernel（arXiv 2602.05885）、daVinci（KernelBench L2）

## 安装与冒烟

```bash
bash setup.sh
python3 smoke_test.py http://<eval-server>:8004
```

## 开源状态

✅ **独立副本已就位（2026-08-30）** — `kernelgym/` 评估引擎、`drkernel/` reward 实现、`evals/` 口径工具均为真实代码副本（非软链、非共享盘），排除：私有数据集、训练 checkpoint、内部日志、verl_patch 内部 overlay。

> ⚠️ 训练栈（`main_grading.py` / verl 集成）依赖 [verl](https://github.com/verl-project/verl) 上游，未随仓库发布，需自行安装对齐版本。

## License

[MIT](LICENSE)
