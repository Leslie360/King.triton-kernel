---
name: Bug report
about: Report a bug to help us improve King.triton-kernel
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

<!--
感谢报告 bug。请尽量补全以下信息，特别是"评测口径"部分——
本项目对评测口径（fast@1 vs fast@1.2、best-of vs best-turn、refcache 状态）非常敏感，
口径不明会导致无法复现与定位。
-->

## 描述

<!-- 清晰、简洁地描述 bug 是什么。 -->

## 复现步骤

1. 执行命令 / 脚本：
   ```bash
   # 粘贴最小复现命令
   ```
2. 输入 / 配置：
3. 观察到的结果：

## 期望行为

<!-- 你期望发生什么？ -->

## 实际行为

<!-- 实际发生了什么？贴出报错堆栈 / 日志片段。 -->

## 环境

- **OS**: （如 Ubuntu 22.04）
- **Python 版本**: （如 3.10.14）
- **torch 版本**: （如 2.4.0，`python -c "import torch; print(torch.__version__)"`）
- **triton 版本**: （如 3.0.0，`python -c "import triton; print(triton.__version__)"`）
- **GPU / 驱动**: （如 A800 / CUDA 12.4；`nvidia-smi`）
- **是否使用 reference_cache**: （ON / OFF）
- **verl / vllm 版本**（若与训练侧相关）:

## 评测口径（如涉及数字）

<!-- 若 bug 涉及正确率/速度等数字, 请务必标注口径, 否则按"口径不明"处理: -->
- 指标: （sample_solve_rate / correctness / fast@1 / fast@1.2 / 其他）
- 采样方式: （best-of N / best-turn / greedy）
- 轮次范围: （如 5 轮修复全轮次 / 仅最终轮）
- speedup 阈值: （≥1.0x / ≥1.2x / 其他）
- 口径工具脚本 SHA: （若用到 evals/agg_eval.py 等）

## 额外上下文

<!-- 截图、相关 issue/PR、任何你觉得有用的信息。 -->
