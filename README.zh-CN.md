# King.triton-kernel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/Leslie360/King.triton-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/Leslie360/King.triton-kernel/actions/workflows/ci.yml)

[English](README.md) | **简体中文**

> **King.triton-kernel** 是一个强化学习训练框架：教会 14B 模型在真实 GPU 上编写、编译并运行更快的 Triton kernel，把算子规格变成 KernelBench L2 上可验证的加速——另附一套独立的 GPU 评估环境，可用于任何需要真实执行校验的代码生成任务。

面向 KernelBench L2 的 **Triton kernel 生成 RLVR 训练管线**：将 14B 模型训练为能自主完成「分析算子 → 编写 Triton 实现 → 编译执行 → 按正确性与速度反馈迭代修复」全流程的 agent，并提供一套可独立复用的 GPU 分布式评估环境。

---

## 目录

- [简介](#简介)
- [背景与问题](#背景与问题)
- [方法](#方法)
- [评测结果](#评测结果)
- [安装与快速开始](#安装与快速开始)
- [仓库结构与分层边界](#仓库结构与分层边界)
- [相关工作](#相关工作)
- [文档索引](#文档索引)
- [开源状态](#开源状态)
- [许可](#许可)

## 简介

本项目包含两个可独立使用的交付物：

| 交付物 | 说明 | 适用场景 |
|---|---|---|
| **kernelgym 评估环境** | 自包含的 GPU 分布式判分服务：子进程隔离执行、CUDA 事件计时、正确性校验、参考计时缓存 | 任何需要「真实执行验证」的代码生成任务，可直接复用，不绑定本项目训练栈 |
| **RLVR 训练方法** | 以「执行正确」为核心奖励信号的多轮修复强化学习管线（训练栈未随本仓库发布，见安装说明） | kernel / 代码生成类 RL 后训练 |

目标读者：

1. **AI Infra / RL 研究者与工程师**——希望复现 kernel 生成 RL 训练，或复用一套带「口径治理」的执行验证评估环境；
2. **相关方向的评估工作者**——关注 KernelBench L2 上不同方法的可比口径与复现纪律。

**硬件要求**：训练需单机 8 卡 A800（或同等显存的 GPU）；单独使用 kernelgym 评估环境只需 1 卡。

## 背景与问题

用 LLM 生成 GPU kernel，有两个绕不开的工程问题：

**其一，编译通过不等于正确。** 一个能通过编译的 kernel 完全可能输出错误结果甚至触发非法内存访问。如果奖励信号只反映「能不能编译」，模型会学会写「编译友好但功能错误」的代码。因此奖励必须来自**真实 GPU 上的执行结果**——这是本项目的核心方法主张（与 Dr.Kernel / daVinci / DRTriton / CUDA Agent 等工作的共识）。

**其二，评测口径容易失真。** 同一批模型产出，换个统计口径（采样预算、轮次范围、加速比阈值、参考实现计时方式）就能差出十几个百分点。缺乏口径纪律的数字既无法复现，也无法与外部工作比较。本项目把口径治理作为一等工程问题对待（见[评测结果](#评测结果)一节的口径说明）。

## 方法

### 核心主张

> 奖励信号是「跑对了没」，不是「编译过了没」。编译率 ≠ 正确率。

落实到奖励设计：

- **编译失败：硬惩罚**（-1.0 量级），直接拦截；
- **奖励构成**：正确性 0.4 + 加速比 0.3 + 编译 0.3；
- **判分铁律**：kernel 必须在真实 GPU 上执行通过且输出与参考实现对齐（容差内），执行失败的提交不进入加速比统计——「败」与「慢」严格区分；
- **多轮修复（默认 5 轮）+ 自适应终止**：模型根据评估环境反馈（编译错误 / 正确性失败 / 加速比不足）迭代改进，速度达标后提前终止，防止后续轮次退化。

### 训练管线

```
SKILL 注入 ──► 修复飞轮 SFT ──► RLVR (TRLOO)
    │              │                │
    │         (正确性主战场)    (性能轴主攻)
    │              │                │
    └──► 多轮自我修复 + 自适应终止 ◄─┘
```

- **SKILL 注入**：将常见错误模式（dtype / mask / 形状 / 数值稳定性 / 边界等 10 类）提炼为技能文档，按轮次反馈注入提示词；
- **修复飞轮 SFT**：收集 RL 过程中的失败-修复轨迹蒸馏回流，主攻正确性；
- **RLVR**（主算法 TRLOO）：在可验证奖励上做强化学习，主攻性能轴。

### 口径治理

为保证对外数字可复现、可比较，本项目执行以下口径纪律（细则见 `docs/ARCHITECTURE.md` §5）：

1. **参考计时缓存**：判分不重跑参考实现的计时，消除 run 级分母抖动；
2. **权威聚合唯一**：`evals/agg_eval.py` 是唯一权威聚合脚本，其余脚本只做交叉验证；
3. **失败轮次不入速度统计**：只有 `correctness=True` 的轮次参与加速比统计；
4. **数字四件套**：每个对外数字必须附脚本版本（含 SHA）、采样预算、提取器版本、口径版本，禁止口径混用。

## 评测结果

> 数字已锁定（2026-09-02，v2 口径：参考缓存开启），三次独立评测取均值。

**主指标 fast@1.2**：逐题取历史最优（per-problem best-of-history，每题所有采样 × 所有轮次中的最优提交）口径下，加速比 ≥1.2× 且正确的题目占比。评测条件：KernelBench L2（100 题），8 采样 × 5 轮修复，A800 单机，TF32 开启。

| 模型 | fast@1.2（逐题历史最优） | 备注 |
|---|---|---|
| **King.triton-kernel（14B，本项目）** | **61.0% ± 6.2pp**（n=3） | 主打数字 |
| SFT v2（从 RL 修复飞轮蒸馏，约 10 分钟 LoRA） | 68.3% ± 6.2pp（n=3） | 与 RL 基线统计上不可区分，训练成本约 1%；其中一次运行达 72%，发布后待复验 |
| Dr.Kernel-14B | 47.8%（最优轮次口径，STTS†） | arXiv 2602.05885，采样预算未披露 |
| daVinci-14B | 27.1%（L2 Fast@1.2，最优轮次口径） | arXiv 2606.16497 |

口径注意事项：

- fast@1 与 fast@1.2、逐题历史最优与最优轮次、加速比 ≥1.0× 与 ≥1.2× 属于不同口径，不可直接比较。daVinci 双口径分开报：27.1% = L2 Fast@1.2，70.6% = L2 Fast@1。
- 测量条件（205 号机，8×A800）：时钟 1155/1410MHz 自然加速（无系统权限不可锁频）；参考实现计时存在 12–20% 的机器间差异（机器固有属性），上表数字在 205 单机内自洽；正确性 82.3% 按 TF32 参考实现判定（TF32 开启口径）。
- 单次评测噪声 ±3–6pp，重要结论以 ≥3 次评测均值报告。

## 安装与快速开始

**环境要求**：Python ≥ 3.10，Linux + NVIDIA GPU（驱动可见即可，CUDA 由 torch/triton 自带），Redis。

```bash
# 1. 克隆并安装依赖
git clone https://github.com/Leslie360/King.triton-kernel.git
cd King.triton-kernel
bash setup.sh          # 等价于 pip install -r requirements.txt

# 2. 以可编辑方式安装（提供 kernelgym-server 等命令行入口）
pip install -e .

# 3. 启动评估服务（默认端口 10907，可用环境变量 API_PORT 覆盖）
kernelgym-server

# 4. 冒烟测试（校验服务健康状态）
python3 smoke_test.py http://localhost:10907
```

命令行入口（`pyproject.toml [project.scripts]`）：

| 命令 | 用途 |
|---|---|
| `kernelgym-server` | 判分 API 服务 |
| `kernelgym-worker` | GPU 执行 worker |
| `kernelgym-worker-monitor` | worker 监控 |
| `kernelgym-single-worker` | 单 worker 模式（调试用） |

> 本仓库仅发布评估环境与口径工具。RL 训练栈（verl 集成）不在本仓库内，它消费本仓库产生的判分结果；`kernelgym/` 评估环境无 verl/ray/vllm 依赖。

## 仓库结构与分层边界

```
King.triton-kernel/
├── kernelgym/   # GPU 分布式评估环境（核心库 + server + worker）
├── evals/       # 评测口径与聚合（agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py）
├── docs/        # ARCHITECTURE.md（架构与口径）/ ROADMAP.md
├── tests/       # 166 个测试函数（CI 跑 CPU 纯逻辑子集）
├── setup.sh      # 依赖安装
└── smoke_test.py
```

分层边界（详见 `docs/ARCHITECTURE.md` §6）：

| 层 | 位置 | 依赖 | 随仓发布 |
|---|---|---|---|
| 评估核心库 | `kernelgym/core|schema|workflow|backend|toolkit|worker/` | torch / triton / numpy | 是 |
| 判分服务 | `kernelgym/server|config|utils/` | fastapi / redis | 是 |
| 评测聚合 | `evals/` | pandas / pyarrow | 是 |

分层铁律：`kernelgym/` 保持零 verl/ray/vllm 依赖，评估环境必须可独立复用；训练侧不内嵌评测逻辑，只读取 server 返回的判定结果。

## 相关工作

| 工作 | 关系 |
|---|---|
| [Dr.Kernel](https://github.com/alpha-beta-wang/Dr.-Kernel)（arXiv 2602.05885） | 本项目参考其「评估环境 + 训练方法」的架构组织，并在评测口径治理上做了差异化工程 |
| daVinci（arXiv 2606.16497） | KernelBench L2 对比基线 |
| DRTriton / CUDA Agent | 同以真实 GPU 执行验证为核心奖励信号的领域工作 |
| [verl](https://github.com/verl-project/verl) | 训练框架上游依赖 |

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构、数据流、评测口径定义、分层边界 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 当前 / 近期 / 中期 / 长期计划 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南（含评测口径纪律） |

## 开源状态

v0.1.0（2026-08-31）首次发布：`kernelgym/` 评估引擎、`drkernel/` 奖励实现、`evals/` 口径工具均为真实代码副本，已排除私有数据集、训练 checkpoint、内部日志与内部路径。部分源自 verl / Dr.Kernel 生态的文件保留 Apache-2.0 版权头，详见 [NOTICE](NOTICE)。

## 许可

[MIT](LICENSE)。含 Apache-2.0 部分文件的授权说明见 [NOTICE](NOTICE)。

## 贡献

欢迎通过 issue 与 PR 参与，提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)——其中评测口径纪律对本项目同样约束外部贡献。
