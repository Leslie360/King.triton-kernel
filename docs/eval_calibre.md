# eval 口径对齐（P0-1, 2026-08-25）

> 目的: 理清我们各指标的精确含义, 与 Dr.Kernel/daVinci 对齐, 避免口径乐观。验收: 任何方法回测 59% 既有口径(3轮连续)。

## 一、我们指标的精确定义（代码核实）

**eval 设置**: 400 题 × 4 采样 × 最多 3 轮修复(best-of-samples × best-of-turns), TP8, mem_util 0.3。

| 指标 | 定义 | gs150 值 | 含义 |
|---|---|---|---|
| **pass@1** | score ≥ 0.99(main_grading.py compute_pass_at_k), 任一采样任一轮达标即算解出 | **59.0%** | reward = 0.5×correct + 0.5×min(speedup,3) → **≈正确且 speedup≥~0.98x** |
| **best_by_turn_3 correctness** | 任一采样任一轮有正确 kernel(不管速度) | 62.75% | 正确性上限 |
| **best_by_turn_3 fast@1** | 正确 kernel 的 performance 字段 ≥ 1.0 | 45.5% | 严格"正确+≥1.0x" |
| **best_by_turn_3 fast@1.2** | 正确 kernel 的 performance 字段 ≥ 1.2 | **19.5%** | 严格"正确+≥1.2x"(与 Dr.Kernel 可比) |
| 编译率(全轮均) | 所有轮次编译通过率 | 91.25% | 地基指标 |
| speedup_positive | 有正加速的轮次比例 | 21.6% | 性能信号 |

**关键发现**:
1. **pass@1(59%) 不是"正确且≥1.0x 加速"**(那是 fast@1=45.5%)。59% 与 45.5% 的 13.5pp 差 = 被判"解出"但严格口径下 speedup<1.0x 的样本。**59% 有口径乐观成分。**
2. **严格加速口径(fast@1.2=19.5%)是我们明显落后的地方**(vs Dr.Kernel 47.8%)——差距在**性能/加速**, 不在正确性(correctness 62.75% 已接近 best-turn 水平)。
3. 性能 reward 权重 0.5 且 speedup 上限 3.0, 使得"正确但仅微加速"即可 score≥0.99 → 训练也在优化"够用即可"而非"快"。

## 二、与公开基准对齐

| 基准 | 指标 | 数值 | 与我们可比口径 |
|---|---|---|---|
| **我们 gs150** | pass@1 / fast@1 / fast@1.2 | 59% / 45.5% / **19.5%** | — |
| Dr.Kernel-14B(2602.05885) | L2 Fast@1.2 best-turn | 47.8% | **我们 19.5% vs 47.8% → 性能轴落后**(若口径一致) |
| Dr.Kernel-14B | L2 Fast@1 best-turn | 80.9% | 我们 45.5% vs 80.9% |
| daVinci-kernel-14B(2606.16497) | L2 Fast1 | 70.6% | 需确认其 Fast1 定义 |
| KernelBench one-shot 前沿 | L2 fast_1 | o1=24% R1=36% | 我们多轮+RLVR 是其 3-6 倍(正确性轴) |

**⚠️ 口径 caveat**: Dr.Kernel/daVinci 的"Fast@1/1.2"精确语义(阈值、best-turn 聚合、TF32 基线)未在代码级对齐, 上表为近似。且 KernelBench-Verified(2607.16241)指出标准协议因 TF32 基线未启用+硬编码 bypass 会**高估速度**——我们的 19.5% 若在 verified 协议下可能更高, 反之亦然。**任何"超越"宣称前必须代码级对齐。**

## 三、建议的标准化汇报

对外统一报 **4 个数**(不再只报 pass@1):
1. pass@1(score≥0.99) —— 现有 headline
2. correctness(best_by_turn) —— 正确性上限
3. fast@1(正确+≥1.0x) —— 严格正确+达标
4. fast@1.2(正确+≥1.2x) —— 与 Dr.Kernel 可比

**对决策的意义**:
- gs180 三组 eval 都要算这 4 个数, 别只看 pass@1。
- 5 轮杠杆主要抬 correctness; **性能 gap(59→45.5→19.5)要靠性能 reward 强化**(提高 perf 权重/阈值), 是独立杠杆(见 ROADMAP P0 VeRPO/性能面)。
- 若 OSS/论文: 明确标注口径, 优先做 KernelBench-Verified 强化让数字更硬。

## 四、待办
- [ ] 代码级对齐 Dr.Kernel/daVinci 的 Fast@1/1.2 精确语义(读其 eval 代码)
- [ ] 把 4 个数标准化到 eval 输出(metrics.json 已有大部分字段)
- [ ] 决定是否调性能 reward 权重(0.5/0.5 → 偏性能)做性能杠杆 A/B

## 五、pass@1 聚合口径发现（2026-08-26, W3.4 审阅）

**关键事实**: 报告的 pass@1 (gs150=59%, gs180-3t=61.33%) 比从 eval_outputs 严格重算高 **8.5-10pp**。
- 严格重算 (`scripts/repro_pass_at_k.py`, 用 eval_outputs 的 score 字段, pass@k 估算器 k=1): gs150=**50.5%**, gs180-3t=**51.5%**
- eval_outputs 的 score = 0.5×correct + 0.5×perf (**无 coverage**, 已用通过样本验证 score=1.004=0.5×1+0.5×1.007+0.5×0)
- 报告值用 verl reward_tensor (可能含 coverage 加成) → 更宽松

**诚实处理**:
1. 对外统一报 4 个数 (pass@1/correctness/fast@1/fast@1.2) + **标注 pass@1 的口径范围** (严格 50-52% vs 报告 59-61%)
2. 用 `scripts/repro_pass_at_k.py` 做代码级复现, 防口径漂移
3. gs210 决策点: 两种口径都算, 报区间
4. **单次 eval 噪声 ±3-6pp**: 59→61.33 的 +2.33pp 在噪声内不显著 (p≈0.5), 需 ≥3 次 eval 平均才能确证

## 六、指标重命名（2026-08-26, 消除"pass@1"误导）

- **原 "pass@1"(报告值 59-65%) → `sample_solve_rate`**(per-sample 通过率, 每题内通过采样占比平均)。不是标准 pass@1!
- **标准 pass@1(每题≥1采样过) → `pass1_best`**(=95%, 能力上限)。
- repro_pass_at_k.py 已改名输出。对外讲清楚: "sample_solve_rate 65%" 是每采样通过率, "pass1_best 95%" 才是每题解出比例。
