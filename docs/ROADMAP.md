# King.triton-kernel 路线图

> 目标：让 14B 模型写出**正确且更快**的 Triton kernel，作为 MIT 开源、可复现、单机 8×A800 即可训练/评估的 RLVR 管线。
> 分层定位：先立住"评估环境 + 诚实口径"这个可信根基，再强化性能轴，最后走向社区共建。
> 相关：`docs/ARCHITECTURE.md`（结构）、`docs/eval_calibre.md` + `docs/CALIBER_TABLE.md`（口径）。

---

## 当前（口径锁定中）

> 一句话：**数字还不硬，先锁死怎么数，再谈更好。**

### 目标
1. **口径冻结并落文档**：`agg_eval.py` 为唯一权威聚合，4 数标准（sample_solve_rate / correctness / fast@1 / fast@1.2），best-of-history 定义冻结。
2. **refcache=ON（v2 口径）全面重测**：消除 run 级分母抖动，所有对外数字统一到 v2。
3. **禁止口径混用**：fast@1 vs fast@1.2、best-of vs best-turn、≥1.0x vs ≥1.2x 各标定义。

### 关键动作
- [ ] `evals/agg_eval.py` 锚定复核（含失败 turn 不入 speedup 纪律）
- [ ] refcache=ON 下 gs300×3 重测，回填 README 结果表（TBD）
- [ ] 交叉验证脚本（`compare_gs_eval.py` / `repro_pass_at_k.py`）只做验证，不产出决策数字
- [ ] 补齐 `docs/CALIBER_TABLE.md` 待核项（Dr.Kernel 预算、daVinci Fast1 定义、CUDA Agent 基线）

### 完成判据
- 每个对外数字附四件套（脚本 SHA / 预算 / 提取器版本 / 口径版本）
- README 无 TBD，数字均来自 v2（refcache=ON）

---

## 近期（SFT v2 训练 / eval×3）

> 一句话：**把模型变强（正确性 + 性能），用 v2 口径三次 eval 验证。**

### 主线
1. **SFT 库 v2 重建**：7:3 配比 + ≥100 题覆盖（依赖 rollout v2 新 dump 全量）。
2. **SFT v2 训练**（6-8h）+ **eval×3**（v2 口径，取均值/std，单次噪声 ±3-6pp）。
3. 修复飞轮 + SKILL 注入迭代，作为正确性主战场。

### 评估矩阵
- 每次训练后：**sample_solve_rate / correctness / fast@1 / fast@1.2** ×（single / best-of-history）×（refcache=ON）
- 对比基线：Dr.Kernel（47.8% best-turn）、daVinci（70.6% Fast1 / 27.1% Fast@1.2），**先查可比性表**。

### 完成判据
- SFT v2 达到或超过当前 best（B2 小胜线），correctness 持续爬升
- 三次 eval 的 fast@1.2 best-of 均值/方差可报告，无明显口径漏洞

---

## 中期（性能轴优化 / 更多后端）

> 一句话：**正确性已接近 best-turn 水平，差距在"快"上——把性能轴立起来。**

### 性能轴（当前最大差距）
- 现状：correctness ~62.75% 已接近 best-turn，但 fast@1.2 best-of 落后 Dr.Kernel（我们 ~39% vs 47.8%，单次落后更大）。
- 方向：
  - [ ] 提高性能 reward 权重 / 阈值（从 0.5/0.5 偏向性能），做性能杠杆 A/B
  - [ ] 性能 reward 上限 3.0 评估（当前训练可能只优化"够用即可"而非"快"）
  - [ ] VeRPO / 性能面强化（参考 `eval_calibre.md` 的独立性能杠杆）
  - [ ] TF32 baseline 对齐（`ENABLE_TF32_BASELINE`，对齐 KernelBench-Verified 协议防高估）

### 更多后端
- [ ] backend/ 抽象已就位（Triton / CUDA），扩展更多后端（如 torch.compile 基线、CUTLASS 片段、多精度）
- [ ] 多后端评估可复现：同一 kernel 跨后端对比，扩大覆盖

### 训练方法
- [ ] 技能库检索 + 执行验证准入（学 daVinci）
- [ ] agentic RL（学 CUDA Agent，多角色/工具）
- [ ] 多轮 reward 设计（学 MusaCoder：首轮锚定 / 动态重试）

### 完成判据
- fast@1.2 best-of 在 v2 口径下追平/超过 47.8% 线（同预算）
- ≥2 个后端稳定运行，评估口径不因后端切换而漂移

---

## 长期（社区共建）

> 一句话：**让这个评估环境 + 诚实口径成为社区可信的公共资源。**

### 目标
1. **评估环境标准化**：`kernelgym/` 独立、可复用、零 verl/ray/vllm 依赖 —— 成为 Triton kernel 生成的公共评测基座。
2. **诚实口径方法论**：best-of-history / 失败 turn 不入 speedup / refcache 固定分母 —— 作为"可复现数字"的行业惯例推广。
3. **开放式比较基准**：统一 4 数报告 + 预算标注，让任何方法都能在相同口径下被公平比较。

### 动作
- [ ] 发布独立副本 + 完整文档（已就位，见 RELEASE_CHECKLIST.md）
- [ ] 公开模型 + 数据 provenance + license
- [ ] 社区 PR：多后端、新基准（KernelBench-Verified）、训练脚本可复现
- [ ] CI / 测试矩阵（`pyproject.toml` 已含 pytest / ruff / mypy / pre-commit 骨架）
- [ ] 跨硬件可移植（非 A800 也能跑，扩大覆盖）

### 完成判据
- 外部项目可独立 clone + `setup.sh` + `smoke_test.py` 跑通
- ≥1 个外部贡献者提交多后端 / 新基准 PR
- 诚实口径成为 README 主张的差异化亮点（"执行正确非编译"已是领域标配，我们的差异化 = 独立评估环境 + 诚实口径方法论）

---

## 里程碑总览

| 阶段 | 时间 | 主线 | 验收 |
|---|---|---|---|
| 当前 | 2026-08 末 | 口径锁定（refcache=ON 重测） | 数字全 v2，无 TBD，四件套齐全 |
| 近期 | 2026-09 | SFT v2 训练 + eval×3 | 正确性/性能达标，三次 eval 可报告 |
| 中期 | 2026 Q4 | 性能轴优化 + 更多后端 | fast@1.2 追平 47.8%，≥2 后端 |
| 长期 | 2027 | 社区共建 | 独立可复用 + 外部 PR + 诚实口径推广 |
