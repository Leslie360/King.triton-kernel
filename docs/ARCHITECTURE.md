# King.triton-kernel 架构

> MIT 开源的 Triton kernel 生成 RLVR（强化学习 + 可验证奖励）管线。
> 本文件描述代码结构、数据流、评测口径与分层边界。面向新贡献者，先读本文件再读 `README.md`。

---

## 0. 一句话定位

让 14B 模型学会**分析 kernel → 写 Triton 实现 → 编译 → 在真实 GPU 上执行验证 → 按正确性与速度反馈迭代修复**。
核心主张：**奖励"执行正确"，编译只是硬门槛**（详见父仓内部文档 `core_insight_reward_execution.md`）。

```
SKILL 注入 ──► 修复飞轮 SFT ──► RLVR (TRLOO)
    │              │                │
    │         (正确性主战场)    (性能轴主攻)
    └──► 多轮自我修复 + 自适应终止 ◄─┘
```

---

## 1. 顶层目录结构

```
King.triton-kernel/
├── kernelgym/   # 独立 GPU 评估环境（subprocess worker / 计时 / correctness / 多后端）
├── drkernel/    # 训练侧（reward 实现 + 代码提取工具；VERL 集成）
├── evals/       # 评测口径与聚合（agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py）
├── docs/        # 架构 / 口径 / 发布 / 路线图
├── pyproject.toml   # 打包：kernelgym-server / worker / worker-monitor / single-worker 入口
├── setup.sh     # 安装 + 冒烟
└── smoke_test.py
```

### 1.1 独立副本原则

`kernelgym/` 与 `drkernel/` 是**真实代码副本**（非软链、非共享盘、不含 .git 历史），
可被外部项目直接复用。排除了私有数据集、训练 checkpoint、内部日志与 verl_patch 内部 overlay。
训练栈（`main_grading.py` / verl 集成）依赖上游 [verl]，不随仓库发布，需自行安装对齐版本。

---

## 2. 模块图 — kernelgym 子包职责

`kernelgym/` 是一个**分层**的独立评估引擎，零 verl / ray / vllm 依赖，
只依赖 torch / triton / redis / fastapi / httpx。

```
                         ┌──────────────────────────────────────────┐
                         │              server (API 层)              │
                         │  api/server.py · models.py · utils.py     │
                         │  code_retry_manager · task_manager        │
                         └───────────────┬──────────────────────────┘
                                         │ SchedulerAPI (submit/wait/status/cancel)
                         ┌───────────────▼──────────────────────────┐
                         │              core (抽象层)                │
                         │  types.py (TaskSpec/Result/Artifact/      │
                         │            Metric/TaskGroup)              │
                         │  workflow.py (WorkflowController)         │
                         │  scheduler.py (SchedulerAPI)              │
                         │  registry.py (通用注册表)                  │
                         └───────────────┬──────────────────────────┘
                                         │ workflow 编排
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
   ┌────────▼────────┐        ┌──────────▼─────────┐      ┌──────────▼─────────┐
   │  workflow/       │        │     schema/        │      │    backend/        │
   │  kernelbench.py  │        │  task.py result.py │      │  base.py           │
   │  kernel_simple   │        │  serialization.py  │      │  triton.py         │
   │  helpers(引用缓存)│        │  simple_task.py    │      │  kernelbench/      │
   └────────┬────────┘        └─────────────────────┘      │  cuda/triton       │
            │ 产生 TaskSpec                                 └──────────┬─────────┘
            │                                                        │ Backend 接口
            ▼                                                        ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          worker (执行层)                                 │
   │  subprocess_pool.py (持久 worker 池 · CUDA 错误自动重启)                  │
   │  task_executor.py     (每任务 spawn 隔离)                                 │
   │  gpu_worker / single_worker / worker_monitor                              │
   └──────────────────────────────────────────────────────────────────────────┘
            │ 调用 toolkit.evaluate(task, backend)
            ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          toolkit (评测逻辑)                               │
   │  kernelbench/pipeline.py: 加载→编译→correctness→Triton检测→计时→coverage  │
   │  kernel_simple/                                                           │
   │  validation.py / base.py / registry.py                                    │
   └──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 各子包职责清单

| 子包 | 职责 | 关键文件 |
|---|---|---|
| **core/** | 领域抽象：任务规格、结果模型、工作流控制器、调度器接口、通用注册表 | `types.py` / `workflow.py` / `scheduler.py` / `registry.py` |
| **schema/** | 请求/响应数据结构与序列化 | `task.py` / `result.py` / `serialization.py` |
| **workflow/** | 端到端编排（一个 eval 请求如何拆成 reference + kernel 两个子任务并合并） | `kernelbench.py` / `kernelbench_helpers.py` |
| **backend/** | 后端执行适配器（编译/加载/运行抽象），目前 Triton / CUDA | `base.py` / `triton.py` / `kernelbench/` |
| **toolkit/** | 评测逻辑（correctness / 计时 / Triton 使用检测 / coverage），通过 registry 按名取 | `kernelbench/pipeline.py` / `base.py` |
| **worker/** | subprocess 隔离执行，CUDA 错误不拖垮主进程 | `subprocess_pool.py` / `task_executor.py` |
| **server/** | FastAPI API 层 + 任务管理 + 引用缓存注册 | `api/server.py` / `task_manager.py` |
| **config/** | 设置（env 驱动，含 reference_cache 开关） | `settings.py` |
| **utils/** | 错误分类 / GPU 诊断 | `error_classifier.py` / `gpu_diagnostics.py` |

> 注：`core/workflow.py` 中的 `WorkflowController` 是**抽象**，具体实现在 `workflow/kernelbench.py`
> （`KernelBenchWorkflowController`），通过 `server/scheduler.py` 的 `TaskManagerScheduler` 适配 `core.scheduler` 接口。

---

## 3. 数据流 — 提交 kernel → reward

一次完整 eval 请求的端到端路径（以 KernelBench workflow 为例）：

```
1. 客户端提交 EvaluationTask
   ├─ task_id / kernel_code / reference_code / uuid / use_reference_cache
   └─ run_correctness / run_performance / measure_performance / enable_profiling ...
        │
        ▼
2. server/api/server.py 收到请求
   └─ 委托 KernelBenchWorkflowController.handle_request()
        │
        ▼
3. _validate_inputs(): 校验代码含 class Model/ModelNew、resources、引用缓存约束
        │
        ▼
4. _create_paired_tasks(): 拆成两个子任务
   ├─ kernel_task    (kind="kernelbench.kernel", 评测自定义 kernel)
   └─ reference_task (kind="kernelbench.ref", 测 reference 分母)
   ★ 若 use_reference_cache 且缓存命中 → 不创建 reference_task（见 §4）
        │
        ▼
5. scheduler.submit() → task_manager → worker pool 调度
   └─ worker (subprocess, spawn 隔离) 执行
        │
        ▼
6. toolkit/kernelbench/pipeline.py 逐阶段执行:
   a. 加载 original model + 输入 (get_init_inputs/get_inputs)
   b. backend.compile(custom_model_src)   ← 编译（硬门槛）
      ├─ 编译失败 → KernelExecResult(compiled=False) → reward=-1.0 量级
      └─ 编译通过 → backend.load + create_model
   c. _run_correctness_step()  ← correctness 校验 (rtol/atol 容差)
      ├─ 不正确 → compiled=True, correctness=False → reward=-0.5
      └─ 正确   → 继续
   d. _run_triton_detection_step()  ← 检测是否真用了 Triton（防 decoy）
      └─ 未用 Triton 却标 triton → decoy_kernel=True
   e. _run_performance_step()  ← CUDA 事件计时 (num_perf_trials 次取 mean)
      └─ kernel_runtime → speedup = reference_runtime / kernel_runtime
        │
        ▼
7. 子任务结果合并 (_combine_results) → EvaluationResult
   ├─ compiled / correctness / decoy_kernel
   ├─ reference_runtime / kernel_runtime / speedup
   └─ status / error_code / metadata (含 profiling, coverage)
        │
        ▼
8. _persist_result() 落盘 (eval_results_path)
   └─ reward 由训练侧 (drkernel) 或 agg_eval 依据 speedup 计算
```

### 3.1 reward 计算（训练侧 / 评估侧一致口径）

核心 reward 映射（口径细则见父仓内部文档 `docs/eval_calibre.md`）：

| 结果 | reward | 说明 |
|---|---|---|
| 编译失败 | **-1.0** | 硬门槛，最负 |
| 编译通过但 wrong | **-0.5** | 非硬门槛但明确为负 |
| 正确但 decoy（未真用 Triton） | **-0.3** 附加 | 软惩罚 |
| 正确 | 正（由 correctness + performance 缩放） | `0.5×correct + 0.5×min(speedup, 3)` |

多轮修复（默认 5 轮）中，模型收到 server 返回的 `compile/correctness/speedup/error` 反馈并迭代改进；
**verifier 自适应终止**（速度达标即早停）防止退化。

---

## 4. reference_cache 设计 — 为何固定分母消除抖动

> 代码：`kernelgym/workflow/kernelbench_helpers.py`（`InMemoryReferenceCache`）+ `kernelgym/server/api/server.py` 注册。

### 4.1 问题

`speedup = reference_runtime / kernel_runtime`。reference 分母每次 eval 都重新计时，
而**同一 reference 在不同 run 间有机器级抖动**（实测 gs300 方差 std≈23.6，同一 reference 差 42%）。
这导致：**跑得快不是模型变强，而是 reference 这次恰好慢** —— 严重污染对外数字。

### 4.2 设计

- `InMemoryReferenceCache`：`key = (uuid, reference_code_hash, is_valid)`。
- reference 首次跑完**写进缓存**（`_put_reference_cache`），后续同题同 reference 同 validation 状态**复用分母**，不再重跑。
- `use_reference_cache=True` 且 `uuid` 存在时才启用；缺 uuid 会校验失败。
- 缓存为**本 server 进程内**内存缓存（server 重启即清空，分母重新计时）——避免跨机器缓存造成口径漂移。

### 4.3 为何如此设计

1. **固定分母**：同题所有样本/轮次用同一个 reference_runtime → 分母抖动从 run 方差中剔除。
2. **诚实**：对外数字不再因"这次 reference 慢"而虚高，消除 refcache=OFF 时 39% 与 refcache=ON 时 ~20-29% 的自相矛盾（发布红线，见父仓内部文档 `RELEASE_CHECKLIST_oss_internal.md`）。
3. **口径可复现**：任何方法重测需在 refcache=ON 下进行，否则数字不可比。

---

## 5. 评测口径 — best-of-history 与失败 turn

> 权威定义冻结于父仓内部文档 `docs/CALIBER_TABLE.md`。唯一权威实现 = `evals/agg_eval.py`。

### 5.1 权威指标

对外统一报 **4 个数**（不再只报 pass@1）：

| 指标 | 定义 | 说明 |
|---|---|---|
| **sample_solve_rate** | per-sample 通过率（正确且达标，每题内通过采样占比平均） | 原"报告 pass@1"（注意：**不是**标准 pass@1） |
| **pass1_best** | 标准 pass@1（每题 ≥1 采样过） | 能力上限（=95% 量级） |
| **fast@1** | 正确 kernel 的 performance ≥ 1.0 | 严格"正确 + ≥1.0x" |
| **fast@1.2** | 正确 kernel 的 performance ≥ 1.2 | 与 Dr.Kernel/daVinci 可比 |

### 5.2 best-of-history

**权威口径 = per-problem best-of-history**：每题取所有采样 × 所有轮次中的最优提交。
即 `∃ (sample, turn): 正确 ∧ speedup ≥ 阈值` 即判定该题通过。

### 5.3 失败 turn 不入 speedup（关键纪律）

**失败 turn（performance=0）不计入 speedup 候选** —— 这是"败"不是"慢"，禁止混进 speedup 统计。
`agg_eval.py` 的 `per()` 逻辑：只有 `correctness=True` 的 turn 才参与 `bs = max(bs, sp)` 统计。
> 2026-08-30 flip 复算曾差一个量级，正是因混入了失败 turn。此纪律不可破。

### 5.4 聚合与方差

- 单次 eval 噪声 **±3-6pp**，重要结论需 ≥3 次 eval 取平均（`agg_eval.py <run1> <run2> <run3> <prefix>`）。
- 报告均值与 std；fast@1.2 best-of-history 是决策锚定指标。

### 5.5 口径纪律（CALIBER_TABLE 摘录）

- **禁止两个"官方"脚本并存**：`agg_eval.py` 是唯一权威；`compare_gs_eval.py` 与手写 parquet 只许交叉验证。
- 每个数字附四件套：**脚本（含 SHA）/ 预算 / 提取器版本 / 口径版本**。
- 对 Dr.Kernel（47.8% best-turn, 预算未披露）、daVinci（70.6% L2 Fast1 / 27.1% L2 Fast@1.2）比较前**必须查可比性表**，不同口径标注"不可比"。

---

## 6. 分层边界 — 纯库 vs server vs 训练侧

| 层 | 位置 | 依赖 | 是否随仓发布 | 说明 |
|---|---|---|---|---|
| **纯库（kernelgym 核心）** | `core/ schema/ workflow/ backend/ toolkit/ worker/` | torch / triton / numpy | ✅ 发布 | 独立评估引擎，无 verl/ray/vllm 耦合，可独立启动判分 server |
| **server（API 服务）** | `server/ config/ utils/` | fastapi / uvicorn / pydantic / redis | ✅ 发布 | 暴露 HTTP 判分接口（`kernelgym-server` 入口），注册 reference_cache |
| **训练侧（drkernel + verl）** | `drkernel/` | verl / ray / vllm / httpx | ⚠️ 依赖 verl，不随仓发布 | reward 实现 + 代码提取工具（`rew_common.py`），需自行安装对齐 verl |
| **评测口径（evals/）** | `evals/` | pandas / pyarrow | ✅ 发布 | `agg_eval.py` 权威聚合 |

### 6.1 分层铁律

1. **kernelgym 保持零 verl/ray/vllm 依赖**——评估环境必须可独立复用，不绑训练栈。
2. **训练侧不内嵌评测逻辑**——reward 读取 server 返回的 correctness/speedup，不在训练进程里重新判分。
3. **evals/ 是唯一对外数字来源**——决策数字只从 `agg_eval.py` 出，禁止脚本外手算。

---

## 7. 相关文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 顶层定位 / 安装 / 开源状态 |
| `docs/ARCHITECTURE.md` | 本文件：架构 / 口径 / 分层边界 |
| `docs/ROADMAP.md` | 当前 / 近期 / 中期 / 长期计划 |
| 父仓内部文档：`core_insight_reward_execution.md` | 核心洞见：奖励执行而非编译 |
| 父仓内部文档：`eval_calibre.md` | 口径定义 / 与基准对齐 / 4 数标准 |
| 父仓内部文档：`CALIBER_TABLE.md` | 对外数字唯一来源 / 权威口径冻结 |
| 父仓内部文档：`RELEASE_CHECKLIST_oss_internal.md` | 发布前清扫 + 数字口径 + 来源行 |
