# King.triton-kernel Architecture

> An MIT-licensed RLVR (reinforcement learning with verifiable rewards) pipeline for Triton kernel generation.
> This document describes the code structure, data flow, evaluation calibers, and layering boundaries. New contributors should read this file before `README.md`.

---

## 0. One-line positioning

Teach a 14B model to **analyze a kernel → write a Triton implementation → compile → verify on a real GPU → iteratively repair based on correctness and speed feedback**.
Core claim: **reward execution correctness; compilation is only a hard gate**.

```
SKILL injection ──► repair-flywheel SFT ──► RLVR (TRLOO)
    │                     │                     │
    │              (correctness front)     (performance front)
    │                     │                     │
    └──► multi-turn self-repair + adaptive termination ◄─┘
```

---

## 1. Top-level layout

```
King.triton-kernel/
├── kernelgym/   # standalone GPU evaluation environment (subprocess workers / timing / correctness / multi-backend)
├── drkernel/    # training side (reward implementations + code extraction tooling; verl integration)
├── evals/       # aggregation & calibers (agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py)
├── docs/        # architecture / calibers / release / roadmap
├── pyproject.toml   # packaging: kernelgym-server / worker / worker-monitor / single-worker entry points
├── setup.sh     # install + smoke
└── smoke_test.py
```

### 1.1 Standalone-copy principle

`kernelgym/` and `drkernel/` are **real code copies** (not symlinks, not shared-disk references, no .git history),
directly reusable by external projects. Private datasets, training checkpoints, internal logs, and the verl_patch
internal overlay are excluded. The training stack (`main_grading.py` / verl integration) depends on upstream
[verl] and is not shipped with this repository; install and align the version yourself.

---

## 2. Module map — kernelgym subpackage responsibilities

`kernelgym/` is a **layered**, standalone evaluation engine with zero verl / ray / vllm dependencies,
depending only on torch / triton / redis / fastapi / httpx.

```
                         ┌──────────────────────────────────────────┐
                         │              server (API layer)          │
                         │  api/server.py · models.py · utils.py     │
                         │  code_retry_manager · task_manager        │
                         └───────────────┬──────────────────────────┘
                                         │ SchedulerAPI (submit/wait/status/cancel)
                         ┌───────────────▼──────────────────────────┐
                         │              core (abstractions)          │
                         │  types.py (TaskSpec/Result/Artifact/      │
                         │            Metric/TaskGroup)               │
                         │  workflow.py (WorkflowController)         │
                         │  scheduler.py (SchedulerAPI)              │
                         │  registry.py (generic registry)           │
                         └───────────────┬──────────────────────────┘
                                         │ workflow orchestration
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
   ┌────────▼────────┐        ┌──────────▼─────────┐      ┌──────────▼─────────┐
   │  workflow/       │        │     schema/        │      │    backend/        │
   │  kernelbench.py  │        │  task.py result.py │      │  base.py           │
   │  kernel_simple   │        │  serialization.py  │      │  triton.py         │
   │  helpers(refcache)│       │  simple_task.py    │      │  kernelbench/      │
   └────────┬────────┘        └─────────────────────┘      │  cuda/triton       │
            │ emits TaskSpec                                 └──────────┬─────────┘
            │                                                           │ Backend interface
            ▼                                                           ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          worker (execution layer)                        │
   │  subprocess_pool.py (persistent worker pool · auto-restart on CUDA error)│
   │  task_executor.py     (per-task spawn isolation)                         │
   │  gpu_worker / single_worker / worker_monitor                            │
   └──────────────────────────────────────────────────────────────────────────┘
            │ calls toolkit.evaluate(task, backend)
            ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          toolkit (evaluation logic)                      │
   │  kernelbench/pipeline.py: load→compile→correctness→Triton-detect→time→cov│
   │  kernel_simple/                                                          │
   │  validation.py / base.py / registry.py                                  │
   └──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Subpackage responsibilities

| Subpackage | Responsibility | Key files |
|---|---|---|
| **core/** | Domain abstractions: task specs, result models, workflow controller, scheduler interface, generic registry | `types.py` / `workflow.py` / `scheduler.py` / `registry.py` |
| **schema/** | Request/response data structures and serialization | `task.py` / `result.py` / `serialization.py` |
| **workflow/** | End-to-end orchestration (how one eval request splits into reference + kernel subtasks and merges) | `kernelbench.py` / `kernelbench_helpers.py` |
| **backend/** | Backend execution adapters (compile/load/run abstraction); currently Triton / CUDA | `base.py` / `triton.py` / `kernelbench/` |
| **toolkit/** | Evaluation logic (correctness / timing / Triton-usage detection / coverage), resolved by name via registry | `kernelbench/pipeline.py` / `base.py` |
| **worker/** | Subprocess-isolated execution; CUDA errors never take down the main process | `subprocess_pool.py` / `task_executor.py` |
| **server/** | FastAPI layer + task management + reference-cache registration | `api/server.py` / `task_manager.py` |
| **config/** | Settings (env-driven, incl. the reference_cache switch) | `settings.py` |
| **utils/** | Error classification / GPU diagnostics | `error_classifier.py` / `gpu_diagnostics.py` |

> Note: `WorkflowController` in `core/workflow.py` is the **abstraction**; the concrete implementation lives in
> `workflow/kernelbench.py` (`KernelBenchWorkflowController`), adapted to the `core.scheduler` interface via
> `TaskManagerScheduler` in `server/scheduler.py`.

---

## 3. Data flow — submit kernel → reward

End-to-end path of one full eval request (KernelBench workflow example):

```
1. Client submits an EvaluationTask
   ├─ task_id / kernel_code / reference_code / uuid / use_reference_cache
   └─ run_correctness / run_performance / measure_performance / enable_profiling ...
        │
        ▼
2. server/api/server.py receives the request
   └─ delegates to KernelBenchWorkflowController.handle_request()
        │
        ▼
3. _validate_inputs(): checks class Model/ModelNew, resources, reference-cache constraints
        │
        ▼
4. _create_paired_tasks(): splits into two subtasks
   ├─ kernel_task    (kind="kernelbench.kernel", evaluates the custom kernel)
   └─ reference_task (kind="kernelbench.ref", times the reference denominator)
   ★ if use_reference_cache and cache hit → no reference_task is created (see §4)
        │
        ▼
5. scheduler.submit() → task_manager → worker pool scheduling
   └─ worker (subprocess, spawn-isolated) executes
        │
        ▼
6. toolkit/kernelbench/pipeline.py runs stage by stage:
   a. load original model + inputs (get_init_inputs/get_inputs)
   b. backend.compile(custom_model_src)   ← compile (hard gate)
      ├─ compile failure → KernelExecResult(compiled=False) → reward ≈ -1.0
      └─ compiled → backend.load + create_model
   c. _run_correctness_step()  ← correctness check (rtol/atol tolerance)
      ├─ incorrect → compiled=True, correctness=False → reward=-0.5
      └─ correct   → continue
   d. _run_triton_detection_step()  ← detects whether Triton was actually used (anti-decoy)
      └─ labeled triton but not used → decoy_kernel=True
   e. _run_performance_step()  ← CUDA-event timing (mean over num_perf_trials)
      └─ kernel_runtime → speedup = reference_runtime / kernel_runtime
        │
        ▼
7. Subtask results merged (_combine_results) → EvaluationResult
   ├─ compiled / correctness / decoy_kernel
   ├─ reference_runtime / kernel_runtime / speedup
   └─ status / error_code / metadata (incl. profiling, coverage)
        │
        ▼
8. _persist_result() writes to disk (eval_results_path)
   └─ reward is computed by the training side (drkernel) or agg_eval from speedup
```

### 3.1 Reward computation (consistent on training and evaluation sides)

Core reward mapping:

| Outcome | Reward | Note |
|---|---|---|
| Compile failure | **-1.0** | Hard gate, most negative |
| Compiled but wrong | **-0.5** | Not a hard gate but clearly negative |
| Correct but decoy (Triton not actually used) | **-0.3** additional | Soft penalty |
| Correct | positive (scaled by correctness + performance) | `0.5×correct + 0.5×min(speedup, 3)` |

During multi-turn repair (5 turns by default), the model receives `compile/correctness/speedup/error` feedback
from the server and iterates; **verifier-side adaptive termination** (early stop once the speed target is met)
prevents degradation.

---

## 4. reference_cache design — why a fixed denominator removes jitter

> Code: `kernelgym/workflow/kernelbench_helpers.py` (`InMemoryReferenceCache`) + registration in `kernelgym/server/api/server.py`.

### 4.1 The problem

`speedup = reference_runtime / kernel_runtime`. If the reference denominator is re-timed on every eval,
**the same reference jitters across runs at machine level** (measured: gs300 std ≈ 23.6; the same reference
varied by 42%). The consequence: **being "fast" is not the model getting stronger, it's the reference happening
to be slow this time** — which badly pollutes public numbers.

### 4.2 Design

- `InMemoryReferenceCache`: `key = (uuid, reference_code_hash, is_valid)`.
- The first completed reference run is **written to the cache** (`_put_reference_cache`); subsequent requests
  for the same problem, same reference, same validation state **reuse the denominator** instead of re-running it.
- Enabled only when `use_reference_cache=True` and `uuid` is present; a missing uuid fails validation.
- The cache is **in-process memory of the server** (cleared on restart, denominators re-timed) — avoiding
  cross-machine caches that would drift calibers.

### 4.3 Rationale

1. **Fixed denominator**: all samples/turns of the same problem share one reference_runtime → denominator
   jitter is removed from run variance.
2. **Honesty**: public numbers no longer inflate because "the reference was slow this time" — eliminating the
   self-contradiction between refcache=OFF (~39%) and refcache=ON (~20–29%) figures.
3. **Reproducible calibers**: any re-measurement must run with refcache=ON, otherwise numbers are not comparable.

---

## 5. Evaluation calibers — best-of-history and failed turns

> Authoritative implementation = `evals/agg_eval.py`.

### 5.1 Authoritative metrics

Four numbers are reported publicly (not just pass@1):

| Metric | Definition | Note |
|---|---|---|
| **sample_solve_rate** | per-sample pass rate (correct and on-target; average of within-problem pass fraction) | formerly "reported pass@1" (note: **not** standard pass@1) |
| **pass1_best** | standard pass@1 (≥1 sample passes per problem) | capability ceiling (~95% level) |
| **fast@1** | correct kernels with performance ≥ 1.0 | strict "correct + ≥1.0x" |
| **fast@1.2** | correct kernels with performance ≥ 1.2 | comparable with Dr.Kernel/daVinci |

### 5.2 best-of-history

**Authoritative caliber = per-problem best-of-history**: each problem takes the best submission across all
samples × all turns, i.e. the problem passes iff `∃ (sample, turn): correct ∧ speedup ≥ threshold`.

### 5.3 Failed turns excluded from speedup (critical discipline)

**Failed turns (performance=0) are never speedup candidates** — "wrong" is not "slow", and must not mix into
speedup statistics. In `agg_eval.py`'s `per()` logic, only turns with `correctness=True` participate in the
`bs = max(bs, sp)` statistic.

> A 2026-08-30 recomputation differed by an order of magnitude precisely because failed turns had been mixed in.
> This discipline is non-negotiable.

### 5.4 Aggregation and variance

- Single-eval noise is **±3–6pp**; conclusions require the mean of ≥3 evals
  (`agg_eval.py <run1> <run2> <run3> <prefix>`).
- Report mean and std; fast@1.2 best-of-history is the decision-anchor metric.

### 5.5 Caliber discipline

- **No two "official" scripts may coexist**: `agg_eval.py` is the single authority; `compare_gs_eval.py` and
  hand-written parquet analyses are for cross-validation only.
- Every number carries a four-piece provenance: **script (incl. SHA) / budget / extractor version / caliber version**.
- Before comparing with Dr.Kernel (47.8% best-turn, budget undisclosed) or daVinci (70.6% L2 Fast1 /
  27.1% L2 Fast@1.2), check comparability; different calibers must be marked "not comparable".

---

## 6. Layering — pure library vs server vs training side

| Layer | Location | Dependencies | Shipped | Note |
|---|---|---|---|---|
| **Pure library (kernelgym core)** | `core/ schema/ workflow/ backend/ toolkit/ worker/` | torch / triton / numpy | ✅ | Standalone evaluation engine, no verl/ray/vllm coupling; can start an independent grading server |
| **server (API service)** | `server/ config/ utils/` | fastapi / uvicorn / pydantic / redis | ✅ | HTTP grading API (`kernelgym-server` entry), registers reference_cache |
| **Training side (drkernel + verl)** | `drkernel/` | verl / ray / vllm / httpx | ⚠️ depends on verl, not shipped | Reward implementations + code extraction tooling (`rew_common.py`); install and align verl yourself |
| **Evaluation calibers** | `evals/` | pandas / pyarrow | ✅ | `agg_eval.py` authoritative aggregation |

### 6.1 Layering rules

1. **kernelgym stays free of verl/ray/vllm dependencies** — the evaluation environment must be independently
   reusable, never bound to a training stack.
2. **The training side embeds no grading logic** — rewards read the correctness/speedup returned by the server;
   they never re-grade inside the training process.
3. **evals/ is the single source of public numbers** — decision numbers come only from `agg_eval.py`;
   no out-of-script hand computation.

---

## 7. Related documentation

| Document | Content |
|---|---|
| `README.md` | Top-level positioning / installation / release status |
| `docs/ARCHITECTURE.md` | This file: architecture / calibers / layering |
| `docs/ROADMAP.md` | Current / near-term / mid-term / long-term plans |
| `CHANGELOG.md` | Release history |
| `CONTRIBUTING.md` | Contributing guide incl. evaluation-caliber discipline |
