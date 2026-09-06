# King.triton-kernel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/Leslie360/King.triton-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/Leslie360/King.triton-kernel/actions/workflows/ci.yml)

**English** | [简体中文](README.zh-CN.md)

> **King.triton-kernel** is a reinforcement-learning training framework that teaches a 14B LLM to write, compile, and run faster Triton kernels on real GPUs — turning operator specs into verified speedups on KernelBench L2 — plus a standalone GPU evaluation environment for any execution-verified code-generation task.

An RLVR training pipeline for Triton kernel generation on KernelBench L2: it trains a 14B model into an agent that autonomously analyzes an operator, writes a Triton implementation, compiles and executes it on a real GPU, and iteratively repairs it based on correctness and speed feedback — plus a standalone, reusable GPU evaluation environment.

---

## Contents

- [Introduction](#introduction)
- [Background](#background)
- [Method](#method)
- [Results](#results)
- [Installation & Quick Start](#installation--quick-start)
- [Repository Layout & Layering](#repository-layout--layering)
- [Related Work](#related-work)
- [Documentation](#documentation)
- [Release Status](#release-status)
- [License](#license)

## Introduction

This project ships two independently usable deliverables:

| Deliverable | Description | Use case |
|---|---|---|
| **kernelgym evaluation environment** | A self-contained distributed GPU grading service: subprocess-isolated execution, CUDA-event timing, correctness checking, reference timing cache | Any code-generation task that needs real-execution verification; reusable without the training stack |
| **RLVR training method** | A multi-turn repair reinforcement-learning pipeline with execution correctness as the core reward signal (training stack not shipped in this repo; see Installation) | RL post-training for kernel / code generation |

Target audience:

1. **AI Infra / RL researchers and engineers** who want to reproduce kernel-generation RL training, or reuse an execution-verified evaluation environment with caliber governance;
2. **Evaluation practitioners** who care about comparable protocols and reproducibility discipline on KernelBench L2.

**Hardware requirements**: training needs a single node with 8×A800 (or GPUs with comparable memory); using the kernelgym evaluation environment alone needs only 1 GPU.

## Background

Two engineering problems are unavoidable when generating GPU kernels with LLMs:

**First, compiling is not the same as being correct.** A kernel that compiles may still produce wrong results or trigger illegal memory access. If the reward signal only reflects compilability, the model learns to write compile-friendly but functionally broken code. The reward must therefore come from **actual execution on a real GPU** — this is the core methodological claim of this project (shared with Dr.Kernel / daVinci / DRTriton / CUDA Agent).

**Second, evaluation calibers distort easily.** The same set of model outputs can differ by double-digit percentage points depending on the aggregation protocol (sampling budget, turn range, speedup threshold, reference timing method). Numbers without caliber discipline are neither reproducible nor comparable to external work. This project treats caliber governance as a first-class engineering problem (see the protocol notes in [Results](#results)).

## Method

### Core claim

> The reward signal is "did it run correctly", not "did it compile". Compile rate ≠ correctness rate.

Concrete design:

- **Compile failure: hard penalty** (−1.0 magnitude), rejected outright;
- **Reward composition**: correctness 0.4 + speedup 0.3 + compilation 0.3;
- **Grading rule**: a kernel must execute on a real GPU and match the reference output within tolerance before entering speed statistics — failed submissions never mix into speedup numbers ("wrong" and "slow" are strictly separated);
- **Multi-turn repair (5 turns by default) + adaptive early stop**: the model iterates on grader feedback (compile error / correctness failure / insufficient speedup) and stops early once the speed target is met, preventing later-turn degradation.

### Training pipeline

```
SKILL injection ──► repair-flywheel SFT ──► RLVR (TRLOO)
    │                     │                     │
    │              (correctness front)     (performance front)
    │                     │                     │
    └──► multi-turn self-repair + adaptive termination ◄─┘
```

- **SKILL injection**: common failure modes (dtype / mask / shape / numerical stability / boundary, 10 categories) are distilled into skill documents and injected into prompts per turn based on feedback;
- **Repair-flywheel SFT**: failure-repair trajectories collected during RL are distilled back into supervised fine-tuning — the main lever for correctness;
- **RLVR** (main algorithm TRLOO): reinforcement learning on verifiable rewards — the main lever for performance.

### Caliber governance

To keep reported numbers reproducible and comparable, the project enforces the following discipline (details in `docs/ARCHITECTURE.md` §5):

1. **Reference timing cache**: the reference implementation is not re-timed during grading, removing run-level denominator jitter;
2. **Single authoritative aggregator**: `evals/agg_eval.py` is the only authoritative aggregation script; all other scripts are for cross-validation only;
3. **Failed turns excluded from speedup**: only turns with `correctness=True` participate in speedup statistics;
4. **Four-piece provenance**: every public number carries script version (incl. SHA), sampling budget, extractor version, and caliber version. Mixing calibers is forbidden.

## Results

> Numbers locked as of 2026-09-02 under the v2 protocol (reference cache ON), reported as the mean of three independent evaluation runs.

**Primary metric — fast@1.2**: fraction of problems where the best submission across all samples × all repair turns (per-problem best-of-history) is correct and achieves speedup ≥ 1.2×. Conditions: KernelBench L2 (100 problems), 8 samples × 5 repair turns, single A800 node, TF32 enabled.

| Model | fast@1.2 (best-of-history) | Notes |
|---|---|---|
| **King.triton-kernel (14B, this project)** | **61.0% ± 6.2pp** (n=3) | headline number |
| SFT v2 (distilled from the RL repair flywheel, ~10-min LoRA) | 68.3% ± 6.2pp (n=3) | statistically indistinguishable from the RL baseline at ~1% training cost |
| Dr.Kernel-14B | 47.8% (best-turn, STTS†) | arXiv 2602.05885; sampling budget not disclosed |
| daVinci-14B | 27.1% (L2 Fast@1.2, best-turn) | arXiv 2606.16497 |

### Measurement provenance

Every headline figure above is a self-measured number with the following fixed provenance; do not reproduce or cite it without the same conditions:

- **Physical machine**: a single 8×A800-SXM4-80GB node (Kubernetes pod). Timed numbers are machine-bound and valid only for the machine on which they were measured — the reference timing cache was rebuilt on that machine (a cache built on a different node is invalid here).
- **Reference cache (refcache)**: **ON** — the reference implementation is not re-timed during grading; the speedup denominator is fixed, removing run-level jitter.
- **Protocol / caliber**: 8 samples × 5 repair turns, **best-of-history** (per-problem best submission across all samples × all turns), KernelBench L2 (100 problems), speedup threshold ≥ 1.2×, TF32 enabled (TF32-ON).
- **Precision**: reported as the **mean of n=3 independent evaluation runs**, with a 3-run sample std of **±6.2pp** (single-run noise is ±3–6pp; conclusions require the mean of ≥3 runs).
- **Aggregation**: `scripts/agg_eval.py` is the single authoritative aggregator (four-piece provenance: script version incl. SHA, sampling budget, extractor version, caliber version).
- **Measurement timestamps**: RL headline locked 2026-09-02; SFT v2 numbers locked 2026-09-05 (eval runs 09-03–09-05).

Status flags on the SFT v2 row:

- **SFT v2 68.3% is locked** (n=3) and is a co-headline alongside the RL baseline.
- The single **72% run is flagged**: it is one run, not a locked figure, and must not be quoted as the SFT v2 result.

Protocol caveats:

- fast@1 vs fast@1.2, best-of-history vs best-turn, and speedup ≥1.0× vs ≥1.2× are different calibers and must not be compared directly. daVinci reports both: 27.1% = L2 Fast@1.2, 70.6% = L2 Fast@1.
- Measurement conditions (single 8×A800 node): clocks at 1155/1410 MHz natural boost (frequency locking unavailable without root); reference timing carries a 12–20% cross-node spread (machine property) — all numbers above are internally consistent within the measuring node; correctness 82.3% is judged against the TF32 reference (TF32-ON caliber).
- Single-run noise is ±3–6pp; conclusions require the mean of ≥3 runs.

## Installation & Quick Start

**Requirements**: Python ≥ 3.10, Linux + NVIDIA GPU (driver-visible is enough; CUDA ships with torch/triton), Redis.

**Verified environment** (the exact stack on which all headline numbers and the 164-test suite were produced):

| Component | Version / note |
|---|---|
| OS / node | Kubernetes pod, 8×A800-SXM4-80GB |
| Python | 3.10 (conda env `drkernel`) |
| PyTorch | ≥ 2.8 (CUDA 12.1 via conda) |
| Triton | ≥ 3.4 |
| vLLM | ≥ 0.8.5 (async rollout) |
| Redis | ≥ 6.2 (grading broker) |
| CUDA | 12.1 (driver-visible) |

> The measuring node runs as a Kubernetes pod and may be recycled/re-scheduled; timed results are machine-bound and valid only for that node (see Measurement provenance above). The `scripts/launch_local.sh` smoke path only needs **one** GPU (`cuda:0`).

```bash
# 1. Clone and install dependencies
git clone https://github.com/Leslie360/King.triton-kernel.git
cd King.triton-kernel
bash setup.sh          # equivalent to: pip install -r requirements.txt

# 2. Editable install (provides kernelgym-server and other CLI entry points)
pip install -e .

# 3. One-click local smoke: starts redis + eval server + a single-GPU worker,
#    then runs the smoke test (health + a real POST /evaluate grading check)
bash scripts/launch_local.sh
python3 smoke_test.py http://localhost:10907
```

`scripts/launch_local.sh` starts everything a fresh machine needs — a local
Redis instance, the KernelGym grading API server, and one GPU worker on
`cuda:0` — and prints the server URL. Default port is **10907** (override with
`API_PORT`, e.g. `API_PORT=10907 bash scripts/launch_local.sh`). On machines
without a CUDA-visible GPU, the server still starts for `/health` but
`smoke_test.py` prints a clear skip message for the grading-chain check.

To run the server and worker manually instead (e.g. when an eval server already
exists elsewhere):

```bash
# Start the evaluation server (default port 10907, override with API_PORT)
kernelgym-server

# Start a single GPU worker in a second terminal (required for grading)
kernelgym-single-worker --worker-id worker-1 --device cuda:0

# Smoke test (health; add --evaluate to also run a real grading task when a
# GPU worker is registered)
python3 smoke_test.py http://localhost:10907 --evaluate
```

CLI entry points (from `pyproject.toml [project.scripts]`):

| Command | Purpose |
|---|---|
| `kernelgym-server` | Grading API server |
| `kernelgym-worker` | GPU execution worker |
| `kernelgym-worker-monitor` | Worker monitor |
| `kernelgym-single-worker` | Single-worker mode (debugging) |

> This repository ships the evaluation environment and caliber tooling only. The RL training stack (verl-integrated) is not part of this repo; it consumes the server-returned verdicts produced here. The `kernelgym/` evaluation environment has no verl/ray/vllm dependency.

## Repository Layout & Layering

```
King.triton-kernel/
├── kernelgym/   # Distributed GPU evaluation environment (core lib + server + worker)
├── evals/       # Aggregation and calibers (agg_eval.py / compare_gs_eval.py / repro_pass_at_k.py)
├── docs/        # ARCHITECTURE.md (architecture & calibers) / ROADMAP.md
├── tests/       # 166 test functions (CI runs the CPU-only logic subset)
├── setup.sh      # Dependency installation
└── smoke_test.py
```

Layering (details in `docs/ARCHITECTURE.md` §6):

| Layer | Location | Dependencies | Shipped |
|---|---|---|---|
| Core evaluation library | `kernelgym/core\|schema\|workflow\|backend\|toolkit\|worker/` | torch / triton / numpy | Yes |
| Grading server | `kernelgym/server\|config\|utils/` | fastapi / redis | Yes |
| Evaluation aggregation | `evals/` | pandas / pyarrow | Yes |

Layering rule: `kernelgym/` stays free of verl/ray/vllm dependencies so the evaluation environment is always independently reusable. The (external) training stack embeds no grading logic of its own and only consumes server-returned verdicts.

## Related Work

| Work | Relationship |
|---|---|
| [Dr.Kernel](https://github.com/alpha-beta-wang/Dr.-Kernel) (arXiv 2602.05885) | This project follows its "evaluation environment + training method" architecture, with differentiated engineering on evaluation-caliber governance |
| daVinci (arXiv 2606.16497) | KernelBench L2 comparison baseline |
| DRTriton / CUDA Agent | Domain works that likewise use real GPU execution as the core reward signal |
| [verl](https://github.com/verl-project/verl) | Upstream training framework (not shipped with this repo) |

## Documentation

| Document | Content |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture, data flow, caliber definitions, layering |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current / near-term / mid-term / long-term plans |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributing guide (incl. evaluation-caliber discipline) |

## Release Status

v0.1.0 (2026-08-31) is the first public release: `kernelgym/`, `evals/`, docs, and tests are real code, with private datasets, training checkpoints, internal logs, and internal paths excluded. The repository ships the evaluation environment and caliber tooling under the MIT License (see [NOTICE](NOTICE)).

## License

[MIT](LICENSE). Attribution for Apache-2.0-licensed files in [NOTICE](NOTICE).

## Contributing

Issues and PRs are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first — the evaluation-caliber discipline applies to external contributions as well.
