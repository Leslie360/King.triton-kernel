# Contributing Guide

Thanks for contributing to **King.triton-kernel**. This guide covers development environment setup, running tests, code style, submitting PRs, and — most importantly — **evaluation-caliber discipline**.

---

## 1. Development Environment

### Prerequisites

- **Python ≥ 3.10** (3.10 recommended; `requires-python = ">=3.10"`)
- Linux + NVIDIA GPU (**required for GPU logic**; pure logic tests also run on CPU)
- CUDA driver (visible via `nvidia-smi` is enough; the CUDA toolkit ships with torch/triton)

### Setup

```bash
git clone <your-fork> && cd King.triton-kernel

# venv recommended
python3.10 -m venv .venv
source .venv/bin/activate

# Install dev dependencies (pytest/ruff etc.) + editable install of this package
pip install -e ".[dev]"
```

> **Heavy dependencies**: `torch` / `triton` are core dependencies (installed via the `[project] dependencies` when running `pip install -e ".[dev]"`). If your machine already has CUDA builds of torch/triton, you can skip that part; on CI they often fail to install due to size, and pure logic tests do not depend on them (see the testing section below).

For the training side (verl / ray / vllm for the RL integration in `drkernel/`), install the `[train]` extra separately and use it in an isolated environment:

```bash
pip install -e ".[train]"
```

---

## 2. Running Tests

### Pure logic tests (recommended for daily runs; also what CI uses)

```bash
pytest -m "not gpu and not redis and not slow"
```

- This command is filtered by the default markers in `pyproject.toml` `addopts`: tests requiring a real GPU, a real redis, or slow / eval-server-dependent tests are **skipped**.
- `tests/conftest.py` provides a `gpu_available` fixture that auto-skips relevant cases when no GPU is present or torch cannot be installed, so **pure logic tests run on CPU-only machines and CI**.

### GPU tests

```bash
# Requires a real NVIDIA GPU and a CUDA-compatible torch/triton
pytest -m "gpu"
```

> ⚠️ GPU tests **compile and execute real Triton kernels**. Do not run them casually on shared/production machines (see the isolation requirements in SECURITY.md).

### Smoke test

```bash
bash setup.sh
python3 smoke_test.py http://<eval-server>:10907
```

---

## 3. Code Style (ruff)

This repository uses [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`, line-length=110, target py310). Both must pass before submitting:

```bash
ruff check .
ruff format --check .
```

The `[dev]` extra already includes ruff; pre-commit (`[dev]`) is also available.

**Style notes**:

- Strictly follow `ruff check` (select: E, F, I, W, UP, B).
- Format with `ruff format`; do not hand-align.
- `extend-exclude` already excludes `drkernel/`, `docs/`, `evals/` — do not bypass lint when changing business code.

---

## 4. PR Workflow

1. **Fork + branch**: create a feature branch from `main` (`feat/xxx` or `fix/xxx`).
2. **Develop**: small commits with clear messages (conventional commits style preferred: `feat(...)` / `fix(...)` / `docs(...)`).
3. **Local verification** (mandatory before submitting):
   ```bash
   ruff check .
   ruff format --check .
   pytest -m "not gpu and not redis and not slow"
   ```
   For changes touching GPU logic, additionally run the relevant GPU tests where possible.
4. **PR checklist**: fill in `.github/pull_request_template.md`, especially the **evaluation-caliber discipline** section.
5. **CI**: PRs automatically run the `lint` and `test` jobs; both must pass (`test` tolerates skipping torch-import modules when torch cannot be installed, but pure logic cases must stay green).
6. **Review**: merge (squash) after at least one maintainer approval.

---

## 5. Evaluation-Caliber Discipline (hard rules — read this)

> **This project is extremely sensitive to evaluation calibers: mixing calibers = dishonest numbers.** This is the project's differentiating claim and a hard gate in review.

Any externally cited number **must be annotated with its full caliber**, otherwise it is invalid:

| Dimension | Must specify |
|------|----------|
| Metric | sample_solve_rate / correctness / fast@1 / fast@1.2 / other |
| Sampling | best-of N / best-turn / greedy (with budget) |
| Turn range | all repair turns / final turn only |
| Speedup threshold | ≥1.0x / ≥1.2x / other |
| reference_cache | ON / OFF |
| Caliber script SHA | if using `evals/agg_eval.py` or similar |

**Specific requirements**:

- When comparing models/methods, **the same caliber must be used**; otherwise direct comparison is forbidden.
- Changes to any aggregation logic under `evals/` must come with regression tests and a before/after explanation.
- Do not "shop" for calibers to make numbers look better (e.g., mixing fast@1 and fast@1.2 reporting).

---

## 6. Release Hygiene

This repository must not contain: private datasets, training checkpoints, internal logs, internal overlays. Self-check before submitting:

- ❌ No absolute paths / internal hostnames / internal IPs.
- ❌ No large files such as `.ckpt` / `.safetensors` / log dumps.
- ✅ Only real code copies (`kernelgym/`, `drkernel/`, `evals/` are standalone copies, not symlinks).
- List and confirm before deleting files; follow project hard rules.

---

## 7. Code of Conduct

By participating in this project you agree to abide by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant 2.1).

---

## 8. Security

Vulnerabilities involving the **code-execution surface** (evaluation / grading / feeding model output to GPU execution) must follow private disclosure — see [SECURITY.md](SECURITY.md); do **not** open a public issue.

---

## Questions & Help

- Feature discussions / questions → GitHub Discussions / issues.
- Bugs → use the `.github/ISSUE_TEMPLATE/bug_report.md` template; fill in environment and caliber.
- Feature suggestions → use the `feature_request.md` template.
