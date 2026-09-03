---
name: Bug report
about: Report a bug to help us improve King.triton-kernel
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

<!--
Thanks for reporting a bug. Please fill in as much of the following as possible,
especially the "Evaluation Caliber" section — this project is highly sensitive to
evaluation calibers (fast@1 vs fast@1.2, best-of vs best-turn, refcache state);
unclear calibers make reproduction and diagnosis impossible.
-->

## Description

<!-- A clear and concise description of the bug. -->

## Steps to Reproduce

1. Command / script executed:
   ```bash
   # paste the minimal reproduction command
   ```
2. Input / configuration:
3. Observed result:

## Expected Behavior

<!-- What you expected to happen. -->

## Actual Behavior

<!-- What actually happened. Paste the error stack / log fragments. -->

## Environment

- **OS**: (e.g., Ubuntu 22.04)
- **Python version**: (e.g., 3.10.14)
- **torch version**: (e.g., 2.4.0, `python -c "import torch; print(torch.__version__)"`)
- **triton version**: (e.g., 3.0.0, `python -c "import triton; print(triton.__version__)"`)
- **GPU / driver**: (e.g., A800 / CUDA 12.4; `nvidia-smi`)
- **reference_cache**: (ON / OFF)
- **verl / vllm version** (if training-side related):

## Evaluation Caliber (if numbers are involved)

<!-- If the bug involves correctness/speed numbers, the caliber must be stated;
otherwise it will be treated as "caliber unclear": -->
- Metric: (sample_solve_rate / correctness / fast@1 / fast@1.2 / other)
- Sampling: (best-of N / best-turn / greedy)
- Turn range: (e.g., all 5 repair turns / final turn only)
- Speedup threshold: (≥1.0x / ≥1.2x / other)
- Caliber tooling script SHA: (if using evals/agg_eval.py etc.)

## Additional Context

<!-- Screenshots, related issues/PRs, anything else useful. -->
