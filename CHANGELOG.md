# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-08-31

Initial public release.

### Core insight

- **Reward "execution correctness", not compilation — compile rate ≠ correctness rate.** The core methodological claim of this project: the reward signal is "did it run correctly", not "did it compile". Compile failure receives a hard penalty (−1.0 magnitude); reward composition is correctness 0.4 + speedup 0.3 + compilation 0.3, combined with multi-turn repair (5 turns by default) and verifier-side adaptive early stop.
- This claim is now a domain consensus (Dr.Kernel / daVinci / DRTriton / CUDA Agent all use real GPU execution as the core reward signal); this project's differentiated engineering is in **evaluation-caliber honesty**.

### Evaluation environment (kernelgym/)

- Standalone distributed GPU evaluation environment: only depends on torch/triton/redis/fastapi/httpx, **no verl/ray/vllm coupling**, can start an independent grading server.
- Subprocess worker pool isolates kernel execution: CUDA errors / illegal memory access are contained, workers stay alive.
- Real-execution verification: CUDA-event timing + correctness checking (rtol/atol tolerance) + multi-backend support.

### Caliber methodology

- **Fixed reference denominator** (`InMemoryReferenceCache`, key = uuid + ref_hash + is_valid): grading does not re-time the reference denominator, eliminating run-level denominator jitter.
- Authoritative caliber = **per-problem best-of-history** (best submission across any sample and any turn, over all turns).
- **Caliber audit discipline**: every public number must carry its caliber (metric / sample count / turns / refcache state / script SHA); mixing calibers is forbidden.

### Release hygiene

- `kernelgym/`, `drkernel/`, `evals/` are **real code copies** (not symlinks, not shared-disk references).
- Excluded: private datasets, training checkpoints, internal logs, verl_patch internal overlay, internal paths/hostnames.
- The training stack (verl integration) depends on upstream verl and is not shipped with this repository; install and align the version yourself.

### Engineering governance

- Added GitHub Actions CI (`.github/workflows/ci.yml`): `lint` (ruff check + ruff format --check) and `test` (CPU-only logic tests, skipping gpu/redis/slow).
- Added issue / PR templates, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md.
- Numbers: evaluation figures are locked under the v2 protocol (reference_cache=ON) — fast@1.2 best-of-history 61.0% ± 6.2pp (n=3); see README.

[0.1.0]: https://github.com/Leslie360/King.triton-kernel/releases/tag/v0.1.0
