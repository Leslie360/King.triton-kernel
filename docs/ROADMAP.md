# King.triton-kernel Roadmap

> Goal: teach a 14B model to write **correct and faster** Triton kernels, as an MIT-licensed, reproducible RLVR pipeline trainable/evaluable on a single 8×A800 node.
> Layered positioning: first establish the trustworthy foundation of "evaluation environment + honest calibers", then push the performance axis, then move toward community co-building.
> Related: `docs/ARCHITECTURE.md` (structure).

---

## Current (caliber lock-in) — largely done as of 2026-09-02

> One line: **numbers were not solid yet — lock down how they are counted first, then talk about getting better.**

### Goals
1. **Freeze calibers and document them**: `agg_eval.py` as the single authoritative aggregator, the 4-metric standard (sample_solve_rate / correctness / fast@1 / fast@1.2), best-of-history definition frozen.
2. **Re-measure everything with refcache=ON (v2 protocol)**: eliminate run-level denominator jitter; all public numbers unified on v2.
3. **Forbid caliber mixing**: fast@1 vs fast@1.2, best-of vs best-turn, ≥1.0x vs ≥1.2x each carry their own definition.

### Key actions
- [x] `evals/agg_eval.py` anchor re-check (incl. the failed-turns-excluded-from-speedup discipline)
- [x] Re-measure under refcache=ON and backfill the README results table (locked 2026-09-02: fast@1.2 best-of-history 61.0% ± 6.2pp, n=3)
- [ ] Cross-validation scripts (`compare_gs_eval.py` / `repro_pass_at_k.py`) remain validation-only, never producing decision numbers
- [ ] Close remaining comparability items (Dr.Kernel budget, daVinci Fast1 definition, CUDA Agent baseline)

### Done criteria
- Every public number carries the four-piece provenance (script SHA / budget / extractor version / caliber version)
- README has no TBD; all numbers come from v2 (refcache=ON)

---

## Near term (SFT v2 training / eval×3)

> One line: **make the model stronger (correctness + performance) and verify with three v2-protocol evals.**

### Main line
1. **Rebuild the SFT library (v2)**: 7:3 easy:hard mix + ≥100-problem coverage.
2. **SFT v2 training** (6–8h) + **eval×3** (v2 protocol, mean/std; single-run noise ±3–6pp).
3. Repair flywheel + SKILL injection iteration as the correctness front.

### Evaluation matrix
- After each training run: **sample_solve_rate / correctness / fast@1 / fast@1.2** × (single / best-of-history) × (refcache=ON)
- Baselines: Dr.Kernel (47.8% best-turn), daVinci (70.6% Fast1 / 27.1% Fast@1.2) — check comparability first.

### Done criteria
- SFT v2 reaches or exceeds the current best; correctness keeps climbing
- fast@1.2 best-of mean/variance reportable over three evals, with no caliber loopholes

---

## Mid term (performance-axis optimization / more backends)

> One line: **correctness is already near best-turn level; the gap is in "fast" — stand up the performance axis.**

### Performance axis (current biggest gap)
- Status: correctness ~62.75% approaches best-turn, but fast@1.2 best-of lags Dr.Kernel (~39% vs 47.8%; single-run gap larger).
- Directions:
  - [ ] Raise the performance reward weight / threshold (shift 0.5/0.5 toward performance); run performance-lever A/B tests
  - [ ] Evaluate the speedup reward cap of 3.0 (training may currently optimize "good enough" rather than "fast")
  - [ ] VeRPO / dedicated performance-side strengthening
  - [ ] TF32 baseline alignment (`ENABLE_TF32_BASELINE`, aligning with the KernelBench-Verified protocol against overestimation)

### More backends
- [ ] backend/ abstraction is in place (Triton / CUDA); extend to more backends (torch.compile baseline, CUTLASS snippets, multi-precision)
- [ ] Reproducible multi-backend evaluation: compare the same kernel across backends, widening coverage

### Training methods
- [ ] Skill-library retrieval + execution-verified admission (following daVinci)
- [ ] Agentic RL (following CUDA Agent: multi-role / tools)
- [ ] Multi-turn reward design (following MusaCoder: first-turn anchoring / dynamic retry)

### Done criteria
- fast@1.2 best-of reaches/exceeds the 47.8% line under the v2 protocol (same budget)
- ≥2 backends running stably, calibers not drifting across backend switches

---

## Long term (community co-building)

> One line: **make this evaluation environment + honest calibers a community-trusted public resource.**

### Goals
1. **Standardized evaluation environment**: `kernelgym/` standalone, reusable, zero verl/ray/vllm dependencies — the public evaluation substrate for Triton kernel generation.
2. **Honest-caliber methodology**: best-of-history / failed-turns-out-of-speedup / refcache fixed denominators — promoted as an industry habit for "reproducible numbers".
3. **Open comparison benchmark**: unified 4-metric reporting + budget annotation, so any method can be fairly compared under the same calibers.

### Actions
- [ ] Publish the standalone copy + full documentation (in place — this repository)
- [ ] Public model + data provenance + license
- [ ] Community PRs: multi-backend, new benchmarks (KernelBench-Verified), reproducible training scripts
- [ ] CI / test matrix (`pyproject.toml` already has the pytest / ruff / mypy / pre-commit skeleton)
- [ ] Cross-hardware portability (runnable beyond A800, widening coverage)

### Done criteria
- External projects can independently clone + `setup.sh` + `smoke_test.py` end-to-end
- ≥1 external contributor lands a multi-backend / new-benchmark PR
- Honest calibers become the README's differentiating highlight ("execution correctness over compilation" is already a domain standard; our differentiation = the standalone evaluation environment + honest-caliber methodology)

---

## Milestone overview

| Phase | Time | Main line | Acceptance |
|---|---|---|---|
| Current | late 2026-08 | Caliber lock-in (refcache=ON re-measurement) | All numbers on v2, no TBD, four-piece provenance complete |
| Near term | 2026-09 | SFT v2 training + eval×3 | Correctness/performance targets met; three evals reportable |
| Mid term | 2026 Q4 | Performance-axis optimization + more backends | fast@1.2 reaches 47.8%; ≥2 backends |
| Long term | 2027 | Community co-building | Independently reusable + external PRs + honest-caliber adoption |
