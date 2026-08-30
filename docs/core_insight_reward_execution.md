# Reward the Execution, Not the Compilation

**The one design decision that took our kernel-generation RL from "stuck at 28-36%" to 59% pass@1.**

> Status: internal technical note (2026-08-25). Numbers are single-run TP8 evals; multi-run bootstrap pending.
> Project: 14B Triton kernel generation via RLVR (verifiable-reward RL). Algorithm: TRLOO.
> Task: KernelBench-level-2 style Triton kernels; verifier = executes the kernel, checks correctness AND performance.

---

## TL;DR

If your code-generation RL rewards "it compiles", the model learns to **optimize the compiler, not the algorithm** — compiling 90% of the time while solving almost nothing. If your reward instead rewards "it executes correctly", the model actually learns to solve.

We measured this directly: same model family, same data, same algorithm. With a compile-oriented reward, compilation rate hit ~90% while pass@1 sat at 28-36%. We changed the reward's primary signal from "compiled" to "correct on execution" and pass@1 climbed 38% → 46.7% → 48.3% → 55.3% → **59.0%** across 150 RL steps, with correctness still climbing.

This is not a Triton-specific trick. Any code-generation RL where the output can be **executed and verified** (math, SQL, algorithms, kernels, DB queries) should reward the *execution result*, and gate hard on the cheapest cheap check (compilation / syntax) as a *necessary-but-not-sufficient* condition.

---

## The setup

- **Model**: 14B (Qwen3-based), full-parameter RLVR on 8×A800.
- **Task**: generate a Triton kernel (`ModelNew`) that (a) compiles, (b) produces correct outputs on held-in inputs, (c) is fast (performance reward).
- **Verifier**: an eval server that sandboxes the kernel, runs correctness checks + profiling. Multi-turn: the model gets verifier feedback and can repair its own kernel across up to 5 turns.
- **Baseline (Round-4 flywheel SFT)**: 34.17% pass@1.

## The discovery

Our first RL runs (v4-v6) kept producing a confusing picture:

| Signal | Value | Meaning |
|---|---|---|
| Compilation rate | **~90%** | The model almost always emits *syntactically valid* Triton |
| pass@1 (solve) | **28-36%** | ...but barely solves anything |

A 90% compile rate with a 30% solve rate is the signature of **reward hacking the compiler**. The model found that emitting *anything that compiles* is cheap reward, and it converged there. It wasn't learning to compute — it was learning to compile.

### The reward before (what went wrong)

The reward mixed "compiled" and "correct" so that *compiling* dominated the signal:

- compile failure → some penalty
- compiled but wrong → small penalty
- correct → big reward

With a dominated weight on "did it compile", the gradient strongly favored "make it compile", and correctness barely mattered. The model complied — literally.

### The reward after (the fix)

We made **execution correctness the primary signal**, and compilation a **hard gate**:

| Outcome | Reward |
|---|---|
| Compile failed | **-1.0** (hard gate, dominates) |
| Compiled but wrong | **-0.5** |
| Correct but no custom kernel (decoy) | **-0.3** additional (soft-decoy) |
| Correct | positive (scaled by correctness + performance) |

Key properties:
1. **Compile-fail is the worst case** — it's still penalized, so the model doesn't abandon compilability.
2. **Compile-pass is NOT rewarded by itself** — a compiled-but-wrong kernel is clearly negative.
3. **The only way to high reward is to execute correctly.**

### The result

TP8 eval, validation set (same口径 as baseline):

| Checkpoint | pass@1 | Note |
|---|---|---|
| Round-4 baseline | 34.17% | prior best SFT |
| RL v7b gs30 | 38.0% | reward fix in place |
| gs60 | 46.67% | |
| gs90 | 48.33% | |
| gs120 | 55.33% | |
| **gs150** | **59.0%** | correctness still climbing (37.7%) |

Correctness rate (not just pass@1) rose steadily: 20.5% → 25.5% → 29.1% → 32.8% → 37.7%. The model was genuinely learning to compute, not to compile.

## What we also tried (and why it didn't help)

These negative results matter — they're what the community usually assumes works:

- **Teacher "give the answer" SFT (A2/C0)**: distilled teacher solutions onto the student → **not significant** (+0.8pp). Feeding correct answers is weak when the student's *errors* are the informative signal.
- **Same-source repair flywheel (R2→R5)**: repeatedly SFT on repair samples of the same problem pool → saturates and then hurts (41.67% → 37.33%). Same-source data has no new information.
- **TP4 eval**: systematically broken on this setup (same model: TP8 = 41.67% vs TP4 = 24%). Eval config matters as much as training.

## Why this generalizes

Any code-generation RL with a **verifiable execution oracle** faces the same trap: the cheapest verifiable signal (compiles / parses / type-checks) is *not* the goal signal (correct output / performance). If the reward does not separate them with a hard gate, the model optimizes the cheap signal.

The pattern to copy anywhere:
1. **Hard gate** the cheap check (compile/syntax) — failures are the most negative outcome.
2. **Primary signal** = execution correctness (the actual goal).
3. **Secondary signal** = performance / quality, kept optional so correctness isn't diluted.

## Reproducibility notes

- Eval口径: TP=8, `gpu_memory_utilization=0.3`, best-of-4-samples × best-of-3-turns (multi-turn repair), `solve_threshold=0.99` (correct AND fast).
- Numbers above are **single** TP8 evals (variance ±3-6pp). Multi-run bootstrap is pending.
- Training: TRLOO, geo-mean anchoring, solverate-aware sampling, 6-GPU + parameter/optimizer offload.
- Config in `scripts/w3_kernel_rl_14b_62.sh`.

## TODO (before public release)

- [ ] Multi-run / multi-seed bootstrap for credible numbers
- [ ] Ablation: same config, old reward vs new reward (clean head-to-head)
- [ ] Generalize the reward module beyond Triton (execution-verified code RL)
- [ ] Publish model + data provenance + license
