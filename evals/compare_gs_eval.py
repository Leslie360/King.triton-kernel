#!/usr/bin/env python3
"""Compare gs-checkpoint eval calibers (alignment table): computes
sample_solve_rate / correctness / fast@1.2 (single) / fast@1.2 (best-of)
from eval_outputs.

Usage: python3 scripts/compare_gs_eval.py <eval_outputs_dir> [max_turn]

Aligned with the gs270 caliber: fast@1.2 = correct and speedup>=1.2;
single = per-sample final turn; best-of = per-problem best across any sample
and any turn.
"""
import sys, os, json, glob
ed = sys.argv[1]
max_turn = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Collect (correctness, performance) per sample per turn
problems = {}  # problem -> list of samples; sample -> list of turns (correct, perf)
samples = []
for stf in sorted(glob.glob(f"{ed}/*/*_eval.json")):
    try:
        d = json.load(open(stf))
    except: continue
    # Parse problem/sample/turn from path
    parts = stf.split("/")
    sp = parts[-2]  # problem_X_sample_Y
    turn = int(parts[-1].split("_")[1])
    correctness = bool(d.get("correctness"))
    perf = d.get("performance") or d.get("speedup") or 0.0
    prob = sp.split("_sample")[0]
    problems.setdefault(prob, {}).setdefault(sp, {})[turn] = (correctness, perf)
    samples.append((sp, turn, correctness, perf))

# 1. correctness (per-sample, correct in any turn)
correct_samples = sum(1 for sp, sdict in problems.items() for s, t in sdict.items() if any(c for c,p in t.values()))
total_samples = sum(len(sdict) for sdict in problems.values())
correctness = correct_samples / total_samples if total_samples else 0

# 2. fast@1.2 single-pass: per-sample, final turn is correct and speedup>=1.2
def last_turn(t): return max(t.keys()) if t else None
fast12_single = sum(1 for sp, sdict in problems.items() for s, t in sdict.items()
                    if (lt:=last_turn(t)) and t[lt][0] and t[lt][1] >= 1.2)
fast12_single_rate = fast12_single / total_samples if total_samples else 0

# 3. fast@1.2 best-of: per-problem, any sample any turn is correct and speedup>=1.2
fast12_best = sum(1 for prob, sdict in problems.items()
                  if any(c and p >= 1.2 for s,t in sdict.items() for c,p in t.values()))
fast12_best_rate = fast12_best / len(problems) if problems else 0

# 4. sample_solve_rate (per-sample pass: correct and speedup>=1.0, approx.)
solve = sum(1 for sp, sdict in problems.items() for s, t in sdict.items()
            if any(c and p >= 1.0 for c,p in t.values()))
solve_rate = solve / total_samples if total_samples else 0

# 5. pass1_best (per-problem, any sample passes)
pass1_best = sum(1 for prob, sdict in problems.items()
                 if any(c and p >= 1.0 for s,t in sdict.items() for c,p in t.values()))
pass1_best_rate = pass1_best / len(problems) if problems else 0

print(f"eval_dir={ed} problems={len(problems)} samples={total_samples}")
print(f"  sample_solve_rate (per-sample, correct&>=1.0): {solve_rate*100:.1f}%")
print(f"  pass1_best (per-problem): {pass1_best_rate*100:.1f}%")
print(f"  correctness (per-sample, any-turn correct): {correctness*100:.1f}%")
print(f"  fast@1.2 single-pass (per-sample final turn): {fast12_single_rate*100:.1f}%")
print(f"  fast@1.2 best-of (per-problem): {fast12_best_rate*100:.1f}%")
