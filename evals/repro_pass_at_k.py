#!/usr/bin/env python3
"""
repro_pass_at_k.py — recompute pass@1 from eval_outputs to surface caliber drift.

Background (2026-08-26, W3.4 review finding):
- Reported pass@1 (gs150=59%, gs180=61.33%) sits ~8-10pp above the strict
  recompute from eval_outputs.
- Reason: the reported value uses the verl reward_tensor (which may include
  a coverage bonus), while eval_outputs' score = 0.5*correct + 0.5*perf
  (no coverage term).
- This script reports both calibers for honest reporting.

Usage:
  python scripts/repro_pass_at_k.py <eval_outputs_dir> <max_turn>
  python scripts/repro_pass_at_k.py w3_results/eval_rl_v7b_global_step_150/eval_outputs 3
"""
import json, glob, sys
from collections import defaultdict

def recompute(eval_dir, max_turn):
    # per problem -> {sample_uid: best_score} (only the first max_turn turns)
    per_problem = defaultdict(dict)
    for f in glob.glob(f'{eval_dir}/problem_*_sample_*/turn_*_eval.json'):
        d = json.load(open(f))
        if d.get('turn_id', 99) > max_turn:
            continue
        pid, uid = d.get('problem_id'), d.get('uid')
        per_problem[pid][uid] = max(per_problem[pid].get(uid, -1), d.get('score', 0))

    n_problem = len(per_problem)
    # A. pass@k estimator (k=1): mean of per-problem per-sample pass rate
    rates = [sum(1 for s in sm.values() if s >= 0.99) / len(sm) for sm in per_problem.values()]
    passk = sum(rates) / n_problem
    # B. best-of-samples: at least one passing sample per problem
    best_any = sum(1 for sm in per_problem.values() if max(sm.values()) >= 0.99) / n_problem
    # C. per-sample pass rate (overall)
    all_scores = [s for sm in per_problem.values() for s in sm.values()]
    per_sample = sum(1 for s in all_scores if s >= 0.99) / len(all_scores)
    return {"sample_solve_rate": passk, "pass1_best": best_any, "per_sample": per_sample, "n_problem": n_problem}

if __name__ == "__main__":
    eval_dir, max_turn = sys.argv[1], int(sys.argv[2])
    r = recompute(eval_dir, max_turn)
    print(f"eval_dir={eval_dir} max_turn={max_turn}  problems={r['n_problem']}")
    print(f"  sample_solve_rate (per-sample pass rate, original reported pass@1): {r['sample_solve_rate']*100:.1f}%")
    print(f"  pass1_best (>=1 passing sample per problem, standard pass@1): {r['pass1_best']*100:.1f}%")
    print(f"  per_sample pass rate: {r['per_sample']*100:.1f}%")
    print("Note: sample_solve_rate, when coverage is permissive, ~= reported pass@1; strict caliber: docs/eval_calibre.md")
