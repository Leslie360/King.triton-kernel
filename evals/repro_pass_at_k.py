#!/usr/bin/env python3
"""
repro_pass_at_k.py — 从 eval_outputs 重算 pass@1, 暴露口径差异

背景 (2026-08-26, W3.4 审阅发现):
- 报告的 pass@1 (gs150=59%, gs180=61.33%) 比从 eval_outputs 严格重算高 ~8-10pp
- 原因: 报告值用 verl reward_tensor (可能含 coverage 加成), eval_outputs 的 score = 0.5*correct+0.5*perf (无 coverage)
- 本脚本给出两种口径, 供诚实汇报

用法:
  python scripts/repro_pass_at_k.py <eval_outputs_dir> <max_turn>
  python scripts/repro_pass_at_k.py w3_results/eval_rl_v7b_global_step_150/eval_outputs 3
"""
import json, glob, sys
from collections import defaultdict

def recompute(eval_dir, max_turn):
    # per problem -> {sample_uid: best_score} (仅前 max_turn 轮)
    per_problem = defaultdict(dict)
    for f in glob.glob(f'{eval_dir}/problem_*_sample_*/turn_*_eval.json'):
        d = json.load(open(f))
        if d.get('turn_id', 99) > max_turn:
            continue
        pid, uid = d.get('problem_id'), d.get('uid')
        per_problem[pid][uid] = max(per_problem[pid].get(uid, -1), d.get('score', 0))

    n_problem = len(per_problem)
    # A. pass@k 估算器 (k=1): 每题内通过样本占比平均
    rates = [sum(1 for s in sm.values() if s >= 0.99) / len(sm) for sm in per_problem.values()]
    passk = sum(rates) / n_problem
    # B. best-of-samples: 每题任一采样通过
    best_any = sum(1 for sm in per_problem.values() if max(sm.values()) >= 0.99) / n_problem
    # C. per-sample 通过率 (整体)
    all_scores = [s for sm in per_problem.values() for s in sm.values()]
    per_sample = sum(1 for s in all_scores if s >= 0.99) / len(all_scores)
    return {"sample_solve_rate": passk, "pass1_best": best_any, "per_sample": per_sample, "n_problem": n_problem}

if __name__ == "__main__":
    eval_dir, max_turn = sys.argv[1], int(sys.argv[2])
    r = recompute(eval_dir, max_turn)
    print(f"eval_dir={eval_dir} max_turn={max_turn}  problems={r['n_problem']}")
    print(f"  sample_solve_rate (per-sample 通过率, 原报告 pass@1): {r['sample_solve_rate']*100:.1f}%")
    print(f"  pass1_best (每题≥1采样过, 标准pass@1): {r['pass1_best']*100:.1f}%")
    print(f"  per-sample 通过率: {r['per_sample']*100:.1f}%")
    print("注: sample_solve_rate 若含 coverage 宽松 ≈ 报告 pass@1; 严格口径见 docs/eval_calibre.md")
