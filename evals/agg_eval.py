#!/usr/bin/env python3
"""Aggregate fast@1.2 best-of-history + sample_solve mean/std across N eval runs.

Usage: python3 agg_eval.py <run1> <run2> <run3> <model_prefix>
Reads w3_results/eval_rl_v7b_gs315_earlystop_t1_0_r<run>/eval_outputs
"""
import os, json, glob, re, statistics, sys
# Project root: defaults to the parent of the script's directory (parent of evals/),
# overridable via the KING_BASE environment variable.
BASE=os.environ.get("KING_BASE") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
runs=sys.argv[1:-1]; prefix=sys.argv[-1]
def per(ED):
    r={}
    if not os.path.isdir(ED): return r
    for n in os.listdir(ED):
        m=re.match(r"problem_(\d+)_sample_(\d+)",n)
        if not m: continue
        p=int(m.group(1)); d=os.path.join(ED,n)
        bc=False;bs=0.
        for tf in glob.glob(os.path.join(d,"turn_*_eval.json")):
            try:j=json.load(open(tf))
            except:continue
            c=bool(j.get("correctness") or (j.get("reward_extra_info") or {}).get("correctness"))
            sp=j.get("performance") or 0.
            if c:bc=True
            if c and sp>bs:bs=sp
        r.setdefault(p,[]).append((bc,bs))
    return r
print(f"=== Aggregating {prefix} runs={runs} ===")
f12_boh=[]; sr=[]; p1=[]
for rn in runs:
    # 2026-08-30: match the eval step automatically (gs300/gs315/gs270...) instead
    # of hardcoding gs315.
    _m=[]
    for _d in sorted(os.listdir(f"{BASE}/w3_results"), reverse=True):
        if f"eval_rl_v7b_gs" in _d and f"earlystop_t1_0_r{rn}" in _d:
            _m.append(f"{BASE}/w3_results/{_d}")
    ED=_m[0]+"/eval_outputs" if _m else f"{BASE}/w3_results/eval_rl_v7b_gs315_earlystop_t1_0_r{rn}/eval_outputs"
    pr=per(ED)
    if not pr: print(f"  {rn}: no data"); continue
    ns=sum(len(v) for v in pr.values())
    s=sum(1 for v in pr.values() for c,sp in v if c)/max(ns,1)*100
    f=sum(1 for v in pr.values() if any(c and sp>=1.2 for c,sp in v))/max(len(pr),1)*100
    pb=sum(1 for v in pr.values() if any(c for c,sp in v))/max(len(pr),1)*100
    f12_boh.append(f); sr.append(s); p1.append(pb)
    print(f"  {rn}: problems={len(pr)} sample={ns} sample_solve={s:.1f}% pass1_best={pb:.0f}% fast@1.2best-of={f:.0f}%")
if len(f12_boh)>=2:
    print(f"  --- mean/std (n={len(f12_boh)}) ---")
    print(f"  sample_solve: mean={statistics.mean(sr):.1f}% std={statistics.stdev(sr) if len(sr)>1 else 0:.1f}")
    print(f"  fast@1.2 best-of-history: mean={statistics.mean(f12_boh):.1f}% std={statistics.stdev(f12_boh) if len(f12_boh)>1 else 0:.1f}")
    print(f"  pass1_best: mean={statistics.mean(p1):.1f}%")
