"""Tests for evals/agg_eval.py aggregation logic.

agg_eval.py executes top-level code on import (parses sys.argv, prints results),
so we drive it as a subprocess against a synthetic ``w3_results`` tree built in
``tmp_path`` and set ``KING_BASE`` to that tree. Only stdlib is used, so this
test has no torch dependency.

Data layout produced by per():
- for each `problem_<p>_sample_<s>` dir, reads `turn_*_eval.json`
- correctness = j["correctness"] or j["reward_extra_info"]["correctness"]
- speedup = j["performance"] or 0.0
- aggregates: sample_solve (per-sample correct %), fast@1.2 best-of-history
  (per-problem any-correct-and-speedup>=1.2 %), pass1_best (per-problem any-correct %).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

AGG_EVAL = Path(__file__).resolve().parents[1] / "evals" / "agg_eval.py"


def _write_problem(root: Path, p: int, samples):
    """samples: list of (dir_name, list_of_json_dicts_for_turn_*_eval.json)."""
    for sample_name, turns in samples:
        d = root / f"problem_{p}_{sample_name}"
        d.mkdir(parents=True, exist_ok=True)
        for i, turn in enumerate(turns):
            (d / f"turn_{i}_eval.json").write_text(json.dumps(turn))


def _build_tree(base: Path):
    """Create w3_results with 3 runs of identical eval_outputs and return base."""
    w3 = base / "w3_results"
    turns_correct_fast = {"correctness": True, "performance": 1.5}
    turns_wrong = {"correctness": False, "performance": 0.9}
    turns_correct_slow = {"correctness": True, "performance": 1.0}
    turns_correct_med = {"correctness": True, "performance": 1.3}
    turns_wrong2 = {"correctness": False, "performance": 0.5}
    # correctness can also live in reward_extra_info
    turns_correct_via_extra = {"reward_extra_info": {"correctness": True},
                               "performance": 2.0}

    # Run names are numeric (agg_eval builds `earlystop_t1_0_r{run}`).
    for rn in ("1", "2", "3"):
        ed = w3 / f"eval_rl_v7b_gs315_earlystop_t1_0_r{rn}" / "eval_outputs"
        _write_problem(ed, 0, [("sample_0", [turns_correct_fast]),
                               ("sample_1", [turns_wrong])])
        _write_problem(ed, 1, [("sample_0", [turns_correct_slow]),
                               ("sample_1", [turns_correct_med])])
        _write_problem(ed, 2, [("sample_0", [turns_wrong2])])
        # an extra problem reading correctness from reward_extra_info
        _write_problem(ed, 3, [("sample_0", [turns_correct_via_extra])])

    # A distractor run dir that should NOT match (different run number)
    (w3 / "eval_rl_v7b_gs315_earlystop_t1_0_r99").mkdir(parents=True, exist_ok=True)
    return base


def _run_agg(base: Path, runs, prefix="model_test"):
    env = dict(Path="/usr/bin:/bin", KING_BASE=str(base))
    proc = subprocess.run(
        [sys.executable, str(AGG_EVAL), *runs, prefix],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_agg_eval_sample_solve_and_fast12(tmp_path):
    base = _build_tree(tmp_path)
    out = _run_agg(base, ["1"])

    # per-run line for run "1"
    line = next(l for l in out.splitlines() if l.startswith("  1:"))
    # 4 problems, 6 samples total (4 correct: p0s0,p1s0,p1s1,p3s0) -> 66.7%
    assert "题=4" in line, line
    assert "sample=6" in line, line
    assert "sample_solve=66.7%" in line, line
    # fast@1.2 best-of: problems with any correct sample at speedup>=1.2:
    #   p0(fast 1.5), p1(med 1.3), p3(2.0) -> 3/4 = 75%; p2 wrong excluded
    assert "fast@1.2best-of=75%" in line, line
    # pass1_best: problems with any correct sample -> p0,p1,p3 = 3/4 = 75%
    assert "pass1_best=75%" in line, line


def test_agg_eval_multirun_mean_std(tmp_path):
    base = _build_tree(tmp_path)
    out = _run_agg(base, ["1", "2", "3"])

    # identical runs -> std should be 0, mean equals single-run values
    assert "sample_solve: mean=66.7% std=0.0" in out, out
    assert "fast@1.2 best-of-history: mean=75.0% std=0.0" in out, out
    assert "pass1_best: mean=75.0%" in out, out


def test_agg_eval_missing_run_reports_no_data(tmp_path):
    base = _build_tree(tmp_path)
    out = _run_agg(base, ["1", "999"])
    assert "999: 无数据" in out


def test_agg_eval_single_run_no_mean_section(tmp_path):
    # mean/std summary only printed when >= 2 runs
    base = _build_tree(tmp_path)
    out = _run_agg(base, ["1"])
    assert "均值/方差" not in out


def test_agg_eval_incorrect_only_problem_excluded_from_fast(tmp_path):
    # Verify problem_2 (all wrong) is excluded from both fast@1.2 and pass1_best.
    base = _build_tree(tmp_path)
    out = _run_agg(base, ["1"])
    line = next(l for l in out.splitlines() if l.startswith("  1:"))
    assert "fast@1.2best-of=75%" in line  # not 100%
    assert "pass1_best=75%" in line
