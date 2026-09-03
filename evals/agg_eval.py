#!/usr/bin/env python3
"""Aggregate N eval runs into fast@1.2 best-of-history + sample_solve + pass1_best.

For each eval run, scans the per-problem/per-sample eval output and computes three
metrics:

- sample_solve:  per-sample correctness rate (correct / total samples)
- fast@1.2 best-of-history:  per-problem "any sample correct AND speedup >= 1.2"
- pass1_best:  per-problem "any sample correct"

Runs are matched by directory name under the results base; the default naming
template is ``eval_rl_v7b_gs*_earlystop_t1_0_r<run>`` (backward compatible with the
legacy layout). Point ``--base-dir`` at a tree that contains a ``w3_results/``
directory, or override the results directory name with ``--results-dir``.

Usage:
    python3 agg_eval.py <run1> [<run2> ...] <model_prefix>
    python3 agg_eval.py --base-dir /path/to/tree 1 2 3 my_model

The results base defaults to ``$KING_BASE`` if set, otherwise to the repository root
(the parent of the ``evals/`` directory this script lives in).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Default name of the per-run subdir that holds the per-problem/per-sample results.
_DEFAULT_RESULTS_DIR = "eval_outputs"
# Substring the run directory name must contain to be considered a match.
_RUN_MARKER = "earlystop_t1_0_r"
# Each problem dir holds per-sample turn_*_eval.json files.
_PROBLEM_RE = re.compile(r"problem_(\d+)_sample_(\d+)")


def _default_base_dir() -> Path:
    """Resolve the results base: $KING_BASE, else the repository root."""
    env = os.environ.get("KING_BASE")
    if env:
        return Path(env)
    # This file lives in evals/; the repo root is its parent's parent.
    return Path(__file__).resolve().parent.parent


def _find_run_eval_dir(base: Path, run: str, results_dir: str) -> Path:
    """Locate the per-run results directory for a run number.

    Prefers a real match under ``base/w3_results``; falls back to the legacy
    ``base/w3_results/eval_rl_v7b_gs315_earlystop_t1_0_r<run>`` path so a missing
    explicit dir still yields a (likely empty) eval dir rather than erroring.
    """
    results_root = base / "w3_results"
    candidates: List[Path] = []
    if results_root.is_dir():
        # Newest-first to pick the most recent run when several match.
        for entry in sorted(results_root.iterdir(), key=lambda p: p.name, reverse=True):
            name = entry.name
            if "eval_rl_v7b_gs" in name and f"{_RUN_MARKER}{run}" in name:
                candidates.append(entry)
    if candidates:
        return candidates[0] / results_dir
    return results_root / f"eval_rl_v7b_gs315_{_RUN_MARKER}{run}" / results_dir


def _parse_turn_file(path: Path) -> Tuple[bool, float]:
    """Extract (correctness, speedup) from a single turn_*_eval.json file.

    correctness may live at the top level or under reward_extra_info; speedup is the
    ``performance`` field. Returns (False, 0.0) on any parse/type failure.
    """
    try:
        with open(path) as f:
            j = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, 0.0
    correct = bool(
        j.get("correctness")
        or (j.get("reward_extra_info") or {}).get("correctness")
    )
    speedup = j.get("performance") or 0.0
    return correct, float(speedup)


def collect_per_problem(eval_dir: Path) -> Dict[int, List[Tuple[bool, float]]]:
    """Group turn results by problem number.

    Returns {problem_id: [(correct, speedup), ...]}. Each problem_id maps to one
    entry per sample (best-observed speedup per sample is not folded here; the
    caller reduces per-problem).
    """
    per_problem: Dict[int, List[Tuple[bool, float]]] = {}
    if not eval_dir.is_dir():
        return per_problem
    for sample_dir in eval_dir.iterdir():
        m = _PROBLEM_RE.match(sample_dir.name)
        if not m:
            continue
        problem = int(m.group(1))
        best_correct = False
        best_speedup = 0.0
        for turn_file in sorted(glob.glob(str(sample_dir / "turn_*_eval.json"))):
            correct, speedup = _parse_turn_file(Path(turn_file))
            if correct:
                best_correct = True
            if correct and speedup > best_speedup:
                best_speedup = speedup
        per_problem.setdefault(problem, []).append((best_correct, best_speedup))
    return per_problem


def summarize_run(per_problem: Dict[int, List[Tuple[bool, float]]]) -> Tuple[int, int, float, float, float]:
    """Aggregate one run's per-problem data.

    Returns (num_problems, num_samples, sample_solve, pass1_best, fast@1.2_boh):
      - num_samples: total samples across all problems
      - sample_solve: percent of samples that are correct
      - pass1_best: percent of problems with at least one correct sample
      - fast@1.2_boh: percent of problems with a correct sample at speedup >= 1.2
    """
    num_problems = len(per_problem)
    num_samples = sum(len(samples) for samples in per_problem.values())
    correct_samples = sum(
        1 for samples in per_problem.values() for correct, _sp in samples if correct
    )
    problems_correct = sum(
        1 for samples in per_problem.values() if any(correct for correct, _sp in samples)
    )
    problems_fast = sum(
        1 for samples in per_problem.values() if any(correct and sp >= 1.2 for correct, sp in samples)
    )
    sample_solve = correct_samples / max(num_samples, 1) * 100
    pass1_best = problems_correct / max(num_problems, 1) * 100
    fast12_boh = problems_fast / max(num_problems, 1) * 100
    return num_problems, num_samples, sample_solve, pass1_best, fast12_boh


def print_run_line(run: str, stats: Tuple[int, int, float, float, float]) -> None:
    """Print the per-run summary line (format is load-bearing for tests)."""
    num_problems, num_samples, sample_solve, pass1_best, fast12_boh = stats
    if num_problems == 0:
        print(f"  {run}: 无数据")
        return
    print(
        f"  {run}: 题={num_problems} sample={num_samples} "
        f"sample_solve={sample_solve:.1f}% pass1_best={pass1_best:.0f}% "
        f"fast@1.2best-of={fast12_boh:.0f}%"
    )


def main(argv: List[str] | None = None) -> int:
    """Entry point: parse args, aggregate runs, print the report."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_default_base_dir(),
        help="Tree containing w3_results/ (default: $KING_BASE or repo root).",
    )
    parser.add_argument(
        "--results-dir",
        default=_DEFAULT_RESULTS_DIR,
        help="Name of the per-run results subdir (default: %(default)s).",
    )
    parser.add_argument(
        "run_and_prefix",
        nargs="+",
        help="One or more eval run numbers followed by the model prefix (the last "
             "positional is the prefix, matching the legacy CLI).",
    )
    args = parser.parse_args(argv)

    if len(args.run_and_prefix) < 2:
        parser.error("need at least one run number and a model prefix")
    runs: List[str] = args.run_and_prefix[:-1]
    prefix: str = args.run_and_prefix[-1]

    base: Path = args.base_dir
    results_dir: str = args.results_dir
    print(f"=== 聚合 {prefix} runs={runs} ===")

    fast12_boh_vals: List[float] = []
    sample_solve_vals: List[float] = []
    pass1_best_vals: List[float] = []

    for run in runs:
        eval_dir = _find_run_eval_dir(base, run, results_dir)
        per_problem = collect_per_problem(eval_dir)
        stats = summarize_run(per_problem)
        print_run_line(run, stats)
        num_problems, _ns, sample_solve, pass1_best, fast12_boh = stats
        if num_problems == 0:
            continue
        fast12_boh_vals.append(fast12_boh)
        sample_solve_vals.append(sample_solve)
        pass1_best_vals.append(pass1_best)

    if len(fast12_boh_vals) >= 2:
        n = len(fast12_boh_vals)
        print(f"  --- 均值/方差(n={n}) ---")
        print(
            f"  sample_solve: mean={statistics.mean(sample_solve_vals):.1f}% "
            f"std={statistics.stdev(sample_solve_vals) if len(sample_solve_vals) > 1 else 0:.1f}"
        )
        print(
            f"  fast@1.2 best-of-history: mean={statistics.mean(fast12_boh_vals):.1f}% "
            f"std={statistics.stdev(fast12_boh_vals) if len(fast12_boh_vals) > 1 else 0:.1f}"
        )
        print(f"  pass1_best: mean={statistics.mean(pass1_best_vals):.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
