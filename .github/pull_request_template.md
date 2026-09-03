## Summary

<!-- One sentence describing what this PR does. -->

## Type

- [ ] Bug fix
- [ ] New feature / new module
- [ ] Docs / governance files
- [ ] Evaluation-caliber tooling
- [ ] Refactor / performance
- [ ] Other (please specify)

## Changes

- [ ] Files / modules touched (list paths and key points)
- [ ] Whether any public scripts or configs were added / modified

## Testing

- [ ] Ran `ruff check .` and `ruff format --check .`
- [ ] Ran `pytest -m "not gpu and not redis and not slow"` (pure logic tests)
- [ ] (If GPU logic is involved) ran the relevant GPU tests / smoke test
- [ ] Test result summary: ___

## Evaluation-Caliber Discipline (required if this PR touches any numbers/metrics)

> Project hard rule: **calibers must not be mixed**. Any public number must carry its full caliber,
> otherwise it is considered dishonest. Please state explicitly which caliber your cited numbers use:

- Metric: sample_solve_rate / correctness / fast@1 / fast@1.2 / other
- Sampling: best-of N / best-turn / greedy
- Turn range: all repair turns / final turn only
- Speedup threshold: ≥1.0x / ≥1.2x / other
- reference_cache: ON / OFF
- Caliber tooling script SHA: ___

## Checklist

- [ ] No internal paths / private data / checkpoints introduced
- [ ] No evaluation-caliber mixing
- [ ] Docs synced (if README / docs are affected)

## Related Issues / PRs

<!-- Link the related issue or PR numbers. -->
