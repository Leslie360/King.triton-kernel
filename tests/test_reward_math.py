"""Tests for the pure reward math in drkernel/kernel/rewards/reward_client.py.

reward_client.py imports ray/httpx/verl at module level, which are stubbed by
the `reward_deps_stub` fixture in conftest.py. We construct a bare
KernelRewardClient via __new__ (skipping __init__ which spins up Ray workers)
and set only the attributes each pure method needs.

Covers:
- _cheap_syntax_filter
- _bucketed_speedup_reward
- _over_replacement_penalty
- calculate_reward_discrete
- calculate_reward_like_kernel
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


_STATE = {}


@pytest.fixture(autouse=True)
def _install_reward_stubs(reward_deps_stub):
    """Load reward_client.py (with ray/httpx/verl stubbed) and stash the module
    so `_client` can build bare KernelRewardClient instances without touching
    the broken drkernel.kernel.rewards package __init__."""
    _STATE["module"] = reward_deps_stub
    return reward_deps_stub


def _client(rc, **attrs):
    KernelRewardClient = _STATE["module"].KernelRewardClient

    obj = KernelRewardClient.__new__(KernelRewardClient)
    obj.reward_config = rc
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _penalties(comp=-0.5, correct=-0.3, perf=-0.1):
    return {"compilation_fail": comp, "correctness_fail": correct, "perf_degrade": perf}


@pytest.fixture
def rc():
    """A minimal reward_config with just enough for the pure methods."""
    return SimpleNamespace(
        reward_policy=SimpleNamespace(penalties=_penalties()),
    )


@pytest.fixture
def client(rc):
    return _client(rc)


# ---------------------------------------------------------------------------
# _cheap_syntax_filter
# ---------------------------------------------------------------------------
def test_cheap_syntax_filter_empty():
    c = _client(None)
    ok, err = c._cheap_syntax_filter("")
    assert ok is False
    assert err == "empty kernel"


def test_cheap_syntax_filter_syntax_error():
    c = _client(None)
    ok, err = c._cheap_syntax_filter("import triton\nif True print('bad')\nclass Model:\n pass\n")
    assert ok is False
    assert "syntax" in err


def test_cheap_syntax_filter_no_triton():
    c = _client(None)
    ok, err = c._cheap_syntax_filter("import torch\nclass Model:\n    pass\n")
    assert ok is False
    assert err == "no-triton"


def test_cheap_syntax_filter_no_model_class():
    c = _client(None)
    ok, err = c._cheap_syntax_filter("import triton\nx = 1\n")
    assert ok is False
    assert err == "no-model-class"


def test_cheap_syntax_filter_accepts_modelnew():
    c = _client(None)
    code = "import triton\nclass ModelNew:\n    def __init__(self):\n        pass\n"
    ok, err = c._cheap_syntax_filter(code)
    assert ok is True
    assert err == ""


def test_cheap_syntax_filter_unicode_normalized():
    # Full-width parens / curly quotes get normalized before ast.parse.
    c = _client(None)
    code = "import triton\nclass ModelNew:\n    def __init__（self）:\n        pass\n"
    ok, err = c._cheap_syntax_filter(code)
    assert ok is True, err


# ---------------------------------------------------------------------------
# _bucketed_speedup_reward (default bands)
# ---------------------------------------------------------------------------
def test_bucketed_below_one_is_zero():
    c = _client(None)
    assert c._bucketed_speedup_reward(0.5) == 0.0
    assert c._bucketed_speedup_reward(0.99) == 0.0


def test_bucketed_at_band_anchors():
    c = _client(None)
    assert c._bucketed_speedup_reward(1.0) == pytest.approx(0.2)
    assert c._bucketed_speedup_reward(1.2) == pytest.approx(0.4)
    assert c._bucketed_speedup_reward(1.5) == pytest.approx(0.6)
    assert c._bucketed_speedup_reward(2.0) == pytest.approx(0.8)
    assert c._bucketed_speedup_reward(3.0) == pytest.approx(1.0)


def test_bucketed_interpolation():
    c = _client(None)
    # 1.1 between 1.0(0.2) and 1.2(0.4) -> 0.3
    assert c._bucketed_speedup_reward(1.1) == pytest.approx(0.3)
    # 1.35 between 1.2(0.4) and 1.5(0.6) -> 0.5
    assert c._bucketed_speedup_reward(1.35) == pytest.approx(0.5)


def test_bucketed_capped_at_three():
    c = _client(None)
    assert c._bucketed_speedup_reward(5.0) == pytest.approx(1.0)
    assert c._bucketed_speedup_reward(100.0) == pytest.approx(1.0)


def test_bucketed_custom_bands():
    custom = SimpleNamespace(bucket_bands=[(1.0, 0.1), (3.0, 1.0)])
    c = _client(custom)
    assert c._bucketed_speedup_reward(2.0) == pytest.approx(0.55)  # mid interpolation
    assert c._bucketed_speedup_reward(1.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _over_replacement_penalty
# ---------------------------------------------------------------------------
def _over_client(rc=None, **attrs):
    default_attrs = dict(
        penalize_over_replacement=True,
        over_replacement_penalty=-0.3,
        over_replacement_speedup_threshold=1.0,
        over_replacement_min_custom_kernels=1,
        over_replacement_opt_backends={"pytorch", "native", "torch"},
    )
    default_attrs.update(attrs)
    return _client(rc or SimpleNamespace(), **default_attrs)


def test_over_penalty_disabled_flag():
    c = _over_client(penalize_over_replacement=False)
    result = {"num_custom_kernel": 2, "num_total_kernels": 2,
              "reference_backend": "pytorch"}
    assert c._over_replacement_penalty(result, speedup=0.5) == 0.0


def test_over_penalty_no_custom_kernel():
    c = _over_client()
    result = {"num_custom_kernel": 0, "num_total_kernels": 2,
              "reference_backend": "pytorch"}
    assert c._over_replacement_penalty(result, speedup=0.5) == 0.0


def test_over_penalty_not_degraded():
    c = _over_client()
    result = {"num_custom_kernel": 2, "num_total_kernels": 2,
              "reference_backend": "pytorch"}
    assert c._over_replacement_penalty(result, speedup=1.5) == 0.0


def test_over_penalty_non_opt_backend():
    c = _over_client()
    result = {"num_custom_kernel": 2, "num_total_kernels": 2,
              "reference_backend": "custom_cpp"}
    assert c._over_replacement_penalty(result, speedup=0.5) == 0.0


def test_over_penalty_base_case():
    # num_total=10 so coverage (1/10=0.1) < 0.5 -> H2 does NOT amplify.
    c = _over_client()
    result = {"num_custom_kernel": 1, "num_total_kernels": 10,
              "reference_backend": "pytorch"}
    assert c._over_replacement_penalty(result, speedup=0.8) == pytest.approx(-0.3)


def test_over_penalty_no_backend_still_triggers():
    # no backend field -> treated as "unable to judge" -> degrades+replace triggers
    c = _over_client()
    result = {"num_custom_kernel": 1, "num_total_kernels": 10}
    assert c._over_replacement_penalty(result, speedup=0.8) == pytest.approx(-0.3)


def test_over_penalty_metadata_plural_read():
    c = _over_client()
    result = {"metadata": {"num_custom_kernels": 1, "num_total_kernels": 10},
              "reference_backend": "pytorch"}
    assert c._over_replacement_penalty(result, speedup=0.8) == pytest.approx(-0.3)


def test_over_penalty_h2_coverage_amplifies():
    c = _over_client()
    result = {"num_custom_kernel": 2, "num_total_kernels": 2,
              "reference_backend": "pytorch"}
    # coverage 1.0 >= 0.5 -> penalty * 1.5 = -0.45
    assert c._over_replacement_penalty(result, speedup=0.5) == pytest.approx(-0.45)


def test_over_penalty_h3_time_extra():
    c = _over_client()
    result = {
        "num_custom_kernel": 1,
        "num_total_kernels": 10,  # low coverage so H2 does not fire
        "custom_kernel_cuda_time_in_profiling_us": 800,
        "total_kernel_run_time_in_profiling_us": 1000,
        "reference_backend": "pytorch",
    }
    # H3 time_coverage 0.8 >= 0.5 -> penalty -0.1 -> -0.4
    assert c._over_replacement_penalty(result, speedup=0.5) == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# calculate_reward_discrete
# ---------------------------------------------------------------------------
def test_discrete_failure():
    c = _client(None)
    r = c.calculate_reward_discrete({"status": "failed", "error_message": "x"})
    assert r["reward"] == -1.0
    assert r["success"] is False


def test_discrete_decoy():
    c = _client(None)
    r = c.calculate_reward_discrete(
        {"status": "completed", "decoy_kernel": True, "correctness": True,
         "compiled": True, "speedup": 5.0}
    )
    assert r["reward"] == -1.0
    assert r["decoy_kernel"] is True


def test_discrete_not_compiled():
    c = _client(None)
    r = c.calculate_reward_discrete(
        {"status": "completed", "correctness": False, "compiled": False,
         "speedup": 0.0}
    )
    assert r["reward"] == -1.0


def test_discrete_compiled_wrong():
    c = _client(None)
    r = c.calculate_reward_discrete(
        {"status": "completed", "correctness": False, "compiled": True,
         "speedup": 0.0}
    )
    assert r["reward"] == 1.0


def test_discrete_correct_slow():
    c = _client(None)
    r = c.calculate_reward_discrete(
        {"status": "completed", "correctness": True, "compiled": True,
         "speedup": 1.0}
    )
    assert r["reward"] == 2.0


def test_discrete_correct_fast():
    c = _client(None)
    r = c.calculate_reward_discrete(
        {"status": "completed", "correctness": True, "compiled": True,
         "speedup": 1.5}
    )
    assert r["reward"] == 3.0


# ---------------------------------------------------------------------------
# calculate_reward_like_kernel
# ---------------------------------------------------------------------------
def test_like_failure():
    c = _client(SimpleNamespace(reward_policy=SimpleNamespace(penalties=_penalties())))
    r = c.calculate_reward_like_kernel({"status": "timeout", "error_message": "took long"})
    assert r["reward"] == -1.0
    assert r["success"] is False


def test_like_decoy():
    c = _client(None)
    r = c.calculate_reward_like_kernel(
        {"status": "completed", "decoy_kernel": True, "correctness": True,
         "compiled": True, "speedup": 3.0}
    )
    assert r["reward"] == -1.0
    assert r["decoy_kernel"] is True


def test_like_compilation_fail(rc):
    c = _client(rc)
    r = c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": False, "correctness": False, "speedup": 0.0}
    )
    assert r["reward"] == pytest.approx(-0.5)


def test_like_correctness_fail(rc):
    c = _client(rc)
    r = c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": False, "speedup": 0.0}
    )
    assert r["reward"] == pytest.approx(-0.3)


def test_like_speedup_bands(rc):
    c = _client(rc)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 3.0}
    )["reward"] == pytest.approx(1.0)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 2.5}
    )["reward"] == pytest.approx(0.8)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.6}
    )["reward"] == pytest.approx(0.6)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.3}
    )["reward"] == pytest.approx(0.4)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 1.1}
    )["reward"] == pytest.approx(0.2)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 0.5}
    )["reward"] == pytest.approx(-0.1)


def test_like_custom_penalties():
    cfg = SimpleNamespace(reward_policy=SimpleNamespace(
        penalties=_penalties(comp=-1.0, correct=-0.2, perf=-0.05)))
    c = _client(cfg)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": False, "correctness": False, "speedup": 0.0}
    )["reward"] == pytest.approx(-1.0)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": False, "speedup": 0.0}
    )["reward"] == pytest.approx(-0.2)
    assert c.calculate_reward_like_kernel(
        {"status": "completed", "compiled": True, "correctness": True, "speedup": 0.5}
    )["reward"] == pytest.approx(-0.05)
