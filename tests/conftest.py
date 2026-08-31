"""Shared pytest fixtures for King.triton-kernel.

Goals:
- Make ``kernelgym`` importable from the repo root without installation.
- Provide a ``gpu_available`` fixture (skip when torch.cuda is unavailable).
- Provide a ``fake_redis`` fixture for tests that only need dict-like redis.
- Stub out the heavy optional RL deps (ray/httpx/verl) so the pure reward
  math in ``drkernel.kernel.rewards.reward_client`` can be unit-tested on a
  CPU-only box without ever touching a real Ray cluster or the verl sandbox.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path: make the repo root importable so `import kernelgym` / `import drkernel`
# work regardless of whether the package is pip-installed.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# gpu_available fixture — skip tests that need a real GPU when torch/cuda
# is unavailable. The fixture itself is import-torch safe: it only touches
# torch inside the try block.
# ---------------------------------------------------------------------------
@pytest.fixture
def gpu_available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _skipif_no_gpu(gpu_available):
    if not gpu_available:
        pytest.skip("GPU required but torch.cuda.is_available() is False")


# A helper used by GPU-marked tests: skip cleanly on CPU-only machines.
@pytest.fixture
def require_gpu(gpu_available):
    _skipif_no_gpu(gpu_available)
    return True


# ---------------------------------------------------------------------------
# fake_redis — minimal dict-backed stand-in for real redis (integration-free).
# ---------------------------------------------------------------------------
class FakeRedis:
    """A tiny dict-backed redis lookalike sufficient for unit tests."""

    def __init__(self):
        self._store = {}

    def set(self, key, value, ex=None):
        self._store[key] = value
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, *keys):
        removed = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                removed += 1
        return removed

    def exists(self, *keys):
        return sum(1 for k in keys if k in self._store)

    def flushall(self):
        self._store.clear()
        return True

    def keys(self, pattern="*"):
        import fnmatch

        return [k for k in self._store if fnmatch.fnmatch(str(k), pattern)]


@pytest.fixture
def fake_redis():
    return FakeRedis()


# ---------------------------------------------------------------------------
# Optional RL dep stubs (ray / httpx / verl).
#
# reward_client.py does module-level `import httpx`, `import ray` and
# `from verl.tools.sandbox_fusion_tools import TokenBucketWorker`. On a
# minimal install none of these exist. We inject lightweight stand-ins into
# sys.modules BEFORE the module is imported so the pure reward methods
# (`_cheap_syntax_filter`, `_bucketed_speedup_reward`, ...) can be exercised.
#
# The @ray.remote decorator is turned into an identity so `_HybridHttpWorker`
# can still be defined (we never instantiate it in unit tests).
# ---------------------------------------------------------------------------
def _install_optional_stubs(monkeypatch):
    # ---- httpx stub (only the symbols used at import time matter) ----
    if "httpx" not in sys.modules:
        httpx_stub = types.ModuleType("httpx")
        httpx_stub.Timeout = object
        httpx_stub.Limits = object
        httpx_stub.Client = object
        httpx_stub.TimeoutException = type("TimeoutException", (Exception,), {})
        httpx_stub.ConnectError = type("ConnectError", (Exception,), {})
        sys.modules["httpx"] = httpx_stub

    # ---- verl stub ----
    if "verl" not in sys.modules:
        verl_stub = types.ModuleType("verl")
        verl_tools = types.ModuleType("verl.tools")
        sandbox_fusion = types.ModuleType("verl.tools.sandbox_fusion_tools")

        class TokenBucketWorker:
            @classmethod
            def options(cls, **kwargs):
                return _OptionsProxy(cls)

            @staticmethod
            def remote(*args, **kwargs):
                return object()

        class _OptionsProxy:
            def __init__(self, cls):
                self._cls = cls

            def remote(self, *args, **kwargs):
                return self._cls.remote(*args, **kwargs)

        sandbox_fusion.TokenBucketWorker = TokenBucketWorker
        verl_tools.sandbox_fusion_tools = sandbox_fusion
        verl_stub.tools = verl_tools
        sys.modules["verl"] = verl_stub
        sys.modules["verl.tools"] = verl_tools
        sys.modules["verl.tools.sandbox_fusion_tools"] = sandbox_fusion

    # ---- ray stub ----
    if "ray" not in sys.modules:
        ray_stub = types.ModuleType("ray")

        def _remote_decorator(*args, **kwargs):
            # Return a decorator that passes the class through unchanged.
            if args and callable(args[0]):
                return args[0]
            return lambda cls: cls

        def _options_remote(*args, **kwargs):
            return object()

        class _RemoteProxy:
            @staticmethod
            def options(*args, **kwargs):
                return _RemoteProxy

            @staticmethod
            def remote(*args, **kwargs):
                return object()

        def _get(ref):
            return ref

        def _wait(refs, **kwargs):
            if isinstance(refs, list):
                return (list(refs), [])
            return ([refs], [])

        class ObjectRef:
            pass

        ray_stub.remote = _remote_decorator
        ray_stub.get = _get
        ray_stub.wait = _wait
        ray_stub.ObjectRef = ObjectRef
        sys.modules["ray"] = ray_stub


@pytest.fixture
def reward_deps_stub(monkeypatch):
    """Install sys.modules stubs for ray/httpx/verl, then load reward_client.py
    directly by file path.

    We load it standalone (importlib.util.spec_from_file_location) rather than
    `from drkernel.kernel.rewards import reward_client` because that package's
    __init__.py contains a broken non-relative import
    (`from kernel.rewards.kernel_reward import ...`) that fails regardless of
    the ray/verl stubs. reward_client.py itself only uses absolute imports, so
    loading it on its own is safe and keeps this test CPU-only."""
    _install_optional_stubs(monkeypatch)

    import importlib.util

    _path = _REPO_ROOT / "drkernel" / "kernel" / "rewards" / "reward_client.py"
    spec = importlib.util.spec_from_file_location("reward_client", _path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_collection_modifyitems(config, items):
    """Do not auto-skip gpu-marked tests here; the fixture handles it.
    Also make 'gpu' marker configurable via -m but let default addopts win."""
    pass
