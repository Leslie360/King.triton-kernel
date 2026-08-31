"""GPU-marker smoke tests.

These exercise the `gpu_available` / `require_gpu` fixtures from conftest.py.
On a CPU-only machine they skip; on a machine with CUDA they run the (cheap)
checks. This file is a template for future GPU integration tests: any test
that actually executes a Triton/CUDA kernel should be marked
``@pytest.mark.gpu`` so the default ``-m "not gpu ..."`` addopts exclude it
on CI runners without a GPU.
"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_gpu_available_flag(gpu_available):
    # Runs only on GPU machines; on CPU it is skipped by the fixture.
    import torch

    assert torch.cuda.is_available() is True
    assert gpu_available is True
    assert torch.cuda.device_count() >= 1


@pytest.mark.gpu
def test_require_gpu_fixture(require_gpu):
    assert require_gpu is True


def test_gpu_fixture_reports_truthy_value(gpu_available):
    # Not GPU-marked: still runs everywhere and just reflects the flag.
    assert isinstance(gpu_available, bool)
