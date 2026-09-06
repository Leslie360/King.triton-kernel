"""Tests for kernelgym/toolkit/validation.py.

Covers:
- validate_code: empty / missing class / present.
- early_kernel_validation: missing class, syntax errors, missing imports, cuda
  backend missing indicator, unicode (curly quotes / full-width punctuation)
  normalization, and a valid triton kernel passing.
"""

from __future__ import annotations

from kernelgym.common import ErrorCode
from kernelgym.toolkit.validation import early_kernel_validation, validate_code


# ---------------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------------
def test_validate_code_empty():
    ok, msg = validate_code("")
    assert ok is False
    assert "required" in msg.lower()


def test_validate_code_missing_class():
    code = "import triton\nx = 1\n"
    ok, msg = validate_code(code, entry_point="Model")
    assert ok is False
    assert "Model" in msg


def test_validate_code_present():
    code = "import triton\nclass Model:\n    pass\n"
    ok, msg = validate_code(code, entry_point="Model")
    assert ok is True
    assert msg == ""


def test_validate_code_custom_entry_point():
    code = "class KernelNew:\n    pass\n"
    ok, _ = validate_code(code, entry_point="KernelNew")
    assert ok is True


# ---------------------------------------------------------------------------
# early_kernel_validation — missing class
# ---------------------------------------------------------------------------
def test_early_missing_modelnew_class():
    # validate_code looks for `class ModelNew` when entry_point="Model"
    code = "import triton\n\nx = 1\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is False
    assert ec is ErrorCode.VALIDATION_ERROR
    assert "ModelNew" in msg


def test_early_empty_code():
    ok, msg, ec = early_kernel_validation("", backend="triton")
    assert ok is False
    assert ec is ErrorCode.VALIDATION_ERROR


# ---------------------------------------------------------------------------
# early_kernel_validation — syntax / unicode normalization
# ---------------------------------------------------------------------------
def test_early_syntax_error():
    code = (
        "import triton\n"
        "class ModelNew:\n"
        "    def __init__(self):\n"
        "        if True print('oops')  # invalid syntax\n"
    )
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is False
    assert ec is ErrorCode.SYNTAX_ERROR
    assert "syntax" in msg.lower()


def test_early_fullwidth_parens_normalized():
    # Full-width parens would be a syntax error unless normalized.
    code = "import triton\nclass ModelNew:\n    def __init__（self）:\n        self.x = 1\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    # After normalization `__init__（self）` -> `__init__(self)` and compiles.
    assert ok is True, (msg, ec)
    assert ec is None


def test_early_curly_quotes_normalized():
    # Curly double quotes inside code are normalized to straight quotes.
    code = "import triton\nclass ModelNew:\n    def __init__(self):\n        self.label = “hello”\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is True, (msg, ec)
    assert ec is None


def test_early_arrow_normalized():
    code = "import triton\nclass ModelNew:\n    def f(self, x):\n        return x → y\n"
    # `→` normalizes to `->`, which is still invalid in this context...
    # but the point is normalization happens; this exercises the replace path.
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is False
    assert ec is ErrorCode.SYNTAX_ERROR


# ---------------------------------------------------------------------------
# early_kernel_validation — import requirements
# ---------------------------------------------------------------------------
def test_early_triton_backend_missing_import():
    code = "class ModelNew:\n    pass\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is False
    assert ec is ErrorCode.IMPORT_ERROR
    assert "import triton" in msg.lower()


def test_early_triton_backend_from_import():
    code = "from triton import language as tl\nclass ModelNew:\n    pass\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    # `from triton import` satisfies the import requirement.
    assert ok is True, (msg, ec)
    assert ec is None


def test_early_cuda_backend_missing_indicator():
    code = "import numpy\nclass ModelNew:\n    pass\n"
    ok, msg, ec = early_kernel_validation(code, backend="cuda")
    assert ok is False
    assert ec is ErrorCode.IMPORT_ERROR
    assert "cuda" in msg.lower()


def test_early_cuda_backend_torch_cuda():
    code = "import torch\nclass ModelNew:\n    def __init__(self):\n        torch.cuda.empty_cache()\n"
    ok, msg, ec = early_kernel_validation(code, backend="cuda")
    # torch.cuda indicator present -> passes import gate.
    assert ok is True, (msg, ec)
    assert ec is None


# ---------------------------------------------------------------------------
# early_kernel_validation — valid triton kernel
# ---------------------------------------------------------------------------
def test_early_valid_triton_kernel():
    code = (
        "import triton\n"
        "import triton.language as tl\n"
        "import torch\n"
        "@triton.jit\n"
        "def add_kernel(x_ptr, y_ptr, n):\n"
        "    pid = tl.program_id(0)\n"
        "    tl.store(y_ptr + pid, 1.0)\n"
        "\n"
        "class ModelNew:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is True, (msg, ec)
    assert ec is None


def test_early_valid_simple_triton_no_jit():
    # No @triton.jit / kernel pattern -> early-return path (backend triton).
    code = "import triton\nclass ModelNew:\n    def __init__(self):\n        pass\n"
    ok, msg, ec = early_kernel_validation(code, backend="triton")
    assert ok is True, (msg, ec)
    assert ec is None
