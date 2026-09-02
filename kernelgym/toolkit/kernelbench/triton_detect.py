import torch
import time
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

def _call_inference_no_grad(run: Callable, *args, **kwargs):
    # Avoid gradient / Autocast interfering with detection
    with torch.inference_mode():
        return run(*args, **kwargs)


def _call_inference_with_grad(run: Callable, *args, **kwargs):
    # Avoid gradient / Autocast interfering with detection.
    # with torch.inference_mode() is not enough, because torch.compile checks grad.
    # Avoids tricky issues like grad checks in torch.compile.
    with torch.enable_grad():
        return run(*args, **kwargs)


def _call_inference(run: Callable, *args, **kwargs):
    """Backward-compatible: defaults to the no-grad behavior."""
    return _call_inference_no_grad(run, *args, **kwargs)

def _resolve_triton_jitfunction():
    """Try to resolve Triton's JITFunction type; returns None on failure."""
    try:
        from triton.runtime.jit import JITFunction  # type: ignore
        return JITFunction
    except Exception:
        return None

def _resolve_triton_autotunedkernel():
    try:
        from triton.runtime.autotuner import AutotunedKernel  # type: ignore
        return AutotunedKernel
    except Exception:
        return None

def _resolve_triton_autotuner():
    """Try to resolve Triton's Autotuner class for objects wrapped by the
    @triton.autotune decorator."""
    for mod_path, name in [
        ("triton.autotune", "Autotuner"),
        ("triton.runtime.autotuner", "Autotuner"),
    ]:
        try:
            m = __import__(mod_path, fromlist=[name])
            obj = getattr(m, name, None)
            if obj is not None:
                return obj
        except Exception:
            continue
    return None

def _resolve_triton_cudakernel():
    # The path varies across versions; try as many as we can.
    for mod_path, name in [
        ("triton.runtime.driver", "CUDAKernel"),
        ("triton.backends.nvidia.driver", "CUDAKernel"),
        ("triton.backends.cuda.driver", "CUDAKernel"),
        ("triton.runtime.code_cache", "CUDAKernel"),
    ]:
        try:
            m = __import__(mod_path, fromlist=[name])
            return getattr(m, name, None)
        except Exception:
            continue
    return None

def _resolve_triton_backend_kernels(candidates):
    """Resolve multiple backend kernel classes at once; return the list of found classes."""
    found = []
    for mod_path, name in candidates:
        try:
            m = __import__(mod_path, fromlist=[name])
            cls = getattr(m, name, None)
            if cls is not None:
                found.append(cls)
        except Exception:
            continue
    return found

def _get_kernel_name(obj: Any, _depth: int = 0) -> str:
    # Multi-strategy kernel-name resolution
    try:
        fn_obj = getattr(obj, "fn", None)
        if fn_obj is not None:
            name = getattr(fn_obj, "__name__", None) or getattr(fn_obj, "kernel_name", None)
            if name:
                return str(name)
    except Exception:
        pass
    if _depth < 2:
        try:
            kernel_obj = getattr(obj, "kernel", None)
            if kernel_obj is not None and kernel_obj is not obj:
                name = _get_kernel_name(kernel_obj, _depth + 1)
                if name and name != "unknown":
                    return str(name)
        except Exception:
            pass
    for attr in ("name", "kernel_name", "cache_key"):
        try:
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val:
                return val
        except Exception:
            pass
    try:
        return obj.__class__.__name__
    except Exception:
        return "unknown"


class TritonKernelLaunchHook:
    """Context manager: temporarily hook Triton's possible entry points to capture kernel names."""

    def __init__(self):
        self._JITFunction = None
        self._AutotunedKernel = None
        self._Autotuner = None
        self._CUDAKernel = None
        self._CUDAKernel_classes = []
        self._HIPKernel_classes = []
        self._KernelInterface_classes = []
        self._Launcher_classes = []
        self._orig_methods: List[Tuple[Any, str, Any]] = []
        self._lock = threading.Lock()
        self.captured: List[str] = []
        self._enabled = False

    def _append_capture(self, name: str, grid: Any, fn_or_obj: Any):
        try:
            module = None
            filename = None
            try:
                module = getattr(fn_or_obj, "__module__", None)
            except Exception:
                pass
            try:
                code = getattr(fn_or_obj, "__code__", None)
                if code is not None:
                    filename = getattr(code, "co_filename", None)
            except Exception:
                pass
            info_extra = []
            if module:
                info_extra.append(f"module={module}")
            if filename:
                info_extra.append(f"file={filename}")
            extra = (" "+" ".join(info_extra)) if info_extra else ""
            with self._lock:
                self.captured.append(f"{name} grid={grid}{extra}")
        except Exception:
            # Fallback: at least record name and grid
            try:
                with self._lock:
                    self.captured.append(f"{name} grid={grid}")
            except Exception:
                pass

    def _patch_method(self, cls: Any, method_name: str, wrapper_factory: Callable[[Any], Callable]):
        try:
            orig = getattr(cls, method_name, None)
            if orig is None:
                return
            # Avoid double-patching
            if getattr(orig, "_kb_patched", False):
                return
            wrapped = wrapper_factory(orig)
            setattr(wrapped, "_kb_patched", True)
            setattr(cls, method_name, wrapped)
            self._orig_methods.append((cls, method_name, orig))
        except Exception:
            pass

    def _get_grid_from_obj(self, obj: Any):
        for attr in ("grid", "launch_grid", "grid_fn", "launch_grid_fn"):
            try:
                val = getattr(obj, attr, None)
                if val is not None:
                    return val
            except Exception:
                continue
        return None

    def _wrap_call_with_grid(
        self,
        name_obj_getter: Optional[Callable[[Any], Any]] = None,
        grid_from_args: bool = False,
        grid_from_obj: bool = False,
    ):
        def _factory(orig):
            def _patched(obj, *args, **kwargs):
                try:
                    name_obj = name_obj_getter(obj) if name_obj_getter else obj
                    name = _get_kernel_name(name_obj)
                    grid = kwargs.get("grid", None)
                    if grid is None and grid_from_obj:
                        grid = self._get_grid_from_obj(obj)
                    if grid is None and grid_from_args and len(args) >= 1:
                        grid = args[0]
                    self._append_capture(name, grid, name_obj)
                except Exception:
                    pass
                return orig(obj, *args, **kwargs)
            return _patched
        return _factory

    def _wrap_getitem(self, name_obj_getter: Optional[Callable[[Any], Any]] = None):
        def _factory(orig):
            def _patched(obj, grid):
                launcher = orig(obj, grid)
                name_obj = name_obj_getter(obj) if name_obj_getter else obj
                name = _get_kernel_name(name_obj)
                def _wrapper(*a, **k):
                    self._append_capture(name, grid, name_obj)
                    return launcher(*a, **k)
                setattr(_wrapper, "_kb_patched", True)
                return _wrapper
            return _patched
        return _factory

    def __enter__(self):
        self._JITFunction = _resolve_triton_jitfunction()
        self._AutotunedKernel = _resolve_triton_autotunedkernel()
        self._Autotuner = _resolve_triton_autotuner()
        self._CUDAKernel = _resolve_triton_cudakernel()
        # Resolve multiple backend classes at once, for full coverage
        self._CUDAKernel_classes = _resolve_triton_backend_kernels([
            ("triton.runtime.driver", "CUDAKernel"),
            ("triton.backends.nvidia.driver", "CUDAKernel"),
            ("triton.backends.cuda.driver", "CUDAKernel"),
            ("triton.runtime.code_cache", "CUDAKernel"),
        ])
        self._HIPKernel_classes = _resolve_triton_backend_kernels([
            ("triton.backends.amd.driver", "HIPKernel"),
            ("triton.backends.rocm.driver", "HIPKernel"),
        ])
        self._KernelInterface_classes = _resolve_triton_backend_kernels([
            ("triton.runtime.jit", "KernelInterface"),
            ("triton.runtime.jit", "Kernel"),
            ("triton.runtime.jit", "CompiledKernel"),
        ])
        self._Launcher_classes = _resolve_triton_backend_kernels([
            ("triton.runtime.launcher", "Launcher"),
            ("triton.runtime.launcher", "KernelLauncher"),
            ("triton.runtime.launcher", "KernelLauncherBase"),
        ])

        # 1) JITFunction path
        jf = self._JITFunction
        if jf is not None:
            # hook legacy launch
            def wrap_launch(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(obj)
                    grid = kwargs.get("grid", None)
                    if grid is None and len(args) >= 1:
                        grid = args[0]
                    self._append_capture(name, grid, getattr(obj, "fn", obj))
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(jf, "launch", wrap_launch)
            self._patch_method(jf, "run", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))

            # Direct call: kernel(..., grid=...)
            def wrap_call(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(obj)
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, getattr(obj, "fn", obj))
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(jf, "__call__", wrap_call)

            # Subscript call: kernel[grid](...)
            def wrap_getitem(orig):
                def _patched(obj, grid):
                    launcher = orig(obj, grid)
                    name = _get_kernel_name(obj)
                    def _wrapper(*a, **k):
                        self._append_capture(name, grid, getattr(obj, "fn", obj))
                        return launcher(*a, **k)
                    setattr(_wrapper, "_kb_patched", True)
                    return _wrapper
                return _patched
            self._patch_method(jf, "__getitem__", wrap_getitem)

        # 2) AutotunedKernel may be returned directly as a decorator
        ak = self._AutotunedKernel
        if ak is not None:
            def wrap_ak_call(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(getattr(obj, "fn", obj))
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, getattr(obj, "fn", obj))
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(ak, "__call__", wrap_ak_call)
            self._patch_method(ak, "run", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))
            self._patch_method(ak, "launch", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
                grid_from_args=True,
            ))

            def wrap_ak_getitem(orig):
                def _patched(obj, grid):
                    launcher = orig(obj, grid)
                    name = _get_kernel_name(getattr(obj, "fn", obj))
                    def _wrapper(*a, **k):
                        self._append_capture(name, grid, getattr(obj, "fn", obj))
                        return launcher(*a, **k)
                    setattr(_wrapper, "_kb_patched", True)
                    return _wrapper
                return _patched
            self._patch_method(ak, "__getitem__", wrap_ak_getitem)

        # 2.5) Autotuner wrapper (newer @triton.autotune)
        at = self._Autotuner
        if at is not None:
            def wrap_at_call(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(getattr(obj, "fn", obj))
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, getattr(obj, "fn", obj))
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(at, "__call__", wrap_at_call)
            self._patch_method(at, "run", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))
            self._patch_method(at, "launch", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
                grid_from_args=True,
            ))

            def wrap_at_getitem(orig):
                def _patched(obj, grid):
                    launcher = orig(obj, grid)
                    name = _get_kernel_name(getattr(obj, "fn", obj))
                    def _wrapper(*a, **k):
                        self._append_capture(name, grid, getattr(obj, "fn", obj))
                        return launcher(*a, **k)
                    setattr(_wrapper, "_kb_patched", True)
                    return _wrapper
                return _patched
            self._patch_method(at, "__getitem__", wrap_at_getitem)

        # 3) Low-level CUDAKernel (if present)
        ck = self._CUDAKernel
        if ck is not None:
            def wrap_ck_call(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(obj)
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, obj)
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(ck, "__call__", wrap_ck_call)
            self._patch_method(ck, "run", self._wrap_call_with_grid())
            self._patch_method(ck, "launch", self._wrap_call_with_grid(
                grid_from_args=True,
            ))

        # 3.5) Multi-path CUDAKernel classes (best-effort full coverage)
        for ck_cls in self._CUDAKernel_classes:
            def wrap_ck_call_gen(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(obj)
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, obj)
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(ck_cls, "__call__", wrap_ck_call_gen)
            self._patch_method(ck_cls, "run", self._wrap_call_with_grid())
            self._patch_method(ck_cls, "launch", self._wrap_call_with_grid(
                grid_from_args=True,
            ))

        # 3.6) HIPKernel (AMD/ROCm backend)
        for hk_cls in self._HIPKernel_classes:
            def wrap_hk_call_gen(orig):
                def _patched(obj, *args, **kwargs):
                    name = _get_kernel_name(obj)
                    grid = kwargs.get("grid", None)
                    self._append_capture(name, grid, obj)
                    return orig(obj, *args, **kwargs)
                return _patched
            self._patch_method(hk_cls, "__call__", wrap_hk_call_gen)
            self._patch_method(hk_cls, "run", self._wrap_call_with_grid())
            self._patch_method(hk_cls, "launch", self._wrap_call_with_grid(
                grid_from_args=True,
            ))

        # 4) KernelInterface/Kernel abstract classes (covers other kernel entry points)
        for ki_cls in self._KernelInterface_classes:
            self._patch_method(ki_cls, "__call__", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))
            self._patch_method(ki_cls, "run", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))
            self._patch_method(ki_cls, "launch", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "fn", o),
                grid_from_args=True,
            ))
            self._patch_method(ki_cls, "__getitem__", self._wrap_getitem(
                name_obj_getter=lambda o: getattr(o, "fn", o),
            ))

        # 5) Launcher (also captures if grid is pre-cached)
        for launcher_cls in self._Launcher_classes:
            self._patch_method(launcher_cls, "__call__", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "kernel", o),
                grid_from_obj=True,
            ))
            self._patch_method(launcher_cls, "launch", self._wrap_call_with_grid(
                name_obj_getter=lambda o: getattr(o, "kernel", o),
                grid_from_obj=True,
            ))

        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return
        # Restore all patched methods
        for cls, method_name, orig in reversed(self._orig_methods):
            try:
                setattr(cls, method_name, orig)
            except Exception:
                pass
        self._orig_methods.clear()
        self._enabled = False


def detect_triton_usage(
    fn_or_model: Union[Callable[..., Any], torch.nn.Module],
    *args: Any,
    warmup: int = 1,
    steps: int = 1,
    use_cuda: Optional[bool] = True,
    return_matches: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[bool, Tuple[bool, List[str]]]:
    """
    Detect whether one or more forward calls actually invoked a Triton kernel
    (via hooking Triton's run entry points).

    Args:
      - fn_or_model: callable, or nn.Module. If nn.Module, internally calls its forward
        under inference_mode().
      - *args, **kwargs: arguments passed to the callable / model's forward.
      - warmup: number of warm-up runs (recommend >=1 to trigger JIT/compile and
        avoid recording only the compilation phase).
      - steps: number of capture runs (recommend 1-3).
      - use_cuda: whether to synchronize CUDA after each call (affects stability
        only, not the hook itself).
      - return_matches: whether to also return the captured Triton kernel names
        (with grid).
      - profile_kwargs: legacy-compatibility placeholder, ignored.

    Returns:
      - If return_matches=False: bool indicating whether any Triton kernel was detected.
      - If return_matches=True: (bool, List[str]); the second item is the
        deduplicated, lexicographically-sorted list of captured kernel descriptions.

    Notes:
      - Only detects Triton custom kernels; does not cover ATen / cuBLAS / cuDNN
        kernel names.
    """
    def _run_detection(run_callable: Callable[..., Any]) -> Tuple[bool, List[str]]:
        # Warm up to trigger any JIT/compile + cache
        for _ in range(max(0, int(warmup))):
            _ = run_callable(*args, **kwargs)
            if (use_cuda is None and torch.cuda.is_available()) or (use_cuda is True):
                torch.cuda.synchronize()

        with TritonKernelLaunchHook() as hook:
            for _ in range(max(1, int(steps))):
                _ = run_callable(*args, **kwargs)
                if (use_cuda is None and torch.cuda.is_available()) or (use_cuda is True):
                    torch.cuda.synchronize()

        matched = sorted(set(hook.captured))
        used = len(matched) > 0
        return used, matched

    # Capture Triton kernel launches
    try:
        import triton  # noqa: F401 — only used to confirm Triton exists
    except Exception:
        return (False, []) if return_matches else False

    if isinstance(fn_or_model, torch.nn.Module):
        run_callable_no_grad = lambda *a, **k: _call_inference_no_grad(fn_or_model, *a, **k)
        run_callable_with_grad = lambda *a, **k: _call_inference_with_grad(fn_or_model, *a, **k)
    else:
        run_callable_no_grad = fn_or_model
        run_callable_with_grad = fn_or_model

    used_no_grad, matches_no_grad = _run_detection(run_callable_no_grad)
    used_with_grad, matches_with_grad = _run_detection(run_callable_with_grad)

    used = used_no_grad and used_with_grad
    all_matches = sorted(set(matches_no_grad) | set(matches_with_grad))

    if return_matches:
        return used, all_matches
    return used


def detect_triton_usage_for_module(
    model: torch.nn.Module,
    *inputs: Any,
    warmup: int = 1,
    steps: int = 1,
    use_cuda: Optional[bool] = None,
    return_matches: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None,
    **forward_kwargs: Any,
) -> Union[bool, Tuple[bool, List[str]]]:
    """
    Convenience wrapper: pass an nn.Module and the inputs to its forward.
    """
    # model.eval()
    return detect_triton_usage(
        model, *inputs,
        warmup=warmup,
        steps=steps,
        use_cuda=use_cuda,
        return_matches=return_matches,
        profile_kwargs=profile_kwargs,
        **forward_kwargs,
    )


# ============================ CUDA Detection ============================

import threading


class TorchOpsCallHook:
    """Hook torch.ops calls to capture custom-op calls outside the core namespace.

    Notes:
      - Monkey-patches OpOverload.__call__ to count the called ops.
      - Only records ops whose namespace is not in the core set AND have at
        least one CUDA-device tensor argument.
    Limitations:
      - Relies on the internal API torch._ops.OpOverload; not guaranteed stable
        across versions.
      - Cannot cover custom kernels called through other indirect paths (most
        common extensions go through torch.ops, though).
    """

    CORE_NAMESPACES = {
        "aten", "prim", "prims", "quantized", "mkldnn", "xnnpack",
        "sparse", "c10", "_caffe2", "_aten", "mps", "xla"
    }

    def __init__(self):
        self._OpOverload = None
        self._orig_call = None
        self._lock = threading.Lock()
        self.captured = []
        self._enabled = False

    def __enter__(self):
        try:
            import torch as _torch
            # Get the type of an OpOverload instance
            sample_overload = getattr(_torch.ops, "aten").add.Tensor
            self._OpOverload = type(sample_overload)
        except Exception:
            return self

        def _patched(op_overload, *args, **kwargs):
            name = None
            try:
                packet = getattr(op_overload, "overloadpacket", None)
                ns = getattr(packet, "namespace", None)
                op = getattr(packet, "name", None)
                overload = getattr(op_overload, "overloadname", None)
                name = f"{ns}::{op}.{overload}" if overload else f"{ns}::{op}"
                # Only count non-core namespaces
                if ns and ns not in TorchOpsCallHook.CORE_NAMESPACES:
                    # Check whether any CUDA tensor is involved
                    has_cuda = False
                    for a in args:
                        try:
                            if isinstance(a, _torch.Tensor) and a.is_cuda:
                                has_cuda = True
                                break
                        except Exception:
                            pass
                    if not has_cuda:
                        for v in kwargs.values():
                            try:
                                if isinstance(v, _torch.Tensor) and v.is_cuda:
                                    has_cuda = True
                                    break
                            except Exception:
                                pass
                    if has_cuda:
                        with self._lock:
                            self.captured.append(f"torch.ops:{name}")
            except Exception:
                pass
            return self._orig_call(op_overload, *args, **kwargs)

        try:
            self._orig_call = self._OpOverload.__call__
            if not getattr(self._orig_call, "_kb_patched", False):
                setattr(self._OpOverload, "__call__", _patched)
                setattr(self._OpOverload.__call__, "_kb_patched", True)
                self._enabled = True
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._OpOverload and self._orig_call is not None:
            try:
                setattr(self._OpOverload, "__call__", self._orig_call)
            except Exception:
                pass
        self._enabled = False


class NumbaCudaLaunchHook:
    """Hook Numba CUDA kernel launch paths to capture kernel calls."""

    def __init__(self):
        self._classes = []  # [(cls, method_name, orig)]
        self._lock = threading.Lock()
        self.captured = []
        self._enabled = False

    def _patch_method(self, cls, method_name):
        try:
            orig = getattr(cls, method_name, None)
            if orig is None:
                return
            if getattr(orig, "_kb_patched", False):
                return

            def _wrap(orig_fn):
                def _patched(obj, *args, **kwargs):
                    try:
                        name = getattr(obj, "py_func", None)
                        if name is None:
                            name = getattr(obj, "__name__", None)
                        if name is None:
                            name = obj.__class__.__name__
                        with self._lock:
                            self.captured.append(f"numba:{name}")
                    except Exception:
                        pass
                    return orig_fn(obj, *args, **kwargs)
                return _patched

            patched = _wrap(orig)
            setattr(patched, "_kb_patched", True)
            setattr(cls, method_name, patched)
            self._classes.append((cls, method_name, orig))
        except Exception:
            pass

    def __enter__(self):
        try:
            import numba.cuda.compiler as ncc  # type: ignore
        except Exception:
            return self

        # Try common call paths
        for cls_name in ("CUDAKernel", "Dispatcher"):
            try:
                cls = getattr(ncc, cls_name, None)
                if cls is None:
                    continue
                self._patch_method(cls, "__call__")
                self._patch_method(cls, "__getitem__")
            except Exception:
                pass

        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return
        for cls, method_name, orig in reversed(self._classes):
            try:
                setattr(cls, method_name, orig)
            except Exception:
                pass
        self._classes.clear()
        self._enabled = False


class CuPyKernelLaunchHook:
    """Hook CuPy custom Kernel call paths."""

    def __init__(self):
        self._records = []  # [(cls, method, orig)]
        self._lock = threading.Lock()
        self.captured = []
        self._enabled = False

    def _patch(self, cls, method_name, label):
        try:
            orig = getattr(cls, method_name, None)
            if orig is None:
                return
            if getattr(orig, "_kb_patched", False):
                return

            def _wrap(orig_fn):
                def _patched(obj, *args, **kwargs):
                    try:
                        name = getattr(obj, "name", None) or obj.__class__.__name__
                        with self._lock:
                            self.captured.append(f"cupy:{label}:{name}")
                    except Exception:
                        pass
                    return orig_fn(obj, *args, **kwargs)
                return _patched

            patched = _wrap(orig)
            setattr(patched, "_kb_patched", True)
            setattr(cls, method_name, patched)
            self._records.append((cls, method_name, orig))
        except Exception:
            pass

    def __enter__(self):
        try:
            import cupy as cp  # type: ignore
        except Exception:
            return self

        for cls, method, label in [
            (getattr(cp, "RawKernel", None), "__call__", "raw"),
            (getattr(cp, "ElementwiseKernel", None), "__call__", "elementwise"),
        ]:
            if cls is not None:
                self._patch(cls, method, label)

        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return
        for cls, method, orig in reversed(self._records):
            try:
                setattr(cls, method, orig)
            except Exception:
                pass
        self._records.clear()
        self._enabled = False


class CudaKernelLaunchHook:
    """Top-level hook: tries TorchOps / Numba / CuPy captures in parallel."""

    def __init__(self):
        self._torch_hook = TorchOpsCallHook()
        self._numba_hook = NumbaCudaLaunchHook()
        self._cupy_hook = CuPyKernelLaunchHook()
        self.captured = []

    def __enter__(self):
        self._torch_hook.__enter__()
        self._numba_hook.__enter__()
        self._cupy_hook.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._torch_hook.__exit__(exc_type, exc, tb)
        self._numba_hook.__exit__(exc_type, exc, tb)
        self._cupy_hook.__exit__(exc_type, exc, tb)
        # Aggregate
        self.captured = (
            list(self._torch_hook.captured)
            + list(self._numba_hook.captured)
            + list(self._cupy_hook.captured)
        )


def detect_cuda_usage(
    fn_or_model,
    *args,
    warmup: int = 1,
    steps: int = 1,
    use_cuda: Optional[bool] = True,
    return_matches: bool = False,
    **kwargs,
):
    """Detect whether a forward call used any "custom CUDA kernel".

    Strategy (best-effort):
      - Capture kernel/op calls via hooks from:
        1) non-core namespaces under torch.ops (treated as custom extensions)
        2) CUDA kernels compiled by numba.cuda.jit
        3) cupy.RawKernel / ElementwiseKernel calls

    Returns:
      - bool or (bool, List[str])
    """
    is_module = isinstance(fn_or_model, torch.nn.Module)
    run_callable = (lambda *a, **k: _call_inference(fn_or_model, *a, **k)) if is_module else fn_or_model

    # Warm up
    for _ in range(max(0, int(warmup))):
        _ = run_callable(*args, **kwargs)
        if (use_cuda is None and torch.cuda.is_available()) or (use_cuda is True):
            torch.cuda.synchronize()

    # Capture
    with CudaKernelLaunchHook() as hook:
        for _ in range(max(1, int(steps))):
            _ = run_callable(*args, **kwargs)
            if (use_cuda is None and torch.cuda.is_available()) or (use_cuda is True):
                torch.cuda.synchronize()

    matched = sorted(set(hook.captured))
    used = len(matched) > 0
    if return_matches:
        return used, matched
    return used


def detect_cuda_usage_for_module(
    model: torch.nn.Module,
    *inputs,
    warmup: int = 1,
    steps: int = 1,
    use_cuda: Optional[bool] = None,
    return_matches: bool = False,
    **forward_kwargs,
):
    # model.eval()
    return detect_cuda_usage(
        model, *inputs,
        warmup=warmup,
        steps=steps,
        use_cuda=use_cuda,
        return_matches=return_matches,
        **forward_kwargs,
    )
