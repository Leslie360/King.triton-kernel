"""KernelBench evaluation pipeline (task-level, toolkit layer)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import torch

from kernelgym.config import settings
from kernelgym.toolkit.kernelbench import triton_detect as detect
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult, get_error_name, set_seed
from kernelgym.toolkit.kernelbench.loading import (
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness
from kernelgym.toolkit.kernelbench.profiling import compute_triton_kernel_coverage
from kernelgym.toolkit.kernelbench.timing import (
    get_timing_stats,
    run_profiling_only,
    time_execution_with_cuda_event,
)

logger = logging.getLogger("kernelgym.pipeline")


def _run_correctness_step(
    original_model,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_correct_trials: int,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
) -> KernelExecResult:
    if verbose:
        logger.info("[Eval] Checking Correctness")
    try:
        return run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def _run_triton_detection_step(
    *,
    enable_triton_detection: bool,
    is_triton: bool,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    seed_num: int,
    device: Union[torch.device, int],
    verbose: bool,
    backend: str,
):
    if not enable_triton_detection:
        return False
    try:
        logger.info("Begin Triton usage detection")
        if kernel_exec_result and kernel_exec_result.correctness:
            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            used, matches = detect.detect_triton_usage_for_module(
                model_new,
                *inputs,
                warmup=1,
                steps=1,
                use_cuda=True,
                return_matches=True,
            )
            metadata["triton_profiler_used"] = used
            metadata["triton_profiler_matches"] = matches
            logger.info("Triton usage detection result: %s", used)
            logger.info("Triton usage detection matches: %s", matches)
            if not used and is_triton:
                logger.warning(
                    "[Eval] Backend is 'triton' but no Triton usage detected, marking as decoy"
                )
                kernel_exec_result.decoy_kernel = True
                kernel_exec_result.runtime = -1.0
                return True
    except Exception as e:
        if verbose:
            logger.error("[Eval] Error in Triton usage detection: %s", e)
        metadata["error_in_triton_detection"] = e
    return False


def _run_performance_step(
    *,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: Dict[str, Any],
    num_perf_trials: int,
    verbose: bool,
    seed_num: int,
    device: Union[torch.device, int],
    enable_profiling: bool,
):
    def _profiling_empty(metrics: Dict[str, Any]) -> bool:
        if not metrics:
            return True
        if "kernels" not in metrics:
            return True
        if len(metrics.get("kernels", [])) == 0:
            return True
        return False

    try:
        if kernel_exec_result and kernel_exec_result.correctness:
            if verbose:
                logger.info("[Eval] Measuring Performance as Sample is Correct")

            torch.cuda.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model_new = custom_model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            elapsed_times, profiling_metrics = time_execution_with_cuda_event(
                model_new,
                *inputs,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
                enable_profiling=enable_profiling,
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if enable_profiling and _profiling_empty(profiling_metrics):
                retry_count = max(0, int(getattr(settings, "profiling_retry_count", 0)))
                for attempt in range(retry_count):
                    logger.warning(
                        "[WARNING] Profiler returned empty results. Retrying (%s/%s)...",
                        attempt + 1,
                        retry_count,
                    )
                    retry_metrics = run_profiling_only(
                        model_new,
                        *inputs,
                        num_trials=max(1, min(num_perf_trials, 10)),
                        verbose=verbose,
                        device=device,
                    )
                    if not _profiling_empty(retry_metrics):
                        profiling_metrics = retry_metrics
                        break
                    profiling_metrics = retry_metrics

            if enable_profiling:
                logger.debug(
                    "[DEBUG] profiling_metrics type: %s, empty: %s",
                    type(profiling_metrics),
                    not profiling_metrics,
                )
                if profiling_metrics.get("profiling_warning"):
                    logger.warning(
                        "[WARNING] Profiling warning: %s",
                        profiling_metrics["profiling_warning"],
                    )

                if _profiling_empty(profiling_metrics):
                    logger.warning("[WARNING] Profiler returned empty results!")
                    logger.warning(
                        "[WARNING] This may be a profiler bug, not a decoy kernel issue."
                    )
                    logger.warning(
                        "[WARNING] Triton hook detected: %s",
                        metadata.get("triton_profiler_used", False),
                    )
                    logger.warning(
                        "[WARNING] Triton matches: %s",
                        len(metadata.get("triton_profiler_matches", [])),
                    )
                    if metadata.get("triton_profiler_used", False):
                        logger.info(
                            "[INFO] Skipping decoy detection due to profiler failure (Triton hook passed)"
                        )

            if profiling_metrics and len(profiling_metrics) > 0:
                metadata["profiling"] = profiling_metrics
                if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
                    kernel_exec_result.metadata["profiling"] = profiling_metrics

                logger.debug(
                    "[DEBUG Profiling] profiling_metrics keys: %s",
                    profiling_metrics.keys(),
                )
                logger.debug(
                    "[DEBUG Profiling] kernel_count: %s",
                    profiling_metrics.get("kernel_count", "N/A"),
                )
                logger.debug(
                    "[DEBUG Profiling] triton_profiler_matches: %s",
                    metadata.get("triton_profiler_matches", []),
                )

                try:
                    coverage_result_dict = compute_triton_kernel_coverage(
                        metadata["triton_profiler_matches"], profiling_metrics
                    )
                except Exception as coverage_error:
                    logger.error(
                        "[ERROR] compute_triton_kernel_coverage failed: %s",
                        coverage_error,
                    )
                    import traceback

                    traceback.print_exc()
                    coverage_result_dict = {
                        "num_custom_kernels": 0,
                        "num_total_kernels": 0,
                        "triton_kernels_not_in_profiling": metadata.get(
                            "triton_profiler_matches", []
                        ),
                        "triton_kernels_in_profiling": [],
                        "total_kernel_run_time_in_profiling_us": 0,
                        "custom_kernel_cuda_time_in_profiling_us": 0,
                    }
                logger.debug(
                    "[DEBUG Coverage] num_custom_kernels: %s",
                    coverage_result_dict["num_custom_kernels"],
                )
                logger.debug(
                    "[DEBUG Coverage] num_total_kernels: %s",
                    coverage_result_dict["num_total_kernels"],
                )
                num_custom_kernels = coverage_result_dict["num_custom_kernels"]
                num_total_kernels = coverage_result_dict["num_total_kernels"]
                triton_kernels_not_in_profiling = coverage_result_dict[
                    "triton_kernels_not_in_profiling"
                ]
                triton_kernels_in_profiling = coverage_result_dict[
                    "triton_kernels_in_profiling"
                ]
                total_kernel_run_time_in_profiling_us = coverage_result_dict[
                    "total_kernel_run_time_in_profiling_us"
                ]
                custom_kernel_cuda_time_in_profiling_us = coverage_result_dict[
                    "custom_kernel_cuda_time_in_profiling_us"
                ]

                metadata["num_custom_kernels"] = num_custom_kernels
                metadata["num_total_kernels"] = num_total_kernels
                ratio = num_custom_kernels / num_total_kernels if num_total_kernels > 0 else 0
                metadata[
                    "triton_kernel_coverage"
                ] = f"Run {num_custom_kernels} custom kernels / Total {num_total_kernels} kernels, Coverage: {ratio:.2%}"
                metadata["triton_kernel_not_in_profiling"] = (
                    triton_kernels_not_in_profiling
                )
                metadata["triton_kernel_in_profiling"] = triton_kernels_in_profiling

                metadata[
                    "total_kernel_run_time_in_profiling_us"
                ] = total_kernel_run_time_in_profiling_us
                metadata[
                    "custom_kernel_cuda_time_in_profiling_us"
                ] = custom_kernel_cuda_time_in_profiling_us
                ratio_time = (
                    custom_kernel_cuda_time_in_profiling_us
                    / total_kernel_run_time_in_profiling_us
                    if total_kernel_run_time_in_profiling_us > 0
                    else 0
                )
                metadata[
                    "custom_kernel_cuda_time_coverage"
                ] = (
                    f"Custom kernel CUDA time: {custom_kernel_cuda_time_in_profiling_us:.2f}us / Total time: {total_kernel_run_time_in_profiling_us:.2f}us, Coverage: {ratio_time:.2%}"
                )

                if kernel_exec_result and isinstance(kernel_exec_result.metadata, dict):
                    kernel_exec_result.metadata["num_custom_kernels"] = num_custom_kernels
                    kernel_exec_result.metadata["num_total_kernels"] = num_total_kernels
                    kernel_exec_result.metadata[
                        "triton_kernel_coverage"
                    ] = f"Run {num_custom_kernels} custom kernels / Total {num_total_kernels} kernels, Coverage: {ratio:.2%}"
                    kernel_exec_result.metadata["triton_profiler_matches"] = metadata[
                        "triton_profiler_matches"
                    ]

                    kernel_exec_result.metadata[
                        "custom_kernel_cuda_time_in_profiling_us"
                    ] = custom_kernel_cuda_time_in_profiling_us
                    kernel_exec_result.metadata[
                        "total_kernel_run_time_in_profiling_us"
                    ] = total_kernel_run_time_in_profiling_us
                    kernel_exec_result.metadata[
                        "custom_kernel_cuda_time_coverage"
                    ] = (
                        f"Custom kernel CUDA time: {custom_kernel_cuda_time_in_profiling_us:.2f}us / Total time: {total_kernel_run_time_in_profiling_us:.2f}us, Coverage: {ratio_time:.2%}"
                    )

                if num_custom_kernels == 0 and num_total_kernels > 0:
                    logger.warning(
                        "[WARNING] Profiler captured %s kernels but 0 custom kernels - marking as decoy",
                        num_total_kernels,
                    )
                    kernel_exec_result.decoy_kernel = True
                elif num_custom_kernels == 0 and num_total_kernels == 0:
                    logger.warning(
                        "[WARNING] Profiler captured 0 total kernels - likely profiler bug, NOT marking as decoy"
                    )
                    logger.info(
                        "[INFO] Relying on Triton hook detection instead (detected: %s)",
                        metadata.get("triton_profiler_used", False),
                    )
            if verbose:
                logger.info("[Eval] Performance Stats: %s", runtime_stats)
            kernel_exec_result.runtime = runtime_stats["mean"]
            kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            logger.error("[Eval] Error in Measuring Performance: %s", e)
        kernel_exec_result.metadata["error_during_performance"] = e

def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = True,
    measure_performance: bool = True,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    backend: str = "cuda",
    entry_point: str = "Model",
    enable_profiling: bool = True,
    enable_triton_detection: bool = True,
    backend_adapter: Optional[Any] = None,
) -> KernelExecResult:
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.cuda.set_device(device)
    is_triton = backend == "triton"
    metadata: Dict[str, Any] = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    if is_triton:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "cuda", "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(f"device must be an int or torch.device, got {type(device)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        logger.info("[Eval] Start Evalulation! on device: %s", device)
        logger.info("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context, entry_point
    )
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    logger.debug("[DEBUG] init inputs: %s", init_inputs)

    if (
        len(init_inputs) > 1
        and hasattr(init_inputs[0], "__len__")
        and not isinstance(init_inputs[0], (str, torch.Tensor))
        and len(init_inputs[0]) == 0
    ):
        init_inputs = init_inputs[1]

    with torch.no_grad():
        set_seed(seed_num)

        if type(init_inputs) == list:
            original_model = Model(*init_inputs)
        else:
            original_model = Model(**init_inputs)

        assert hasattr(original_model, "forward")
        if verbose:
            logger.info("[Eval] Original Model Loaded")
    if verbose:
        logger.info("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    tempfile_handle = None
    backend_handle = None
    backend_session = None

    def _cleanup():
        if backend_session is not None:
            backend_session.close()
            return
        if backend_adapter is not None and backend_handle is not None:
            backend_adapter.cleanup(backend_handle)
            return
        graceful_eval_cleanup(context, device, tempfile_handle)

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if backend_adapter is not None:
            artifact = backend_adapter.compile(
                custom_model_src,
                device=device,
                backend=backend,
                entry_point=f"{entry_point}New",
                build_dir=build_dir,
            )
            if not artifact.get("compiled"):
                error = artifact.get("error", "Unknown compile error")
                if "lock" in str(error) or "No such file or directory" in str(error):
                    logger.error(
                        "[Eval] Lock file error during compilation, Please retry. Error: %s",
                        error,
                    )
                    _cleanup()
                    return None
                metadata["compilation_error_name"] = "compile_error"
                metadata["compilation_error"] = error
                _cleanup()
                return KernelExecResult(compiled=False, metadata=metadata)

            backend_handle = backend_adapter.load(
                artifact,
                device=device,
                context=context,
                build_dir=build_dir,
            )
            backend_session = backend_adapter.open_session(backend_handle, device=device)
            tempfile_handle = backend_handle.get("tempfile_handle")
        else:
            if is_triton:
                ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                    custom_model_src, entry_point=f"{entry_point}New"
                )
                if verbose:
                    logger.info("[Eval] Model with Triton Loaded")
            else:
                ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        logger.error(
            "Failed to compile custom CUDA kernel: Record as compilation failure. \nError: %s",
            e,
        )

        if "lock" in str(e) or "No such file or directory" in str(e):
            logger.error(
                "[Eval] Lock file error during compilation, Please retry. Error: %s",
                e,
            )
            _cleanup()
            return None
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e
        _cleanup()
        return KernelExecResult(compiled=False, metadata=metadata)

    try:
        def _create_custom_model():
            if backend_session is not None:
                return backend_session.create_model(
                    init_inputs,
                    no_grad=True,
                    synchronize=False,
                )
            if type(init_inputs) == list:
                return ModelNew(*init_inputs)
            return ModelNew(**init_inputs)

        with torch.no_grad():
            set_seed(seed_num)
            custom_model = _create_custom_model()

            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            logger.info("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        logger.error(
            "Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \n"
            "Error: %s",
            e,
        )
        _cleanup()
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    kernel_exec_result = None

    kernel_exec_result = _run_correctness_step(
        original_model,
        custom_model,
        get_inputs,
        metadata,
        num_correct_trials,
        verbose,
        seed_num,
        device,
    )

    decoy_detected = _run_triton_detection_step(
        enable_triton_detection=enable_triton_detection,
        is_triton=is_triton,
        kernel_exec_result=kernel_exec_result,
        custom_model=custom_model,
        get_inputs=get_inputs,
        metadata=metadata,
        seed_num=seed_num,
        device=device,
        verbose=verbose,
        backend=backend,
    )
    if decoy_detected:
        _cleanup()
        return kernel_exec_result

    if measure_performance:
        _run_performance_step(
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            seed_num=seed_num,
            device=device,
            enable_profiling=enable_profiling,
        )

    _cleanup()
    return kernel_exec_result




def eval_reference_only(
    original_model_src: str,
    seed_num: int = 42,
    num_perf_trials: int = 10,
    verbose: bool = False,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),
    entry_point: str = "Model",
    reference_backend: Optional[str] = None,
    backend_adapter: Optional[Any] = None,
) -> KernelExecResult:
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    # 2026-08-26 TF32-baseline switch (aligned with the KernelBench-Verified protocol):
    # the reference runs pure FP32 eager by default, while the candidate runs on
    # tensor cores → speedup is systematically overestimated.
    # Set ENABLE_TF32_BASELINE=True so the reference also uses TF32. Default off
    # (keeps the current caliber).
    if os.environ.get("ENABLE_TF32_BASELINE", "False").lower() == "true":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.cuda.set_device(device)
    metadata: Dict[str, Any] = {}
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    context: Dict[str, Any] = {}

    if verbose:
        logger.info("[Eval] Start Evaluation! on device: %s", device)
        logger.info("[Eval] Loading Original Model")

    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
            original_model_src, context, entry_point
        )
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        with torch.no_grad():
            set_seed(seed_num)
            if type(init_inputs) == list:
                original_model = Model(*init_inputs)
            else:
                original_model = Model(**init_inputs)
            assert hasattr(original_model, "forward")
        if verbose:
            logger.info("[Eval] Original Model Loaded")

    except Exception as e:
        logger.error("Failed to load original model: %s", e)
        metadata["model_load_error"] = e
        metadata["model_load_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    kernel_exec_result = KernelExecResult(compiled=True, correctness=True, metadata=metadata)

    try:
        if verbose:
            logger.info("[Eval] Measuring Performance of Original Model")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        model = original_model.cuda(device=device)
        if reference_backend:
            backend_name = reference_backend.lower()
            metadata["reference_backend"] = backend_name
            logger.info("[Eval] reference_backend=%s", backend_name)
            if backend_name in ("torch_compile", "torch-compile", "compile"):
                try:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("torch.compile is not available")
                    model = torch.compile(model)
                    metadata["reference_backend_compiled"] = True
                    logger.info("[Eval] torch.compile succeeded")
                except Exception as e:
                    metadata["reference_backend_error"] = str(e)
                    logger.error("[Eval] torch.compile failed: %s", e)
                    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        torch.cuda.synchronize(device=device)

        elapsed_times, _ = time_execution_with_cuda_event(
            model,
            *inputs,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
            enable_profiling=False,
        )
        runtime_stats = get_timing_stats(elapsed_times, device=device)

        if verbose:
            logger.info("[Eval] Performance Stats: %s", runtime_stats)
        kernel_exec_result.runtime = runtime_stats["mean"]
        kernel_exec_result.runtime_stats = runtime_stats
    except Exception as e:
        if verbose:
            logger.error("[Eval] Error in Measuring Performance: %s", e)
        kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device, None)
    return kernel_exec_result
