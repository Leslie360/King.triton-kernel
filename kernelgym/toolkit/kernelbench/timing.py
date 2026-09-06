"""KernelBench timing helpers (toolkit layer)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from kernelgym.toolkit.kernelbench.profiling import (
    extract_profiling_metrics,
    profiling_context,
)

logger = logging.getLogger(__name__)


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
    enable_profiling: bool = False,
) -> tuple[list[float], dict[str, Any]]:
    if device is None:
        if verbose:
            logger.info("Using current device: %s", torch.cuda.current_device())
        device = torch.cuda.current_device()

    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    logger.info(
        "[Profiling] Using device: %s %s, warm up %s, trials %s",
        device,
        torch.cuda.get_device_name(device),
        num_warmup,
        num_trials,
    )
    elapsed_times = []

    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        torch.cuda.synchronize(device=device)

        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            logger.info("Trial %s: %s ms", trial + 1, f"{elapsed_time_ms:.3g}")
        elapsed_times.append(elapsed_time_ms)

    profiling_metrics: dict[str, Any] = {}
    if enable_profiling:
        try:
            torch.cuda.synchronize(device=device)

            num_profiling_trials = min(10, num_trials)
            logger.info(
                "[Profiling] Running %s additional iterations for profiling...",
                num_profiling_trials,
            )

            with profiling_context(True) as prof:
                for _ in range(num_profiling_trials):
                    kernel_fn(*args)
                torch.cuda.synchronize(device=device)

            profiling_metrics = extract_profiling_metrics(prof)
            if profiling_metrics:
                logger.info(
                    "[Profiling] Captured %s CUDA kernels",
                    profiling_metrics.get("kernel_count", 0),
                )
                logger.info(
                    "[Profiling] Total CUDA time: %s us",
                    f"{profiling_metrics.get('total_cuda_time_us', 0):.2f}",
                )

        except Exception as e:
            logger.error("[Profiling] Warning: Profiling failed: %s", e)
            profiling_metrics = {"profiling_error": str(e)}

    return elapsed_times, profiling_metrics


def run_profiling_only(
    kernel_fn: callable,
    *args,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> dict[str, Any]:
    if device is None:
        if verbose:
            logger.info("Using current device: %s", torch.cuda.current_device())
        device = torch.cuda.current_device()

    profiling_metrics: dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device=device)
        logger.info("[Profiling] Running %s iterations (profiling-only)...", num_trials)
        with profiling_context(True) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        profiling_metrics = extract_profiling_metrics(prof)
        if profiling_metrics:
            logger.info(
                "[Profiling] Captured %s CUDA kernels",
                profiling_metrics.get("kernel_count", 0),
            )
    except Exception as e:
        logger.error("[Profiling] Warning: Profiling-only failed: %s", e)
        profiling_metrics = {"profiling_error": str(e)}

    return profiling_metrics


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)

    return stats
