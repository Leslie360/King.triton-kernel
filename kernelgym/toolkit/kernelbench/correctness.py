"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)

logger = logging.getLogger(__name__)


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        logger.info("[Exception %s] %s ", exception_type, str(exception_msg))

    metadata[exception_type] = exception_msg
    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                logger.info("[Eval] Generating Random Input with seed %s", trial_seed)

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            logger.info("device: %s", device)
            logger.info("inputs: %s", inputs[0].device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        logger.info(
                            "[FAIL] trial %s: Output shape mismatch: Expected %s, got %s",
                            trial,
                            output.shape,
                            output_new.shape,
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # Handle bool tensors: cast to float before diffing, since bool
                # does not support subtraction
                output_f = output.to(torch.float32)
                output_new_f = output_new.to(torch.float32)
                if not torch.allclose(output_f, output_new_f, atol=1e-02, rtol=1e-02):
                    max_diff = torch.max(torch.abs(output_f - output_new_f)).item()
                    avg_diff = torch.mean(torch.abs(output_f - output_new_f)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        logger.info("[FAIL] trial %s: Output mismatch", trial)
                else:
                    pass_count += 1
                    if verbose:
                        logger.info("[PASS] trial %s: New Model matches Model", trial)

            except Exception as e:
                logger.error("[Error] Exception happens during correctness check")
                logger.error("Error in launching kernel for ModelNew: %s", e)

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=False
                )
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    if verbose:
        logger.info(
            "[Eval] Pass count: %s, num_correct_trials: %s",
            pass_count,
            num_correct_trials,
        )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
