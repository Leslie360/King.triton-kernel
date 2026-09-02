# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel reward function implementation.

Integrates with KernelServer to evaluate kernel code quality and performance.
"""

import asyncio
import logging
import os
import re
from typing import Dict, Any
from kernel.rewards.reward_client import KernelRewardClient


# Global client instance and its config; reuse connection and rebuild on config change
_global_client = None
_global_client_cfg = {}


def _resolve_use_reference_cache(reward_config) -> bool:
    """
    A4: Resolve whether to enable reference_cache (scoring no longer re-runs the
    reference denominator).
    Priority (high → low):
      1. ENABLE_REFERENCE_CACHE env var (A/B calibration; explicitly overrides config)
      2. reward_config.reference_cache.enable (the top-level reference_cache section
         is merged into reward_model by main_kernel/main_grading and threaded through
         reward_config to here)
      3. Default False (preserves original behavior, no fallback)
    """
    env = os.environ.get("ENABLE_REFERENCE_CACHE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        rc = getattr(reward_config, "reference_cache", None)
        if rc is not None:
            enable = getattr(rc, "enable", False)
            if enable:
                return True
    except Exception:
        pass
    return False


def attach_reference_cache(reward_model_cfg, full_cfg):
    """
    A4: Merge the top-level reference_cache section into reward_model config so it
    threads through reward_config to kernel_reward.py (Hydra config is frozen by
    default, so use OmegaConf.merge to return a new config instead of mutating in
    place).
    Returns the new reward_model config; on exception returns the original — never
    block training/eval because of A4.
    """
    if reward_model_cfg is None or full_cfg is None:
        return reward_model_cfg
    try:
        rc = full_cfg.get("reference_cache", None)
        if rc is None:
            return reward_model_cfg
        from omegaconf import OmegaConf
        merged = OmegaConf.merge(reward_model_cfg, {"reference_cache": rc})
        return merged
    except Exception:
        return reward_model_cfg


def extract_reference_code(solution_str: str) -> str:
    """
    Extract reference code from the solution string.

    Args:
        solution_str: Full string containing prompt and response.

    Returns:
        The extracted reference code.
    """
    # Look for reference-implementation markers
    patterns = [
        r"# Reference Implementation\s*\n(.*?)(?=# Your Task|# Generate|$)",
        r"```python\s*# Reference\s*\n(.*?)```",
        r"# PyTorch Reference:\s*\n(.*?)(?=# Task|# Generate|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no specific marker, try to extract the first Python code block
    code_block_match = re.search(r"```python\s*\n(.*?)```", solution_str, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Fall back to the whole string
    return solution_str


def extract_kernel_code(solution_str: str) -> str:
    """
    Extract kernel code from the solution string.

    Args:
        solution_str: Full string containing prompt and response.

    Returns:
        The extracted kernel code.
    """
    # Look for kernel-implementation markers
    patterns = [
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no specific marker, use the shared robust extractor (take last block +
    # strip prose, aligned with eval-side main_grading, 2026-08-27 L4 fix to
    # prevent "wrong-engine modification" regression)
    try:
        from ..rew_common import extract_code_robust
    except ImportError:
        from kernel.rew_common import extract_code_robust
    return extract_code_robust(solution_str)

def compute_kernel_reward_batch(solution_strs: list, ground_truths: list, entry_points: str, **kwargs) -> list:
    """
    Compute kernel-code rewards in batch.

    Args:
        solution_strs: List of solution strings.
        ground_truths: List of reference implementations.
        **kwargs: Other arguments.

    Returns:
        List of reward results.
    """
    try:
        # Prepare task data
        tasks = []

        # Read client config uniformly from reward_config
        reward_config = kwargs.get("reward_config", None)
        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model
        uuids = kwargs.get("uuids", None)
        is_valid = kwargs.get("is_valid", False)

        try:
            task_timeout = getattr(reward_config, "task_timeout", None)
            task_timeout_in_client = getattr(reward_config, "task_timeout_in_client", None)
        except Exception:
            task_timeout = None
            task_timeout_in_client = None

        num_perf_trials = getattr(reward_config, "num_perf_trials")
        num_correct_trials = getattr(reward_config, "num_correct_trials")
        enable_profiling = getattr(reward_config, "enable_profiling")
        verbose_errors = getattr(reward_config, "verbose_errors")
        detect_decoy_kernel = getattr(reward_config, "detect_decoy_kernel")
        reference_backend = getattr(reward_config, "reference_backend")

        # A4: Resolve reference_cache switch (config threading + env override)
        use_reference_cache = _resolve_use_reference_cache(reward_config)
        if use_reference_cache:
            logging.info("A4: reference_cache ENABLED (scoring no longer re-runs reference denominator)")

        for i, solution_str in enumerate(solution_strs):
            # reference_code = extract_reference_code(solution_str)
            reference_code = ground_truths[i]
            kernel_code = extract_kernel_code(solution_str)
            entry_point = entry_points[i]

            if uuids is not None:
                uuid = uuids[i]
            

            
            tasks.append({
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": entry_point,
                "use_reference_cache": use_reference_cache,
                "uuid": uuid if uuids is not None else "",
                "is_valid": is_valid,
                "task_timeout": task_timeout,
                "task_timeout_in_client": task_timeout_in_client,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "enable_profiling": enable_profiling,
                "verbose_errors": verbose_errors,
                "detect_decoy_kernel": detect_decoy_kernel,
                "reference_backend": reference_backend,
            })
        
        # Synchronously call the async function
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get the client and compute rewards in batch (read values only from reward_config)
        if reward_config is None:
            raise ValueError("reward_config is required")

        server_url = getattr(reward_config, "server_url", None)
        if not server_url:
            raise ValueError("server_url is required and cannot be None or empty")

        global _global_client, _global_client_cfg
        if _global_client is None or _global_client_cfg is not reward_config:
            _global_client = KernelRewardClient(reward_config=reward_config)
            _global_client_cfg = reward_config
            
        client = _global_client
        
        # Call passing task_timeout; A4: use_reference_cache resolved from config/env
        results = loop.run_until_complete(
            client.compute_batch_rewards(tasks, use_reference_cache=use_reference_cache,
                                       is_valid=is_valid, task_timeout=task_timeout,
                                       task_timeout_in_client=task_timeout_in_client)
        )
        
        return results
        
    except Exception as e:
        logging.error(f"Error in compute_kernel_reward_batch: {e}")
        # Return a list of error results
        return [
            {
                "score": reward_config.reward_policy.penalties.penalty_score,
                "reward": reward_config.reward_policy.penalties.penalty_score,
                "correctness": False,
                "success": False,
                "error": str(e)
            }
            for _ in solution_strs
        ]
