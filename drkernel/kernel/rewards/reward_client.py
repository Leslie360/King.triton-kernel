"""
Hybrid Kernel reward client (composed implementation):
- External API matches KernelServer (/evaluate submit, /status poll, /results fetch).
- Concurrency and rate limiting use the sandbox fusion Ray worker pool + global token bucket.
- Does not inherit Enhanced; it directly reuses the core request/poll/reward logic to keep behavior aligned,
  leaving observability to be added later if needed.

Two-level timeout design:
1. task_timeout (in payload["timeout"]): Server-side execution limit for kernel evaluation
2. task_timeout_in_client: Client-side polling timeout including queue wait time
Invariant: task_timeout_in_client >= task_timeout (client waits longer due to queuing)
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
import random
from uuid import uuid4

import httpx
import ray

from verl.tools.sandbox_fusion_tools import TokenBucketWorker


logger = logging.getLogger(__name__)


@ray.remote
class _HybridHttpWorker:
    def __init__(self, server_url: str, rate_limit: int, default_timeout: int, acquire_timeout: int) -> None:
        self.server_url = server_url
        # print(f"[DEBUG] Default timeout: {default_timeout}")
        self.default_timeout = int(default_timeout)
        self.acquire_timeout = int(acquire_timeout)
        self._limits = httpx.Limits(max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.default_timeout, write=10.0, pool=5.0),
            limits=self._limits,
            headers={"Content-Type": "application/json"},
        )
        self._rate_limit_worker = TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def _backoff(self, attempt: int, base: int = 2, cap: int = 30) -> float:
        return min(base ** attempt, cap)

    def get_token_in_use(self) -> int:
        try:
            return ray.get(self._rate_limit_worker.get_current_count.remote())
        except Exception:
            return -1

    def submit_and_poll(self, task_data: Dict[str, Any], client_timeout: int, max_retries: Optional[int], requeue_max: int = 0) -> Dict[str, Any]:
        """Submit task and poll for results, with requeue on result-wait timeout.

        Args:
            task_data: Task payload including server-side timeout in task_data["timeout"]
            client_timeout: Client-side total timeout including queue wait + execution time
            max_retries: Max retry attempts for submission failures
            requeue_max: Max requeues when the result never reaches a terminal state
                within client_timeout (lost-task recovery, A2).
        """
        # ===== 2026-08-28 A2: requeue on result-wait timeout =====
        # 270 eval freeze incident: worker restart dropped in-flight tasks → server-side
        # task stuck in PROCESSING forever; old behavior returned timeout on poll exhaustion
        # and silently dropped the task.
        # Fix: on timeout (task still non-terminal), re-submit with a **new task_id**, up to
        # requeue_max times. Only after exhausting retries do we return a timeout failure —
        # no more infinite waits, no more silent drops.
        # ===== 2026-08-29 A2 blind-spot extension: "stall-then-failed" =====
        # step305 incident: task stalled on the server for a long time then returned failed
        # (not timeout). Old A2 only requeued status=timeout and missed this case. Now:
        # if status=failed AND elapsed > stall_threshold (likely stall, not real failure),
        # also requeue. Real failures usually complete within seconds (compile/runtime error);
        # stall-then-failed runs far longer. A misjudged requeue only costs one extra run
        # (capped by requeue_max); requeueing beats silently dropping.
        requeue_max = int(requeue_max or 0)
        stall_threshold = max(client_timeout * 0.5, 120.0)  # failed but elapsed > 50% or >120s => suspected stall
        attempt = 0
        while True:
            result = self._submit_and_poll_once(task_data, client_timeout, max_retries)
            elapsed = float(result.get("elapsed", 0))
            is_timeout = result.get("status") == "timeout"
            is_stall_failed = (result.get("status") == "failed" and elapsed > stall_threshold)
            if (is_timeout or is_stall_failed) and attempt < requeue_max:
                attempt += 1
                # Fresh task_id: the old one is stuck in PROCESSING on the server,
                # re-submitting the same id would silently no-op / never complete.
                new_task = dict(task_data)
                new_task["task_id"] = f"parallel_task_retry_{uuid4().hex[:8]}"
                try:
                    reason = "timeout" if is_timeout else f"stall-failed({elapsed:.0f}s)"
                    print(f"[HybridWorker] A2 requeue #{attempt}/{requeue_max} reason={reason} old_task_id={task_data.get('task_id','')} new_task_id={new_task['task_id']}")
                except Exception:
                    pass
                task_data = new_task
                continue
            return result

    def _submit_and_poll_once(self, task_data: Dict[str, Any], client_timeout: int, max_retries: Optional[int]) -> Dict[str, Any]:
        """Single submit-and-poll attempt (no requeue). Returns a result dict."""
        start_ts = time.time()
        # Rate-limit only during submission; polling does not consume tokens.
        try:
            # Submit with limited retries: 429/503/timeout/connect errors.
            attempt = 0
            unlimited = max_retries is None or max_retries == -1
            while unlimited or attempt < (max_retries or 0):
                try:
                    # Acquire token with timeout.
                    acquire_ref = self._rate_limit_worker.acquire.remote()
                    ready, _ = ray.wait([acquire_ref], timeout=self.acquire_timeout)
                    if not ready:
                        try:
                            curr = ray.get(self._rate_limit_worker.get_current_count.remote())
                        except Exception:
                            curr = -1
                        print(f"[HybridWorker] acquire timeout tokens_in_use={curr}")
                        return {"status": "failed", "error_message": "rate limiter acquire timeout"}
                    # Log once on first attempt to help debug "server did not receive request".
                    if attempt == 0:
                        print(f"[HybridWorker] POST /evaluate task_id={task_data.get('task_id', '')} url={self.server_url}")
                    resp = self._client.post(f"{self.server_url}/evaluate", json=task_data)
                    # Log status code to help diagnose non-200 responses.
                    try:
                        print(f"[HybridWorker] POST /evaluate resp={resp.status_code} task_id={task_data.get('task_id','')}")
                    except Exception:
                        pass
                    # Release token immediately after submission.
                    try:
                        self._rate_limit_worker.release.remote()
                    except Exception:
                        pass
                    if resp.status_code == 200:
                        break
                    if resp.status_code in (429, 503):
                        time.sleep(self._backoff(attempt, base=2 if resp.status_code == 429 else 5))
                        attempt += 1
                        continue
                    resp.raise_for_status()
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    try:
                        self._rate_limit_worker.release.remote()
                    except Exception:
                        pass
                    if unlimited or attempt < (max_retries or 0) - 1:
                        time.sleep(self._backoff(attempt))
                        attempt += 1
                        continue
                    return {"status": "failed", "error_message": str(e)}
                except Exception as e:
                    try:
                        self._rate_limit_worker.release.remote()
                    except Exception:
                        pass
                    return {"status": "failed", "error_message": str(e)}

            # Poll status at a fixed 1s interval.
            task_id = task_data.get("task_id", "")
            last_status = None
            while time.time() - start_ts < client_timeout:
                try:
                    s = self._client.get(f"{self.server_url}/status/{task_id}")
                    if s.status_code == 200:
                        data = s.json()
                        status = data.get("status", "unknown")
                        if status != last_status:
                            last_status = status
                            try:
                                print(f"[HybridWorker] STATUS task_id={task_id} -> {status}")
                            except Exception:
                                pass
                        if status in ("completed", "failed", "timeout", "cancelled"):
                            if status == "completed":
                                r = self._client.get(f"{self.server_url}/results/{task_id}")
                                if r.status_code == 200:
                                    result = r.json()
                                    result["status"] = status
                                    result["elapsed"] = time.time() - start_ts
                                    return result
                                return {"status": status, "error_message": f"Failed to fetch results: HTTP {r.status_code}", "elapsed": time.time() - start_ts}
                            return {"status": status, "error_message": data.get("error_message", f"Task {status}"), "elapsed": time.time() - start_ts}
                except Exception:
                    pass
                time.sleep(1.0)

            return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)", "elapsed": time.time() - start_ts}
        finally:
            # No need to release here (already released during submission).
            pass


class KernelRewardClient:
    def __init__(self, *, reward_config: Any) -> None:
        # Allow passing a wrapper config object.
        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model

        # Read required fields from reward_config.
        self.server_url = str(reward_config.server_url)
        self.timeout = float(reward_config.timeout)
        # task_timeout_in_client: client-side timeout including queue wait (should >= task_timeout)
        self.task_timeout_in_client = int(getattr(reward_config, 'task_timeout_in_client', self.timeout))
        self.max_retries = reward_config.max_retries
        # ===== 2026-08-28 A2: requeue on eval-result-wait timeout =====
        # 270 eval freeze incident: worker restart dropped in-flight tasks → redis task
        # stuck in "processing" forever. The engine polled until task_timeout_in_client
        # expired, then only returned "timeout"; the task was silently dropped, never
        # requeued.
        # Fix: when the result-wait times out, re-submit the same task with a **new
        # task_id** (the old id is stuck on the server side), bounded by
        # result_requeue_max. Only after exhausting requeues do we return a timeout
        # failure.
        # Enabled by default (3 attempts); override via result_requeue_enabled=False /
        # result_requeue_max=N.
        self.result_requeue_enabled = bool(getattr(reward_config, "result_requeue_enabled", True))
        self.result_requeue_max = int(getattr(reward_config, "result_requeue_max", 3))
        if self.result_requeue_max < 0:
            self.result_requeue_max = 0
        self.rate_limit = int(reward_config.rate_limit)
        if self.rate_limit <= 0:
            self.rate_limit = 1
        # Use max_concurrent as worker concurrency.
        self.num_workers = int(reward_config.max_concurrent)
        self.task_counter = 0
        self.acquire_timeout = int(reward_config.acquire_timeout)

        # Reward policy (aligned with KernelRewardClient); use defaults if not set.
        self.reward_config = reward_config

        # Ray worker (persistent httpx.Client + global token bucket).
        self._worker = _HybridHttpWorker.options(max_concurrency=self.num_workers).remote(
            self.server_url, self.rate_limit, int(self.timeout), self.acquire_timeout
        )
        # Keep a separate token bucket handle for heartbeat water-level checks.
        self._rate_limit_worker = TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(self.rate_limit)

        # Reward function weights and parameters.
        self.reward_func_name = reward_config.reward_func_name
        self.init_correct_weight = float(reward_config.init_correct_weight)
        self.init_performance_weight = float(reward_config.init_performance_weight)
        self.speedup_eps = float(reward_config.speedup_eps)
        self.penalty_score = float(reward_config.reward_policy.penalties.penalty_score)
        self.speedup_reward_upper_bound = float(reward_config.speedup_reward_upper_bound)
        self.speedup_reward_lower_bound = float(reward_config.speedup_reward_lower_bound)
        # 2026-08-26 performance reward A/B (Step1): correct-gate × speedup
        #   use_correct_gate=True: reward = min(speedup, cap) if correct else -0.5
        #   (CUDA-L1 community rationale: linear weighting makes the model sacrifice
        #   correctness for speed; default False preserves existing behavior)
        self.use_correct_gate = bool(getattr(reward_config, "use_correct_gate", False))
        # 2026-08-26 performance reward Step2: bucketed speedup
        # (distinguishes speedup tiers per user proposal)
        #   use_bucketed_speedup=True: when correct, apply per-tier interpolation
        #   (1.0->0.2 ... 3.0->1.0)
        self.use_bucketed_speedup = bool(getattr(reward_config, "use_bucketed_speedup", False))
        # ===== 2026-08-28 L6 over-replacement penalty (perf package, batch B) =====
        # Confirmed #1 perf killer: 40% of samples <1.0x — the model replaces the
        # reference's optimized-library ops (cuDNN/cuBLAS, ...) with naive serial
        # Triton kernels, making things slower with each edit. This flag penalizes
        # the reward for "over-replacement".
        # NOTE: this is the 4th "previously no-op" flag bit in reward_client.py — it
        # MUST be actually consumed by calculate_reward_speedup (see
        # _over_replacement_penalty); smoke test: docs/L6_OVERREPLACE_PATCH.md.
        self.penalize_over_replacement = bool(getattr(reward_config, "penalize_over_replacement", False))
        # Over-replacement scoring hyperparameters (all configurable, no hardcoding;
        # smoke test verifies each one takes effect)
        self.over_replacement_penalty = float(getattr(reward_config, "over_replacement_penalty", -0.3))
        # speedup below this threshold counts as "regressed" (slower than reference);
        # default 1.0 means anything <1.0x counts as regression
        self.over_replacement_speedup_threshold = float(
            getattr(reward_config, "over_replacement_speedup_threshold", 1.0)
        )
        # how many reference ops must be replaced at minimum to trigger (protects
        # against single-op jitter)
        self.over_replacement_min_custom_kernels = int(
            getattr(reward_config, "over_replacement_min_custom_kernels", 1)
        )
        # "reference is an optimized-library call" detection: when reference_backend ==
        # pytorch, the reference op runs through cuDNN/cuBLAS underneath
        # (runtime config reference_backend="pytorch", see kernel_trainer.yaml:18).
        # Use a set to support multiple optimized backends; an empty set means
        # "don't check backend, only speedup regression triggers".
        _opt = getattr(reward_config, "over_replacement_opt_backends", None)
        self.over_replacement_opt_backends = set(_opt) if _opt else {"pytorch", "native", "torch"}

    def _get_reward_func(self):
        """Select reward function based on config; default to calculate_reward_like_kernel."""
        try:
            func = getattr(self, str(self.reward_func_name), None)
            if callable(func):
                return func
        except Exception:
            pass
        try:
            print(f"[HybridClient] invalid reward_func_name={self.reward_func_name}, fallback to calculate_reward_like_kernel")
        except Exception:
            pass
        return self.calculate_reward_like_kernel

    def _next_task_id(self, prefix: str) -> str:
        try:
            self.task_counter += 1
        except Exception:
            # Fallback: still guarantee uniqueness.
            self.task_counter = int(time.time() * 1000) % 1000000
        return f"{prefix}_{self.task_counter:06d}_{uuid4().hex[:8]}"

    def _cheap_syntax_filter(self, kernel_code: str) -> Tuple[bool, str]:
        """Cheap pre-filter (no eval cost): parseable syntax + triton kernel markers + Model class."""
        import ast
        k = kernel_code or ""
        if not k.strip():
            return False, "empty kernel"
        # Normalize common Unicode characters that LLMs emit (harmless in
        # comments/strings, but ast.parse rejects them in code positions).
        # 2026-08-21 expansion: curly quotes + full-width punctuation + dashes +
        # arrows (U+2014 / U+2192 etc. garbage sources, see RL_ANALYSIS §1.4)
        k = k.replace("’", "'").replace("‘", "'")
        k = k.replace("“", '"').replace("”", '"')
        k = k.replace("—", "-").replace("–", "-").replace("―", "-").replace("—", "-")  # dashes
        k = k.replace("→", "->").replace("←", "<-").replace("⇒", "=>").replace("→", "->")  # arrows
        k = k.replace("、", ",").replace("，", ",").replace("：", ":").replace("；", ";")  # full-width punctuation
        k = k.replace("（", "(").replace("）", ")").replace("［", "[").replace("］", "]")
        k = k.replace("…", "...")
        try:
            ast.parse(k)
        except SyntaxError as e:
            return False, f"syntax-error({e.msg})"
        if "import triton" not in k and "@triton" not in k:
            return False, "no-triton"
        if "class ModelNew" not in k and "class Model" not in k:
            return False, "no-model-class"
        return True, ""

    def _preflight_validate(self, reference_code: str, kernel_code: str, entry_point: str) -> Tuple[bool, str]:
        """Minimal preflight: verify entry point / syntax exists to avoid wasteful (expensive) requests.
        Efficiency optimization: cheap syntax filter (ast.parse) intercepts garbage,
        avoiding sending it to the eval server for full compilation (saves ~97% of
        useless compilations).
        """
        try:
            # Cheap syntax / key-element filter: don't send garbage kernels to the
            # eval server.
            syn_ok, syn_err = self._cheap_syntax_filter(kernel_code)
            if not syn_ok:
                return False, f"cheap-filter: {syn_err}"
            ref_required = f"class {entry_point}"
            ker_required = f"class {entry_point}New"
            ref_ok = ref_required in (reference_code or "")
            # Relaxed: the model often names its implementation `class Model` (copying
            # the reference name) instead of `class ModelNew`. The server-side
            # _resolve_entry_point already tolerates this and unloads the correct
            # entry_point; only this preflight was over-strict and was killing valid
            # kernels (reward=0). Accept either class name; let the server handle it.
            ker_ok = (
                ker_required in (kernel_code or "")
                or ref_required in (kernel_code or "")
            )
            if ref_ok and ker_ok:
                return True, ""
            missing = []
            if not ref_ok:
                missing.append(ref_required)
            if not ker_ok:
                missing.append(f"{ker_required} or {ref_required}")
            return False, ", ".join(missing)
        except Exception as e:
            logger.debug(f"preflight skipped due to error: {e}")
            return True, ""

    def calculate_reward_like_kernel(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
        # "Server returned a decoy kernel; force -1 and carry the marker."
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": -1.0,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)

        penalties = self.reward_config.reward_policy.penalties
        compilation_fail_penalty = float(penalties.get("compilation_fail", -0.5))
        correctness_fail_penalty = float(penalties.get("correctness_fail", -0.3))
        perf_degrade_penalty = float(penalties.get("perf_degrade", -0.1))

        if not compiled:
            reward = compilation_fail_penalty
        elif not correctness:
            reward = correctness_fail_penalty
        else:
            if speedup >= 3.0:
                reward = 1.0
            elif speedup >= 2.0:
                reward = 0.8
            elif speedup >= 1.5:
                reward = 0.6
            elif speedup >= 1.2:
                reward = 0.4
            elif speedup >= 1.0:
                reward = 0.2
            else:
                reward = perf_degrade_penalty
        return {
            "reward": reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": reward,
        }

    def compute_coverage_reward(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Some server versions put coverage fields in metadata, possibly with plural names; normalize here.
        metadata = result.get("metadata") or {}

        def _get_field(*keys: str, default: int = 0) -> int:
            for k in keys:
                if k in metadata:
                    return metadata.get(k) or default
                if k in result:
                    return result.get(k) or default
            return default

        num_custom_kernel = _get_field("num_custom_kernels", "num_custom_kernel")
        num_total_kernels = _get_field("num_total_kernels", "num_total_kernel", "num_total_kernels")
        custom_kernel_cuda_time_in_profiling_us = _get_field("custom_kernel_cuda_time_in_profiling_us")
        total_kernel_run_time_in_profiling_us = _get_field("total_kernel_run_time_in_profiling_us")

        # Only log keys once when all fields are missing to aid debugging.
        if (
            not num_custom_kernel
            and not num_total_kernels
            and "num_custom_kernel" not in result
            and "num_total_kernels" not in result
            and "num_custom_kernels" not in metadata
            and "num_total_kernels" not in metadata
        ):
            try:
                print(f"[HybridClient] coverage fields missing, fallback to 0: keys={list(result.keys())}")
            except Exception:
                pass

        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels


        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us

        if self.reward_config.coverage_reward.reward_type == "time_coverage":
            coverage = time_coverage
        elif self.reward_config.coverage_reward.reward_type == "number_coverage":
            coverage = num_coverage
        else:
            raise ValueError(f"Invalid reward type: {self.reward_config.coverage_reward.reward_type}")

        return {
            "coverage": coverage,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def calculate_reward_weighted(self, result: Dict[str, Any]) -> Dict[str, Any]:

        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")

            return_result = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }

            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]

            return return_result
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        # In fact, profiling is always None here since it actually lives in metadata
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        is_speedup_positive = speedup >= (1 + self.speedup_eps) # ignore too small speedup

        reward = self.init_correct_weight * correctness + self.init_performance_weight * is_speedup_positive

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0
        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]
            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")
            if self.reward_config.coverage_reward.enable:
                final_reward += self.reward_config.coverage_reward.weight * coverage

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def _bucketed_speedup_reward(self, speedup: float) -> float:
        """Bucketed + intra-bucket-interpolated speedup reward (2026-08-26, user proposal;
        2026-08-27 B1 made the curve configurable to a convex shape):
        Default 1.0x->0.2 / 1.2x->0.4 / 1.5x->0.6 / 2.0x->0.8 / 3.0x->1.0;
        B1 convex curve (strongest incentive to push toward 3x):
        1.0->0.02 / 1.2->0.08 / 1.5->0.20 / 2->0.35 / 2.5->0.65 / 3->1.0
        Linear interpolation within each bucket. Override via reward_config.bucket_bands.
        """
        spd = min(speedup, 3.0)
        bands = getattr(getattr(self, "reward_config", None), "bucket_bands", None)
        if not bands:
            bands = [(1.0, 0.2), (1.2, 0.4), (1.5, 0.6), (2.0, 0.8), (3.0, 1.0)]
        if spd < 1.0:
            return 0.0
        for i, (lo, base) in enumerate(bands):
            hi, hi_base = bands[i + 1] if i + 1 < len(bands) else (3.0, 1.0)
            if spd < hi:
                frac = (spd - lo) / (hi - lo) if hi > lo else 0.0
                return base + frac * (hi_base - base)
        return 1.0

    def _over_replacement_penalty(self, result: Dict[str, Any], speedup: float) -> float:
        """Over-replacement penalty (2026-08-28 L6): penalizes the model for replacing
        optimized-library ops in the reference with naive slow implementations.

        Returns a penalty value <= 0 (0 = not triggered). When flag
        `penalize_over_replacement` is off, always returns 0.
        Composable heuristics (each reads real signals from result/metadata, individually
        toggleable):
          H1 replace + regress (main criterion): the model wrote a custom kernel
             (num_custom_kernel>0) AND overall speedup regressed vs reference (<1.0)
             AND the reference was an optimized-library call → deduct
             over_replacement_penalty.
             Physical meaning: the reference was an optimized-library op
             (cuDNN/cuBLAS, fast), and the model replaced it with a naive serial
             Triton kernel, making overall performance worse = over-replacement.
          H2 large-coverage replacement: num_custom_kernel/num_total_kernels is high
             (replaced most ops) AND still overall regression → larger deduction
             (default × 1.5 factor, configurable via over_replacement_coverage_factor).
             Physical meaning: replacing nearly every optimized op in the graph with a
             slow kernel — the wider the replacement, the more harmful.
          H3 time-share mismatch: the custom kernel dominates profiling time
             (high time_coverage) yet the overall result regressed → small extra
             deduction (default +0.1, configurable).

        "Reference is an optimized-library call" detection: reference_backend ∈
        over_replacement_opt_backends (default {pytorch, native, torch}, because pytorch
        reference ops go through cuDNN/cuBLAS underneath). When reference backend
        field is missing, treat as "indeterminate" and only trigger H1 by regression
        + replacement.

        Anti no-op: this function is the sole real consumer of the flag; smoke test
        in docs/L6_OVERREPLACE_PATCH.md.
        """
        if not self.penalize_over_replacement:
            return 0.0

        def _num(field: str, default: int = 0) -> int:
            for scope in (result, result.get("metadata") or {}):
                if not isinstance(scope, dict):
                    continue
                for k in (field, field + "s"):  # accept singular or plural
                    v = scope.get(k)
                    if v:
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            return 0
            return default

        num_custom_kernel = _num("num_custom_kernel")
        num_total_kernels = _num("num_total_kernels")
        custom_us = _num("custom_kernel_cuda_time_in_profiling_us")
        total_us = _num("total_kernel_run_time_in_profiling_us")

        # No replacement signal → don't trigger
        if num_custom_kernel <= 0:
            return 0.0
        # Regression criterion: overall speedup below threshold (default <1.0 means slower than reference)
        degraded = speedup < self.over_replacement_speedup_threshold
        if not degraded:
            return 0.0

        # Whether the reference was an optimized-library call (backend detection;
        # when backend info is absent, only regression + replacement trigger)
        backend = str(result.get("reference_backend") or result.get("metadata", {}).get("reference_backend") or "")
        lib_backed = (not backend) or (backend.lower() in {b.lower() for b in self.over_replacement_opt_backends})

        # H1 main criterion: replacement + regression + reference library call
        if not lib_backed:
            return 0.0
        penalty = self.over_replacement_penalty

        # H2: large-coverage replacement → amplify
        if num_total_kernels > 0:
            coverage = num_custom_kernel / num_total_kernels
            cov_factor = float(getattr(self.reward_config, "over_replacement_coverage_factor", 1.5))
            if coverage >= 0.5:
                penalty *= cov_factor
                try:
                    print(f"[HybridClient] over-replacement H2: coverage={coverage:.2f} penalty={penalty:.3f}")
                except Exception:
                    pass

        # H3: slow impl dominates runtime → add extra
        if total_us > 0 and custom_us / total_us >= 0.5:
            extra = float(getattr(self.reward_config, "over_replacement_time_extra", 0.1))
            penalty -= extra
            try:
                print(f"[HybridClient] over-replacement H3: time_cov={custom_us/total_us:.2f} extra={extra}")
            except Exception:
                pass

        try:
            print(f"[HybridClient] over-replacement penalty: speedup={speedup:.3f} "
                  f"custom={num_custom_kernel}/{num_total_kernels} backend={backend!r} penalty={penalty:.3f}")
        except Exception:
            pass
        return penalty

    def calculate_reward_speedup(self, result: Dict[str, Any]) -> Dict[str, Any]:
        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")

            # === RL v4 syntax/compile penalty (2026-08-21) ===
            # Goal: let the gradient distinguish "syntax garbage" (-1) vs "compile
            # failure" (-0.5) vs "wrong but correct-shape" (0). Addresses the main
            # RL drift cause (previously everything was uniformly penalty_score=0,
            # so garbage and "wrong but correct-shape" scored the same).
            # Criterion: cheap-filter / compile-error markers in error_message.
            reward_fail = penalty_score  # default 0 (wrong but correct-shape / runtime failure)
            em_lower = str(error_message).lower()
            if "syntax-error" in em_lower or "client validation failed" in em_lower \
               or "syntaxerror" in em_lower or "unterminated string" in em_lower \
               or "invalid character" in em_lower or "invalid decimal" in em_lower \
               or "unmatched" in em_lower:
                reward_fail = -1.0   # syntax garbage (cheap-filter / ast failed)
            elif "kernel evaluation failed" in em_lower or "compilation" in em_lower \
                 or "compile" in em_lower:
                reward_fail = -1.0   # compile failure = hard barrier (2026-08-23: -0.5 → -1.0, same level as syntax; prevents farming compile failures)
            # Otherwise (runtime error / timeout / task failure) → penalty_score (0)

            return_result = {
                "reward": reward_fail,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }

            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]

            return return_result
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        # In fact, profiling is always None here since it is actually inside metadata
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        # is_speedup_positive = speedup >= (1 + self.speedup_eps) # ignore too small speedup

        reward_speedup = speedup
        if speedup > self.speedup_reward_upper_bound:
            reward_speedup = self.speedup_reward_upper_bound

        if reward_speedup < self.speedup_reward_lower_bound:
            reward_speedup = 0.0

        # 2026-08-26 A/B (Step1, correct-gate): reward = min(speedup, cap) if correct else -0.5
        # Default False preserves linear weighting; with the gate on, correct rewards
        # scale with speedup, incorrect gets a hard -0.5.
        # 2026-08-27 B1: + correct_bonus (always added when correct) + wrong_floor
        # (configurable, default -0.5, B1 uses -0.2).
        _correct_bonus = float(getattr(self.reward_config, "correct_bonus", 0.0) or 0.0)
        _wrong_floor = float(getattr(self.reward_config, "wrong_floor", -0.5) or -0.5)
        if self.use_correct_gate:
            if self.use_bucketed_speedup and correctness:
                reward = self._bucketed_speedup_reward(reward_speedup) + _correct_bonus
            else:
                reward = (reward_speedup if correctness else _wrong_floor) + (_correct_bonus if correctness else 0.0)
            if compiled and not correctness:
                print(f"[HybridClient] gate penalty: compiled-but-wrong reward {reward}")
        else:
            reward = self.init_correct_weight * correctness + self.init_performance_weight * reward_speedup

        # 2026-08-23 (reward fix): compiled but execution-wrong = explicit negative (-0.5),
        # no longer scored the same as 0.
        # Goal: keep the main gradient signal on "going from wrong to correct", not on
        # "going from syntax-fail to compile-pass" (compile is a hard barrier, correctness
        # is the main signal).
        if compiled and not correctness and not self.use_correct_gate:
            reward = -0.5
            print(f"[HybridClient] compiled-but-wrong penalty: reward {reward}")

        # P2 fix (2026-08-21 final review): num_custom_kernel read from result/metadata,
        # not hardcoded to 0 (otherwise the fake-compile penalty always fires, also
        # penalizing "wrote a kernel but it's wrong", which is semantically wrong)
        def _nck() -> int:
            for scope in (result, result.get("metadata") or {}):
                for k in ("num_custom_kernels", "num_custom_kernel"):
                    v = scope.get(k) if isinstance(scope, dict) else None
                    if v:
                        try:
                            return int(v)
                        except (TypeError, ValueError):
                            return 0
            return 0

        num_custom_kernel = _nck()
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0

        # Fake-compile penalty (2026-08-21 review §3.1 / §4.2): compiled but did not
        # actually execute any custom kernel = soft decoy (only visible when profiling
        # is on; the model writes kernels that "compile but are empty" — penalize, not
        # score 0).
        if compiled and num_custom_kernel == 0 and not correctness:
            reward -= 0.3
            print(f"[HybridClient] soft-decoy penalty: compiled=True num_custom_kernel=0 (reward {reward})")

        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]

            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")

            if self.reward_config.coverage_reward.enable:
                final_reward += self.reward_config.coverage_reward.weight * coverage

        # ===== 2026-08-28 L6 over-replacement penalty (perf package, batch B, real consumer) =====
        # Over-replacement = #1 perf killer (40% of samples <1.0x). For samples that are
        # correct but slow because the model replaced optimized-library ops with naive
        # serial Triton kernels, deduct on top of the base reward. With the flag off
        # this is always 0 (preserves existing behavior exactly).
        if self.penalize_over_replacement and correctness:
            _over_penalty = self._over_replacement_penalty(result, speedup)
            if _over_penalty < 0:
                final_reward += _over_penalty
                print(f"[HybridClient] over-replacement applied: reward {final_reward:.3f} "
                      f"(delta {_over_penalty:.3f}) speedup={speedup:.3f}")

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def calculate_reward_discrete(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """CUDA-Agent-style discrete reward {-1, 1, 2, 3} (for A/B, doesn't break existing).

        Tiers (per the paper, failure is always -1):
          - syntax / compile failure (status != completed)         -> -1
          - decoy (reward hacking)                                 -> -1
          - compiled but correctness=False                         ->  1
          - correctness=True                                       ->  2
          - correctness=True AND speedup > 1.2 (vs reference)     ->  3

        "Speedup vs reference": the paper's caliber normalizes against reference
        runtime (server-returned speedup is already reference-normalized).
        speedup<=1.2 counts as "correct but not significantly faster", stays at 2.
        """
        # failure (syntax / compile / runtime / timeout) always -1
        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
        # decoy (reward hacking) -> -1
        if result.get("decoy_kernel", False):
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": -1.0,
            }

        correctness = bool(result.get("correctness", False))
        speedup = result.get("speedup", 0.0)
        if speedup is None:
            speedup = 0.0
        compiled = bool(result.get("compiled", False))

        if not compiled:
            reward = -1.0
        elif not correctness:
            reward = 1.0
        elif speedup > 1.2:
            reward = 3.0
        else:
            reward = 2.0

        return {
            "reward": float(reward),
            "speedup": float(speedup),
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": float(reward),
        }

    def _merge_reward_result(self, raw_result: Dict[str, Any], reward_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge raw KernelServer response payload with derived reward summary."""
        merged: Dict[str, Any] = {}
        if raw_result:
            merged.update(raw_result)
        if reward_summary:
            merged.update(reward_summary)
        return merged

    async def compute_batch_rewards(
        self,
        tasks: List[Dict[str, Any]],
        *,
        use_reference_cache: Optional[bool] = None,
        is_valid: Optional[bool] = None,
        task_timeout: Optional[int] = None,
        task_timeout_in_client: Optional[int] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        penalty_score = self.penalty_score
        if not tasks:
            return []
        # print(f"[DEBUG] Task timeout: {task_timeout or self.timeout}")
        effective_timeout = int(task_timeout or self.timeout)
        effective_timeout_in_client = int(task_timeout_in_client or self.task_timeout_in_client)

        # Validate timeout invariant: client timeout should be >= server timeout
        if effective_timeout_in_client < effective_timeout:
            print(f"[WARNING] task_timeout_in_client ({effective_timeout_in_client}s) < task_timeout ({effective_timeout}s)")
            print(f"[WARNING] Adjusting task_timeout_in_client to match task_timeout to respect timeout invariant")
            effective_timeout_in_client = effective_timeout
        obj_refs: List[ray.ObjectRef] = []
        index_map: List[int] = []  # worker submission order -> original index
        prefilled: Dict[int, Dict[str, Any]] = {}
        submitted: int = 0
        skipped: int = 0
        # Map obj_ref index -> task metadata for heartbeat tracking of pending tasks.
        idx_to_task_info: Dict[int, Dict[str, Any]] = {}
        
        for idx, task in enumerate(tasks):
            kcode = task.get("kernel_code", "")
            ep = task.get("entry_point", "Model")
            ok, missing = self._preflight_validate(task.get("reference_code", ""), kcode, ep)
            if not ok:
                try:
                    print(f"[HybridClient] preflight failed(idx={idx}): missing {missing} entry_point={ep}")
                except Exception:
                    pass
                prefilled[idx] = {
                    "reward": penalty_score,
                    "speedup": 0.0,
                    "success": False,
                    "correctness": False,
                    "compiled": False,
                    "error": f"Client validation failed: missing {missing}",
                }
                continue

            # Per-task timeout handling: fall back to batch default when explicitly None.
            # Server-side task execution timeout.
            per_task_timeout_raw = task.get("task_timeout", None)
            per_task_timeout = (
                effective_timeout if per_task_timeout_raw is None else int(per_task_timeout_raw)
            )

            # Client-side task execution timeout.
            # Client-side timeout should be >= server-side timeout.
            # Because queuing is involved, client timeout should be >= server timeout.
            per_task_timeout_in_client_raw = task.get("task_timeout_in_client", None)
            per_task_timeout_in_client = (
                effective_timeout_in_client if per_task_timeout_in_client_raw is None else int(per_task_timeout_in_client_raw)
            )

            # Validate per-task timeout invariant
            if per_task_timeout_in_client < per_task_timeout:
                per_task_timeout_in_client = per_task_timeout
            # print(f"[DEBUG] Per task timeout: {per_task_timeout}")

            # Randomly log one task (~5%).
            try:
                if random.random() < 0.05:
                    def _clip2(s: Optional[str], n: int = 600) -> str:
                        try:
                            return (s or "")[:n]
                        except Exception:
                            return str(s)[:n]
                    print(f"[HybridClient] DEBUG(entry_point={ep})\n[ref]\n{task.get('reference_code','')}\n[kernel]\n{kcode}")
            except Exception:
                pass

            payload = {
                "task_id": task.get("task_id") or self._next_task_id("parallel_task"),
                "reference_code": task.get("reference_code", ""),
                "kernel_code": kcode,
                "backend": "triton",
                "num_correct_trials": task.get("num_correct_trials", 5),
                "num_perf_trials": task.get("num_perf_trials", 100),
                "timeout": per_task_timeout,
                "priority": "normal",
                "entry_point": ep,
                "is_valid": task.get("is_valid", is_valid),
                "verbose_errors": task.get("verbose_errors", True),
                "enable_profiling": task.get("enable_profiling", True),
                "detect_decoy_kernel": task.get("detect_decoy_kernel", True),
                "reference_backend": task.get("reference_backend", None),
            }

            # enforce detect decoy kernel if validate
            if payload["is_valid"]:
                print(f"Enforce detect decoy kernel if validate: {payload['detect_decoy_kernel']}")
                payload["detect_decoy_kernel"] = True

            ucache = task.get("use_reference_cache", use_reference_cache)
            if ucache:
                payload["use_reference_cache"] = True
                if task.get("uuid"):
                    payload["uuid"] = task["uuid"]
            
            # Record task metadata for heartbeat tracking.
            obj_ref_idx = len(obj_refs)  # Index of the obj_ref about to be appended to the list.
            idx_to_task_info[obj_ref_idx] = {
                "task_id": payload["task_id"],
                "entry_point": ep,
                "uuid": payload.get("uuid", ""),
                "orig_idx": idx,
            }
            
            obj_refs.append(
                self._worker.submit_and_poll.remote(
                    payload,
                    per_task_timeout_in_client,
                    self.max_retries,
                    self.result_requeue_max if self.result_requeue_enabled else 0,
                )
            )
            index_map.append(idx)
            submitted += 1

        try:
            skipped = len(prefilled)
            print(f"[HybridClient] batch submitted={submitted} skipped={skipped}")
        except Exception:
            pass

        # Heartbeat: report progress and token-bucket level every 60s.
        pending = set(range(len(obj_refs)))
        results: List[Tuple[int, Dict[str, Any]]] = []
        start_ts = time.time()
        # Track remaining refs to avoid reprocessing completed tasks and looping.
        remaining_refs: List[ray.ObjectRef] = list(obj_refs)
        ref_to_idx = {ref: i for i, ref in enumerate(obj_refs)}
        
        while remaining_refs:
            done, remaining_refs = await asyncio.to_thread(ray.wait, remaining_refs, num_returns=1, timeout=60)
            if done:
                ref = done[0]
                idx = ref_to_idx.get(ref, None)
                try:
                    res = await asyncio.to_thread(ray.get, ref)
                except Exception as e:
                    res = {"status": "failed", "error_message": str(e)}
                if idx is not None and idx in pending:
                    results.append((idx, res))
                    pending.discard(idx)
            else:
                elapsed = time.time() - start_ts
                in_use = -1
                try:
                    in_use = await asyncio.to_thread(ray.get, self._rate_limit_worker.get_current_count.remote())
                except Exception:
                    pass
                
                # Collect detailed info for pending tasks for logging.
                pending_tasks_info = []
                for p_idx in sorted(list(pending))[:10]:  # Log at most the first 10 pending tasks.
                    if p_idx in idx_to_task_info:
                        info = idx_to_task_info[p_idx]
                        pending_tasks_info.append(f"task_id={info['task_id']} entry={info['entry_point']} uuid={info['uuid'][:8] if info['uuid'] else 'N/A'}")
                
                pending_summary = "; ".join(pending_tasks_info) if pending_tasks_info else "N/A"
                if len(pending) > 10:
                    pending_summary += f" ... (+{len(pending)-10} more)"
                
                print(f"[BatchHeartbeat] hybrid: completed={len(results)}/{len(obj_refs)}, pending={len(pending)}, elapsed={elapsed:.1f}s tokens_in_use={in_use}/{self.rate_limit}")
                print(f"[BatchHeartbeat] pending_tasks: {pending_summary}")
        # Merge back to original order.
        merged: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        # Fill prefilled first.
        for i, v in prefilled.items():
            merged[i] = v
        # Then fill worker results.
        for idx_in_obj, data in results:
            orig_idx = index_map[idx_in_obj]
        # === Flatten profiling fields from metadata to top-level (2026-08-21 coverage bug fix) ===
        # The eval server puts num_custom_kernels (plural) / custom_kernel_cuda_time etc.
        # under metadata, while kernel_async + the reward function reads from top-level
        # (singular num_custom_kernel) — without this flattening, coverage is always 0.
            _md = data.get("metadata") if isinstance(data, dict) else None
            if isinstance(_md, dict):
                if "num_custom_kernels" in _md and "num_custom_kernel" not in data:
                    data["num_custom_kernel"] = _md["num_custom_kernels"]
                if "num_total_kernels" in _md and "num_total_kernels" not in data:
                    data["num_total_kernels"] = _md["num_total_kernels"]
                for _k in ("custom_kernel_cuda_time_in_profiling_us",
                           "total_kernel_run_time_in_profiling_us", "time_coverage"):
                    if _k in _md and _k not in data:
                        data[_k] = _md[_k]
            reward_func = self._get_reward_func()
            reward_summary = reward_func(data)
            merged[orig_idx] = self._merge_reward_result(data, reward_summary)
        # Fallback for any missing entries.
        for i, v in enumerate(merged):
            if v is None:
                merged[i] = {
                    "reward": penalty_score,
                    "speedup": 0.0,
                    "success": False,
                    "correctness": False,
                    "compiled": False,
                    "error": "Unknown error",
                    "num_custom_kernel": 0,
                    "num_total_kernels": 0,
                    "custom_kernel_cuda_time_in_profiling_us": 0,
                    "total_kernel_run_time_in_profiling_us": 0,
                }
        return merged  # type: ignore[return-value]
