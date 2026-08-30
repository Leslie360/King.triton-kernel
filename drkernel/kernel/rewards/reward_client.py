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
        # ===== 2026-08-28 A2: 结果等待超时重投 =====
        # 270 eval 卡死事故: worker 重启丢在途任务 → server 侧任务永远卡在 processing,
        # 轮询到 client_timeout 超时后旧逻辑只返回 timeout、任务静默丢失。
        # 修复: 超时(任务仍非终态)即用**新 task_id** 重投, 上限 requeue_max 次;
        # 只有耗尽重投次数才返回 timeout 失败, 不再无限等待也不静默丢弃。
        # ===== 2026-08-29 A2 盲区扩展: "卡顿后 failed" =====
        # step305 事故: 任务在 server 卡很久最终返回 failed(非 timeout), 旧 A2 只重投
        # status=timeout 不覆盖此类。现: failed 且耗时 > stall_threshold(疑似卡顿而非
        # 真失败) 也重投。真 failed 通常秒级(编译/运行错), 卡顿后 failed 耗时远超正常。
        # 误判重投只是多跑一次(requeue_max 上限), 不致命; 宁可重投不静默丢。
        requeue_max = int(requeue_max or 0)
        stall_threshold = max(client_timeout * 0.5, 120.0)  # failed 但耗时过半或>120s 疑似卡顿
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
        # ===== 2026-08-28 A2: eval 结果等待超时重投 =====
        # 270 eval 卡死事故: worker 重启丢在途任务 → redis 里任务永远卡在 processing,
        # 引擎轮询到 task_timeout_in_client 超时后只返回 timeout, 任务静默丢失、无重投。
        # 修复: 结果等待超时后, 用**新 task_id** 重新提交该任务 (旧 id 在 server 侧已卡死),
        # 由 result_requeue_max 限制重投次数; 只有耗尽重投次数才返回 timeout 失败。
        # 默认启用(3 次), 可用 result_requeue_enabled=False / result_requeue_max=N 覆盖。
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
        # 2026-08-26 性能 reward A/B (Step1): correct-gate × speedup
        #   use_correct_gate=True 时: reward = min(speedup,cap) if correct else -0.5
        #   (CUDA-L1 社区依据: 线性加权会让模型牺牲正确换速度; 默认 False 不改变现有行为)
        self.use_correct_gate = bool(getattr(reward_config, "use_correct_gate", False))
        # 2026-08-26 性能 reward Step2: 分档加速 (区分加速档次, 用户提议)
        #   use_bucketed_speedup=True 时: correct 按 speedup 分档+档内插值 (1.0->0.2 ... 3.0->1.0)
        self.use_bucketed_speedup = bool(getattr(reward_config, "use_bucketed_speedup", False))
        # ===== 2026-08-28 L6 over-replacement 惩罚 (批次 B 性能包) =====
        # 头号性能杀手实锤: 40% 样本 <1.0x —— 模型把参考里的优化库算子(cuDNN/cuBLAS 等)
        # 换成 naive 串行 Triton kernel → 越改越慢。本 flag 使 reward 对"过度替换"扣分。
        # 注意: 这是 reward_client.py 第 4 个"曾 no-op"的 flag 位 —— 必须被 calculate_reward_speedup
        # 真实消费 (见 _over_replacement_penalty), 冒烟见 docs/L6_OVERREPLACE_PATCH.md。
        self.penalize_over_replacement = bool(getattr(reward_config, "penalize_over_replacement", False))
        # 过度替换的判分超参 (全部可配置, 防硬编码; 冒烟里逐一验证生效)
        self.over_replacement_penalty = float(getattr(reward_config, "over_replacement_penalty", -0.3))
        # speedup 低于该阈值视为"退化"(相对参考变慢); 默认 1.0 即任何 <1.0x 都算退化
        self.over_replacement_speedup_threshold = float(
            getattr(reward_config, "over_replacement_speedup_threshold", 1.0)
        )
        # 至少替换了多少个参考算子才触发 (防单算子抖动误伤)
        self.over_replacement_min_custom_kernels = int(
            getattr(reward_config, "over_replacement_min_custom_kernels", 1)
        )
        # "参考是优化库调用"的判定: reference_backend == pytorch 时参考 op 底层走 cuDNN/cuBLAS 等
        # (运行配置 reference_backend="pytorch", 见 kernel_trainer.yaml:18)。
        # 用 set 支持多个优化 backend; 空集表示不检查 backend, 只按速度退化判。
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
        """便宜预筛 (无 eval 成本): 语法可解析 + 含 triton kernel 标记 + 含 Model 类。"""
        import ast
        k = kernel_code or ""
        if not k.strip():
            return False, "empty kernel"
        # 归一化 LLM 常见 Unicode 字符 (在注释/字符串里无害, 但 ast.parse 在代码位会报 invalid character)
        # 2026-08-21 扩展: 弯引号 + 全角标点 + 破折号 + 箭头 (U+2014/U+2192 等垃圾源, 见 RL_ANALYSIS §1.4)
        k = k.replace("’", "'").replace("‘", "'")
        k = k.replace("“", '"').replace("”", '"')
        k = k.replace("—", "-").replace("–", "-").replace("―", "-").replace("—", "-")  # 破折号
        k = k.replace("→", "->").replace("←", "<-").replace("⇒", "=>").replace("→", "->")  # 箭头
        k = k.replace("、", ",").replace("，", ",").replace("：", ":").replace("；", ";")  # 全角标点
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
        """Minimal preflight: verify entry point/语法 exists to avoid meaningless (昂贵) requests.
        效率优化: 便宜语法筛 (ast.parse) 拦截垃圾, 避免发 eval server 全额编译。
        """
        try:
            # 便宜语法/关键要素筛: 垃圾 kernel 不发 eval server (省 97% 无效编译)
            syn_ok, syn_err = self._cheap_syntax_filter(kernel_code)
            if not syn_ok:
                return False, f"cheap-filter: {syn_err}"
            ref_required = f"class {entry_point}"
            ker_required = f"class {entry_point}New"
            ref_ok = ref_required in (reference_code or "")
            # 放宽: 模型常把实现命名为 `class Model`(抄 reference 名)而非 `class ModelNew`。
            # server 端 _resolve_entry_point 本就容忍并卸载正确 entry_point, 只有这里 preflight 太严
            # 导致有效 kernel 被误杀(reward=0)。接受任一类名, 由 server 处理。
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
                missing.append(f"{ker_required} 或 {ref_required}")
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
        # Server returned a decoy kernel; force -1 and carry the marker.
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
        # In fact, profiling is always None here since it is actually inside metadata
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
        """分档+档内插值加速 reward (2026-08-26, 用户提议; 2026-08-27 B1 改可配置凸曲线):
        默认 1.0x->0.2 / 1.2x->0.4 / 1.5x->0.6 / 2.0x->0.8 / 3.0x->1.0;
        B1 凸曲线(冲3x激励最强): 1.0->0.02 / 1.2->0.08 / 1.5->0.20 / 2->0.35 / 2.5->0.65 / 3->1.0
        档内线性插值。可经 reward_config.bucket_bands 覆盖。"""
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
        """过度替换惩罚 (2026-08-28 L6): 模型把参考里的优化库算子替换成 naive 慢实现时扣分。

        返回 <= 0 的惩罚值 (0 = 不触发)。flag `penalize_over_replacement` 关闭时恒返回 0。
        可组合启发式 (每条都从 result/metadata 读真实信号, 可单独关):
          H1 替换+退化 (主判据): 模型写了 custom kernel (num_custom_kernel>0) 且
             overall speedup 相对参考退化 (<1.0) 且参考是优化库调用 → 扣 over_replacement_penalty。
             物理含义: 参考里本来是 cuDNN/cuBLAS 等优化库算子(快), 模型用 naive Triton
             串行核替换后整体反而变慢 = 过度替换。
          H2 大覆盖替换: num_custom_kernel/num_total_kernels 高 (替换了多数算子) 仍整体退化
             → 更大扣分 (默认乘 1.5 系数, 可配 over_replacement_coverage_factor)。
             物理含义: 把整张图几乎所有优化 op 都换成慢核, 替换面越广危害越大。
          H3 时间占比不匹配: custom kernel 占 profiling 时间主导 (time_coverage 高) 却整体退化
             → 说明慢实现是主耗时来源, 追加小扣分 (默认 +0.1 叠加, 可配)。

        "参考是优化库调用"判定: reference_backend ∈ over_replacement_opt_backends
        (默认 {pytorch,native,torch}, 因为 pytorch 参考 op 底层走 cuDNN/cuBLAS)。
        无参考 backend 字段时视为"无法判定", 仅靠速度退化+替换触发 H1 主判据。

        防 no-op: 本函数是 flag 唯一真实消费者, 冒烟见 docs/L6_OVERREPLACE_PATCH.md。
        """
        if not self.penalize_over_replacement:
            return 0.0

        def _num(field: str, default: int = 0) -> int:
            for scope in (result, result.get("metadata") or {}):
                if not isinstance(scope, dict):
                    continue
                for k in (field, field + "s"):  # 兼容单/复数
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

        # 没有任何替换信号 → 不触发
        if num_custom_kernel <= 0:
            return 0.0
        # 退化判据: overall speedup 低于阈值 (默认 <1.0 即比参考慢)
        degraded = speedup < self.over_replacement_speedup_threshold
        if not degraded:
            return 0.0

        # 参考是否优化库调用 (backend 判定; 无 backend 信息时仅靠退化+替换触发)
        backend = str(result.get("reference_backend") or result.get("metadata", {}).get("reference_backend") or "")
        lib_backed = (not backend) or (backend.lower() in {b.lower() for b in self.over_replacement_opt_backends})

        # H1 主判据: 替换 + 退化 + 参考库调用
        if not lib_backed:
            return 0.0
        penalty = self.over_replacement_penalty

        # H2: 大覆盖替换 → 放大
        if num_total_kernels > 0:
            coverage = num_custom_kernel / num_total_kernels
            cov_factor = float(getattr(self.reward_config, "over_replacement_coverage_factor", 1.5))
            if coverage >= 0.5:
                penalty *= cov_factor
                try:
                    print(f"[HybridClient] over-replacement H2: coverage={coverage:.2f} penalty={penalty:.3f}")
                except Exception:
                    pass

        # H3: 慢实现主耗时 → 追加
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

            # === RL v4 语法/编译惩罚 (2026-08-21) ===
            # 目的: 让梯度区分"语法垃圾"(-1) vs "编译失败"(-0.5) vs "错但对"(0),
            # 治 RL 漂移主因 (之前全统一 penalty_score=0, 垃圾和错但对同分)。
            # 判据: 基于 error_message 的 cheap-filter / 编译错误标记。
            reward_fail = penalty_score  # 默认 0 (错但对/运行时失败)
            em_lower = str(error_message).lower()
            if "syntax-error" in em_lower or "client validation failed" in em_lower \
               or "syntaxerror" in em_lower or "unterminated string" in em_lower \
               or "invalid character" in em_lower or "invalid decimal" in em_lower \
               or "unmatched" in em_lower:
                reward_fail = -1.0   # 语法垃圾 (cheap-filter / ast 不过)
            elif "kernel evaluation failed" in em_lower or "compilation" in em_lower \
                 or "compile" in em_lower:
                reward_fail = -1.0   # 编译失败 = 硬门槛 (2026-08-23: -0.5→-1.0, 与语法同级, 防刷编译)
            # 其余 (运行时错误/超时/任务失败) → penalty_score (0)

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

        # 2026-08-26 A/B (Step1, correct-gate): reward = min(speedup,cap) if correct else -0.5
        # 默认 False 保留线性加权; 开 gate 后正确时按 speedup 奖, 不正确时硬 -0.5
        # 2026-08-27 B1: + correct_bonus(正确恒加) + wrong_floor(可配置, 默认 -0.5, B1 用 -0.2)
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

        # 2026-08-23 (reward 修复): 编译成功但执行错误 = 显式负分 (-0.5), 不再同分 0
        # 目的: 让模型主梯度来自"从错误变正确", 而非"从语法错变编译对" (编译=硬门槛, 正确=主信号)
        if compiled and not correctness and not self.use_correct_gate:
            reward = -0.5
            print(f"[HybridClient] compiled-but-wrong penalty: reward {reward}")

        # P2 fix (2026-08-21 终审): num_custom_kernel 从 result/metadata 真实读取, 不再硬编码 0
        # (否则伪编译惩罚恒真, 对"有 kernel 但错"的样本也罚, 语义不准确)
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

        # 伪编译惩罚 (2026-08-21 review §3.1/§4.2): compiled 但没真正执行 custom kernel = soft decoy
        # (profiling 开后才真实可见; 模型在写"能编译但空实现"的 kernel, 应惩罚而非同分 0)
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

        # ===== 2026-08-28 L6 over-replacement 惩罚 (批次 B 性能包, flag 真实消费点) =====
        # 过度替换 = 头号性能杀手 (40% 样本 <1.0x)。正确但慢的样本若把优化库算子换成
        # naive Triton 串行核, 在 base reward 之上再扣分。flag 关时恒 0 (与现有行为完全一致)。
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
        """CUDA Agent 风格离散奖励 {-1, 1, 2, 3} (供 A/B, 不破坏现有)。

        分级 (按论文, 失败一律 -1):
          - 语法/编译失败 (status != completed)            -> -1
          - decoy (奖励破解)                               -> -1
          - compiled 但 correctness=False                  ->  1
          - correctness=True                               ->  2
          - correctness=True 且 speedup > 1.2 (相对参考)   ->  3

        相对参考 speedup: 论文口径按 reference runtime 计算 (server 返回的 speedup
        已相对 reference 归一)。speedup<=1.2 视为"对但无显著加速", 留在 2。
        """
        # 失败 (语法/编译/运行时/超时) 一律 -1
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
        # decoy (奖励破解) -> -1
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
            # === 拍平 metadata 的 profiling 字段到顶层 (2026-08-21 coverage bug 修复) ===
            # eval server 把 num_custom_kernels(复数)/custom_kernel_cuda_time 等放 metadata,
            # 而 kernel_async + reward 函数从顶层(单数 num_custom_kernel)读 → 不加这个 coverage 恒为 0
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
