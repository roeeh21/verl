# Copyright 2025 AI21 Labs
"""

Integration module for AI21-Evaluators with VeRL reward system.

"""

import asyncio
import gc
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import psutil
import ray
from ai21_evaluators.backend.utils.inference_utils import DIRECT_VLLM_CLIENT_TYPE_ENV_VAR, USE_DIRECT_VLLM_CLIENT
from ai21_evaluators.evaluators.registry import (
    _EVALUATORS,
    EVALUATORS_RESGITERED_ENV_VAR,
    register_all_evaluators,
    register_all_parsers,
    warmup_registered_evaluators,
)
from ai21_evaluators.evaluators.structures import EvaluatorQueryContext
from ai21_evaluators.file_formats.verifiable_task_dataset import VerifiableTask, run_evaluation_entries
from filelock import FileLock
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

_DEFAULT_WARMUP_DIR = Path.home() / ".ai21_evaluators_warmup"
_WARMUP_DIR = Path(os.environ.get("AI21_EVALUATORS_WARMUP_DIR", _DEFAULT_WARMUP_DIR))


def _warmup_evaluators_with_lock():
    """
    Warmup all registered evaluators exactly once per shared filesystem using file locking;
    Mind that different actors might not be sharing the same filesystem.

    This triggers _prerun_import() for all evaluators via warmup_registered_evaluators(),
    which loads lazy dependencies (e.g., sentence_splitting downloads NLTK data at import)
    in case the evaluator has implemeneted warmup() method.

    Uses filelock (https://py-filelock.readthedocs.io/) for platform-independent locking.

    Flow:
    - First process: acquires lock, does warmup, creates marker file, releases
    - Other processes: acquire lock, find marker file exists, release immediately
    """
    _WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = _WARMUP_DIR / ".lock"
    warmup_marker = _WARMUP_DIR / ".done"
    with FileLock(lock_file):
        if warmup_marker.exists():
            print("[EVALUATORS_WARMUP] Already warmed up by another process")
        else:
            print("[EVALUATORS_WARMUP] First process on lock, warming up...")
            warmup_registered_evaluators()
            warmup_marker.touch()
            print("[EVALUATORS_WARMUP] Completed successfully")


asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

DEFAULT_MODELS_GATEWAY_CLIENT_URL = "https://tools.algo-studio.ai21.com/models-gateway"
EFFECTIVE_MODELS_GATEWAY_CLIENT_URL = os.environ.get("MODELS_GATEWAY_CLIENT_URL", DEFAULT_MODELS_GATEWAY_CLIENT_URL)

VERBOSE = os.environ.get("AI21_VERL_EVALUATORS_VERBOSE", "False").lower() == "true"
if VERBOSE:
    os.environ.setdefault("MODELS_INFERENCE_VLLM_CLIENT_VERBOSE", "True")
    os.environ.setdefault("OPENAI_VLLM_CLIENT_VERBOSE", "True")

MODEL_INFERENCE_VLLM_CLIENT_VERBOSE = os.environ.get("MODELS_INFERENCE_VLLM_CLIENT_VERBOSE", "False").lower() == "true"
OPENAI_VLLM_CLIENT_VERBOSE = os.environ.get("OPENAI_VLLM_CLIENT_VERBOSE", "False").lower() == "true"

OPENAI_VLLM_CLIENT_TIMEOUT_SECONDS = int(os.environ.get("OPENAI_VLLM_CLIENT_TIMEOUT_SECONDS", 420))
OPENAI_VLLM_CLIENT_MAX_RETRIES = int(os.environ.get("OPENAI_VLLM_CLIENT_MAX_RETRIES", 5))

MANUAL_GC = os.environ.get("AI21_VERL_EVALUATORS_MANUAL_GC", "False").lower() == "true"
MODEL_INFERENCE_CLIENT_REQUEST_TIMEOUT_IN_SECONDS = float(
    os.environ.get("MODEL_INFERENCE_CLIENT_REQUEST_TIMEOUT_IN_SECONDS", 240)
)

EVALUATOR_RAY_METRICS_ENABLED = os.environ.get("EVALUATOR_RAY_METRICS_ENABLED", "False").lower() == "true"

assert "VLLM_MODEL_INTERNAL_URL" not in os.environ and "VLLM_MODEL_INTERNAL_NAME" not in os.environ, (
    '"VLLM_MODEL_INTERNAL_NAME" or "VLLM_MODEL_INTERNAL_URL" should not be set directly, '
    "use jlm_model_name variable with format model_name@model_address"
)

EXCLUDED_EVALUATOR_SCORE = 0.0

# AI21 Note: Why do we need RAY_TASK_ENV?
# This is because not all ray workers are created as subprocesses of the main process,
# therefore they don't automatically inherit all env vars that the main process defines.
# This is even trickier, as usually the "first wave" of ray workers is created as subprocesses,
# so the side effects of not defining env vars in RAY_TASK_ENV are only seen after several steps.
# https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/general-debugging.html?#environment-variables-aren-t-passed-from-the-driver-process-to-worker-processes
RAY_TASK_ENV = {
    "env": {
        "DISABLE_NEST_ASYNCIO": "True",
        "AI21_VERL_EVALUATORS_VERBOSE": VERBOSE,
        "MODELS_INFERENCE_VLLM_CLIENT_VERBOSE": MODEL_INFERENCE_VLLM_CLIENT_VERBOSE,
        "OPENAI_VLLM_CLIENT_VERBOSE": OPENAI_VLLM_CLIENT_VERBOSE,
        "AI21_VERL_EVALUATORS_MANUAL_GC": MANUAL_GC,
        "MODELS_GATEWAY_CLIENT_URL": EFFECTIVE_MODELS_GATEWAY_CLIENT_URL,
        "EVALUATOR_RAY_METRICS_ENABLED": EVALUATOR_RAY_METRICS_ENABLED,
        "MODEL_INFERENCE_CLIENT_REQUEST_TIMEOUT_IN_SECONDS": MODEL_INFERENCE_CLIENT_REQUEST_TIMEOUT_IN_SECONDS,
        "OPENAI_VLLM_CLIENT_TIMEOUT_SECONDS": OPENAI_VLLM_CLIENT_TIMEOUT_SECONDS,
        "OPENAI_VLLM_CLIENT_MAX_RETRIES": OPENAI_VLLM_CLIENT_MAX_RETRIES,
    }
}

for optional_env_var in [
    USE_DIRECT_VLLM_CLIENT,
    DIRECT_VLLM_CLIENT_TYPE_ENV_VAR,
    "OPENAI_VLLM_CLIENT_DEFAULT_REASONING_EFFORT",
    "AI21_EVALUATORS_SEND_CPU_BOUND_TO_PROCESS_POOL",
    "AI21_EVALUATORS_MAX_CPU_BOUND_TASK_TIME",
    "AI21_EVALUATORS_NUM_CPU_BOUND_WORKERS",
    "AI21_EVALUATORS_NUM_CPU_BOUND_TASKS_PER_WORKER",
]:
    if optional_env_var in os.environ:
        RAY_TASK_ENV["env"][optional_env_var] = os.environ[optional_env_var]


def _log_multiline(prefix: str, data: dict[str, Any]) -> None:
    """
    Helper function to log multiple key-value pairs in a single print statement.
    Args:
        prefix: The logging prefix/category (e.g., "[EVALUATORS_TIMEOUT]")
        data: Dictionary of key-value pairs to log
    """
    lines = [f"{prefix} {key}: {value}" for key, value in data.items()]
    print("\n".join(lines))


def _log_timeout_error(task: VerifiableTask, completion: str, data_source: str, timeout_duration: float) -> None:
    """Helper function to log timeout errors with all relevant context."""
    _log_multiline(
        "[EVALUATORS_TIMEOUT]",
        {"task": task, "completion": completion, "data_source": data_source, "timeout_duration": timeout_duration},
    )


def _log_evaluation_exception(task: VerifiableTask, completion: str, data_source: str, exception: Exception) -> None:
    """Helper function to log evaluation exceptions with all relevant context."""
    _log_multiline(
        "[EVALUATORS_EXCEPTION]",
        {"task": task, "completion": completion, "data_source": data_source, "exception": exception},
    )


def _log_verbose_memory_info(stage: str, memory_mb: float, additional_info: Optional[dict[str, Any]] = None) -> None:
    """Helper function to log verbose memory information if VERBOSE is enabled."""
    if VERBOSE:
        log_data = {"stage": stage, "memory_usage_mb": f"{memory_mb:.2f} MB"}
        if additional_info:
            log_data.update(additional_info)
        _log_multiline("[RAY_TASK_MEMORY]", log_data)


if VERBOSE:
    _log_multiline("[RAY_TASK_ENV]", RAY_TASK_ENV.get("env", {}))


@ray.remote(runtime_env=RAY_TASK_ENV, max_restarts=0)
class AI21RayEvaluatorActor:
    def __init__(self, registry: ray.ObjectRef | dict[str, Any]):
        self._registry = ray.get(registry) if isinstance(registry, ray.ObjectRef) else registry
        assert len(self._registry) > 0, "No evaluators registered"
        start_time = time.time()
        register_all_parsers()
        end_time = time.time()
        if VERBOSE:
            print(f"[AI21_EVALUATORS] Register all parsers time: {end_time - start_time:.3f} seconds")
        start_time = time.time()
        _warmup_evaluators_with_lock()
        end_time = time.time()
        if VERBOSE:
            print(f"[AI21_EVALUATORS] Warmup evaluators with lock time: {end_time - start_time:.3f} seconds")

    def ping(self) -> bool:
        """Health check method - returns True if actor is alive and responsive."""
        return True

    def __ray_shutdown__(self):
        # Clean up process pool
        from ai21_evaluators.backend.utils.cpu_subprocessing import cleanup_process_pool

        cleanup_process_pool()

    async def _ai21_evaluation_task(
        self,
        completion: str,
        task: VerifiableTask,
        data_source: str,
        timeout: float,
    ) -> tuple[dict[str, float], bool, dict[str, Any]]:
        """
        Ray remote task that runs AI21-Evaluators evaluation.

        This task runs in a fresh Ray worker process where no event loop is running,
        making asyncio.run() safe to use. This avoids all the event loop conflicts
        that occur when trying to run async code in the main Ray process.

        Args:
            completion: Model completion to evaluate
            task: VerifiableTask
            data_source: Dataset source identifier
            timeout: Timeout in seconds for individual evaluator calls

        Returns:
            Tuple of (Dict mapping evaluator names to their computed scores, do_exclude flag, timing_info)
        """

        # Memory monitoring setup
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Timing setup
        task_start_time = time.time()
        timing_info = {
            "task_start_time": task_start_time,
            "initial_memory_mb": initial_memory,
            "evaluator_count": len(task.evaluations),
            "individual_evaluator_times": {},
        }
        evaluation_info = {"response_details": None}

        _log_verbose_memory_info("initial", initial_memory)

        async def run_evaluation():
            """Run the AI21-Evaluators evaluation asynchronously."""
            aggregate_do_exclude = False
            non_success_count = 0
            timeout_count = 0
            try:
                # Check memory before evaluation
                pre_eval_memory = process.memory_info().rss / 1024 / 1024  # MB
                timing_info["pre_eval_memory_mb"] = pre_eval_memory

                _log_verbose_memory_info("pre_evaluation", pre_eval_memory)

                # Run evaluation with timeout and time it
                start_time = time.time()
                if VERBOSE:
                    print(
                        {
                            "category": "[EVALUATORS_TIME]",
                            "task_id": task.id,
                            "description": f"Starting evaluation with timeout {timeout} seconds",
                            "completion": completion,
                        }
                    )
                elif MODEL_INFERENCE_VLLM_CLIENT_VERBOSE:
                    print(f"[EVALUATORS_TIME] task_id: {task.id}, starting evaluation with timeout {timeout} seconds")

                # Create query context
                query_context = EvaluatorQueryContext(completion=completion, messages=task.messages)

                # Note that asyncio.timout is necessary to coerce the evaluation
                # coroutine to actually get cancelled (as opposed to wait_for).
                async with asyncio.timeout(timeout):
                    responses = await run_evaluation_entries(
                        evaluation_entries=task.evaluations,
                        query_context=query_context,
                        registry=self._registry,
                        aggregation_config=task.aggregation_config,
                    )

                end_time = time.time()
                # Record timing info
                evaluation_time = end_time - start_time
                timing_info["evaluation_time"] = evaluation_time
                timing_info["avg_time_per_evaluator"] = (
                    evaluation_time / len(task.evaluations) if len(task.evaluations) > 0 else 0
                )

                if VERBOSE:
                    print(f"[EVALUATORS_TIME] Ray task: Evaluation time: {evaluation_time} seconds")

                # Check memory after evaluation
                post_eval_memory = process.memory_info().rss / 1024 / 1024  # MB
                delta = post_eval_memory - pre_eval_memory
                timing_info["post_eval_memory_mb"] = post_eval_memory
                timing_info["memory_delta_mb"] = delta

                if delta > 0:
                    _log_verbose_memory_info("post_evaluation", post_eval_memory, {"delta_mb": f"{delta:.2f} MB"})

                # ai21-evaluators should return a single aggregated score
                if len(responses) != 1:
                    raise ValueError(f"Expected single aggregated response, got {len(responses)} responses")
                response = responses[0]

                # Capture response.details for dumping
                try:
                    evaluation_info["response_details"] = response.details.model_dump()
                except Exception as e:
                    evaluation_info["response_details"] = f"Could not extract response.details: {e}"
                    if VERBOSE:
                        print(f"[RESPONSE_DETAILS] Could not extract response.details: {e}")

                if response.result is None:
                    # Handle errors - ai21-evaluators uses None for errors
                    aggregate_do_exclude = True
                    non_success_count += 1
                    aggregated_score = EXCLUDED_EVALUATOR_SCORE
                    if VERBOSE:
                        print(
                            f"[AGGREGATED_SCORE] Ray task: Error in evaluation, "
                            f"error_code={response.error_code}, excluding example"
                        )
                else:
                    # Success - result is the aggregated score
                    aggregate_do_exclude = False
                    aggregated_score = response.result
                    if VERBOSE:
                        print(f"[AGGREGATED_SCORE] Ray task: Aggregated score={aggregated_score}")
                    # Record individual evaluator timing if available
                    for evaluator_name, evaluator_result in zip(
                        response.details.aggregated_evaluators,
                        response.details.aggregated_evaluators_results,
                        strict=False,
                    ):
                        if hasattr(evaluator_result, "processing_seconds"):
                            timing_info["individual_evaluator_times"][evaluator_name] = (
                                evaluator_result.processing_seconds
                            )

            except (asyncio.TimeoutError, TimeoutError):
                timeout_duration = time.time() - start_time
                print(
                    f"[ASYNCIO_TIMEOUT_DEBUG] asyncio.TimeoutError occurred after {timeout_duration:.3f}s "
                    f"Evaluators involved: {[evaluator.evaluator_name for evaluator in task.evaluations]}"
                    f"(timeout was {timeout}s)"
                )
                timing_info["timeout_occurred"] = True
                timing_info["timeout_duration"] = timeout_duration
                print(
                    f"[ASYNCIO_TIMEOUT_DEBUG] Timeout occurred after {timeout_duration:.3f}s (timeout was {timeout}s)"
                )
                if VERBOSE:
                    _log_timeout_error(task, completion, data_source, timeout_duration)
                aggregate_do_exclude = True
                aggregated_score = EXCLUDED_EVALUATOR_SCORE
                timeout_count += 1
            except Exception as e:
                print(f"[EXCEPTION_DEBUG] Caught exception: type={type(e).__name__}, str={str(e)}, repr={repr(e)}")
                timing_info["exception_occurred"] = True
                timing_info["exception_details"] = str(e)
                timing_info["exception_duration"] = time.time() - start_time
                if VERBOSE:
                    _log_evaluation_exception(task, completion, data_source, e)
                    print(f"[EVALUATORS_EXCEPTION_STACKTRACE] Full stacktrace:\n{traceback.format_exc()}")
                aggregate_do_exclude = True
                aggregated_score = EXCLUDED_EVALUATOR_SCORE
            finally:
                # Force garbage collection to clean up any leaked memory
                if MANUAL_GC:
                    gc.collect()

            timing_info["non_success_count"] = non_success_count
            timing_info["timeout_count"] = timeout_count

            return aggregated_score, aggregate_do_exclude, timing_info, evaluation_info

        # Ray task starts in a fresh process => no loop is running yet.
        # This makes asyncio.run() safe to use.
        try:
            aggregated_score, aggregate_do_exclude, timing_info, evaluation_info = await run_evaluation()

            # Final memory check and timing
            task_end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            timing_info["task_end_time"] = task_end_time
            timing_info["total_task_time"] = task_end_time - task_start_time
            timing_info["final_memory_mb"] = final_memory
            timing_info["total_memory_delta_mb"] = final_memory - initial_memory

            if VERBOSE:
                _log_verbose_memory_info(
                    "final", final_memory, {"total_delta_mb": f"{final_memory - initial_memory:.2f} MB"}
                )
                print(f"[RAY_TASK_TIMING] Total task time: {timing_info['total_task_time']:.3f} seconds")

            return aggregated_score, aggregate_do_exclude, timing_info, evaluation_info
        except Exception as e:
            timing_info["task_exception_occurred"] = True
            timing_info["task_exception_details"] = str(e)
            timing_info["task_exception_duration"] = time.time() - task_start_time

            _log_multiline(
                "[RAY_TASK_EXCEPTION]",
                {
                    "error": f"Ray task asyncio.run failed: {e}",
                    "action": f"Returning failure score {EXCLUDED_EVALUATOR_SCORE}",
                },
            )
            return EXCLUDED_EVALUATOR_SCORE, True, timing_info, evaluation_info


class AI21EvaluatorComputeScoreFactory:
    def __init__(
        self,
        timeout: float,
        ai21_evaluators_actors_amount: int,
        ai21_evaluators_actor_cpus: float,
        ai21_evaluators_spread_per_node: bool = True,
    ):
        self.timeout = timeout
        self.actor_pool_size = ai21_evaluators_actors_amount
        if ai21_evaluators_actor_cpus <= 0:
            raise ValueError("ai21_evaluators_actor_cpus must be greater than 0")
        self.actor_cpus = ai21_evaluators_actor_cpus
        self.spread_per_node = ai21_evaluators_spread_per_node

        # Note: Avoid `del os.environ[...]` as it's not thread-safe and can cause SIGSEGV
        # when multiple Ray workers spawn simultaneously. Use pop() with default instead.
        os.environ.pop(EVALUATORS_RESGITERED_ENV_VAR, None)
        register_all_evaluators()
        assert len(_EVALUATORS) > 0, "No evaluators registered"
        print(f"[AI21_RM] Registered {len(_EVALUATORS)} evaluators")
        self._registry = ray.put(_EVALUATORS)
        self._total_evaluation_time = 0.0
        self._evaluation_count = 0
        self._last_timing_metrics = {}
        self._future_actor_map: dict[str, tuple[Any, int]] = {}
        self.actor_pool_req_idx = 0

        self._placement_group = None
        if self.spread_per_node and self.actor_pool_size > 1:
            assert self.actor_pool_size == len(ray.nodes()), (
                f"Not enough nodes for STRICT_SPREAD: {self.actor_pool_size} actors, {len(ray.nodes())} nodes"
            )
            bundles = [{"CPU": self.actor_cpus} for _ in range(self.actor_pool_size)]
            pg_name = f"ai21_evaluator_pg_{uuid.uuid4().hex[:8]}"
            self._placement_group = placement_group(bundles, strategy="STRICT_SPREAD", name=pg_name)
            ray.get(self._placement_group.ready())
            print(f"[AI21_RM] Created STRICT_SPREAD placement group with {self.actor_pool_size} bundles (1 per node)")

        self.actor_pool = [self._get_or_create_actor(idx) for idx in range(self.actor_pool_size)]

    def _get_or_create_actor(self, actor_idx: int):
        """Get existing named actor or create new one. Enables sharing across processes."""
        actor_name = f"ai21_evaluator_actor_{actor_idx}"
        options = {
            "name": actor_name,
            "num_cpus": self.actor_cpus,
            "get_if_exists": True,
        }
        if self._placement_group is not None:
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=self._placement_group,
                placement_group_bundle_index=actor_idx,
            )
            print(f"[AI21_RM] Creating actor {actor_idx} with STRICT_SPREAD placement (bundle {actor_idx})")
        return AI21RayEvaluatorActor.options(**options).remote(self._registry)

    # currently naive return just the next actor, doesn't balance according to load
    def _get_next_actor(self) -> tuple[Any, int]:
        actor_idx = self.actor_pool_req_idx % self.actor_pool_size
        actor = self.actor_pool[actor_idx]
        self.actor_pool_req_idx += 1
        return actor, actor_idx

    def _register_future_actor(self, future: ray.ObjectRef, actor: Any, actor_idx: int) -> None:
        self._future_actor_map[future.hex()] = (actor, actor_idx)

    def release_future_actor(self, future: ray.ObjectRef) -> None:
        self._future_actor_map.pop(future.hex(), None)

    def retire_actor_for_future(self, future: ray.ObjectRef) -> None:
        entry = self._future_actor_map.pop(future.hex(), None)
        if entry is None:
            return
        actor, actor_idx = entry

        current_actor = self.actor_pool[actor_idx]
        if actor is not current_actor:
            print(f"[AI21_EVALUATORS] Actor {actor_idx} already retired, skipping kill and respawn")
            return

        try:
            ray.kill(actor, no_restart=True)
            print(f"[AI21_EVALUATORS] Successfully killed actor {actor_idx}")
        except Exception as err:
            print(f"[AI21_EVALUATORS] Failed to kill actor after timeout: {err}")
        self.actor_pool[actor_idx] = self._get_or_create_actor(actor_idx)
        print(f"[AI21_EVALUATORS] Successfully created new actor {actor_idx}")

    def reset_unresponsive_actors(self, timeout: float = 5.0) -> int:
        """
        Health check all actors in the pool and reset unresponsive actors. MUST be called when actors are idle

        Any actor that doesn't respond to ping within the timeout is retired and respawned.

        Args:
            timeout: Timeout in seconds for each actor's ping response.

        Returns:
            Number of actors that were retired and respawned.
        """
        retired_count = 0
        ping_futures = []

        # Ping all actors
        for actor_idx, actor in enumerate(self.actor_pool):
            try:
                future = actor.ping.remote()
                ping_futures.append((actor_idx, actor, future))
            except Exception as e:
                print(f"[AI21_EVALUATORS_HEALTH] Actor {actor_idx} failed to accept ping: {e}")
                self.actor_pool[actor_idx] = self._get_or_create_actor(actor_idx)
                retired_count += 1

        # Wait for all pings with timeout
        if ping_futures:
            futures_list = [f for _, _, f in ping_futures]
            try:
                ready, _ = ray.wait(futures_list, num_returns=len(futures_list), timeout=timeout)
            except Exception as e:
                print(f"[AI21_EVALUATORS_HEALTH] ray.wait failed: {e}")
                ready = []

            # Check results for ready futures
            ready_set = set(ready)
            for actor_idx, actor, future in ping_futures:
                if future in ready_set:
                    try:
                        ray.get(future, timeout=0.1)
                        # Actor responded successfully
                    except Exception as e:
                        print(f"[AI21_EVALUATORS_HEALTH] Actor {actor_idx} ping failed: {e}. Retiring...")
                        try:
                            ray.kill(actor, no_restart=True)
                        except Exception:
                            pass
                        self.actor_pool[actor_idx] = self._get_or_create_actor(actor_idx)
                        retired_count += 1
                else:
                    # Actor didn't respond in time
                    print(
                        f"[AI21_EVALUATORS_HEALTH] Actor {actor_idx} did not respond to ping in {timeout}s. Retiring.."
                    )
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
                    self.actor_pool[actor_idx] = self._get_or_create_actor(actor_idx)
                    retired_count += 1

        if retired_count > 0:
            print(f"[AI21_EVALUATORS_HEALTH] Health check complete: retired and respawned {retired_count} actors")
        else:
            print(f"[AI21_EVALUATORS_HEALTH] Health check complete: all {len(self.actor_pool)} actors healthy")

        return retired_count

    def get_timing_stats(self) -> dict[str, Any]:
        """Get cumulative timing statistics."""
        return {
            "total_evaluation_time": self._total_evaluation_time,
            "evaluation_count": self._evaluation_count,
            "avg_time_per_evaluation": (
                self._total_evaluation_time / self._evaluation_count if self._evaluation_count > 0 else 0
            ),
            "timeout_setting": self.timeout,
        }

    def get_last_timing_metrics(self) -> dict[str, Any]:
        """Get timing metrics from the last evaluation call."""
        return self._last_timing_metrics.copy()

    def reset_timing_stats(self):
        """Reset cumulative timing statistics."""
        self._total_evaluation_time = 0.0
        self._evaluation_count = 0

    def get_ai21_evaluators_compute_score_ray_future(
        self,
        data_source: str,
        completion: str,
        task: VerifiableTask,
    ) -> ray.ObjectRef:
        """
        VeRL-compatible function that uses AI21-Evaluators for reward computation via Ray tasks.

        This version uses Ray's task system instead of threads, which is more efficient
        and avoids event loop conflicts.

        Corresponding evaluations to the EvaluationEntry class and VerifiableTask class in ai21_evaluators

        class EvaluationEntry(BaseModel):
            query_type: Optional[str] = None
            evaluator_name: Optional[str] = None
            evaluator_config: Optional[dict[str, Any]] = None
            query_args: Optional[dict[str, Any]] = None

        class VerifiableTask(BaseModel):
            id: Optional[str | int] = None
            messages: list[dict[str, str]]
            evaluations: list[EvaluationEntry]
            metadata: Optional[dict[str, Any]] = None

        Args:
            data_source: Dataset source identifier
            completion: The model's generated solution/completion
            task: VerifiableTask

        Returns:
            Tuple of (Dict mapping evaluator names to their computed scores, do_exclude flag)
        """
        actor, actor_idx = self._get_next_actor()
        future = actor._ai21_evaluation_task.remote(
            completion=completion,
            task=task,
            data_source=data_source,
            timeout=self.timeout,
        )
        self._register_future_actor(future, actor, actor_idx)
        return future
