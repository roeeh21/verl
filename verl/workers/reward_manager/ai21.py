# Copyright 2025 AI21 Labs
import json
import multiprocessing
import os
import time
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import ray
import torch

# TODO: temporary setting evaluators process pool on, remove when it's on by default in evaluators side
# Note: Use setdefault to avoid race conditions when multiple Ray workers import simultaneously.
# Direct os.environ modifications at import time can cause SIGSEGV in concurrent scenarios.
os.environ.setdefault("AI21_EVALUATORS_SEND_CPU_BOUND_TO_PROCESS_POOL", "True")
os.environ.setdefault("AI21_EVALUATORS_NUM_CPU_BOUND_WORKERS", str(min(int(multiprocessing.cpu_count()), 50)))

from ai21_evaluators.file_formats.flexible_aggregation import AggregationConfig
from ai21_evaluators.file_formats.verifiable_task_dataset import (
    EvaluationEntry,
    VerifiableTask,
    VerifiableTaskDataset,
    apply_evaluators_config_to_verifiable_dataset,
    change_verifiable_task_model_name,
)

from verl import DataProto
from verl.utils.checkpoint.file_utils import sync_file_gs
from verl.utils.reward_score.ai21.evaluators_integration import AI21EvaluatorComputeScoreFactory
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Use the same environment variable as evaluators_integration.py
VERBOSE = os.environ.get("AI21_VERL_EVALUATORS_VERBOSE", "False").lower() == "true"
TIME_THRESHOLD = 2.0  # Threshold for timing logs
RAY_CANCEL_GRACE_TIMEOUT = 0.5  # Seconds to wait for cooperative cancellation

DEBUG_DUMP_DIR = os.environ.get("AI21_DEBUG_DUMP_DIR", "/tmp/ai21_exclusion_dumps")


def _dump_exclusion_debug_info(
    task_info: dict,
    task_idx: int,
    aggregated_score: float,
    ray_timing_info: dict,
    ray_evaluation_info: dict,
    remote_save_path: Optional[str] = None,
) -> str:
    """Dump exclusion debug info to a JSON file and return the file path."""
    os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dump_file = os.path.join(DEBUG_DUMP_DIR, f"exclusion_{timestamp}_idx{task_idx}.json")

    # Prepare serializable debug info
    debug_info = {
        "timestamp": timestamp,
        "task_idx": task_idx,
        "aggregated_score": aggregated_score,
        "ray_timing_info": ray_timing_info,
        "ray_evaluation_info": ray_evaluation_info,
        "task": task_info["task"].model_dump(),
        "prompt_str": task_info["prompt_str"],
        "response_str": task_info["response_str"],
        "metadata": task_info["metadata"],
        "valid_response_length": int(task_info["valid_response_length"]),
        "data_source": task_info["data_source"],
        "nullify_scores": task_info["nullify_scores"],
    }

    with open(dump_file, "w") as f:
        json.dump(debug_info, f, indent=2, default=str)

    print(f"[AI21_RM_DEBUG] Dumped exclusion debug info to: {dump_file}")

    # Upload to GCS if remote_save_path is provided
    if remote_save_path:
        remote_dump_path = os.path.join(remote_save_path, "exclusion_dumps", os.path.basename(dump_file))
        try:
            sync_file_gs(dump_file, remote_dump_path)
            print(f"[AI21_RM_DEBUG] Uploaded exclusion debug info to: {remote_dump_path}")
        except Exception as e:
            print(f"[AI21_RM_DEBUG] Failed to upload to GCS: {e}")

    return dump_file


@register("ai21")
class AI21RewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        ai21_evaluators_config_file=None,
        ai21_evaluators_timeout=None,
        ai21_clean_thinking_trace=False,
        ai21_enforce_thinking_trace=False,
        ai21_evaluators_async=True,
        ai21_evaluators_batch_size=0,
        ai21_evaluators_actors_amount=None,  # Default to number of Ray nodes
        ai21_evaluators_actor_cpus=1,  # No CPU restriction per actor
        ai21_evaluators_spread_per_node=True,
        stop_seq_to_strip=None,
        remote_save_path=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.ai21_evaluators_config_file = ai21_evaluators_config_file
        self.ai21_evaluators_timeout = ai21_evaluators_timeout
        self.ai21_clean_thinking_trace = ai21_clean_thinking_trace
        # TODO: TAMERG either remove this or remove the data passed flag.
        self.ai21_enforce_thinking_trace = ai21_enforce_thinking_trace
        self.ai21_evaluators_actors_amount = ai21_evaluators_actors_amount
        self.ai21_evaluators_actor_cpus = ai21_evaluators_actor_cpus
        self.ai21_evaluators_spread_per_node = ai21_evaluators_spread_per_node
        self.ai21_evaluators_async = ai21_evaluators_async
        self.ai21_evaluators_batch_size = ai21_evaluators_batch_size
        self.stop_seq_to_strip = stop_seq_to_strip
        self.remote_save_path = remote_save_path
        if not ai21_evaluators_async:
            print("[AI21_RM] ai21_evaluators_async is False, setting ai21_evaluators_batch_size to 1")
            self.ai21_evaluators_batch_size = 1
        # log the parameters
        print("[AI21_RM] AI21RewardManager initialized with parameters:")
        print(f"[AI21_RM] ai21_evaluators_config_file: {ai21_evaluators_config_file}")
        print(f"[AI21_RM] ai21_evaluators_timeout: {ai21_evaluators_timeout}")
        print(f"[AI21_RM] ai21_evaluators_async: {ai21_evaluators_async}")
        print(f"[AI21_RM] ai21_evaluators_batch_size: {ai21_evaluators_batch_size}")
        print(f"[AI21_RM] ai21_evaluators_actors_amount: {ai21_evaluators_actors_amount}")
        print(f"[AI21_RM] ai21_evaluators_actor_cpus: {ai21_evaluators_actor_cpus}")
        print(f"[AI21_RM] stop_seq_to_strip: {stop_seq_to_strip}")
        print(f"[AI21_RM] remote_save_path: {remote_save_path}")
        self._setup_compute_score(compute_score)

    def _setup_compute_score(self, compute_score: Optional[Callable]):
        """Initialize the compute score function and factory."""
        # Store factory to access timing metrics, fallback to provided compute_score function
        self.evaluator_factory: Optional[AI21EvaluatorComputeScoreFactory] = None
        if compute_score is None:
            self.evaluator_factory = AI21EvaluatorComputeScoreFactory(
                timeout=self.ai21_evaluators_timeout,
                ai21_evaluators_actors_amount=self.ai21_evaluators_actors_amount,
                ai21_evaluators_actor_cpus=self.ai21_evaluators_actor_cpus,
                ai21_evaluators_spread_per_node=self.ai21_evaluators_spread_per_node,
            )
            self.compute_score = self.evaluator_factory.get_ai21_evaluators_compute_score_ray_future
        else:
            self.compute_score = compute_score

    def _release_future_actor(self, future: Optional[ray.ObjectRef]) -> None:
        """Release actor bookkeeping for a completed future."""
        if future is None or self.evaluator_factory is None:
            return
        self.evaluator_factory.release_future_actor(future)

    def _retire_future_actor(self, future: Optional[ray.ObjectRef]) -> None:
        """Retire and respawn the actor responsible for the provided future."""
        if future is None or self.evaluator_factory is None:
            return
        self.evaluator_factory.retire_actor_for_future(future)

    def _cancel_future_or_retire(self, future: Optional[ray.ObjectRef]) -> None:
        """Attempt graceful cancellation before retiring an actor."""
        if future is None:
            return

        cancelled = False
        try:
            ray.cancel(future)
            ready, _ = ray.wait([future], timeout=RAY_CANCEL_GRACE_TIMEOUT)
            cancelled = len(ready) > 0
        except Exception as err:
            print(f"[AI21_RM] Failed to cancel future cooperatively: {err}")

        if cancelled:
            self._release_future_actor(future)
        else:
            self._retire_future_actor(future)

    def _reset_unresponsive_actors(self) -> int:
        """
        Reset unresponsive actors in the pool.
        MUST be called when actors are idle.
        Returns the number of actors that were retired and respawned.
        """
        if self.evaluator_factory is None:
            return 0
        return self.evaluator_factory.reset_unresponsive_actors(timeout=5.0)

    def __call__(self, data: DataProto, return_dict=False):
        """Compute aggregated rewards for a batch of data using ai21-evaluators."""
        if "aggregated_score" in data.non_tensor_batch:
            # We are in agent rollout mode, the reward was already calculated,
            # we just need to prepare the batch data structure
            dict_result = self._convert_agent_rollout_reward_extra_infos_dict(data)
            return dict_result

        start_time = time.time()
        if VERBOSE:
            print(f"[TIMING_AI21_RM] Starting AI21RewardManager.__call__ with {len(data)} items at {start_time:.6f}")

        # Initialize data structures
        reward_tensor, reward_extra_info = self._initialize_data_structures(data)
        already_print_data_sources: dict[str, int] = {}

        init_time = time.time()
        if VERBOSE and init_time - start_time > TIME_THRESHOLD:
            print(f"[TIMING_AI21_RM] Initialization completed in {init_time - start_time:.6f}s")

        # Compute aggregated rewards and get timing metrics
        timing_metrics = self._compute_example_reward(
            data, reward_tensor, reward_extra_info, already_print_data_sources
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Add batch-level timing to metrics
        if timing_metrics:
            timing_metrics["ai21_evaluators/total_batch_time"] = total_time
            timing_metrics["ai21_evaluators/initialization_time"] = init_time - start_time
            timing_metrics["ai21_evaluators/processing_time"] = end_time - init_time
            timing_metrics["ai21_evaluators/total_batch_size"] = len(data)

        if VERBOSE and total_time > TIME_THRESHOLD:
            print(
                f"[TIMING_AI21_RM] AI21RewardManager.__call__ completed all {len(data)} items in {total_time:.6f}s "
                f"(avg: {total_time / len(data):.6f}s per item)"
            )

        # Health check and reset unresponsive actors now that they should be idle
        retired_count = self._reset_unresponsive_actors()
        if timing_metrics and retired_count > 0:
            timing_metrics["ai21_evaluators/actors_retired_in_health_check"] = retired_count

        # Prepare result - simplified structure with single reward tensor
        result = {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }

        # Add timing metrics captured from Ray tasks
        if timing_metrics:
            result["reward_extra_info"]["timing_metrics"] = timing_metrics

        return result

    def _initialize_data_structures(self, data: DataProto) -> tuple[torch.Tensor, dict]:
        """Initialize data structures for reward computation."""
        # Initialize single reward tensor for aggregated scores
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, Any] = {}

        # Initialize reward_extra_info structures
        reward_extra_info["do_exclude_example"] = [False] * len(data)
        reward_extra_info["aggregated_score"] = [0.0] * len(data)
        reward_extra_info["response_details"] = [None] * len(data)

        return reward_tensor, reward_extra_info

    def _compute_example_reward(self, data, reward_tensor, reward_extra_info, already_print_data_sources):
        """Compute aggregated rewards for examples in the batch with optional rate limiting."""
        # Prepare all task info first
        tasks_info = []
        for i in range(len(data)):
            task_info = self._process_single_data_item(data[i], i)
            tasks_info.append(task_info)

        # Special case: ai21_evaluators_batch_size=0 means no rate limiting (original behavior)
        if self.ai21_evaluators_batch_size == 0:
            return self._process_all_tasks_at_once(
                data, tasks_info, reward_tensor, reward_extra_info, already_print_data_sources
            )
        else:
            # Rate-limited processing using simple batching
            return self._process_tasks_in_batches(
                data, tasks_info, reward_tensor, reward_extra_info, already_print_data_sources
            )

    def _process_all_tasks_at_once(
        self, data, tasks_info, reward_tensor, reward_extra_info, already_print_data_sources
    ):
        """Process all tasks at once (original behavior when rate limiting is disabled)."""
        if VERBOSE:
            print(
                f"[AI21_RM] Rate limiting disabled - processing all {len(tasks_info)} tasks at once "
                f"(treating whole data as single batch)"
            )

        timing_metrics = self._submit_and_collect_batch(
            tasks_info, reward_tensor, reward_extra_info, already_print_data_sources
        )

        # When batch size is 0, consider the whole data as one batch
        effective_batch_size = len(data)
        if VERBOSE:
            print(f"[AI21_RM] Completed processing single batch of {effective_batch_size} items")
        return self._finalize_timing_metrics(timing_metrics, effective_batch_size)

    def _process_tasks_in_batches(self, data, tasks_info, reward_tensor, reward_extra_info, already_print_data_sources):
        """Process tasks in batches to limit concurrent tasks."""
        timing_metrics = self._initialize_timing_metrics()
        total_tasks = len(tasks_info)
        batch_size = self.ai21_evaluators_batch_size

        if VERBOSE:
            print(f"[AI21_RM] Starting batch processing with batch size {batch_size}")

        # Process tasks in batches
        for batch_start in range(0, total_tasks, batch_size):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_tasks = tasks_info[batch_start:batch_end]

            if VERBOSE:
                print(
                    f"[AI21_RM] Processing batch {batch_start // batch_size + 1}/"
                    f"{(total_tasks + batch_size - 1) // batch_size} (tasks {batch_start + 1}-{batch_end})"
                )

            # Submit and collect this batch
            batch_timing = self._submit_and_collect_batch(
                batch_tasks, reward_tensor, reward_extra_info, already_print_data_sources, batch_start
            )

            # Merge timing metrics
            self._merge_timing_metrics(timing_metrics, batch_timing)

        return self._finalize_timing_metrics(timing_metrics, batch_size)

    def _submit_and_collect_batch(
        self, tasks_info, reward_tensor, reward_extra_info, already_print_data_sources, batch_offset=0
    ):
        """Submit a batch of tasks and collect their aggregated results."""
        timing_metrics = self._initialize_timing_metrics()

        # Phase 1: Submit all tasks in the batch
        futures = []
        future_to_task = {}
        future_start_times: dict[ray.ObjectRef, float] = {}

        for i, task_info in enumerate(tasks_info):
            actual_index = batch_offset + i

            if not task_info["nullify_scores"]:
                future = self._submit_task_for_evaluation(task_info)
                futures.append(future)
                future_to_task[future] = (actual_index, task_info)
                future_start_times[future] = time.time()
            else:
                # Process nullified scores immediately
                self._process_nullified_task_simple(
                    actual_index,
                    task_info,
                    reward_tensor,
                    reward_extra_info,
                    already_print_data_sources,
                    timing_metrics,
                )

        # Phase 2: Wait for all submitted tasks to complete
        if futures:
            if VERBOSE:
                print(f"[AI21_RM] Waiting for {len(futures)} Ray tasks to complete")
            pending_futures = set(futures)
            wait_timeout = self.ai21_evaluators_timeout + 5
            while pending_futures:
                wait_list = list(pending_futures)
                wait_count = min(self.ai21_evaluators_actors_amount, len(wait_list))
                wait_count = max(1, wait_count)
                ready_futures, _ = ray.wait(
                    wait_list,
                    num_returns=wait_count,
                    timeout=wait_timeout,
                )
                # Handle completed futures first
                for future in ready_futures:
                    if future not in pending_futures:
                        continue
                    pending_futures.discard(future)
                    task_idx, task_info = future_to_task.get(future, (None, None))
                    if task_idx is None or task_info is None:
                        continue
                    aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info = self._get_ray_task_result(
                        future, task_idx
                    )
                    self._process_task_results_internal(
                        task_idx,
                        task_info,
                        aggregated_score,
                        do_exclude,
                        ray_timing_info,
                        ray_evaluation_info,
                        reward_tensor,
                        reward_extra_info,
                        already_print_data_sources,
                        timing_metrics,
                    )
                    self._release_future_actor(future)
                    future_start_times.pop(future, None)
                    future_to_task.pop(future, None)

                # Handle timed-out futures without blocking on ray.get again
                timed_out_futures = []
                if pending_futures:
                    now = time.time()
                    timed_out_futures = [
                        future
                        for future in list(pending_futures)
                        if now - future_start_times.get(future, now) >= wait_timeout
                    ]
                if timed_out_futures:
                    print(
                        f"[ERROR_AI21_RM] {len(timed_out_futures)} Ray tasks timed out after "
                        f"{self.ai21_evaluators_timeout}s during ray.wait()"
                    )
                    print(
                        f"[ERROR_AI21_RM] These tasks likely also hit the internal "
                        f"{self.ai21_evaluators_timeout}s timeout. "
                        "Consider increasing ai21_evaluators_timeout."
                    )
                    for future in timed_out_futures:
                        pending_futures.discard(future)
                        task_idx, task_info = future_to_task.get(future, (None, None))
                        if task_idx is None or task_info is None:
                            continue
                        self._handle_timed_out_future(
                            future,
                            task_idx,
                            task_info,
                            reward_tensor,
                            reward_extra_info,
                            already_print_data_sources,
                            timing_metrics,
                        )
                        future_start_times.pop(future, None)
                        future_to_task.pop(future, None)

        return timing_metrics

    def _process_nullified_task_simple(
        self,
        task_idx: int,
        task_info: dict,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
        already_print_data_sources: dict,
        timing_metrics: dict,
    ):
        """Process a task with nullified scores."""
        # Nullified scores are set to 0
        aggregated_score = 0.0
        do_exclude = False  # Nullified scores are not excluded, just scored as 0
        ray_timing_info: dict[str, Any] = {}
        ray_evaluation_info = {"response_details": None}

        self._process_task_results_internal(
            task_idx,
            task_info,
            aggregated_score,
            do_exclude,
            ray_timing_info,
            ray_evaluation_info,
            reward_tensor,
            reward_extra_info,
            already_print_data_sources,
            timing_metrics,
        )

    def _process_task_results_internal(
        self,
        task_idx: int,
        task_info: dict,
        aggregated_score: float,
        do_exclude: bool,
        ray_timing_info: dict,
        ray_evaluation_info: dict,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
        already_print_data_sources: dict,
        timing_metrics: dict,
    ):
        """Internal method to process aggregated task results and update data structures."""
        # Debug on first exclusion if enabled
        if do_exclude and VERBOSE:
            try:
                _dump_exclusion_debug_info(
                    task_info,
                    task_idx,
                    aggregated_score,
                    ray_timing_info,
                    ray_evaluation_info,
                    remote_save_path=self.remote_save_path,
                )
            except Exception as e:
                print(f"[ERROR_AI21_RM] Failed to dump exclusion debug info: {e}")

        # Update timing metrics
        self._update_timing_metrics(timing_metrics, ray_timing_info, do_exclude)

        reward_extra_info["do_exclude_example"][task_idx] = do_exclude

        # Set the aggregated reward score
        reward_tensor[task_idx, task_info["valid_response_length"] - 1] = aggregated_score
        reward_extra_info["aggregated_score"][task_idx] = aggregated_score

        # Capture response_details from ray_evaluation_info if available
        if "response_details" in ray_evaluation_info:
            reward_extra_info["response_details"][task_idx] = ray_evaluation_info["response_details"]

        # Print examination results
        self._print_examination_results(task_info, aggregated_score, do_exclude, already_print_data_sources)

    def _initialize_timing_metrics(self) -> dict:
        """Initialize timing metrics structure."""
        return {
            "total_ray_evaluation_time": 0.0,
            "total_ray_task_time": 0.0,
            "total_evaluations": 0,
            "individual_evaluator_times": {},
            "excluded_examples_count": 0,
            "non_success_count": 0,
            "timeout_count": 0,
        }

    def _update_timing_metrics(self, timing_metrics: dict, ray_timing_info: dict, do_exclude: bool):
        """Update timing metrics with results from a Ray task."""
        if ray_timing_info:
            timing_metrics["total_ray_evaluation_time"] += ray_timing_info.get("evaluation_time", 0)
            timing_metrics["total_ray_task_time"] += ray_timing_info.get("total_task_time", 0)
            timing_metrics["total_evaluations"] += ray_timing_info.get("evaluator_count", 0)
            timing_metrics["non_success_count"] += ray_timing_info.get("non_success_count", 0)
            timing_metrics["timeout_count"] += ray_timing_info.get("timeout_count", 0)

            # Aggregate individual evaluator times
            if "individual_evaluator_times" in ray_timing_info:
                for evaluator_name, eval_time in ray_timing_info["individual_evaluator_times"].items():
                    if evaluator_name not in timing_metrics["individual_evaluator_times"]:
                        timing_metrics["individual_evaluator_times"][evaluator_name] = []
                    timing_metrics["individual_evaluator_times"][evaluator_name].append(eval_time)

        if do_exclude:
            timing_metrics["excluded_examples_count"] += 1

    def _finalize_timing_metrics(self, timing_metrics: dict, batch_size: int) -> dict:
        """Finalize timing metrics and return."""
        # Build aggregated timing metrics
        aggregated_timing_metrics = {
            "ai21_evaluators/batch_size": batch_size,
            "ai21_evaluators/excluded_examples_count": timing_metrics["excluded_examples_count"],
            "ai21_evaluators/non_success_count": timing_metrics["non_success_count"],
            "ai21_evaluators/timeout_count": timing_metrics["timeout_count"],
        }

        # Add batch-level ratios and averages
        if batch_size > 0:
            aggregated_timing_metrics["ai21_evaluators/excluded_examples_ratio"] = (
                timing_metrics["excluded_examples_count"] / batch_size
            )

        if timing_metrics["total_evaluations"] > 0:
            aggregated_timing_metrics.update(
                {
                    "ai21_evaluators/total_ray_evaluation_time": timing_metrics["total_ray_evaluation_time"],
                    "ai21_evaluators/total_ray_task_time": timing_metrics["total_ray_task_time"],
                    "ai21_evaluators/total_evaluations": timing_metrics["total_evaluations"],
                    "ai21_evaluators/avg_ray_evaluation_time": timing_metrics["total_ray_evaluation_time"]
                    / timing_metrics["total_evaluations"],
                    "ai21_evaluators/avg_ray_task_time_per_example": timing_metrics["total_ray_task_time"] / batch_size,
                    "ai21_evaluators/evaluations_per_example": timing_metrics["total_evaluations"] / batch_size,
                    "ai21_evaluators/non_success_ratio": timing_metrics["non_success_count"]
                    / timing_metrics["total_evaluations"],
                    "ai21_evaluators/timeout_ratio": timing_metrics["timeout_count"]
                    / timing_metrics["total_evaluations"],
                }
            )

            # Add individual evaluator timing averages
            for evaluator_name, times in timing_metrics["individual_evaluator_times"].items():
                if times:
                    aggregated_timing_metrics[f"ai21_evaluators/individual_time/{evaluator_name}"] = sum(times) / len(
                        times
                    )

        if VERBOSE:
            print(f"[TIMING_AI21_RM] Aggregated timing metrics: {aggregated_timing_metrics}")

        if timing_metrics["excluded_examples_count"] > 0:
            print(
                f"[AI21_RM] Excluded {timing_metrics['excluded_examples_count']}/"
                f"{batch_size} examples due to evaluation errors"
            )

        return aggregated_timing_metrics

    def _merge_timing_metrics(self, main_metrics: dict, batch_metrics: dict):
        """Merge timing metrics from a batch into the main metrics."""
        main_metrics["total_ray_evaluation_time"] += batch_metrics.get("total_ray_evaluation_time", 0)
        main_metrics["total_ray_task_time"] += batch_metrics.get("total_ray_task_time", 0)
        main_metrics["total_evaluations"] += batch_metrics.get("total_evaluations", 0)
        main_metrics["excluded_examples_count"] += batch_metrics.get("excluded_examples_count", 0)
        main_metrics["non_success_count"] += batch_metrics.get("non_success_count", 0)
        main_metrics["timeout_count"] += batch_metrics.get("timeout_count", 0)

        # Merge individual evaluator times
        batch_individual_times = batch_metrics.get("individual_evaluator_times", {})
        for evaluator_name, times in batch_individual_times.items():
            if evaluator_name not in main_metrics["individual_evaluator_times"]:
                main_metrics["individual_evaluator_times"][evaluator_name] = []
            main_metrics["individual_evaluator_times"][evaluator_name].extend(times)

    def _process_single_data_item(self, data_item, item_index: int) -> dict:
        """Process a single data item and extract relevant information."""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        force_thinking = data_item.non_tensor_batch.get("force_thinking", False)
        decode_start_time = time.time()
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True, force_thinking=force_thinking)
        response_str = self.tokenizer.decode(
            valid_response_ids, skip_special_tokens=True, force_thinking=force_thinking
        )
        if self.stop_seq_to_strip:
            _split = response_str.rsplit(self.stop_seq_to_strip, 1)
            response_str = _split[0]
            if VERBOSE and len(_split) > 1:
                print(f"[AI21_RM] Stripped {self.stop_seq_to_strip} from response_str")

        decode_end_time = time.time()
        if VERBOSE and decode_end_time - decode_start_time > TIME_THRESHOLD:
            print(
                f"[TIMING_AI21_RM] Tokenizer decode for item {item_index + 1} took "
                f"{decode_end_time - decode_start_time:.6f}s"
            )

        metadata = data_item.non_tensor_batch.get("extra_info", None)

        # Extract messages from raw_prompt if available, otherwise create from prompt_str
        messages = data_item.non_tensor_batch.get("raw_prompt")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif (isinstance(messages, np.ndarray) and messages.shape[0] == 0) or messages is None:
            # Fallback to creating messages from the decoded prompt
            messages = [{"role": "user", "content": prompt_str}]

        # Validate evaluator configuration before creating the task
        evaluations_src = data_item.non_tensor_batch["reward_model"]
        if isinstance(evaluations_src, np.ndarray):
            evaluations_src = evaluations_src.tolist()
        if len(evaluations_src) == 0:
            raise ValueError(
                f"AI21 data error: example {item_index} has no evaluators configured. "
                f"data_item: {str(evaluations_src)} \n"
            )
        eval_names = []
        for entry in evaluations_src:
            assert isinstance(entry, dict), f"AI21 data error: evaluator entry must be dict, got {type(entry)}"
            name = entry.get("evaluation_id") or entry.get("evaluator_name")
            if name is None:
                raise ValueError(
                    f"AI21 data error: example {item_index} has an evaluator without a name and id: {entry}. "
                    f"data_item: {data_item}"
                )
            eval_names.append(name)
        if len(eval_names) != len(set(eval_names)):
            raise ValueError(
                f"AI21 data error: example {item_index} contains duplicate evaluator names: "
                f"{eval_names}. data_item: {data_item}"
            )

        # create a task from the data_item
        task = self._create_verifiable_task(data_item, messages, metadata, item_index)

        # Handle thinking trace processing
        response_str, nullify_scores = self._process_thinking_trace(response_str, item_index, force_thinking)

        return {
            "task": task,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "metadata": metadata,
            "valid_response_length": valid_response_length,
            "data_source": data_item.non_tensor_batch[self.reward_fn_key],
            "data_item": data_item,
            "nullify_scores": nullify_scores,
        }

    def _create_verifiable_task(self, data_item, messages: list, metadata: Any, item_index: int) -> VerifiableTask:
        return AI21RewardManager.create_verifiable_task(
            self.ai21_evaluators_config_file, data_item, messages, metadata, item_index
        )

    @staticmethod
    def create_verifiable_task(
        ai21_evaluators_config_file: Optional[str], data_item, messages: list, metadata: Any, item_index: int
    ) -> VerifiableTask:
        """Create a VerifiableTask from a data item."""
        task_creation_start = time.time()

        task = VerifiableTask(
            id=data_item.non_tensor_batch["id"],
            messages=messages,
            evaluations=[
                EvaluationEntry(
                    evaluator_name=entry.get("evaluator_name"),
                    evaluator_config=entry.get("evaluator_config", {}),
                    query_args=entry.get("query_args", {}),
                    evaluation_id=entry.get("evaluation_id"),
                )
                for entry in data_item.non_tensor_batch["reward_model"]
            ],
            metadata=metadata,
            aggregation_config=AggregationConfig(**data_item.non_tensor_batch["aggregation_config"]),
        )

        # create a VerifiableTaskDataset with the task to enable the in-place apply of the config file
        one_task_dataset = VerifiableTaskDataset([task])
        task_creation_end = time.time()
        if VERBOSE and task_creation_end - task_creation_start > TIME_THRESHOLD:
            print(
                f"[TIMING_AI21_RM] Task creation for item {item_index + 1} took "
                f"{task_creation_end - task_creation_start:.6f}s"
            )

        # in-place apply of the config file to the task if there is a config file
        if ai21_evaluators_config_file:
            config_start_time = time.time()
            apply_evaluators_config_to_verifiable_dataset(ai21_evaluators_config_file, one_task_dataset)
            config_end_time = time.time()
            if VERBOSE and config_end_time - config_start_time > TIME_THRESHOLD:
                print(
                    f"[TIMING_AI21_RM] Config application for item {item_index + 1} took "
                    f"{config_end_time - config_start_time:.6f}s"
                )

        # override the jlm_model_name in the task
        if data_item.non_tensor_batch["jlm_model_name"] is not None:
            change_verifiable_task_model_name(task, data_item.non_tensor_batch["jlm_model_name"])

        return task

    def _process_thinking_trace(self, response_str: str, item_index: int, force_thinking: bool) -> tuple[str, bool]:
        """Process thinking trace and determine if scores should be nullified."""
        nullify_scores = False
        if force_thinking:
            # in force_thinking the opening <think> comes from the chat template
            # we make sure that the closing </think> exists exactly oncein the response
            if not response_str.count("</think>") == 1:
                # scores should be 0, set all evaluator scores to 0
                nullify_scores = True

        if self.ai21_clean_thinking_trace:
            thinking_clean_start = time.time()
            response_str = response_str.split("</think>")[-1]
            thinking_clean_end = time.time()
            if VERBOSE and thinking_clean_end - thinking_clean_start > TIME_THRESHOLD:
                print(
                    f"[TIMING_AI21_RM] Thinking trace cleaning for item {item_index + 1} took "
                    f"{thinking_clean_end - thinking_clean_start:.6f}s"
                )

        return response_str, nullify_scores

    def _submit_task_for_evaluation(self, task_info: dict) -> ray.ObjectRef:
        """Submit a task for evaluation."""
        one_task_dataset = VerifiableTaskDataset([task_info["task"]])
        if VERBOSE:
            print(f"[AI21_RM] Submitting task for evaluation: {one_task_dataset.examples[0].messages}")
            print(f"[AI21_RM] Response: {task_info['response_str']}")
        return self.compute_score(
            data_source=task_info["data_source"],
            completion=task_info["response_str"],
            task=one_task_dataset.examples[0],
        )

    def _get_ray_task_result(self, future: ray.ObjectRef, item_index: int) -> tuple[float, bool, dict, dict]:
        """Get aggregated result from a Ray task with error handling."""
        try:
            # Get Ray task result - returns (aggregated_score, do_exclude, timing_info, evaluation_info)
            aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info = ray.get(
                future, timeout=self.ai21_evaluators_timeout
            )
            return aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info
        except ray.exceptions.GetTimeoutError:
            print(f"[ERROR_AI21_RM] Timeout in score computation for item {item_index + 1}")
            aggregated_score = 0.0  # Default score for timeout
            do_exclude = True  # Timeout should mark example for exclusion
            self._cancel_future_or_retire(future)
            ray_timing_info = {"timeout_count": 1, "timeout_occurred": True}
            ray_evaluation_info = {"response_details": None}
            return aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info
        except ray.exceptions.ActorDiedError as err:
            # in case we try to get the result after the actor has died,
            # we mark the example for exclusion
            print(
                f"[ERROR_AI21_RM] Evaluator actor died while scoring item {item_index + 1}: {err}. "
                "Marking example for exclusion."
            )
            aggregated_score = 0.0
            do_exclude = True
            self._retire_future_actor(future)
            ray_timing_info = {"actor_died": True}
            ray_evaluation_info = {"response_details": None}
            return aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info

    def _handle_timed_out_future(
        self,
        future: ray.ObjectRef,
        task_idx: int,
        task_info: dict,
        reward_tensor: torch.Tensor,
        reward_extra_info: dict,
        already_print_data_sources: dict,
        timing_metrics: dict,
    ):
        """Handle futures that exceeded the driver-side ray.wait timeout without blocking."""
        self._cancel_future_or_retire(future)
        aggregated_score = 0.0
        do_exclude = True
        ray_timing_info: dict[str, Any] = {"timeout_count": 1, "timeout_occurred": True}
        ray_evaluation_info = {"response_details": None}
        self._process_task_results_internal(
            task_idx,
            task_info,
            aggregated_score,
            do_exclude,
            ray_timing_info,
            ray_evaluation_info,
            reward_tensor,
            reward_extra_info,
            already_print_data_sources,
            timing_metrics,
        )

    def _print_examination_results(
        self, task_info: dict, aggregated_score: float, do_exclude: bool, already_print_data_sources: dict
    ):
        """Print examination results for debugging."""
        data_source = task_info["data_source"]
        if data_source not in already_print_data_sources:
            already_print_data_sources[data_source] = 0

        if already_print_data_sources[data_source] < self.num_examine:
            already_print_data_sources[data_source] += 1
            print("[prompt]", task_info["prompt_str"])
            print("[response]", task_info["response_str"])
            print("[id]", task_info["data_item"].non_tensor_batch["id"])
            print("[evaluations]", task_info["data_item"].non_tensor_batch["reward_model"])
            print("[metadata]", task_info["metadata"])
            print("[do_exclude]", do_exclude)
            print("[aggregated_score]", aggregated_score)

    def _convert_agent_rollout_reward_extra_infos_dict(self, data: DataProto) -> dict:
        timing_metrics = self._initialize_timing_metrics()
        reward_tensor, reward_extra_info = self._initialize_data_structures(data)
        already_print_data_sources: dict[str, int] = {}

        for task_idx in range(len(data)):
            aggregated_score = data.non_tensor_batch["aggregated_score"][task_idx].item()
            do_exclude = data.non_tensor_batch["do_exclude_example"][task_idx].item()
            ray_timing_info = data.non_tensor_batch["timing_metrics"][task_idx]
            ray_evaluation_info = data.non_tensor_batch["reward_extra_info"][task_idx]
            task_info = data.non_tensor_batch["task_info"][task_idx]

            self._process_task_results_internal(
                task_idx,
                task_info,
                aggregated_score,
                do_exclude,
                ray_timing_info,
                ray_evaluation_info,
                reward_tensor,
                reward_extra_info,
                already_print_data_sources,
                timing_metrics,
            )

        effective_batch_size = len(data)
        final_timing_metrics = self._finalize_timing_metrics(timing_metrics, effective_batch_size)
        reward_extra_info["timing_metrics"] = final_timing_metrics

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
