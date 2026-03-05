# Copyright 2025 AI21 Labs
import asyncio
import inspect
import multiprocessing
import os
import time
from typing import Any, Optional

# TODO: temporary setting evaluators process pool on, remove when it's on by default in evaluators side
os.environ.setdefault("AI21_EVALUATORS_SEND_CPU_BOUND_TO_PROCESS_POOL", "True")
os.environ.setdefault("AI21_EVALUATORS_NUM_CPU_BOUND_WORKERS", str(min(int(multiprocessing.cpu_count()), 50)))

import numpy as np
import ray
from ai21_evaluators.file_formats.verifiable_task_dataset import (
    VerifiableTask,
)

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score.ai21.evaluators_integration import AI21EvaluatorComputeScoreFactory
from verl.workers.reward_manager.ai21 import AI21RewardManager

# Use the same environment variable as evaluators_integration.py
VERBOSE = os.environ.get("AI21_VERL_EVALUATORS_VERBOSE", "False").lower() == "true"
TIME_THRESHOLD = 2.0  # Threshold for timing logs
RAY_CANCEL_GRACE_TIMEOUT = 0.5  # Seconds to wait for cooperative cancellation


@register("ai21")
class AI21RewardLoopManager(RewardLoopManagerBase):
    """AI21 reward loop manager using ai21-evaluators for single data item processing."""

    def __init__(
        self,
        config,
        tokenizer,
        reward_fn,
        reward_router_address,
        reward_model_tokenizer,
        compute_score=None,
        reward_fn_key="data_source",
        **kwargs,
    ):
        super().__init__(config, tokenizer)

        # AI21-specific configuration
        self.ai21_evaluators_config_file = config.reward_model.get("ai21_evaluators_config_file", None)
        self.ai21_evaluators_timeout = config.reward_model.get("ai21_evaluators_timeout", 60.0)
        self.ai21_clean_thinking_trace = config.reward_model.get("ai21_clean_thinking_trace", False)
        self.ai21_enforce_thinking_trace = config.reward_model.get("ai21_enforce_thinking_trace", False)
        self.ai21_evaluators_actors_amount = config.reward_model.get("ai21_evaluators_actors_amount")
        self.ai21_evaluators_actor_cpus = config.reward_model.get("ai21_evaluators_actor_cpus", 1)
        self.ai21_evaluators_spread_per_node = config.reward_model.get("ai21_evaluators_spread_per_node", True)
        self.stop_seq_to_strip = config.reward_model.get("stop_seq_to_strip")  # TODO AI21: Is this the right path?
        self.ai21_evaluators_format_reward = config.reward_model.get("ai21_evaluators_format_reward", 0.0)
        self.reward_fn_key = reward_fn_key

        # Setup compute score function
        self._setup_compute_score(compute_score)

        # Log parameters
        if VERBOSE:
            print("[AI21_RLM] AI21RewardLoopManager initialized with parameters:")
            print(f"[AI21_RLM] ai21_evaluators_config_file: {self.ai21_evaluators_config_file}")
            print(f"[AI21_RLM] ai21_evaluators_timeout: {self.ai21_evaluators_timeout}")
            print(f"[AI21_RLM] stop_seq_to_strip: {self.stop_seq_to_strip}")

    def _setup_compute_score(self, compute_score: Optional[Any]):
        """Initialize the compute score function and factory."""
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

        self.is_async_compute_score = inspect.iscoroutinefunction(self.compute_score)

    def _release_future_actor(self, future: Optional[ray.ObjectRef]) -> None:
        if future is None or self.evaluator_factory is None:
            return
        self.evaluator_factory.release_future_actor(future)

    def _retire_future_actor(self, future: Optional[ray.ObjectRef]) -> None:
        if future is None or self.evaluator_factory is None:
            return
        self.evaluator_factory.retire_actor_for_future(future)

    def _cancel_future_or_retire(self, future: Optional[ray.ObjectRef]) -> None:
        if future is None:
            return

        cancelled = False
        try:
            ray.cancel(future)
            ready, _ = ray.wait([future], timeout=RAY_CANCEL_GRACE_TIMEOUT)
            cancelled = len(ready) > 0
        except Exception:
            cancelled = False

        if cancelled:
            self._release_future_actor(future)
        else:
            self._retire_future_actor(future)

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item and return reward score and extra info."""
        assert len(data) == 1, "AI21RewardLoopManager only supports single data item processing"

        start_time = time.time()
        if VERBOSE:
            print(f"[TIMING_AI21_RLM] Starting AI21RewardLoopManager.run_single at {start_time:.6f}")

        data_item = data[0]

        # Process the data item
        task_info = await self._process_single_data_item(data_item)
        # Handle nullified scores (e.g., invalid thinking traces)
        if task_info["nullify_scores"]:
            if VERBOSE:
                print("[AI21_RLM] Nullifying scores for this example")
            return {
                "reward_score": 0.0,
                "reward_extra_info": {
                    "aggregated_score": 0.0,
                    "do_exclude_example": False,  # Nullified scores are not excluded, just scored as 0
                    "response_details": None,
                    "nullified": True,
                },
            }

        # Submit task for evaluation
        future: Optional[ray.ObjectRef] = None
        try:
            if self.is_async_compute_score:
                future = await self.compute_score(
                    data_source=task_info["data_source"],
                    completion=task_info["response_str"],
                    task=task_info["task"],
                )
            else:
                future = await self.loop.run_in_executor(
                    None,
                    lambda: self.compute_score(
                        data_source=task_info["data_source"],
                        completion=task_info["response_str"],
                        task=task_info["task"],
                    ),
                )

            # Unpack result from AI21 evaluators
            aggregated_score, do_exclude, timing_info, evaluation_info = await self._get_ray_task_result(future)
            self._release_future_actor(future)

        except Exception as e:
            print(f"[ERROR_AI21_RLM] Error in score computation: {e}")
            self._cancel_future_or_retire(future)
            aggregated_score = 0.0
            do_exclude = True
            timing_info = {}
            evaluation_info = {"response_details": None}

        # Prepare reward extra info
        reward_extra_info = {
            "aggregated_score": aggregated_score,
            "do_exclude_example": do_exclude,
            "response_details": evaluation_info.get("response_details"),
            "task_info": task_info,
            "id": task_info["task"].id,
        }

        # Add timing metrics if available
        if timing_info:
            reward_extra_info["timing_metrics"] = timing_info
        else:
            reward_extra_info["timing_metrics"] = {}
        end_time = time.time()
        total_time = end_time - start_time

        if VERBOSE and total_time > TIME_THRESHOLD:
            print(f"[TIMING_AI21_RLM] AI21RewardLoopManager.run_single completed in {total_time:.6f}s")

        return {
            "reward_score": aggregated_score,
            "reward_extra_info": reward_extra_info,
        }

    async def _process_single_data_item(self, data_item) -> dict:
        """Process a single data item and extract relevant information."""
        # Extract prompt and response information
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode prompt and response
        force_thinking = data_item.non_tensor_batch.get("force_thinking", False)

        decode_start_time = time.time()
        prompt_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True, force_thinking=force_thinking),
        )
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True, force_thinking=force_thinking),
        )

        # Strip stop sequence if configured
        if self.stop_seq_to_strip:
            _split = response_str.rsplit(self.stop_seq_to_strip, 1)
            response_str = _split[0]
            if VERBOSE and len(_split) > 1:
                print(f"[AI21_RLM] Stripped {self.stop_seq_to_strip} from response_str")

        decode_end_time = time.time()
        if VERBOSE and decode_end_time - decode_start_time > TIME_THRESHOLD:
            print(f"[TIMING_AI21_RLM] Tokenizer decode took {decode_end_time - decode_start_time:.6f}s")

        # Extract metadata and messages
        metadata = data_item.non_tensor_batch.get("extra_info", None)

        # Extract messages from raw_prompt if available, otherwise create from prompt_str
        messages = data_item.non_tensor_batch.get("raw_prompt")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif (isinstance(messages, np.ndarray) and messages.shape[0] == 0) or messages is None:
            # Fallback to creating messages from the decoded prompt
            messages = [{"role": "user", "content": prompt_str}]

        # Validate evaluator configuration
        evaluations_src = data_item.non_tensor_batch["reward_model"]
        if isinstance(evaluations_src, np.ndarray):
            evaluations_src = evaluations_src.tolist()
        if len(evaluations_src) == 0:
            raise ValueError("AI21 data error: example has no evaluators configured")

        eval_names = []
        for entry in evaluations_src:
            assert isinstance(entry, dict), f"AI21 data error: evaluator entry must be dict, got {type(entry)}"
            name = entry.get("evaluation_id") or entry.get("evaluator_name")
            if name is None:
                raise ValueError(f"AI21 data error: evaluator without a name and id: {entry}")
            eval_names.append(name)

        if len(eval_names) != len(set(eval_names)):
            raise ValueError(f"AI21 data error: duplicate evaluator names: {eval_names}")

        # Create verifiable task
        task = self._create_verifiable_task(data_item, messages, metadata)

        # Handle thinking trace processing
        response_str, nullify_scores = self._process_thinking_trace(response_str, force_thinking)

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

    def _create_verifiable_task(self, data_item, messages: list, metadata: Any) -> VerifiableTask:
        """Create a VerifiableTask from a data item."""
        item_index = 0
        return AI21RewardManager.create_verifiable_task(
            self.ai21_evaluators_config_file, data_item, messages, metadata, item_index
        )

    def _process_thinking_trace(self, response_str: str, force_thinking: bool) -> tuple[str, bool]:
        """Process thinking trace and determine if scores should be nullified."""
        nullify_scores = False

        if force_thinking:
            # In force_thinking the opening <think> comes from the chat template
            # We make sure that the closing </think> exists exactly once in the response
            if not response_str.count("</think>") == 1:
                # Scores should be 0, set all evaluator scores to 0
                nullify_scores = True

        if self.ai21_clean_thinking_trace:
            thinking_clean_start = time.time()
            response_str = response_str.split("</think>")[-1]
            thinking_clean_end = time.time()
            if VERBOSE and thinking_clean_end - thinking_clean_start > TIME_THRESHOLD:
                print(
                    f"[TIMING_AI21_RLM] Thinking trace cleaning took {thinking_clean_end - thinking_clean_start:.6f}s"
                )

        return response_str, nullify_scores

    async def _get_ray_task_result(self, future: ray.ObjectRef) -> tuple[float, bool, dict, dict]:
        """Get aggregated result from a Ray task with error handling."""
        try:
            ray_timeout = self.ai21_evaluators_timeout + 5.0
            # Get Ray task result - returns (aggregated_score, do_exclude, timing_info, evaluation_info)
            aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info = await asyncio.wait_for(
                future, timeout=ray_timeout
            )
            return aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info
        except (ray.exceptions.GetTimeoutError, asyncio.TimeoutError):
            print("[ERROR_AI21_RLM] Timeout in score computation")
            aggregated_score = 0.0  # Default score for timeout
            do_exclude = True  # Timeout should mark example for exclusion
            self._cancel_future_or_retire(future)
            ray_timing_info = {"timeout_count": 1, "timeout_occurred": True}
            ray_evaluation_info = {"response_details": None}
            return aggregated_score, do_exclude, ray_timing_info, ray_evaluation_info
