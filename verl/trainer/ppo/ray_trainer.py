# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.ai21_utils import (
    convert_numpy_to_list,
    convert_reward_extra_infos_dict_ai21_reward_manager,
    ensure_concat_compatibility,
    filter_ai21_excluded_examples,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.checkpoint.file_utils import check_file_exists_gs, sync_file_gs, sync_file_gs_threaded
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import threaded
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    last_total_available_gpus: Optional[int] = None

    def _wait_for_resource_available(self, timeout_s: int, poll_interval_s: float = 5.0):
        deadline = time.time() + timeout_s
        while True:
            try:
                self._check_resource_available()
                break
            except ValueError as e:
                # If we've exceeded the timeout, re-raise the original error
                now = time.time()
                if now >= deadline:
                    raise e

                time.sleep(poll_interval_s)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        # Allow some time for Ray workers/nodes to register before enforcing the resource check.
        # Reuse the existing Ray worker registration timeout env var if set; otherwise default to 600s.
        timeout_s = 2 * int(os.environ.get("RAY_worker_register_timeout_seconds") or 0)
        if timeout_s > 0:
            self._wait_for_resource_available(timeout_s)
        else:
            self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            if self.last_total_available_gpus != total_available_gpus:
                print(f"[resource-check] GPUs available: {int(total_available_gpus)}/{int(total_required_gpus)}. ")
            self.last_total_available_gpus = total_available_gpus
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, config: AlgoConfig) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        config (AlgoConfig): Configuration for algorithm settings. Defaults to None.
    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    adv_estimator = config.adv_estimator
    gamma = config.gamma
    lam = config.lam
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # Check if we have multiple evaluator rewards
    has_multi_evaluator_rewards = "evaluator_reward_tensors" in data.batch
    if adv_estimator != AdvantageEstimator.GRPO and has_multi_evaluator_rewards:
        raise ValueError(f"Multi-evaluator rewards not supported with {adv_estimator} advantage estimator")

    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.PASS_AT_K:
        # AI21 implementation of Pass@k
        advantages, returns = core_algos.compute_pass_at_k_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            K=config.algorithm.pass_at_k.get("K"),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_UNLIKELY:
        advantages, returns = core_algos.compute_grpo_unlikely_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            old_log_probs=data.batch["old_log_probs"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            beta_rank=config.algorithm.grpo_unlikely.get("beta_rank", 0.25),
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            log_probs_agg_method=config.algorithm.get("log_probs_agg_method", "sum"),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        # Deliberately not deleting rollouts from previous runs if they exist to avoid various edge cases
        # This is a memory optimization, so the possible side effects aren't worth the risk
        # Most old rollouts should be overridden by new ones, so this is transient and capped by the number
        # of CP save frequency + number of restarts.
        self.saved_rollout_paths = []

    def _print_amalgam_stats(self):
        """Print statistics about the amalgam dataset."""
        print("Training Amalgam Dataset stats:")
        for stat_dict in self.train_dataset.get_dataset_stats():
            print("  " + ", ".join(f"{k.title()}: {v}" for k, v in stat_dict.items()))

        print("Validation Amalgam Dataset stats:")
        for stat_dict in self.val_dataset.get_dataset_stats():
            print("  " + ", ".join(f"{k.title()}: {v}" for k, v in stat_dict.items()))

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        is_amalgam = self.config.data.custom_cls.name == "AmalgamDataset"
        if is_amalgam:
            # Amalgam dataset is an infinite dataset, so we need to set total_training_steps instead of total_epochs
            assert self.config.trainer.total_training_steps is not None, (
                "total_training_steps must be set for amalgam dataset"
            )
            assert self.config.trainer.total_epochs == 1, (
                "total_epochs must be 1 for amalgam dataset as it is an infinite dataset, "
                "set total_training_steps instead"
            )

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
                **({"is_train": True} if is_amalgam else {}),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
                **({"is_train": False} if is_amalgam else {}),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None and not is_amalgam:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"] if not is_amalgam else 0

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=not is_amalgam,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            # Note: we are ignoring val_batch_size from config, as batching is not desired during inference
            # TODO: consider the design of single controller with a large val dataset in multi-modal scenarios
            # may lead to oom issues
            if self.config.trainer.val_examples_limit is not None:
                val_batch_size = self.config.trainer.val_examples_limit
            else:
                val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True) if not is_amalgam else False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        total_training_steps = None
        if is_amalgam:
            self._print_amalgam_stats()
        else:
            assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )

            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        assert total_training_steps is not None, "total_training_steps must be set"

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _unpack_reward_result(self, reward_result, batch, metrics):
        # unpacks reward_result into batch.batch["token_level_scores"] and batch.non_tensor_batch
        if isinstance(reward_result, tuple) and len(reward_result) == 2:
            # Standard case: (reward_tensor, reward_extra_infos_dict)
            reward_tensor, reward_extra_infos_dict = reward_result
            batch.batch["token_level_scores"] = reward_tensor
            if "timing_metrics" in reward_extra_infos_dict and self.config.reward_model.reward_manager == "ai21":
                ai21_timing_metrics = reward_extra_infos_dict.pop("timing_metrics")
                metrics.update(ai21_timing_metrics)
        elif isinstance(reward_result, torch.Tensor):
            # Fallback for single tensor result
            reward_tensor = reward_result
            reward_extra_infos_dict = {}
        else:
            raise ValueError(f"Unexpected reward_result format: {type(reward_result)}")

        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            print(f"{list(reward_extra_infos_dict.keys())=}")
            if self.config.reward_model.reward_manager == "ai21":
                # Handle special cases to ensure all fields are indexable
                batch_updates = convert_reward_extra_infos_dict_ai21_reward_manager(reward_extra_infos_dict, batch)
                batch.non_tensor_batch.update(batch_updates)
            else:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    def _dump_generations(
        self,
        dump_path,
        inputs,
        inputs_messages,
        outputs,
        outputs_with_special_tokens,
        gts,
        scores,
        reward_extra_infos_dict,
        train_token_ids=None,
        response_logprobs=None,
        advantages_mean=None,
        metadata=None,
        rollout_logprobs=None,
        length_rewards=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        if (
            self.config.trainer.generations_save_freq < 0
            or self.global_steps % self.config.trainer.generations_save_freq != 0
        ):
            return

        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "input_messages": inputs_messages,
            "output_with_special_tokens": outputs_with_special_tokens,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
            "metadata": metadata,
        }
        if gts:
            base_data["gts"] = gts
        if train_token_ids is not None:
            base_data["train_token_ids"] = train_token_ids
        if response_logprobs is not None:
            base_data["response_logprobs"] = response_logprobs
        if advantages_mean is not None:
            base_data["advantage_mean"] = advantages_mean
        if rollout_logprobs is not None:
            base_data["rollout_logprobs"] = rollout_logprobs
        if length_rewards is not None:
            base_data["length_reward"] = length_rewards
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[f"reward_extra_info_{k}"] = v

        @threaded
        def local_dump_and_remote_sync(filename, n, base_data, remote_save_path):
            lines = []
            for i in range(n):
                entry = {k: convert_numpy_to_list(v[i]) for k, v in base_data.items()}
                lines.append(json.dumps(entry, ensure_ascii=False))

            with open(filename, "w") as f:
                f.write("\n".join(lines) + "\n")
            self.saved_rollout_paths.append(filename)

            if remote_save_path:
                remote_path = os.path.join(
                    remote_save_path, os.path.dirname(dump_path).split("/")[-1], os.path.basename(dump_path)
                )
                sync_file_gs(filename, f"{remote_path}/{os.path.basename(filename)}")

            max_rollouts_to_keep = self.config.trainer.max_rollouts_to_keep
            if max_rollouts_to_keep is not None:
                while len(self.saved_rollout_paths) > max_rollouts_to_keep:
                    path_to_remove = self.saved_rollout_paths.pop(0)
                    # There is a possible race condition here - the rollout can be removed before it is synced to remote
                    # but this is highly unlikely (rollout N+1 needs to be saved before rollout N is synced to remote)
                    if os.path.exists(path_to_remove):
                        os.remove(path_to_remove)

        print(f"Dumping generations to {filename} asynchronously")
        local_dump_and_remote_sync(filename, n, base_data, self.config.trainer.remote_save_path)

    def _log_rollout_data(self, batch: DataProto, timing_raw: dict, rollout_data_dir: str):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            if "reward_model" in batch.non_tensor_batch and isinstance(batch.non_tensor_batch["reward_model"][0], dict):
                sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            else:
                # TODO AI21: handle also for ai21 reward manager
                sample_gts = None

            print(f"{batch.batch.keys()=}")
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            inputs_messages = batch.non_tensor_batch.get("raw_prompt", np.array([None] * len(inputs))).tolist()
            length_rewards = batch.non_tensor_batch.get("length_rewards", np.array([None] * len(inputs))).tolist()
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            outputs_with_special_tokens = [
                self.tokenizer.decode(batch.non_tensor_batch["unpadded_responses"][i], skip_special_tokens=False)
                for i in range(len(inputs))
            ]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            train_token_ids = [
                batch.batch["input_ids"][i][batch.batch["attention_mask"][i] == 1].tolist() for i in range(len(inputs))
            ]
            response_logprobs = [
                batch.batch["old_log_probs"][i][batch.batch["response_mask"][i] == 1].tolist()
                for i in range(len(inputs))
            ]
            rollout_logprobs = [
                batch.batch["rollout_log_probs"][i][batch.batch["response_mask"][i] == 1].tolist()
                for i in range(len(inputs))
            ]
            advantages_mean = (
                (batch.batch["advantages"] * batch.batch["response_mask"]).sum(dim=1)
                / batch.batch["response_mask"].sum(dim=1)
            ).tolist()

            metadata = [{"reward_model": reward_model} for reward_model in batch.non_tensor_batch["reward_model"]]
            for key in ["id", "data_source", "jlm_model_name"]:
                if key in batch.non_tensor_batch.keys():
                    for metadata_item, value in zip(metadata, batch.non_tensor_batch[key], strict=False):
                        metadata_item[key] = value
            reward_extra_infos_dict_for_dump = {}
            for key in ["do_exclude_example", "aggregated_score", "response_details"]:
                if key in batch.non_tensor_batch:
                    reward_extra_infos_dict_for_dump[key] = batch.non_tensor_batch[key].tolist()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict_for_dump["request_id"] = batch.non_tensor_batch["request_id"].tolist()
            with marked_timer("dump_rollout_generations_dump_generations", timing_raw):
                self._dump_generations(
                    dump_path=rollout_data_dir,
                    inputs=inputs,
                    inputs_messages=inputs_messages,
                    outputs=outputs,
                    outputs_with_special_tokens=outputs_with_special_tokens,
                    gts=sample_gts,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict_for_dump,
                    train_token_ids=train_token_ids,
                    response_logprobs=response_logprobs,
                    advantages_mean=advantages_mean,
                    metadata=metadata,
                    length_rewards=length_rewards,
                    rollout_logprobs=rollout_logprobs,
                )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_inputs_messages = []
        sample_outputs = []
        sample_outputs_with_special_tokens = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        aggregated_rm_metadata = []
        sample_uids = []

        examples_counter = 0
        print("Starting validation loop...")
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            batch_size = len(test_batch)

            # Check if we should process this batch based on val_examples_limit
            if self.config.trainer.val_examples_limit is not None:
                if examples_counter >= self.config.trainer.val_examples_limit:
                    print(
                        "Stopping validation: examples_counter="
                        f"{examples_counter}, "
                        f"batch_size={batch_size}, "
                        "limit="
                        f"{self.config.trainer.val_examples_limit}"
                    )
                    break
                else:
                    val_set_size = (
                        len(self.val_dataset)
                        if hasattr(self.config.data, "val_datasources_finite")
                        and self.config.data.val_datasources_finite
                        else None
                    )
                    remaining_limit = self.config.trainer.val_examples_limit - examples_counter
                    remaining_limit = (
                        min(remaining_limit, val_set_size) if val_set_size is not None else remaining_limit
                    )
                    if (
                        remaining_limit > 0
                        and remaining_limit < batch_size
                        and not (self.config.data.val_datasources_finite and examples_counter > 0)
                    ):
                        # Truncate the batch to fit within the limit using slicing
                        test_batch = test_batch[:remaining_limit]
                        batch_size = len(test_batch)
                    elif remaining_limit <= 0:
                        # No more examples allowed, skip this batch
                        break

            # in case of amalgam dataset, we need to limit the number of examples
            examples_counter += batch_size
            print(f"Processing validation batch: batch_size={batch_size}, total_examples={examples_counter}")

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            input_messages = test_batch.non_tensor_batch.get("raw_prompt", np.array([None] * len(input_ids))).tolist()
            sample_inputs_messages.extend(input_messages)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            if "reward_model" in test_batch.non_tensor_batch and isinstance(
                test_batch.non_tensor_batch["reward_model"][0], dict
            ):
                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
                ]
            else:
                # TODO AI21: handle also for ai21 reward manager
                ground_truths = None
            if ground_truths is not None:
                sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            output_texts_with_special_tokens = [
                self.tokenizer.decode(ids, skip_special_tokens=False) for ids in output_ids
            ]
            sample_outputs_with_special_tokens.extend(output_texts_with_special_tokens)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            aggregated_rm_metadata.extend(test_batch.non_tensor_batch["reward_model"])

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)

            if self.config.reward_model.reward_manager == "ai21":
                # AI21 format: aggregated reward tensor (aggregation happens in ai21-evaluators)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                del result["reward_extra_info"]["timing_metrics"]

                # Add aggregated scores to reward_extra_infos_dict for detailed analysis
                aggregated_scores = result["reward_extra_info"].get("aggregated_score", [])
                reward_extra_infos_dict["aggregated_score_reward"].extend(aggregated_scores)
            else:
                # Original format: single reward tensor
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()

            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        print("Validation loop completed.")

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        metadata = [
            {"reward_model": aggregated_rm_metadata[i], "data_source": data_sources[i]}
            for i in range(len(sample_scores))
        ]
        if val_data_dir:
            self._dump_generations(
                dump_path=val_data_dir,
                inputs=sample_inputs,
                inputs_messages=sample_inputs_messages,
                outputs=sample_outputs,
                outputs_with_special_tokens=sample_outputs_with_special_tokens,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                metadata=metadata,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            if self.config.reward_model.reward_manager == "ai21" and key_info in ["do_exclude_example"]:
                print(
                    f"[VALIDATION_INFO] AI21 per-task field '{key_info}': "
                    f"{len(lst)} unique tasks, {len(sample_scores)} repeated samples"
                )
                # This is expected behavior - skip validation for per-task fields
                continue
            else:
                assert len(lst) == 0 or len(lst) == len(sample_scores), (
                    f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                )

        # Handle case when no validation batches were processed
        if not data_source_lst:
            print("Warning: No validation batches were processed. Returning empty metrics.")
            return {}

        # Filter out non-metric fields before processing validation metrics
        metrics_dict = {}
        for key_info, lst in reward_extra_infos_dict.items():
            # Skip fields that are not per-sample metrics
            if self.config.reward_model.reward_manager == "ai21" and key_info in [
                "do_exclude_example",
                "evaluator_usage",
                "response_details",
            ]:
                continue
            # Only include fields that have the correct length for per-sample metrics
            if len(lst) == len(sample_scores):
                metrics_dict[key_info] = lst
            else:
                print(
                    f"[VALIDATION_WARNING] Skipping field '{key_info}' due to length mismatch: "
                    f"{len(lst)} vs {len(sample_scores)}"
                )

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, metrics_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            print("Init critic model")
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            print("Init ref model")
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            print("Init reward model")
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        print("Init actor_rollout model")
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model(can_delete_source_model=True)

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"checkpoints/global_step_{self.global_steps}"
        )

        remote_gcs_global_step_folder = (
            None
            if self.config.trainer.remote_save_path is None
            else os.path.join(self.config.trainer.remote_save_path, f"checkpoints/global_step_{self.global_steps:05}")
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        print(f"remote_gcs_global_step_folder: {remote_gcs_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"checkpoints/global_step_{self.global_steps:05}", "actor"
            )
        )
        actor_remote_gcs_path = (
            None
            if self.config.trainer.remote_save_path is None
            else os.path.join(remote_gcs_global_step_folder, "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
            remote_path=actor_remote_gcs_path,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"checkpoints/global_step_{self.global_steps:05}",
                    str(Role.Critic),
                )
            )
            critic_remote_gcs_path = (
                None
                if self.config.trainer.remote_save_path is None
                else os.path.join(remote_gcs_global_step_folder, "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
                remote_path=critic_remote_gcs_path,
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        if self.config.trainer.remote_save_path:
            dataloader_remote_gcs_path = os.path.join(remote_gcs_global_step_folder, "data.pt")
            sync_file_gs_threaded(dataloader_local_path, dataloader_remote_gcs_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "checkpoints/latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
        if self.config.trainer.remote_save_path:
            remote_gcs_latest_checkpointed_iteration = os.path.join(
                self.config.trainer.remote_save_path, "checkpoints/latest_checkpointed_iteration.txt"
            )
            sync_file_gs_threaded(local_latest_checkpointed_iteration, remote_gcs_latest_checkpointed_iteration)

    def _setup_remote_global_step_path(self) -> str | None:
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "checkpoints/latest_checkpointed_iteration.txt"
        )
        os.makedirs(os.path.dirname(local_latest_checkpointed_iteration), exist_ok=True)

        if self.config.trainer.remote_save_path:
            remote_gcs_latest_checkpointed_iteration = os.path.join(
                self.config.trainer.remote_save_path, "checkpoints/latest_checkpointed_iteration.txt"
            )
            if check_file_exists_gs(remote_gcs_latest_checkpointed_iteration):
                sync_file_gs(remote_gcs_latest_checkpointed_iteration, local_latest_checkpointed_iteration)
                with open(local_latest_checkpointed_iteration) as f:
                    step = int(f.read())
                global_step_subfolder = f"checkpoints/global_step_{step:05}"
                os.makedirs(os.path.join(self.config.trainer.default_local_dir, global_step_subfolder), exist_ok=True)
                return os.path.join(self.config.trainer.remote_save_path, global_step_subfolder)

        if self.config.trainer.remote_load_path:
            assert self.config.trainer.remote_load_step is not None, (
                "remote_load_step must be set if remote_load_path is set"
            )
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(self.config.trainer.remote_load_step))
            global_step_subfolder = f"checkpoints/global_step_{self.config.trainer.remote_load_step:05}"
            os.makedirs(os.path.join(self.config.trainer.default_local_dir, global_step_subfolder), exist_ok=True)
            return os.path.join(self.config.trainer.remote_load_path, global_step_subfolder)

        return None

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            remote_global_step_path = self._setup_remote_global_step_path()
            local_checkpoints_folder = os.path.join(self.config.trainer.default_local_dir, "checkpoints")
            if not os.path.isabs(local_checkpoints_folder):
                working_dir = os.getcwd()
                local_checkpoints_folder = os.path.join(working_dir, local_checkpoints_folder)
            global_step_folder = find_latest_ckpt_path(
                local_checkpoints_folder, "global_step_{:05}"
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "checkpoints/global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("checkpoints/global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, str(Role.Actor))
        remote_actor_path = (
            None if remote_global_step_path is None else os.path.join(remote_global_step_path, str(Role.Actor))
        )
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        remote_critic_path = (
            None if remote_global_step_path is None else os.path.join(remote_global_step_path, str(Role.Critic))
        )
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            remote_path=remote_actor_path,
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                remote_path=remote_critic_path,
            )

        # load dataloader
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if remote_global_step_path is not None:
            dataloader_remote_gcs_path = os.path.join(remote_global_step_path, "data.pt")
            sync_file_gs(dataloader_remote_gcs_path, dataloader_local_path)
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _filter_groups(self, batch: DataProto) -> tuple[DataProto, int, DataProto]:
        orig_non_tensor_keys = list(batch.non_tensor_batch.keys())
        metric_name = self.config.algorithm.filter_groups.metric
        if metric_name == "seq_final_reward":
            # Turn to numpy for easier filtering
            batch.non_tensor_batch["seq_final_reward"] = batch.batch["token_level_rewards"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            batch.non_tensor_batch["seq_reward"] = batch.batch["token_level_scores"].sum(dim=-1).numpy()

        # Collect the sequence reward for each trajectory
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=False):
            prompt_uid2metric_vals[uid].append(metric_val)

        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        # Keep if not all the rewards are the same (or if there was only one trajectory for that prompt)
        kept_prompt_uids = [
            uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]

        kept_traj_idxs = []
        filtered_out_idxs = []
        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)
            else:
                filtered_out_idxs.append(idx)

        easy_prompt_ids = []
        if self.config.algorithm.filter_groups.num_times_to_snooze_easy_examples:
            # "uid" is a unique id for a prompt in one specific batch, it is not preserved across batches
            # We need to use "id" to snooze filtered examples

            batch_for_snoozing = batch[filtered_out_idxs]
            # Sanity: We must have "id" in non_tensor_batch to snooze filtered examples
            if "id" not in batch_for_snoozing.non_tensor_batch:
                raise ValueError("id is required for snoozing filtered examples")
            # Sanity: Make sure that all examples with the same uid have the same id
            uid2promptid = {}
            for uid, prompt_id in zip(
                batch_for_snoozing.non_tensor_batch["uid"], batch_for_snoozing.non_tensor_batch["id"], strict=False
            ):
                if uid in uid2promptid and uid2promptid[uid] != prompt_id:
                    raise ValueError(
                        f"all examples with the same uid must have the same id. uid: {uid}, id: {prompt_id}"
                    )
                uid2promptid[uid] = prompt_id
            for prompt_uid, prompt_id in uid2promptid.items():
                metric_vals = prompt_uid2metric_vals[prompt_uid]
                metric_mean = np.mean(metric_vals)
                is_all_success = metric_mean >= self.config.algorithm.filter_groups.snooze_mean_score_threshold
                if is_all_success:
                    prompt_id = uid2promptid[prompt_uid]
                    easy_prompt_ids.append(prompt_id)

        batch = batch.select(non_tensor_batch_keys=orig_non_tensor_keys)
        filtered_out_batch = batch[filtered_out_idxs]
        batch = batch[kept_traj_idxs]

        # Track skipped examples per data_source
        if len(filtered_out_batch) > 0:
            skipped_data_sources = filtered_out_batch.non_tensor_batch.get(
                "data_source", np.array(["unknown"] * len(filtered_out_batch))
            )
            # Get unique UIDs to count prompts (not trajectories)
            skipped_uids = filtered_out_batch.non_tensor_batch.get("uid", np.array(range(len(filtered_out_batch))))
            uid_to_data_source = {}
            for uid, ds in zip(skipped_uids, skipped_data_sources, strict=False):
                if uid not in uid_to_data_source:
                    uid_to_data_source[uid] = ds
            skipped_dataset_counts = Counter(uid_to_data_source.values())
        else:
            skipped_dataset_counts = {}
        return batch, len(kept_prompt_uids), filtered_out_batch, easy_prompt_ids, skipped_dataset_counts

    def _snooze_easy_examples(self, easy_prompt_ids: list[str]):
        if self.config.algorithm.filter_groups.num_times_to_snooze_easy_examples == 0:
            return

        for easy_prompt_id in easy_prompt_ids:
            self.train_dataset.snooze_example(
                easy_prompt_id, self.config.algorithm.filter_groups.num_times_to_snooze_easy_examples
            )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        timing_raw = defaultdict(float)
        batch = None
        unfiltered_rollouts_batch = None
        leftover_rollouts_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        step_processed_examples = defaultdict(int)
        total_processed_examples = defaultdict(int)
        step_skipped_examples = defaultdict(int)
        total_skipped_examples = defaultdict(int)
        step_easy_prompts = 0
        encountered_data_sources = set()

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                num_gen_prompts = len(new_batch.batch)

                # add uid to batch
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            if not self.config.reward_model.launch_reward_fn_async:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences_keep_rollout_mode_active(
                                    gen_batch_output
                                )
                            else:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if not self.config.reward_model.launch_reward_fn_async:
                                gen_batch_output = (
                                    self.async_rollout_manager.generate_sequences_keep_rollout_mode_active(
                                        gen_batch_output
                                    )
                                )
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Track processed examples per data_source
                    data_sources_in_batch = new_batch.non_tensor_batch.get(
                        "data_source", np.array(["unknown"] * num_gen_prompts)
                    )
                    for data_source in data_sources_in_batch:
                        encountered_data_sources.add(data_source)
                        step_processed_examples[data_source] += 1
                        total_processed_examples[data_source] += 1

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_result = None
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                            if self.config.algorithm.filter_groups.enable:
                                reward_result = ray.get(future_reward)
                        else:
                            reward_result = compute_reward(new_batch, self.reward_fn)

                        self._unpack_reward_result(reward_result, new_batch, metrics)

                        unfiltered_rollouts_batch = (
                            DataProto.concat([unfiltered_rollouts_batch, new_batch])
                            if unfiltered_rollouts_batch is not None
                            else new_batch.deepcopy()
                        )

                    ai21_exclusion_enabled = (
                        self.config.reward_model.reward_manager == "ai21"
                        and self.config.algorithm.filter_groups.filter_ai21_eval_errors
                    )

                    # the filters_group logic is adapted from RayDAPOTrainer
                    if not self.config.algorithm.filter_groups.enable:
                        # When filter_groups is disabled we no longer allow AI21 exclusion path.
                        # Use batch as-is (no filtering) for this branch.
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        # Apply AI21 filtering first if enabled (before group filtering)
                        if ai21_exclusion_enabled:
                            new_batch = filter_ai21_excluded_examples(self.config, new_batch)

                        # Apply group filtering
                        (
                            new_batch,
                            num_prompts_in_new_batch,
                            filtered_out_batch,
                            easy_prompt_ids,
                            skipped_dataset_counts,
                        ) = self._filter_groups(new_batch)

                        step_easy_prompts += len(easy_prompt_ids)
                        self._snooze_easy_examples(easy_prompt_ids)

                        for data_source, count in skipped_dataset_counts.items():
                            step_skipped_examples[data_source] += count
                            total_skipped_examples[data_source] += count

                        if self.config.reward_model.reward_manager == "ai21":
                            batch, new_batch = ensure_concat_compatibility(batch, new_batch)
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        num_prompt_in_batch += num_prompts_in_new_batch

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(
                                f"[FILTER_GROUPS] {num_prompt_in_batch=} < {prompt_bsz=} "
                                "after both AI21 and group filtering"
                            )
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"[FILTER_GROUPS] {num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                if self.config.algorithm.filter_groups.strict:
                                    raise ValueError(
                                        f"[FILTER_GROUPS] {num_gen_batches=} >= {max_num_gen_batches=}. "
                                        "Generated too many. Please check if your data are too difficult. "
                                        "You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                                else:
                                    print(
                                        "Warning: filter_groups: didn't succeed to generate enough filtered "
                                        f"groups in {max_num_gen_batches=}. Padding the batch with filtered-out groups."
                                    )
                                    if self.config.reward_model.reward_manager == "ai21":
                                        batch, filtered_out_batch = ensure_concat_compatibility(
                                            batch, filtered_out_batch
                                        )
                                    batch = DataProto.concat([batch, filtered_out_batch])

                        # Align the batch
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = _sorted_by_group(batch)
                        if self.config.algorithm.filter_groups.carry_leftovers_to_next_step:
                            leftover_rollouts_batch = batch[traj_bsz:] if traj_bsz < len(batch) else None
                        else:
                            leftover_rollouts_batch = None  # Discard leftovers - don't carry to next step
                        batch = batch[:traj_bsz]

                    if not self.config.reward_model.launch_reward_fn_async:
                        if self.async_rollout_mode:
                            self.async_rollout_manager.exit_rollout_mode()
                        else:
                            self.actor_rollout_wg.activate_trainer_mode()

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_result = ray.get(future_reward)
                            self._unpack_reward_result(reward_result, batch, metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.reward_model.length_reward.enable:
                            length_reward_tensor = core_algos.compute_length_reward(
                                batch,
                                self.config.reward_model.length_reward.coeff,
                                self.config.reward_model.length_reward.min_length_for_reward,
                            )
                            batch.batch["token_level_rewards"][:, -1] += length_reward_tensor

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        batch = compute_advantage(batch, config=self.config.algorithm)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics_keys = [k for k in actor_output.non_tensor_batch.keys() if k.startswith("metric/")]
                        metrics_batch = actor_output.select(non_tensor_batch_keys=metrics_keys).non_tensor_batch
                        metrics_batch = {k.lstrip("metric/"): v for k, v in metrics_batch.items()}
                        actor_output_metrics = reduce_metrics(metrics_batch)
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, timing_raw, rollout_data_dir)

                    # TODO AI21: response_logprobs and rollout_logprobs are computed also in _log_rollout_data,
                    #  can avoid duplication
                    response_logprobs = [
                        batch.batch["old_log_probs"][i][batch.batch["response_mask"][i] == 1].tolist()
                        for i in range(len(batch.batch["old_log_probs"]))
                    ]
                    rollout_logprobs = [
                        batch.batch["rollout_log_probs"][i][batch.batch["response_mask"][i] == 1].tolist()
                        for i in range(len(batch.batch["rollout_log_probs"]))
                    ]
                    all_rollout_logprobs = np.concatenate(rollout_logprobs)
                    all_response_logprobs = np.concatenate(response_logprobs)
                    rollout_actor_logprobs_abs_diff = np.abs(all_rollout_logprobs - all_response_logprobs)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/step_processed_examples": sum(step_processed_examples.values()),
                        "training/step_skipped_examples": sum(step_skipped_examples.values()),
                        # TODO: these total_ metrics will be reset upon checkpoint resume
                        "training/total_processed_examples": sum(total_processed_examples.values()),
                        "training/total_skipped_examples": sum(total_skipped_examples.values()),
                        "training/rollout_actor_logprobs_abs_diff_mean": rollout_actor_logprobs_abs_diff.mean(),
                        "training/rollout_actor_logprobs_abs_diff_max": rollout_actor_logprobs_abs_diff.max(),
                        "training/rollout_actor_logprobs_correlation": np.corrcoef(
                            all_rollout_logprobs, all_response_logprobs
                        )[0, 1],
                    }
                )

                # Add per-dataset metrics
                for dataset in encountered_data_sources:
                    metrics[f"per-dataset/{dataset}/training/step_processed_examples"] = step_processed_examples[
                        dataset
                    ]
                for dataset in encountered_data_sources:
                    metrics[f"per-dataset/{dataset}/training/step_skipped_examples"] = step_skipped_examples[dataset]
                for dataset in encountered_data_sources:
                    metrics[f"per-dataset/{dataset}/training/total_processed_examples"] = total_processed_examples[
                        dataset
                    ]
                for dataset in encountered_data_sources:
                    metrics[f"per-dataset/{dataset}/training/total_skipped_examples"] = total_skipped_examples[dataset]

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(
                    compute_data_metrics(batch=unfiltered_rollouts_batch, use_critic=self.use_critic, is_prefilter=True)
                )

                for dataset in set(batch.non_tensor_batch["data_source"]):
                    indices = batch.non_tensor_batch["data_source"] == dataset
                    dataset_batch = batch.select_idxs(indices)
                    metrics.update(
                        compute_data_metrics(
                            batch=dataset_batch, use_critic=self.use_critic, prefix="per-dataset/" + dataset
                        )
                    )
                for dataset in set(unfiltered_rollouts_batch.non_tensor_batch["data_source"]):
                    indices = unfiltered_rollouts_batch.non_tensor_batch["data_source"] == dataset
                    dataset_unfiltered_rollouts_batch = unfiltered_rollouts_batch.select_idxs(indices)
                    metrics.update(
                        compute_data_metrics(
                            batch=dataset_unfiltered_rollouts_batch,
                            use_critic=self.use_critic,
                            is_prefilter=True,
                            prefix="per-dataset/" + dataset,
                        )
                    )

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                metrics["train/num_gen_batches"] = num_gen_batches

                if (
                    self.config.algorithm.filter_groups.enable
                    and self.config.algorithm.filter_groups.num_times_to_snooze_easy_examples
                ):
                    num_snooze_skips = self.train_dataset.num_snooze_skips
                    metrics["train/step_snooze_skips"] = num_snooze_skips
                    self.train_dataset.reset_num_snooze_skips_counter()
                    metrics["train/step_newly_snoozed_prompts"] = step_easy_prompts
                    step_easy_prompts = 0
                    prefilter_metric_names = [
                        "critic/score/frac_prompts_const_score_in_0.75-1.00/prefilter",
                        "critic/score/mean/prefilter",
                    ]
                    for src_metric_name in prefilter_metric_names:
                        if src_metric_name in metrics:
                            dst_metric_name = src_metric_name + "/unsnoozed"
                            # Calculate the reward that we would have received if we had not snoozed the easy prompts
                            mean_reward = float(metrics[src_metric_name])
                            # This could be an issue if mean_score_threshold < 1, but any solution isn't 100% reliable
                            snoozed_prompts_reward = 1.0
                            num_prompts_in_unfiltered_batch = len(
                                set(unfiltered_rollouts_batch.non_tensor_batch["uid"])
                            )
                            mean_reward_with_snoozed_prompts = (
                                mean_reward * num_prompts_in_unfiltered_batch
                                + snoozed_prompts_reward * num_snooze_skips
                            ) / (num_prompts_in_unfiltered_batch + num_snooze_skips)
                            metrics[dst_metric_name] = mean_reward_with_snoozed_prompts

                batch = leftover_rollouts_batch
                unfiltered_rollouts_batch = leftover_rollouts_batch
                # Recompute prompt count from leftover using integer division;
                #  under allowed configs leftover is a multiple of n
                if leftover_rollouts_batch is not None:
                    num_prompt_in_batch = len(leftover_rollouts_batch.batch) // self.config.actor_rollout_ref.rollout.n
                else:
                    num_prompt_in_batch = 0
                num_gen_batches = 0
                step_skipped_examples = defaultdict(int)
                step_processed_examples = defaultdict(int)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


def _sorted_by_group(batch: DataProto) -> DataProto:
    uids = batch.non_tensor_batch["uid"]
    grouped_idxs = np.argsort(uids)
    return batch[grouped_idxs]
