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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import PaddingMode, prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits, pad_along_dim, pad_and_stack
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, mask=None) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None:
        return 0

    if not isinstance(gate_logits, tuple | list):
        gate_logits = (gate_logits,)
    # LB is done on dim1, so make sure layers are on dim0
    if len(gate_logits[0].shape) == 2:
        gate_logits = [t.unsqueeze(0) for t in gate_logits]
    else:
        assert len(gate_logits[0].shape) == 3

    # -> [num_layers, num_groups, tokens_per_group, num_experts]
    gate_logits = torch.stack(gate_logits, dim=0)
    if mask is not None:
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        assert mask.shape == gate_logits.shape[1:-1], (mask.shape, gate_logits.shape)
        mask = mask[None, :, :, None]  # shape: [1, groups, tokens, 1]
        mask = mask.repeat(
            [gate_logits.shape[0], 1, 1, gate_logits.shape[3]]
        )  # shape: [layers, groups, tokens, experts]
        assert mask.shape == gate_logits.shape

    router_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
    assert len(router_probs.shape) == 4

    expert_indices = torch.topk(router_probs, k=top_k, dim=-1).indices
    num_experts = router_probs.shape[-1]

    # Shape: [num_layers, num_groups, tokens_per_group, num_selected_experts, num_experts].
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [num_layers, num_groups, tokens_per_group, num_experts]
    expert_mask = torch.max(expert_mask, dim=-2).values

    def mean_with_optional_mask(t, mask, dim):
        assert mask is None or mask.shape == t.shape, (mask.shape, t.shape)
        if mask is None:
            return torch.mean(t, dim=dim, dtype=torch.float)
        return verl_F.masked_mean(t.float(), mask, axis=dim)

    tokens_per_group_and_expert = mean_with_optional_mask(expert_mask, mask, dim=-2)

    router_prob_per_group_and_expert = mean_with_optional_mask(router_probs, mask, dim=-2)

    mean_val = torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert,
        dtype=torch.float,
    )
    return mean_val * (num_experts**2)


def _z_loss(logits, mask=None) -> float:
    # Note: Based on the implementation in : https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py
    """Compute logits z-loss.

    The router z-loss was introduced in Designing Effective Sparse Expert Models
    (https://arxiv.org/abs/2202.08906). It encourages logits to remain
    small in an effort to improve stability.
    Relevant for both router-logits and vocab/output logits

    Args:
    logits: <float>[..., num_experts/vocab_size] logits.
    mask: optional mask of shape [...] of valid indices over which to compute

    Returns:
    Scalar z-loss.
    """
    log_z = torch.logsumexp(logits, dim=-1)
    z_loss = log_z**2
    if mask is None:
        return torch.mean(z_loss, dtype=torch.float)
    else:
        assert mask.shape == z_loss.shape, (mask.shape, z_loss.shape)
        return verl_F.masked_mean(z_loss.float(), mask)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.use_remove_padding
        self.move_left_padding_right = self.config.move_left_padding_right
        self.minimize_padding = self.config.minimize_padding
        self.truncate_padding = self.config.truncate_padding

        if self.minimize_padding:
            self.move_left_padding_right = True

        if self.truncate_padding and self.use_remove_padding:
            raise ValueError("Cannot enable both truncate_padding and use_remove_padding")

        if self.use_remove_padding and (self.minimize_padding or self.move_left_padding_right):
            raise ValueError(
                "When using use_remove_padding=True,"
                "enabling either one of minimize_padding or move_left_padding_right will have no effect."
            )

        if self.truncate_padding:
            self.padding_mode = PaddingMode.TRUNCATE_PADDING
        elif self.use_remove_padding:
            self.padding_mode = PaddingMode.REMOVE_PADDING
        elif self.minimize_padding:
            self.padding_mode = PaddingMode.MINIMIZE_PADDING
        else:
            self.padding_mode = PaddingMode.MAX_SEQUENCE_LENGTH_PADDING

        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
            print(f"{role} move_left_padding_right={self.move_left_padding_right}")
            print(f"{role} minimize_padding={self.minimize_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        if self.use_fused_kernels and (self.minimize_padding or self.move_left_padding_right):
            raise ValueError("use_fused_kernels cannot be used with minimize_padding or move_left_padding_right")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        use_torch_compile = self.config.get("use_torch_compile", True)
        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True) if use_torch_compile else entropy_from_logits
        )
        self.compute_z_loss = torch.compile(_z_loss, dynamic=True) if use_torch_compile else _z_loss
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        micro_batch_size, padded_response_length = micro_batch["responses"].shape

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        entropy = None
        max_activations = None
        router_lbloss = None
        router_z_loss = None

        if micro_batch["position_ids"].dim() == 3:  # qwen2vl mrope
            micro_batch["position_ids"] = micro_batch["position_ids"].transpose(
                0, 1
            )  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            if self.use_remove_padding:  # sequence packing
                input_ids = micro_batch["input_ids"]
                batch_size, seqlen = input_ids.shape
                attention_mask = micro_batch["attention_mask"]
                position_ids = micro_batch["position_ids"]

                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # pad_to = input_ids.numel()
                # TODO: re-enable if needed (example hangs), but update aux_loss
                # to ignore the rmpad
                pad_to = input_ids_rmpad.shape[-1]

                assert pad_to >= input_ids_rmpad.shape[-1]
                pad_to_right_input_ids = torch.nn.functional.pad(
                    input_ids_rmpad, (0, pad_to - input_ids_rmpad.shape[-1]), value=0
                )
                pad_to_right_position_ids_rmpad = torch.nn.functional.pad(
                    position_ids_rmpad, (0, pad_to - input_ids_rmpad.shape[-1]), value=0
                )
                extra_args = {}
                if self.config.router_aux_loss:
                    extra_args["output_router_logits"] = True
                # only pass input_ids and position_ids to enable flash_attn_varlen

                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=pad_to_right_input_ids,
                    attention_mask=None,
                    position_ids=pad_to_right_position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    max_activations = {}
                    for i in range(len(output.hidden_states)):
                        max_activations[f"actor/max_act_{i}"] = (
                            output.hidden_states[i][:, : input_ids_rmpad.shape[-1]].abs().max().detach().item()
                        )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    # TODO AI21 (RH): move router loss and logits z-loss calculation to update_policy
                    if self.config.router_aux_loss:
                        router_z_loss, router_lbloss = self._calculate_router_losses_rmpad(
                            router_logits=output.router_logits
                        )

                    # TODO TAMERG check this
                    logits_zloss = self.compute_z_loss(logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -padded_response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -padded_response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using use_remove_padding (sequence packing) and no ulysses sp
                model_inputs_keys = ["input_ids", "attention_mask", "position_ids"]
                model_inputs = {k: micro_batch[k] for k in model_inputs_keys}

                """
                Initially, all sequences have the prompt-response transition at the same index
                due to left and right padding that aligns them.
                """
                seq_length = model_inputs["input_ids"].size(1)
                start_idx_value = seq_length - padded_response_length - 1
                response_start_idx = torch.full(
                    size=(micro_batch_size,),
                    fill_value=start_idx_value,
                    device=model_inputs["input_ids"].device,
                )
                response_end_idx = torch.full(
                    size=(micro_batch_size,),
                    fill_value=seq_length - 1,
                    device=model_inputs["input_ids"].device,
                )

                if self.move_left_padding_right:

                    def roll_with_different_shift_per_row(t: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
                        # Like torch.roll, but with a different shift for each row
                        out = torch.empty_like(t)
                        for i, row in enumerate(t):
                            out[i] = torch.roll(row, int(shifts[i]), dims=0)
                        return out

                    left_padding_lengths = model_inputs["attention_mask"].argmax(dim=1)
                    shifts = -left_padding_lengths
                    model_inputs = {
                        key: roll_with_different_shift_per_row(t, shifts=shifts) for key, t in model_inputs.items()
                    }
                    response_start_idx = response_start_idx - left_padding_lengths
                    response_end_idx = response_end_idx - left_padding_lengths

                if self.minimize_padding:
                    common_padding_left = model_inputs["attention_mask"].argmax(dim=1).min().item()
                    common_padding_right = model_inputs["attention_mask"].flip(dims=[1]).argmax(dim=1).min().item()
                    model_inputs = {
                        key: t[:, common_padding_left : seq_length - common_padding_right]
                        for key, t in model_inputs.items()
                    }
                    response_start_idx = response_start_idx - common_padding_left
                    response_end_idx = response_end_idx - common_padding_left

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if self.config.router_aux_loss:
                    extra_args["output_router_logits"] = True

                output = self.actor_module(
                    **model_inputs,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    # TODO AI21 (RH): support use_fused_kernels with move_left_padding_right and minimize_padding
                    log_probs = output.log_probs[:, -padded_response_length - 1 : -1]
                    entropy = output.entropy[:, -padded_response_length - 1 : -1]  # (bsz, response_length)

                else:
                    response_logits_list = [
                        output.logits[i, response_start_idx[i] : response_end_idx[i], :]
                        for i in range(micro_batch_size)
                    ]  # list of length bsz, each element is a tensor of shape (response_length, vocab_size)

                    response_logits = pad_and_stack(
                        response_logits_list, pad_value=0
                    )  # (bsz, longest_response_length, vocab_size)

                    response_logits.div_(temperature)

                    labels = micro_batch["responses"][:, : response_logits.shape[1]]

                    log_probs = logprobs_from_logits(logits=response_logits, labels=labels)
                    log_probs = pad_along_dim(
                        tensor=log_probs, target_length=padded_response_length, pad_value=0, dim=1
                    )
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(response_logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, response_logits
                            )
                        entropy = pad_along_dim(
                            tensor=entropy, target_length=padded_response_length, pad_value=0, dim=1
                        )

                    """ TODO AI21 (RH): move router loss and logits z-loss calculation to update_policy
                        (requires outputting router_logits that match the original attention_mask,
                        and unifying it with the use_remove_padding [sequence packing] case)"""
                    if self.config.router_aux_loss:
                        router_z_loss, router_lbloss = self._calculate_router_losses(
                            router_logits=output.router_logits,
                            attention_mask=model_inputs["attention_mask"],
                        )
                    logits_zloss = self.compute_z_loss(response_logits)

            return (
                entropy,
                log_probs,
                router_lbloss if self.config.router_aux_loss else None,
                max_activations,
                router_z_loss,
                output.router_logits if self.config.router_aux_loss else None,
                logits_zloss,
            )

    def _calculate_router_losses(self, router_logits, attention_mask):
        router_z_loss = sum(
            _z_loss(
                router_logits,
                mask=attention_mask.reshape(*router_logits.shape[:-1]),
            )
            for router_logits in router_logits
        ) / len(router_logits)
        # router_logits are [1, batch*seqlen, experts]. Reshape per example
        # so we reshape them to [batch, seqlen, num_expers] according to the attention mask
        # Note that attention_mask is still rolled here in case of
        # move_left_padding_right which matches the router logits
        router_logits = [t.reshape(*attention_mask.shape, t.shape[-1]) for t in router_logits]
        num_experts_conf = (
            self.actor_module.config.num_local_experts
            if hasattr(self.actor_module.config, "num_local_experts")
            else self.actor_module.config.num_experts
        )
        router_lbloss = load_balancing_loss_func(
            router_logits,
            num_experts_conf,
            self.actor_module.config.num_experts_per_tok,
            mask=attention_mask,
        )

        return router_z_loss, router_lbloss

    def _calculate_router_losses_rmpad(self, router_logits):
        router_z_loss = sum(_z_loss(r_logits) for r_logits in router_logits) / len(router_logits)
        num_experts_conf = (
            self.actor_module.config.num_local_experts
            if hasattr(self.actor_module.config, "num_local_experts")
            else self.actor_module.config.num_experts
        )
        router_lbloss = load_balancing_loss_func(
            router_logits,
            num_experts_conf,
            self.actor_module.config.num_experts_per_tok,
        )
        return router_z_loss, router_lbloss

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data,
                max_token_len=max_token_len,
                padding_mode=self.padding_mode,
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                (entropy, log_probs, _, _, _, _, _) = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto, optim_onload_fn=None, optim_offload_fn=None):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss and self.config.kl_loss_coef > 0:
            select_keys.append("ref_log_prob")

        if self.config.filter_disagreement_logprobs["enable"]:
            select_keys.append("rollout_log_probs")

        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch,
                        max_token_len=max_token_len,
                        padding_mode=self.padding_mode,
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    (entropy, log_prob, router_lbloss, max_act, router_z_loss, router_logits, logits_zloss) = (
                        self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                    )
                    if max_act is not None:
                        metrics = {**metrics, **max_act}
                    if self.config.maskout_clipped_responses:
                        clipped_responses = response_mask[:, -1] == 1
                        response_mask[clipped_responses] = 0
                        micro_batch_metrics["actor/clipped_responses"] = clipped_responses.detach().sum().item()
                    if self.config.filter_disagreement_logprobs["enable"]:
                        disagreement_mask = (
                            torch.abs(model_inputs["rollout_log_probs"] - log_prob)
                            > self.config.filter_disagreement_logprobs["threshold"]
                        )
                        response_mask_total = response_mask.detach().sum().item()
                        micro_batch_metrics["actor/disagreement_masked_perc"] = 0
                        if response_mask_total > 0:
                            micro_batch_metrics["actor/disagreement_masked_perc"] = (
                                response_mask[disagreement_mask].detach().sum().item() / response_mask_total
                            )
                        response_mask[disagreement_mask] = 0

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss and self.config.kl_loss_coef > 0:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    if self.config.router_aux_loss:
                        metrics["actor/router_lbloss"] = router_lbloss.detach().item()
                        policy_loss = policy_loss + router_lbloss * self.config.router_lbloss_coef
                        metrics["actor/router_z_loss"] = router_z_loss.detach().item()
                        policy_loss = policy_loss + router_z_loss * self.config.router_zloss_alpha
                        for i in range(len(router_logits)):
                            metrics[f"actor/router_logits_mean_{i}"] = router_logits[i].mean().item()

                    if self.config.logits_zloss_alpha:
                        policy_loss = policy_loss + logits_zloss * self.config.logits_zloss_alpha

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor

                    for thresh in [-5.0, -10.0, -20.0, -40.0]:
                        micro_batch_metrics[f"actor/oldlogprob_lt_{thresh}_frac"] = (
                            verl_F.masked_mean(old_log_prob < thresh, response_mask).detach().item()
                        )

                    if self.config.logits_zloss_alpha:
                        micro_batch_metrics["actor/logits_z_loss"] = logits_zloss.detach().item()

                    append_to_dict(metrics, micro_batch_metrics)

                if optim_onload_fn is not None:
                    optim_onload_fn()
                grad_norm = self._optimizer_step()
                if optim_offload_fn is not None:
                    optim_offload_fn()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
