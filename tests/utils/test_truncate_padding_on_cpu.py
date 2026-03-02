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

"""Tests for the truncate_padding feature."""

import os

import torch
import torch.distributed as dist
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from verl import DataProto
from verl.utils.seqlen_balancing import PaddingMode, prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_dynamic_batch_all_padding_modes():
    """prepare_dynamic_batch round-trips correctly for every PaddingMode."""
    # Setup
    lengths = [30, 25, 20, 15, 10, 5]
    dataproto = _make_dataproto(lengths, max_seq_len=30)

    for mode in PaddingMode:
        # Run
        micro_batches, idx_list = prepare_dynamic_batch(dataproto, max_token_len=60, padding_mode=mode)

        # Assert
        assert len(micro_batches) > 0, f"{mode}: should produce at least one micro-batch"
        all_indices = sorted(idx for batch_indices in idx_list for idx in batch_indices)
        assert all_indices == list(range(len(lengths))), f"{mode}: all sample indices must be covered"
        input_ids = torch.cat([mb.batch["input_ids"] for mb in micro_batches], dim=0)
        restored = restore_dynamic_batch(input_ids, idx_list)
        torch.testing.assert_close(restored, dataproto.batch["input_ids"], msg=f"{mode}: round-trip failed")


def test_forward_micro_batch_truncate_padding():
    """_forward_micro_batch with truncate_padding must produce identical
    log_probs to the standard path on non-padding response positions."""
    # Setup
    _init_dist()
    model, model_config = _make_model()
    micro_batch = _make_micro_batch(vocab_size=model_config.vocab_size)
    standard_actor = _make_actor(model, truncate_padding=False)
    truncate_padding_actor = _make_actor(model, truncate_padding=True)

    # Run
    with torch.no_grad():
        standard_result = standard_actor._forward_micro_batch(micro_batch, temperature=1.0)
        truncate_padding_result = truncate_padding_actor._forward_micro_batch(micro_batch, temperature=1.0)

    # Assert
    mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(
        truncate_padding_result["log_probs"][mask],
        standard_result["log_probs"][mask],
        msg="truncate_padding log_probs must match standard path on real response positions",
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _make_dataproto(lengths: list[int], max_seq_len: int) -> DataProto:
    """Create a DataProto with left-padded sequences of the given real lengths."""
    batch_size = len(lengths)
    input_ids = torch.randint(1, 100, (batch_size, max_seq_len))
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, max_seq_len - length :] = 1
    return DataProto.from_single_dict({"input_ids": input_ids, "attention_mask": attention_mask})


def _init_dist():
    """Initialize a single-process gloo group if not already initialized."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _make_model() -> tuple[nn.Module, GPT2Config]:
    """Create a tiny GPT-2 model for fast CPU testing."""
    config = GPT2Config(vocab_size=100, n_embd=64, n_head=2, n_layer=1, n_positions=64)
    return GPT2LMHeadModel(config).eval(), config


def _make_micro_batch(
    vocab_size: int,
    max_prompt_length: int = 12,
    padded_response_length: int = 20,
    actual_prompt_lengths: list[int] = None,
    response_lengths: list[int] = None,
) -> dict[str, torch.Tensor]:
    """Create a micro-batch dict matching dp_actor's expected layout."""
    if actual_prompt_lengths is None:
        actual_prompt_lengths = [10, 8, 12, 6]
    if response_lengths is None:
        response_lengths = [15, 20, 10, 18]

    batch_size = len(actual_prompt_lengths)
    max_seq_len = max_prompt_length + padded_response_length

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    responses = torch.zeros(batch_size, padded_response_length, dtype=torch.long)
    response_mask = torch.zeros(batch_size, padded_response_length)

    for i in range(batch_size):
        left_pad = max_prompt_length - actual_prompt_lengths[i]
        total_real = actual_prompt_lengths[i] + response_lengths[i]
        tokens = torch.randint(1, vocab_size, (total_real,))
        input_ids[i, left_pad : left_pad + total_real] = tokens
        attention_mask[i, left_pad : max_prompt_length + response_lengths[i]] = 1
        position_ids[i, left_pad : max_prompt_length + response_lengths[i]] = torch.arange(total_real)
        responses[i, : response_lengths[i]] = tokens[actual_prompt_lengths[i] :]
        response_mask[i, : response_lengths[i]] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
    }


def _make_actor(model: nn.Module, truncate_padding: bool = False) -> DataParallelPPOActor:
    """Create a DataParallelPPOActor with minimal config."""
    config = FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=4,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        truncate_padding=truncate_padding,
    )
    return DataParallelPPOActor(config=config, actor_module=model)
