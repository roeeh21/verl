# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import copy
import heapq
from enum import Enum, auto
from itertools import chain
from typing import Optional

import torch
from tensordict import TensorDict
from torch import distributed as dist

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name


def calculate_workload(seqlen_list: list[int]):
    """
    Calculate the workload for a dense transformer block based on sequence length.
    FLOPs = 12 * hidden_size^2 * seqlen + 2 * hidden_size * seqlen^2
    Hardcodes the constants by a 7B model (hidden_size=4096),
    so the FLOPs are propotional to (6 * 4096 * seqlen + seqlen^2).
    """
    return 24576 * seqlen_list + seqlen_list**2


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


def synchronize_micro_batches_num_across_ranks(
    micro_batches_idx: list[list[int]],
    dp_group: Optional[dist.ProcessGroup] = None,
) -> list[list[int]]:
    """
    Ensure all ranks have the same number of micro-batches by splitting larger batches.

    Args:
        micro_batches_idx: List of index lists for micro-batches.
        dp_group: The distributed group for data-parallel sync.

    Returns:
        List of index lists with the same count across all ranks.
    """
    if not dist.is_initialized():
        return micro_batches_idx

    num_micro_batches = len(micro_batches_idx)
    num_micro_batches_tensor = torch.tensor([num_micro_batches], device=get_device_name())
    dist.all_reduce(num_micro_batches_tensor, op=dist.ReduceOp.MAX, group=dp_group)
    max_num_micro_batches = num_micro_batches_tensor.cpu().item()

    while len(micro_batches_idx) < max_num_micro_batches:
        largest_batch_idx = max(range(len(micro_batches_idx)), key=lambda i: len(micro_batches_idx[i]))
        largest_batch = micro_batches_idx[largest_batch_idx]
        if len(largest_batch) <= 1:
            _raise_cannot_synchronize_micro_batches_across_ranks_error(
                micro_batches_idx, max_num_micro_batches, dp_group
            )
        mid = len(largest_batch) // 2
        micro_batches_idx[largest_batch_idx] = largest_batch[:mid]
        micro_batches_idx.append(largest_batch[mid:])

    return micro_batches_idx


def _raise_cannot_synchronize_micro_batches_across_ranks_error(
    micro_batches_idx: list[list[int]], max_num_micro_batches: int, dp_group: dist.ProcessGroup
):
    """
    This error should not happen in theory, since the total batch size should be equal across ranks,
    so it should always be possible to split micro-batches in ranks with fewer micro-batches until
    all ranks have the same number.
    However, it was observed once and could not be reproduced.
    Added as a temporary error handler to provide better diagnostics if it occurs again.
    """
    num_sequences = sum(len(batch) for batch in micro_batches_idx)
    rank = dist.get_rank(dp_group)
    raise RuntimeError(
        f"Cannot split micro-batches further without creating empty batches. "
        f"Need {max_num_micro_batches} micro-batches but only have {num_sequences} sequences in rank {rank}. "
        "Total batch size should be equal across ranks."
    )


def get_minimize_padding_micro_batches(
    batch: TensorDict,
    max_token_len: int,
    dp_group: Optional[dist.ProcessGroup] = None,
    same_micro_num_in_dp: bool = True,
) -> list[list[int]]:
    """
    Create micro-batch index lists that try to minimize padding.

    Uses a simple greedy algorithm to pack sequences into micro-batches.
    Sorts sequences by actual length (descending) and packs them into batches
    such that when padded to the longest sequence in each batch, total tokens <= max_token_len,
    and the padding is minimized.

    Args:
        batch (TensorDict): The input data containing batch with attention_mask.
        max_token_len (int): Maximum number of tokens per micro batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        same_micro_num_in_dp (bool): if True and dp_group set, ensure same count across ranks.

    Returns:
        List[List[int]]: List of index lists, one per micro-batch.
    """

    sequence_lengths = batch["attention_mask"].sum(dim=1).cpu().tolist()

    sorted_sequence_lengths_with_idx = sorted(
        [(length, idx) for idx, length in enumerate(sequence_lengths)], key=lambda x: x[0], reverse=True
    )

    micro_batches_idx = []

    longest_sequence_length, longest_sequence_idx = sorted_sequence_lengths_with_idx[0]
    current_micro_batch_idx = [longest_sequence_idx]
    current_micro_batch_max_len = longest_sequence_length

    for sequence_length, idx in sorted_sequence_lengths_with_idx[1:]:
        new_micro_batch_size = len(current_micro_batch_idx) + 1
        new_total_tokens = new_micro_batch_size * current_micro_batch_max_len

        if new_total_tokens <= max_token_len:
            current_micro_batch_idx.append(idx)
        else:
            micro_batches_idx.append(current_micro_batch_idx)
            current_micro_batch_idx = [idx]
            current_micro_batch_max_len = sequence_length

    if current_micro_batch_idx:
        micro_batches_idx.append(current_micro_batch_idx)

    if same_micro_num_in_dp:
        micro_batches_idx = synchronize_micro_batches_num_across_ranks(micro_batches_idx, dp_group)

    return micro_batches_idx


def get_truncate_padding_micro_batches_jagged(
    batch: TensorDict,
    max_token_len: int,
    dp_group: Optional[dist.ProcessGroup] = None,
    same_micro_num_in_dp: bool = True,
) -> list[list[int]]:
    """Create micro-batch index lists that minimize padding for jagged (nested) tensors.

    Same greedy algorithm as get_truncate_padding_micro_batches, but extracts
    sequence lengths from nested tensor offsets instead of attention_mask.

    Args:
        batch: TensorDict containing nested "input_ids" with jagged layout.
        max_token_len: Maximum number of tokens per micro-batch.
        dp_group: torch.distributed group for data-parallel sync.
        same_micro_num_in_dp: If True, ensure same micro-batch count across DP ranks.

    Returns:
        List of index lists, one per micro-batch.
    """
    input_ids = batch["input_ids"]
    assert input_ids.is_nested, "get_truncate_padding_micro_batches_jagged requires nested input_ids"
    sequence_lengths = input_ids.offsets().diff().cpu().tolist()

    sorted_sequence_lengths_with_idx = sorted(
        [(length, idx) for idx, length in enumerate(sequence_lengths)], key=lambda x: x[0], reverse=True
    )

    micro_batches_idx = []

    if not sorted_sequence_lengths_with_idx:
        if same_micro_num_in_dp:
            micro_batches_idx = synchronize_micro_batches_num_across_ranks(micro_batches_idx, dp_group)
        return micro_batches_idx

    longest_sequence_length, longest_sequence_idx = sorted_sequence_lengths_with_idx[0]
    current_micro_batch_idx = [longest_sequence_idx]
    current_micro_batch_max_len = longest_sequence_length

    for sequence_length, idx in sorted_sequence_lengths_with_idx[1:]:
        new_micro_batch_size = len(current_micro_batch_idx) + 1
        new_total_tokens = new_micro_batch_size * current_micro_batch_max_len

        if new_total_tokens <= max_token_len:
            current_micro_batch_idx.append(idx)
        else:
            micro_batches_idx.append(current_micro_batch_idx)
            current_micro_batch_idx = [idx]
            current_micro_batch_max_len = sequence_length

    if current_micro_batch_idx:
        micro_batches_idx.append(current_micro_batch_idx)

    if same_micro_num_in_dp:
        micro_batches_idx = synchronize_micro_batches_num_across_ranks(micro_batches_idx, dp_group)

    return micro_batches_idx


def get_max_sequence_length_padding_micro_batches(
    batch: TensorDict,
    max_token_len: int,
    dp_group: Optional[dist.ProcessGroup] = None,
    same_micro_num_in_dp: bool = True,
) -> list[list[int]]:
    """
    Create micro-batches for the case where all sequences are padded to the same length.

    Args:
        batch: TensorDict containing the batch data.
        max_token_len: The maximum number of tokens per micro batch.
        dp_group: The distributed group for data-parallel sync.
        **kwargs: Additional keyword arguments.

    Returns:
        List[list[int]]: The index lists for the micro-batches.
    """
    padded_seq_len = batch["input_ids"].shape[-1]
    batch_size = batch["input_ids"].shape[0]
    micro_batch_size = max_token_len // padded_seq_len

    micro_batches_idx = []
    for start_idx in range(0, batch_size, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, batch_size)
        micro_batches_idx.append(list(range(start_idx, end_idx)))

    if same_micro_num_in_dp:
        micro_batches_idx = synchronize_micro_batches_num_across_ranks(micro_batches_idx, dp_group)

    return micro_batches_idx


class PaddingMode(Enum):
    MAX_SEQUENCE_LENGTH_PADDING = auto()
    REMOVE_PADDING = auto()
    MINIMIZE_PADDING = auto()


def rearrange_micro_batches(
    batch,
    max_token_len,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    use_dynamic_bsz_balance=True,
):
    """
    Split a batch into micro-batches by total token count, with optional DP sync.
    This assumes that there's no padding at all (i.e. use_remove_padding is True, i.e. using sequence packing).

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
        max_token_len (int): maximum number of tokens per micro batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches

    Returns:
        List[TensorDict]: the micro-batches.
        List[List[int]]: index lists mapping each micro-batch back to original positions.
    """
    # this is per local micro_bsz
    input_ids = batch["input_ids"]
    if input_ids.is_nested:
        seq_len_effective: torch.Tensor = input_ids.offsets().diff()
        max_seq_len = max(seq_len_effective)
    else:
        max_seq_len = batch["attention_mask"].shape[-1]
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)

    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )
    total_seqlen = seq_len_effective.sum().item()
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if min_num_micro_batch is not None:
        # used to support pp
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)
    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()
    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    assert num_micro_batches <= len(seq_len_effective)

    workloads = calculate_workload(seq_len_effective)
    micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)

    if use_dynamic_bsz_balance:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(workloads[idx] for idx in partition),
                partition[0] if partition else 0,
            ),
            reverse=True,
        )
        # Place smaller micro-batches at both ends to reduce the bubbles exposed during the warm-up and cool-down.
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]

    micro_batches = []

    for partition in micro_bsz_idx:
        curr_micro_batch = tu.index_select_tensor_dict(batch, partition)
        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map


def prepare_dynamic_batch(
    data: DataProto,
    max_token_len: int,
    dp_group=None,
    same_micro_num_in_dp=True,
    padding_mode: PaddingMode = PaddingMode.REMOVE_PADDING,
    **kwargs,
) -> tuple[list[DataProto], list[list[int]]]:
    """
    Prepare a batch for dynamic batching.

    Args:
        data (DataProto): The input data.
        max_token_len (int): The maximum number of tokens per micro batch.
        dp_group (optional): torch.distributed group for data-parallel sync.
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
        padding_mode (PaddingMode, optional): padding mode for the batch.
        kwargs (optional): additional arguments for the micro-batching function.
            Currently only used when padding_mode is PaddingMode.REMOVE_PADDING.
            See rearrange_micro_batches for more details.

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
    """
    max_seq_len = data.batch["input_ids"].shape[-1]
    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the max sequence length (max_prompt_length + max_response_length)."
        f"Got {max_token_len=} and {max_seq_len=}."
    )

    if padding_mode == PaddingMode.MINIMIZE_PADDING:
        micro_batches_idx = get_minimize_padding_micro_batches(
            batch=data.batch,
            max_token_len=max_token_len,
            dp_group=dp_group,
            same_micro_num_in_dp=same_micro_num_in_dp,
        )
    elif padding_mode == PaddingMode.REMOVE_PADDING:
        _, micro_batches_idx = rearrange_micro_batches(
            data.batch,
            max_token_len=max_token_len,
            dp_group=dp_group,
            same_micro_num_in_dp=same_micro_num_in_dp,
            **kwargs,
        )
    elif padding_mode == PaddingMode.MAX_SEQUENCE_LENGTH_PADDING:
        micro_batches_idx = get_max_sequence_length_padding_micro_batches(
            batch=data.batch,
            max_token_len=max_token_len,
            dp_group=dp_group,
            same_micro_num_in_dp=same_micro_num_in_dp,
        )
    else:
        raise ValueError(f"Invalid padding mode: {padding_mode}")

    micro_batches = []
    for micro_batch_idx in micro_batches_idx:
        tensors = tu.index_select_tensor_dict(data.batch, micro_batch_idx)
        non_tensors = {key: value[micro_batch_idx] for key, value in data.non_tensor_batch.items()}
        meta_info = copy.deepcopy(data.meta_info)
        micro_batches.append(DataProto.from_dict(tensors=dict(tensors), non_tensors=non_tensors, meta_info=meta_info))

    return micro_batches, micro_batches_idx


def restore_dynamic_batch(data: torch.Tensor, batch_idx_list: list[list[int]]) -> torch.Tensor:
    """
    Restore a batch from dynamic batching.

    Args:
        data (torch.Tensor): The input data.
        batch_idx_list (List[List[int]]): The list of index lists.

    Returns:
        torch.Tensor: The restored data.
    """
    indices = list(chain.from_iterable(batch_idx_list))
    batch_size = data.shape[0]
    assert len(indices) == batch_size, f"{len(indices)} vs. {batch_size}"
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)

    if data.is_nested:
        tensors = [data[i] for i in revert_indices]
        reverted_data = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    else:
        reverted_data = data[revert_indices]

    return reverted_data
