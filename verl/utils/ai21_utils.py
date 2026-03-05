# Copyright 2025 AI21 Labs

from typing import Optional

import numpy as np
import torch

from verl import DataProto


def filter_ai21_excluded_examples(config: dict, batch: DataProto) -> DataProto:
    """
    Filter out examples marked for exclusion by AI21 reward manager.
    This provides true "skipping" - excluded examples are completely removed from training.

    Args:
        batch: DataProto batch to filter

    Returns:
        Filtered batch with excluded examples removed
    """
    if (
        config.reward_model.reward_manager != "ai21"
        or not config.algorithm.filter_groups.filter_ai21_eval_errors
        or "do_exclude_example" not in batch.non_tensor_batch
    ):
        return batch

    exclude_mask = batch.non_tensor_batch["do_exclude_example"].astype(bool)

    # Nothing to exclude
    if not np.any(exclude_mask):
        return batch

    # Always perform group-level exclusion
    assert "uid" in batch.non_tensor_batch, "uid is required for group-level exclusion"
    uids = batch.non_tensor_batch["uid"]
    unique_uids = np.unique(uids)

    uid_exclude = {uid: bool(np.any(exclude_mask[uids == uid])) for uid in unique_uids}
    keep_mask = np.array([not uid_exclude[uid] for uid in uids], dtype=bool)
    keep_indices = np.where(keep_mask)[0]

    num_examples_excluded = len(batch) - len(keep_indices)
    num_groups_excluded = sum(uid_exclude.values())
    print(
        f"[AI21_RM_WARNING] Group exclusion removed {num_groups_excluded}/{len(unique_uids)} "
        f"groups and {num_examples_excluded}/{len(batch)} examples"
    )

    filtered_batch = batch[keep_indices]
    filtered_batch.check_consistency()
    print(f"[FILTER_AI21_EXCLUDED_EXAMPLES] post filter length {len(filtered_batch)=}")
    return filtered_batch


def ensure_concat_compatibility(
    batch: Optional[DataProto], new_batch: Optional[DataProto]
) -> tuple[DataProto, DataProto]:
    """
    Ensure batch has all evaluator keys by adding missing ones with zeros
    and that reward_model and evaluator_usage are flattened
    """

    def __squeeze_all_non_tensor(dp: DataProto) -> DataProto:
        for k, v in list(dp.non_tensor_batch.items()):
            dp.non_tensor_batch[k] = np.squeeze(v)
        return dp

    def __harmonize_non_tensor_dims(
        dp1: Optional[DataProto], dp2: Optional[DataProto]
    ) -> tuple[Optional[DataProto], Optional[DataProto]]:
        if dp1 is None or dp2 is None:
            return dp1, dp2
        keys = set(dp1.non_tensor_batch.keys()) & set(dp2.non_tensor_batch.keys())
        for k in keys:
            v1 = dp1.non_tensor_batch[k]
            v2 = dp2.non_tensor_batch[k]
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if v1.ndim != v2.ndim:
                    v1_sq = np.squeeze(v1)
                    v2_sq = np.squeeze(v2)
                    assert v1_sq.shape == v2_sq.shape, (
                        f"v1 and v2 of {k} are not the same shape: {v1_sq.shape} != {v2_sq.shape}"
                    )
                    dp1.non_tensor_batch[k] = v1_sq
                    dp2.non_tensor_batch[k] = v2_sq
            else:
                assert type(v1) is type(v2), f"v1 and v2 of {k} are not the same type: {type(v1)} != {type(v2)}"
        return dp1, dp2

    def __add_missing_evaluators(batch: DataProto, all_evaluator_names: set):
        if "evaluator_reward_tensors" in batch.batch:
            current_evaluators = set(batch.batch["evaluator_reward_tensors"].keys())
            missing_evaluators = all_evaluator_names - current_evaluators
            if missing_evaluators:
                batch_size = len(batch.batch)
                # Add missing evaluators with zero tensors
                for evaluator_name in missing_evaluators:
                    # Use shape from any existing evaluator
                    if current_evaluators:
                        reference_tensor = batch.batch["evaluator_reward_tensors"][list(current_evaluators)[0]]
                        batch.batch["evaluator_reward_tensors"][evaluator_name] = torch.zeros_like(reference_tensor)

                    # Also add corresponding score field to non_tensor_batch
                    score_field_name = f"{evaluator_name}_score"
                    if score_field_name not in batch.non_tensor_batch:
                        batch.non_tensor_batch[score_field_name] = np.zeros(batch_size, dtype=np.float32)

    if batch is None or new_batch is None:
        return batch, new_batch

    # Collect all evaluator names from both batches
    batch_evaluators = set(batch.batch.get("evaluator_reward_tensors", {}).keys())
    new_batch_evaluators = set(new_batch.batch.get("evaluator_reward_tensors", {}).keys())
    all_evaluators_names = batch_evaluators | new_batch_evaluators
    __add_missing_evaluators(batch, all_evaluators_names)
    __add_missing_evaluators(new_batch, all_evaluators_names)

    # Squeeze/normalize non-tensor entries (guard against stray extra dims)
    batch = __squeeze_all_non_tensor(batch)
    new_batch = __squeeze_all_non_tensor(new_batch)

    # Harmonize dims across both batches to avoid concat errors
    batch, new_batch = __harmonize_non_tensor_dims(batch, new_batch)

    _strip_filter_keys(batch)
    _strip_filter_keys(new_batch)
    return batch, new_batch


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and tensors to lists in nested data structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return type(obj)(convert_numpy_to_list(item) for item in obj)
    else:
        return obj


def convert_reward_extra_infos_dict_ai21_reward_manager(reward_extra_infos_dict: dict, batch: DataProto) -> dict:
    """
    Convert reward_extra_infos_dict fields to indexable arrays for AI21 reward manager.

    Args:
        reward_extra_infos_dict: Dictionary of extra reward information
        batch: DataProto batch to get batch size and task IDs from

    Returns:
        Dictionary with converted fields suitable for indexing
    """
    batch_updates = {}
    batch_size = len(batch.batch)

    for k, v in reward_extra_infos_dict.items():
        # Handle specific known fields from AI21 reward manager
        if k == "evaluator_usage":
            # Handle both dict format and list format for evaluator_usage
            if isinstance(v, dict):
                usage_list = []
                for i in range(batch_size):
                    usage_list.append(v.get(i, []))
                # Build a 1-D object array where each element is a list (avoid accidental flattening)
                batch_updates[k] = np.fromiter((list(item) for item in usage_list), dtype=object, count=batch_size)
            elif isinstance(v, list) and len(v) == batch_size:
                # Handle list format with potentially mixed shapes
                flattened_values = []
                for item in v:
                    if isinstance(item, (list | tuple | np.ndarray)):
                        if len(item) == 1:
                            flattened_values.append([item[0]])  # Keep as single-element list
                        else:
                            flattened_values.append(list(item))  # Convert to list
                    else:
                        flattened_values.append(item)
                # Build a 1-D object array where each element is a list (avoid accidental flattening)
                batch_updates[k] = np.fromiter(
                    ((item if isinstance(item, list) else [item]) for item in flattened_values),
                    dtype=object,
                    count=batch_size,
                )
            else:
                raise TypeError(f"Unexpected type for evaluator_usage: {type(v)}")

        elif k == "do_exclude_example":
            if isinstance(v, list) and len(v) == batch_size:
                batch_updates[k] = np.array(v, dtype=bool).reshape(-1)
            else:
                raise TypeError(
                    f"Unexpected type/length for do_exclude_example: type={type(v)}, "
                    f"len={len(v) if hasattr(v, '__len__') else 'N/A'}, expected_len={batch_size}"
                )

        elif k == "response_details":
            # response_details: list of dict/None values from ai21-evaluators response.details
            if isinstance(v, list) and len(v) == batch_size:
                batch_updates[k] = np.array(v, dtype=object).reshape(-1)
            else:
                raise TypeError(
                    f"Unexpected format for response_details: type={type(v)}, "
                    f"len={len(v) if hasattr(v, '__len__') else 'N/A'}, expected_len={batch_size}"
                )

        elif k.endswith("_score"):
            # Score fields: lists of numerical scores
            if isinstance(v, list) and len(v) == batch_size:
                # Flatten any nested structure and ensure consistent scalar elements
                flattened_values = []
                for item in v:
                    if isinstance(item, (list | tuple | np.ndarray)) and len(item) == 1:
                        flattened_values.append(float(item[0]))
                    elif isinstance(item, int | float | np.number):
                        flattened_values.append(float(item))
                    else:
                        flattened_values.append(item)
                # Ensure all elements are consistently shaped by storing in a properly shaped array
                batch_updates[k] = np.array(flattened_values, dtype=object).reshape(-1)
            else:
                raise TypeError(
                    f"Unexpected format for score field {k}: type={type(v)}, "
                    f"len={len(v) if hasattr(v, '__len__') else 'N/A'}, expected_len={batch_size}"
                )

        else:
            raise TypeError(f"Unknown field '{k}' with unsupported type {type(v)} in reward_extra_infos_dict")

    return batch_updates


def _strip_filter_keys(batch: Optional[DataProto]) -> None:
    if batch is None:
        return
    keys = ("seq_reward", "seq_final_reward")
    for key in keys:
        if key in batch.non_tensor_batch:
            batch.non_tensor_batch.pop(key, None)
