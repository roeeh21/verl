# Copyright 2025 AI21 Labs
from typing import Any, Iterator, Optional

from torch.utils.data import IterableDataset


class SnoozingDataset(IterableDataset):
    """
    A wrapper dataset that allows snoozing specific examples by their ID.

    This dataset wraps another dataset and provides functionality to skip
    specific examples for a specified number of times when they are encountered.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        id_field_name: str,
        verbose: bool = False,
        store_resume_skipped_ids: bool = False,
    ):
        """
        Initialize the SnoozingDataset.

        Args:
            dataset: The underlying dataset to wrap
        """
        self.dataset = dataset
        self.id_field_name = id_field_name
        self._snooze_counts: dict[str, int] = {}
        self.num_snooze_skips = 0
        """How many times have we skipped examples due to snoozing?"""
        self._num_times_yielded_from_inner_dataset = 0
        """This is the number of times we have yielded examples from the inner dataset, regardless of snoozing. 
        It is used for fast-forwarding the inner dataset when resuming, if the inner dataset is not stateful."""
        self.verbose = verbose
        self._resume_skipped_ids = [] if store_resume_skipped_ids else None

    def snooze_example(self, id: str, num_times: int) -> None:
        """
        Snooze an example by its ID for a specified number of times.

        Args:
            id: The ID of the example to snooze
            num_times: Number of times to skip this example when encountered
        """
        if num_times < 0:
            raise ValueError("num_times must be non-negative")
        if id not in self._snooze_counts:
            self._snooze_counts[id] = 0
        self._snooze_counts[id] += num_times
        if self.verbose:
            print(
                f"[SNOOZING_DATASET] Snoozing example {id} for {num_times} times"
                f"(total snoozes: {self._snooze_counts[id]})"
            )

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the dataset, skipping snoozed examples.

        Yields:
            Examples  from the underlying dataset, skipping snoozed ones
        """
        dataset_iter = iter(self.dataset)
        empty_sentinel = object()
        if not self._is_inner_dataset_stateful() and self._num_times_yielded_from_inner_dataset > 0:
            print(
                f"[SNOOZING_DATASET] Fast-forwarding {self._num_times_yielded_from_inner_dataset} "
                "examples from the inner dataset for resuming because it is not stateful"
            )
            # In order to correctly resume with snoozing, we need to implement state_dict / load_state_dict. However,
            # if the inner dataset did not also implement these functions, we mimic the behavior of StatefulDataLoader
            # by fast-forwarding the inner dataset.
            # This is the code that runs if the dataset is not stateful:
            # https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/stateful_dataloader.py#L576
            # The main() of this file checks this equivalence.
            for _ in range(self._num_times_yielded_from_inner_dataset):
                skipped_id = self._get_example_id(next(dataset_iter))
                if self._resume_skipped_ids is not None:
                    self._resume_skipped_ids.append(skipped_id)
            print("[SNOOZING_DATASET] Done fast-forwarding, last skipped id: ", skipped_id)
        while True:
            example = next(dataset_iter, empty_sentinel)
            if example == empty_sentinel:
                # If we reached the end of the inner dataset, verl will create a new iterator to continue training.
                # We need to start from the beginning of the inner dataset when that happens.
                # We don't reset the snooze count, because we want to snooze the examples when they appear in the
                # next iteration of the dataset.
                self._num_times_yielded_from_inner_dataset = 0
                return
            self._num_times_yielded_from_inner_dataset += 1
            example_id = self._get_example_id(example)
            if self._snooze_counts.get(example_id, 0) > 0:
                # Skip this example and decrement the snooze count
                self._snooze_counts[example_id] -= 1
                if self._snooze_counts[example_id] == 0:
                    del self._snooze_counts[example_id]
                self.num_snooze_skips += 1
                if self.verbose:
                    print(
                        f"[SNOOZING_DATASET] Skipping example {example_id}. "
                        f"{self._snooze_counts.get(example_id, 0)} snoozes remaining."
                    )
                continue
            yield example

    def _get_example_id(self, example: Any) -> Optional[str]:
        """
        Extract the ID from an example.

        Args:
            example: The example to extract ID from

        Returns:
            The ID of the example, raises exception if no ID is found
        """
        if isinstance(example, dict):
            if self.id_field_name in example:
                return str(example[self.id_field_name])
        else:
            if hasattr(example, self.id_field_name):
                return str(getattr(example, self.id_field_name))

        raise ValueError(f"No ID found in the example {example}")

    def reset_num_snooze_skips_counter(self) -> None:
        """
        Reset the number of snooze skips.
        """
        self.num_snooze_skips = 0

    def get_dataset_stats(self) -> dict[str, Any]:
        return self.dataset.get_dataset_stats()

    def state_dict(self) -> dict[str, Any]:
        state_dict = {
            "snooze_counts": self._snooze_counts,
            "num_snooze_skips": self.num_snooze_skips,
            "num_times_yielded_from_inner_dataset": self._num_times_yielded_from_inner_dataset,
        }
        if self._is_inner_dataset_stateful():
            state_dict["dataset_state_dict"] = self.dataset.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._snooze_counts = state_dict.get("snooze_counts", {})
        self.num_snooze_skips = state_dict.get("num_snooze_skips", 0)
        self._num_times_yielded_from_inner_dataset = state_dict.get("num_times_yielded_from_inner_dataset", 0)
        if self._is_inner_dataset_stateful():
            self.dataset.load_state_dict(state_dict.get("dataset_state_dict", {}))

    def _is_inner_dataset_stateful(self) -> bool:
        return hasattr(self.dataset, "state_dict") and hasattr(self.dataset, "load_state_dict")
