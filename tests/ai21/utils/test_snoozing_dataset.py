# Copyright 2025 AI21 Labs
from copy import deepcopy
from types import SimpleNamespace

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from verl.utils.dataset.amalgam_dataset import AmalgamDataset
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.snoozing_dataset import SnoozingDataset


def test_snoozing_dataset_resume():
    # Create a config object similar to what RayPPOTrainer would pass
    config = SimpleNamespace(
        prompt_key="messages",
        max_prompt_length=8192,
        filter_overlong_prompts=True,
        truncation="left",
        shuffle=False,
        return_raw_chat=True,
        finite=True,
        cache_dir="/dev/shm/.cache/verl/rlhf",
    )

    # Add get method to mimic OmegaConf behavior
    config.get = lambda key, default=None: getattr(config, key, default)

    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Reasoning-3B")
    pth = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_small.json"
    amalgam_dataset = AmalgamDataset(
        data_files=pth,
        tokenizer=tokenizer,
        processor=None,  # Not used but kept for interface compatibility
        config=config,
    )

    dataset = SnoozingDataset(amalgam_dataset, "id", verbose=True)

    print(dataset.get_dataset_stats())
    dataloader = StatefulDataLoader(
        dataset=dataset, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn, shuffle=False
    )

    dl_iter = iter(dataloader)
    ids_loaded_in_first_iter = []
    for batch_idx in range(10):
        batch = next(dl_iter)
        ids_loaded_in_first_iter.append(batch["id"][0])
        if batch_idx == 0:
            # Test note: we snooze an example AFTER it has been yielded from the inner dataset
            # because naive fast forwarding would skip the example as its ID is in the snooze set
            dataset.snooze_example(batch["id"][0], 1)

    dataloader_state_dict = deepcopy(dataloader.state_dict())

    original_next_ids = []
    for batch_idx in range(2):
        batch = next(dl_iter)
        original_next_ids.append(batch["id"][0])

    amalgam_dataset_2 = AmalgamDataset(
        data_files=pth,
        tokenizer=tokenizer,
        processor=None,  # Not used but kept for interface compatibility
        config=config,
    )
    dataset_2 = SnoozingDataset(amalgam_dataset_2, "id", verbose=True, store_resume_skipped_ids=True)
    loaded_dataloader = StatefulDataLoader(
        dataset=dataset_2, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn, shuffle=False
    )
    loaded_dataloader.load_state_dict(dataloader_state_dict)

    loaded_iter = iter(loaded_dataloader)
    loaded_next_ids = []
    for batch_idx in range(2):
        batch = next(loaded_iter)
        loaded_next_ids.append(batch["id"][0])

    assert ids_loaded_in_first_iter == dataset_2._resume_skipped_ids
    assert original_next_ids == loaded_next_ids
    assert dict(dataset._snooze_counts) == dict(dataset_2._snooze_counts)
    assert len(dataset_2._snooze_counts) == 1


def test_amalgam_load_consistency():
    # Create a config object similar to what RayPPOTrainer would pass
    config = SimpleNamespace(
        prompt_key="messages",
        max_prompt_length=8192,
        filter_overlong_prompts=True,
        truncation="left",
        shuffle=False,
        return_raw_chat=True,
        finite=True,
        cache_dir="/dev/shm/.cache/verl/rlhf",
    )

    # Add get method to mimic OmegaConf behavior
    config.get = lambda key, default=None: getattr(config, key, default)

    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Reasoning-3B")
    pth = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_small.json"

    reference_ids = None

    for i in range(5):
        amalgam_dataset = AmalgamDataset(
            data_files=pth,
            tokenizer=tokenizer,
            processor=None,  # Not used but kept for interface compatibility
            config=config,
        )

        dataloader = StatefulDataLoader(
            dataset=amalgam_dataset, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn, shuffle=False
        )

        dl_iter = iter(dataloader)
        ids_loaded_in_first_iter = []
        for _ in range(10):
            batch = next(dl_iter)
            ids_loaded_in_first_iter.append(batch["id"][0])

        if reference_ids is None:
            reference_ids = ids_loaded_in_first_iter
        else:
            if reference_ids != ids_loaded_in_first_iter:
                raise AssertionError(
                    "Mismatch between first ids returned by amalgam dataset when loading it several times"
                )
