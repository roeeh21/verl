# Copyright 2025 AI21 Labs

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from verl.utils.dataset.amalgam_dataset import AmalgamDataset


def init_amalgam_dataset_with_mock_config(data_path: str, max_prompt_length: int = 8 * 1024) -> AmalgamDataset:
    # Create a config object similar to what RayPPOTrainer would pass
    config = SimpleNamespace(
        prompt_key="messages",
        max_prompt_length=max_prompt_length,
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
    return AmalgamDataset(
        data_files=data_path,
        tokenizer=tokenizer,
        processor=None,  # Not used but kept for interface compatibility
        config=config,
    )


def test_dataset_filtering_with_invalid_tools():
    # contain `tools` with `1337` as value (should raise a ValueError)
    pth_invalid_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_invalid_tools.json"

    with pytest.raises(ValueError):
        _ = init_amalgam_dataset_with_mock_config(pth_invalid_tools)


def test_dataset_filtering_with_empty_tools():
    # contain `tools` with `null`
    pth_null_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_null_tools.json"
    amalgam_dataset_with_null_tools = init_amalgam_dataset_with_mock_config(pth_null_tools)
    df_with_null_tools = amalgam_dataset_with_null_tools.datasets["nemotron_tool_use_test"].dataframe

    # contain `tools` with empty list
    pth_empty_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_empty_tools.json"
    amalgam_dataset_with_empty_tools = init_amalgam_dataset_with_mock_config(pth_empty_tools)
    df_with_empty_tools = amalgam_dataset_with_empty_tools.datasets["nemotron_tool_use_test"].dataframe

    # no `tools` field
    pth_without_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_no_tools.json"
    amalgam_dataset_without_tools = init_amalgam_dataset_with_mock_config(pth_without_tools)
    df_without_tools = amalgam_dataset_without_tools.datasets["nemotron_tool_use_test"].dataframe

    # all dataframes should have equal length
    assert len(df_with_null_tools) == len(df_with_empty_tools)
    assert len(df_with_null_tools) == len(df_without_tools)


def test_dataset_filtering_with_tools():
    # no `tools` field
    pth_without_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_no_tools.json"
    amalgam_dataset_without_tools = init_amalgam_dataset_with_mock_config(pth_without_tools)
    df_without_tools = amalgam_dataset_without_tools.datasets["nemotron_tool_use_test"].dataframe

    # contain `tools` with non-empty list
    pth_with_tools = "gs://ai21-publishing-studio-experiments/verl-tests/amalgam_8k_with_added_tools.json"
    amalgam_dataset_with_tools = init_amalgam_dataset_with_mock_config(pth_with_tools)
    df_with_tools = amalgam_dataset_with_tools.datasets["nemotron_tool_use_test"].dataframe

    assert len(df_without_tools) == 100
    assert len(df_with_tools) < 100
