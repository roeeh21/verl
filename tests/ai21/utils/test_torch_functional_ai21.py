# Copyright 2025 AI21 Labs
import torch

from verl.utils.torch_functional import get_response_mask


def test_get_response_mask():
    response_id = torch.tensor([[10, 20, 1, 0], [40, 50, 60, 70], [80, 1, 11, 22]])

    expected_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]])

    # Test with eos_token
    eos_token = 1
    mask = get_response_mask(response_id, eos_token)
    assert torch.equal(mask, expected_mask)

    # Test with responses_lengths
    responses_lengths = torch.tensor([3, 4, 2])
    mask = get_response_mask(response_id, responses_lengths=responses_lengths)
    assert torch.equal(mask, expected_mask)
