# Copyright 2025 AI21 Labs
from verl.utils.reward_score.ai21.jlm_relative_judgement_metric import (
    _BAD_FORMAT_REWARD,
    _CORRECT_JUDGEMENT_REWARD,
    _INCORRECT_JUDGEMENT_REWARD,
    jlm_relative_judgement_metric,
)


def test_relative_judgement_incorrect_format():
    completion = "The first one is better [A]"
    metric_args = {"ground_truth": 0}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_relative_judgement_incorrect_format_duplicate_result():
    completion = "The first one is better [RESULT] A . Actually, the second one is better. [RESULT] B"
    metric_args = {"ground_truth": 0}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_relative_judgement_first_better_correct():
    completion = "The first one is better [RESULT] A"
    metric_args = {"ground_truth": 0}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_second_better_correct():
    completion = "The second one is better [RESULT] B"
    metric_args = {"ground_truth": 1}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_tie_correct():
    completion = "They are equal [RESULT] Tie"
    metric_args = {"ground_truth": -1}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_first_better_incorrect():
    completion = "The first one is better [RESULT] A"
    metric_args = {"ground_truth": 1}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_relative_judgement_second_better_incorrect():
    completion = "The second one is better [RESULT] B"
    metric_args = {"ground_truth": 0}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_relative_judgement_incorrect_format_sfr():
    completion = "The first one is better A"
    metric_args = {"ground_truth": 0, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_relative_judgement_first_better_correct_sfr():
    completion = "The first one is better **Result:** A"
    metric_args = {"ground_truth": 0, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_second_better_correct_sfr():
    completion = "The second one is better **Result:** B"
    metric_args = {"ground_truth": 1, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_tie_correct_sfr():
    completion = "They are equal **Result:** Tie"
    metric_args = {"ground_truth": -1, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_relative_judgement_first_better_incorrect_sfr():
    completion = "The first one is better **Result:** A"
    metric_args = {"ground_truth": 1, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_relative_judgement_second_better_incorrect_sfr():
    completion = "The second one is better **Result:** B"
    metric_args = {"ground_truth": 0, "jlm_protocol": "sfr"}
    result = jlm_relative_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD
