# Copyright 2025 AI21 Labs

from verl.utils.reward_score.ai21.jlm_absolute_judgement_metric import (
    _BAD_FORMAT_REWARD,
    _CORRECT_JUDGEMENT_REWARD,
    _INCORRECT_JUDGEMENT_REWARD,
    jlm_absolute_judgement_metric,
)


def test_absolute_judgement_incorrect_format():
    completion = "This is a test"
    metric_args = {"ground_truth": "5"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_absolute_judgement_incorrect_format_duplicate_result():
    completion = "Great job! [RESULT] 5. Actually, bad job. [RESULT] 1"
    metric_args = {"ground_truth": 0}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_absolute_judgement_positive_ground_truth_correct_judgement():
    completion = "Great job! [RESULT] 5"
    metric_args = {"ground_truth": 1}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_negative_ground_truth_incorrect_judgement():
    completion = "Great job! [RESULT] 5"
    metric_args = {"ground_truth": 0}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_negative_ground_truth_correct_judgement():
    completion = "Bad job! [RESULT] 3"
    metric_args = {"ground_truth": 0}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_positive_ground_truth_incorrect_judgement():
    completion = "Bad job. [RESULT] 3"
    metric_args = {"ground_truth": 1}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_incorrect_format_sfr():
    completion = "This is a test"
    metric_args = {"ground_truth": "5", "jlm_protocol": "sfr"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _BAD_FORMAT_REWARD


def test_absolute_judgement_positive_ground_truth_correct_judgement_sfr():
    completion = "Great job! **Result:** 5"
    metric_args = {"ground_truth": 1, "jlm_protocol": "sfr"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_negative_ground_truth_incorrect_judgement_sfr():
    completion = "Great job! **Result:** 5"
    metric_args = {"ground_truth": 0, "jlm_protocol": "sfr"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_negative_ground_truth_correct_judgement_sfr():
    completion = "Bad job! **Result:** 3"
    metric_args = {"ground_truth": 0, "jlm_protocol": "sfr"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _CORRECT_JUDGEMENT_REWARD


def test_absolute_judgement_positive_ground_truth_incorrect_judgement_sfr():
    completion = "Bad job. **Result:** 3"
    metric_args = {"ground_truth": 1, "jlm_protocol": "sfr"}
    result = jlm_absolute_judgement_metric(completion, metric_args)
    assert result == _INCORRECT_JUDGEMENT_REWARD
