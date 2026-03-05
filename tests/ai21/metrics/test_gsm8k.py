# Copyright 2025 AI21 Labs
import pytest

from verl.utils.reward_score.gsm8k import compute_score


# fmt: off
@pytest.mark.parametrize(
    "solution_str, ground_truth, expected_score",
    [
        ("#### 7", "7", 1.0),
        ("#### 7$", "7", 1.0),
        ("#### 7 kg", "7", 1.0),
        ("#### 7 and some", "7", 0.0),
        ("#### 6 #### 7", "6", 0.0),
        ("#### 6 #### 7", "7", 0.0),
        ("""foo bar
#### 12,345.6$ baz """, "12345.6", 1.0), 
        ("""foo bar
#### 12,345.6$ baz lala""", "12345.6", 0.0), 
        ("""foo bar
#### 12,345.6$\n""", "12345.6", 1.0), 
        ("""foo bar
#### 12,345.6$\ndone""", "12345.6", 0.0), 
    ],
)
# fmt: on
def test_gsm8k_strict(solution_str, ground_truth, expected_score):
    score = compute_score(solution_str, ground_truth)
    assert score == expected_score
