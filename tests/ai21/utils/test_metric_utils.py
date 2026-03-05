# Copyright 2025 AI21 Labs
from verl.trainer.ppo.metric_utils import _get_bucket_counts


def test_get_bucket_counts():
    scores = [0, 0.01, 0.25, 1, 0.99]
    n_buckets = 4
    assert _get_bucket_counts(scores, n_buckets) == [2, 1, 0, 2]

    n_buckets = 1
    assert _get_bucket_counts(scores, n_buckets) == [5]

    scores = [0, 0, 0.5]
    n_buckets = 2
    assert _get_bucket_counts(scores, n_buckets) == [2, 1]
