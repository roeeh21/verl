# Copyright 2025 AI21 Labs
from verl.utils.reward_score.math_dapo import compute_score


def math_dapo_metric(completion: str, metric_args: dict) -> float:
    res = compute_score(completion, metric_args["ground_truth"])
    if isinstance(res, dict):
        print(f"math_dapo_metric: {res}")
        return res.get("score", 0)
    else:
        return res
