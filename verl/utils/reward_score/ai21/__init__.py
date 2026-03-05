# Copyright 2025 AI21 Labs
import numpy as np

from verl.utils.reward_score.ai21.exact_match_metric import exact_match_metric
from verl.utils.reward_score.ai21.float_exact_match_metric import float_exact_match
from verl.utils.reward_score.ai21.jlm_absolute_judgement_metric import jlm_absolute_judgement_metric
from verl.utils.reward_score.ai21.jlm_relative_judgement_metric import jlm_relative_judgement_metric
from verl.utils.reward_score.ai21.math_dapo_metric import math_dapo_metric

# For now, we have a static list of metrics and not dynamic, to make it stateless and avoid multiprocessing issues.
ai21_metrics = {
    "prometheus_absolute_judgement": jlm_absolute_judgement_metric,
    "prometheus_relative_judgement": jlm_relative_judgement_metric,
    "critic_absolute_judgement": jlm_absolute_judgement_metric,
    "critic_relative_judgement": jlm_relative_judgement_metric,
    "jlm_absolute_judgement": jlm_absolute_judgement_metric,
    "jlm_relative_judgement": jlm_relative_judgement_metric,
    "float_exact_match": float_exact_match,
    "exact_match": exact_match_metric,
    "math_dapo": math_dapo_metric,  # Add math_dapo metric
}


def ai21_compute_score(data_source, solution_str, ground_truth: np.ndarray, extra_info=None):
    """
    Compute score using the legacy VeRL metrics system.

    This maintains backward compatibility with existing VeRL reward computations.
    """
    return _compute_legacy_score(data_source, solution_str, ground_truth, extra_info)


def _compute_legacy_score(data_source, solution_str, ground_truth: np.ndarray, extra_info=None):
    """
    Compute score using the legacy VeRL metrics system.

    This maintains backward compatibility with existing VeRL reward computations.
    """
    # Handle regular AI21 metrics (ground_truth is an np.ndarray but is actually a list of dicts)
    total_score = 0
    for metric_dict in ground_truth:
        metric_name: str = metric_dict["metric_name"]
        metric_args: dict = metric_dict["metric_args"]
        metric_weight: float = metric_dict.get("metric_weight", 1.0)
        if metric_name not in ai21_metrics:
            raise ValueError(f"Metric {metric_name} not supported")
        metric_score = ai21_metrics[metric_name](solution_str, metric_args)
        total_score += metric_score * metric_weight

    return total_score


# Backward compatibility - ensure existing metric names are still accessible
def get_legacy_metric_function(metric_name: str):
    """Get a legacy metric function by name for backward compatibility."""
    return ai21_metrics.get(metric_name)


# Export list for the module
__all__ = [
    # Main compute score functions
    "ai21_compute_score",  # Legacy/compatible version
    # Legacy metric functions
    "jlm_absolute_judgement_metric",
    "jlm_relative_judgement_metric",
    "float_exact_match",
    "exact_match_metric",
    "math_dapo_metric",
    # Utility functions
    "get_legacy_metric_function",
    # Data structures
    "ai21_metrics",
]
