# Copyright 2025 AI21 Labs
# Copy pasted from alignment-evaluations prometheus_jlm.py . TODO: Connect to alignment-evaluations?
def _get_prediction_integer(generation, jlm_protocol: str, flip=False):
    """From CompletionRankingTask: The `ground_truth` should be:
    * 0 if the first completion is better.
    * 1 if the second completion is better.
    * -1 if they are equal."""

    prediction = ""
    if jlm_protocol == "prometheus":
        result_separator = "[RESULT]"
    elif jlm_protocol == "sfr":
        result_separator = "**Result:**"
    else:
        raise ValueError(f"Unknown JLM protocol: {jlm_protocol}")
    if result_separator in generation:
        result_parts = generation.split(result_separator)
        if len(result_parts) == 2:
            prediction = result_parts[-1].strip()
        else:
            return None
    else:
        return None

    if prediction == "A":
        if not flip:
            return 0
        else:
            return 1
    elif prediction == "B":
        if not flip:
            return 1
        else:
            return 0
    elif prediction == "Tie":
        return -1

    return None


_BAD_FORMAT_REWARD = 0.0
_INCORRECT_JUDGEMENT_REWARD = 0.2
_CORRECT_JUDGEMENT_REWARD = 1.0


def jlm_relative_judgement_metric(completion: str, metric_args: dict) -> float:
    ground_truth = metric_args["ground_truth"]
    jlm_protocol = metric_args.get("jlm_protocol", "prometheus")
    prediction = _get_prediction_integer(completion, jlm_protocol)
    if prediction is None:
        # print("bad format")
        # print(completion)
        return _BAD_FORMAT_REWARD
    is_correct = prediction == ground_truth
    return _CORRECT_JUDGEMENT_REWARD if is_correct else _INCORRECT_JUDGEMENT_REWARD
