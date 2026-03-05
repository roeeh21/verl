# Copyright 2025 AI21 Labs
# Copy pasted from alignment-evaluations prometheus_jlm.py . TODO: Connect to alignment-evaluations?
def _get_prediction_integer(generation: str, jlm_protocol: str):
    # Look for rating in format "[RESULT] X" where X is 1-5
    if jlm_protocol == "prometheus":
        result_separator = "[RESULT]"
    elif jlm_protocol == "sfr":
        result_separator = "**Result:**"
    else:
        raise ValueError(f"Unknown JLM protocol: {jlm_protocol}")
    prediction = ""
    generation = generation.strip()
    if result_separator in generation:
        result_parts = generation.split(result_separator)
        # We require exactly two parts, to also discard cases where the model repeats the rating.
        if len(result_parts) == 2:
            prediction = result_parts[-1].strip()

    prediction = prediction.rstrip(".")  # Allow for a trailing period.
    if prediction.isdigit():
        return int(prediction)

    return -1


_MINIMAL_GOOD_ANSWER_THRESHOLD = 4

_BAD_FORMAT_REWARD = 0.0
_INCORRECT_JUDGEMENT_REWARD = 0.2
_CORRECT_JUDGEMENT_REWARD = 1.0


def jlm_absolute_judgement_metric(completion: str, metric_args: dict) -> float:
    ground_truth = metric_args["ground_truth"]
    jlm_protocol = metric_args.get("jlm_protocol", "prometheus")
    prediction = _get_prediction_integer(completion, jlm_protocol)
    if prediction == -1:
        # print("bad format")
        # print(completion)
        return _BAD_FORMAT_REWARD
    ground_truth_compatible_value = int(prediction >= _MINIMAL_GOOD_ANSWER_THRESHOLD)
    is_correct = ground_truth_compatible_value == ground_truth
    return _CORRECT_JUDGEMENT_REWARD if is_correct else _INCORRECT_JUDGEMENT_REWARD
