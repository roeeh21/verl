# Copyright 2025 AI21 Labs
import math
import re

DEFAULT_ANSWER_REGEX = "<answer>(.*)</answer>"
DEFAULT_ALLOWED_FLOAT_DIFFERENCE = 0.001
# Often the answer format is demonstrated in the prompt, which causes unwanted detections by the regex
# Default answers to remove is ment to overcome this issue
DEFAULT_ANSWERS_TO_REMOVE = ["answer here", "Paris"]


def float_exact_match(solution_str, metric_args: dict) -> int:
    ground_truth = metric_args["ground_truth"]
    answer_regex = metric_args.get("answer_regex", DEFAULT_ANSWER_REGEX)
    allowed_float_difference = metric_args.get("allowed_float_difference", DEFAULT_ALLOWED_FLOAT_DIFFERENCE)
    answers_to_remove = metric_args.get("answers_to_remove", DEFAULT_ANSWERS_TO_REMOVE)
    final_answer = re.findall(answer_regex, solution_str)
    # solution str contains the prompt
    final_answer = [x for x in final_answer if x not in answers_to_remove]
    if len(final_answer) != 1:
        return 0

    final_answer = final_answer[0].strip().lower()
    ground_truth = ground_truth.strip().lower()

    try:
        is_valid = math.isclose(float(final_answer), float(ground_truth), abs_tol=allowed_float_difference)
        return int(is_valid)
    except ValueError:
        pass

    return int(final_answer == ground_truth)
