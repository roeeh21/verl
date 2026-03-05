# Copyright 2025 AI21 Labs
import re

PATTERN = r"""(?x)                                # Enable verbose mode for readability
<\|begin_of_thought\|>                # Match thought section start
\s*                                   # Optional whitespace
(?P<thought>                          # Named capture group for thought content
    [\s\S]*?                         # Any characters including newlines (non-greedy)
    (?:\n\n|\r\n\r\n|$)             # Matches double newlines or end of string
    .*?                              # More content (non-greedy)
)
\s*                                   # Optional whitespace
<\|end_of_thought\|>                  # Match thought section end
\s*                                   # Optional whitespace
<\|begin_of_solution\|>               # Match solution section start
\s*                                   # Optional whitespace
(?P<solution>                         # Named capture group for solution content
    [\s\S]*?                         # Any characters including newlines (non-greedy)
    (?:\\\boxed\{[^}]*\})?          # Optional boxed content
    [\s\S]*?                         # More content (non-greedy)
)
\s*                                   # Optional whitespace
<\|end_of_solution\|>                 # Match solution section end
"""


def extract_xml_answer(text: str) -> str:
    matches = list(re.finditer(r"boxed{([^}]+)}", text))
    if len(matches) == 1:
        return matches[0].group(1).strip()
    return text.strip()


# Reward functions
def correctness_reward_func(completion, answer, **kwargs) -> list[float]:
    extracted_response = extract_xml_answer(completion)
    return 0.8 if extracted_response == answer else 0.0


def strict_format_reward_func(completion, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    match = re.match(PATTERN, completion)
    return 0.2 if match else 0.0


def format_reward_func(completion, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a boxed solution."""
    matches = list(re.finditer(r"boxed{([^}]+)}", completion))
    return 0.2 if len(matches) == 1 else 0.0


reward_functions = [
    correctness_reward_func,
    format_reward_func,
]


def exact_match_metric(completion: str, metric_args: dict) -> float:
    results = [f(completion=completion, answer=metric_args["ground_truth"]) for f in reward_functions]
    print(f"Results: {[f'{f.__name__}: {r}' for f, r in zip(reward_functions, results, strict=False)]}")
    return sum(results)
