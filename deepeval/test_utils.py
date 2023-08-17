import functools
import time
from typing import Any
from .metrics.randomscore import RandomMetric
from .metrics.metric import Metric
from .metrics.bertscore_metric import BertScoreMetric
from .metrics.entailment_metric import EntailmentScoreMetric


def assert_llm_output(input: Any, output: Any, metric: Any = "entailment"):
    if metric == "exact":
        assert_exact_match(input, output)
    elif metric == "random":
        metric: RandomMetric = RandomMetric()
        metric(input, output)
    elif metric == "bertscore":
        metric: BertScoreMetric = BertScoreMetric()
        metric(input, output)
    elif metric == "entailment":
        metric: EntailmentScoreMetric = EntailmentScoreMetric()
        metric(input, output)
    elif isinstance(metric, Metric):
        metric(input, output)
    else:
        raise ValueError("Inappropriate metric")
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def assert_exact_match(text_input, text_output):
    assert text_input == text_output, f"{text_output} != {text_input}"


class TestEvalCase:
    pass


def timing_decorator(func):
    @functools.wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result

    return wrapper


def tags(tags: list):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # TODO - add logging for tags
            print(f"Tags are: {tags}")
            return result

        return wrapper

    return decorator
