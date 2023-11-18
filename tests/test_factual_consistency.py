import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FactualConsistencyMetric

from deepeval.evaluator import assert_test


def test_factual_consistency_metric():
    metric = FactualConsistencyMetric(minimum_score=0.8)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=[
            "Python is a high-level, versatile, and interpreted programming language known for its simplicity and readability."
        ],
    )
    assert_test(test_case, [metric])


def test_factual_consistency_metric_2():
    metric = FactualConsistencyMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is NOT a programming language."],
    )
    with pytest.raises(AssertionError):
        assert_test(test_case, [metric])


def test_factual_consistency_metric_3():
    metric = FactualConsistencyMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is a snake."],
    )
    with pytest.raises(AssertionError):
        assert_test(test_case, [metric])
