import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval.evaluator import assert_test


def test_hallucination_metric():
    metric = HallucinationMetric(minimum_score=0.5)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="A blond drinking water in public.",
        context=[
            "A man with blond-hair, and a brown shirt drinking out of a public water fountain."
        ],
    )
    assert_test(test_case, [metric])


def test_hallucination_metric_2():
    metric = HallucinationMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is NOT a programming language."],
    )
    with pytest.raises(AssertionError):
        assert_test(test_case, [metric])


def test_hallucination_metric_3():
    metric = HallucinationMetric(minimum_score=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is a snake."],
    )
    with pytest.raises(AssertionError):
        assert_test(test_case, [metric])
