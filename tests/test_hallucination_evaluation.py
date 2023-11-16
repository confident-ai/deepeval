import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.vectara_hallucination_evaluation import (
    HallucinationEvaluationMetric,
)
from deepeval.evaluator import assert_test


def test_hallucination_evaluation_metric():
    metric = HallucinationEvaluationMetric(minimum_score=0.5)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="A person on a horse jumps over a broken down airplane.",
        context=[
            "A person is at a diner, ordering an omelette."
        ],
    )
    assert_test(test_case, [metric])