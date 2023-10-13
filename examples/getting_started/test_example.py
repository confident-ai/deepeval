import pytest
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test
from deepeval.metrics.llm_eval_metric import LLMEvalMetric
from deepeval.metrics.metric import Metric


def test_factual_consistency():
    input = "What if these shoes don't fit?"
    context = (
        "All customers are eligible for a 30 day full refund at no extra cost."
    )

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.7)
    test_case = LLMTestCase(
        input=input, actual_output=actual_output, context=context
    )
    assert_test(test_case, [factual_consistency_metric])


def test_humor():
    input = "What if these shoes don't fit?"

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
    funny_metric = LLMEvalMetric(
        name="Funny Metric", criteria="How funny it is", minimum_score=0.3
    )
    test_case = LLMTestCase(input=input, actual_output=actual_output)
    assert_test(test_case, [funny_metric])


class LengthMetric(Metric):
    # This metric checks if the output length is greater than 10 characters
    def __init__(self, max_length: int = 10):
        self.max_length = max_length

    def measure(self, test_case: LLMTestCase):
        self.success = len(test_case.actual_output) > self.max_length
        if self.success:
            score = 1
        else:
            score = 0
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"


def test_length():
    input = "What if these shoes don't fit?"

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
    length_metric = LengthMetric(max_length=10)
    test_case = LLMTestCase(input=input, actual_output=actual_output)
    assert_test(test_case, [length_metric])


def test_everything():
    input = "What if these shoes don't fit?"
    context = (
        "All customers are eligible for a 30 day full refund at no extra cost."
    )

    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.7)
    length_metric = LengthMetric(max_length=10)
    funny_metric = LLMEvalMetric(
        name="Funny Metric", criteria="How funny it is", minimum_score=0.3
    )

    test_case = LLMTestCase(
        input=input, actual_output=actual_output, context=context
    )
    assert_test(
        test_case, [factual_consistency_metric, length_metric, funny_metric]
    )
