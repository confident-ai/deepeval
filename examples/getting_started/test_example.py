import pytest
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluator import assert_test
from deepeval.metrics.llm_eval_metric import LLMEvalMetric
from deepeval.metrics.base_metric import BaseMetric
import deepeval


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


def test_summarization():
    input = "What if these shoes don't fit? I want a full refund."

    # Replace this with the actual output from your LLM application
    actual_output = "If the shoes don't fit, the customer wants a full refund."

    summarization_metric = LLMEvalMetric(
        name="Summarization",
        criteria="Summarization - determine if the actual output is an accurate and concise summarization of the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        minimum_score=0.5,
    )
    test_case = LLMTestCase(input=input, actual_output=actual_output)
    assert_test(test_case, [summarization_metric])


class LengthMetric(BaseMetric):
    # This metric checks if the output length is greater than 10 characters
    def __init__(self, max_length: int = 10):
        self.minimum_score = max_length

    def measure(self, test_case: LLMTestCase):
        self.success = len(test_case.actual_output) > self.minimum_score
        if self.success:
            self.score = 1
        else:
            self.score = 0
        return self.score

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
    summarization_metric = LLMEvalMetric(
        name="Summarization",
        criteria="Summarization - determine if the actual output is an accurate and concise summarization of the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        minimum_score=0.5,
    )

    test_case = LLMTestCase(
        input=input, actual_output=actual_output, context=context
    )
    assert_test(
        test_case,
        [factual_consistency_metric, length_metric, summarization_metric],
    )


@deepeval.set_hyperparameters
def hyperparameters():
    return {
        "model": "GPT-4",
        "prompt_template": """You are a helpful assistant, answer the following question in a non-judgemental tone.

        Question:
        {question}
        """,
    }
