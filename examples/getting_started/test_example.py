import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric, GEval, AnswerRelevancyMetric

# To run this file: deepeval test run <file_name>.py


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."
        ],
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_coherence():
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit? I want a full refund.",
        # Replace this with the actual output from your LLM application
        actual_output="If the shoes don't fit, the customer wants a full refund.",
    )
    assert_test(test_case, [coherence_metric])


class LengthMetric(BaseMetric):
    # This metric checks if the output length is greater than 10 characters
    def __init__(self, max_length: int = 10):
        self.threshold = max_length

    def measure(self, test_case: LLMTestCase):
        self.success = len(test_case.actual_output) > self.threshold
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
    length_metric = LengthMetric(max_length=10)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
    )
    assert_test(test_case, [length_metric])


def test_everything():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    length_metric = LengthMetric(max_length=10)
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is logical, has flow, and is easy to understand and follow.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."
        ],
    )
    assert_test(
        test_case, [answer_relevancy_metric, coherence_metric, length_metric]
    )


# Just an example of prompt_template
prompt_template = """You are a helpful assistant, answer the following question in a non-judgemental tone.

Question:
{question}
"""


# Although the values in this example are hardcoded,
# you should ideally pass in variables to keep things dynamic
@deepeval.log_hyperparameters(model="gpt-4", prompt_template=prompt_template)
def hyperparameters():
    return {"chunk_size": 500, "temperature": 0}
