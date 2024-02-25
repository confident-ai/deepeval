import pytest
from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

input = "Who won the FIFA World Cup in 2018 and what was the score?"
actual_output = (
    "Winners of the FIFA world cup were the French national football team"
)
expected_output = "French national football team"
retrieval_context = [
    "The FIFA World Cup in 2018 was won by the French national football team.",
    "I am birdy",
    "I am a froggy",
    "The French defeated Croatia 4-2 in the final FIFA match to claim the championship.",
]


@pytest.mark.skip(reason="openai is expensive")
def test_rag_metrics():
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
    )
    metric1 = AnswerRelevancyMetric(threshold=0.5)
    metric2 = FaithfulnessMetric(threshold=0.5)
    metric3 = ContextualRelevancyMetric(threshold=0.5)
    metric4 = ContextualPrecisionMetric(threshold=0.5)
    metric5 = ContextualRecallMetric(threshold=0.5)
    assert_test(test_case, [metric1, metric2, metric3, metric4, metric5])
