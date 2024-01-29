import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.skip(reason="openai is expensive")
def test_answer_relevancy_again():
    input = "What if these shoes don't fit?"
    actual_output = "We offer a 45 day full-refund at no extra cost"
    retrieval_context = [
        "All customers a eligible for a 30 day full refund at no extra cost"
    ]
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)
    assert_test(test_case, [faithfulness_metric, relevancy_metric])
