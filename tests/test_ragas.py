import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    RagasMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    ContextRecallMetric,
    ConcisenessMetric,
    CorrectnessMetric,
    CoherenceMetric,
    MaliciousnessMetric,
)
from deepeval.metrics.ragas_metric import AnswerRelevancyMetric
from deepeval.evaluator import assert_test

query = "Who won the FIFA World Cup in 2018?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = [
    "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."
]


@pytest.mark.skip(reason="openai is expensive")
def test_ragas_score():
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        context=context,
    )
    metric = RagasMetric()

    with pytest.raises(AssertionError):
        assert_test(
            test_case=[test_case],
            metrics=[metric],
        )


@pytest.mark.skip(reason="openai is expensive")
def test_everything():
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        context=context,
    )
    metric1 = ContextualRelevancyMetric()
    metric2 = FaithfulnessMetric()
    metric3 = ContextRecallMetric()
    metric4 = ConcisenessMetric()
    metric5 = CorrectnessMetric()
    metric6 = CoherenceMetric()
    metric7 = MaliciousnessMetric()
    metric8 = AnswerRelevancyMetric()
    assert_test(
        test_case,
        [
            metric1,
            metric2,
            metric3,
            metric4,
            metric5,
            metric6,
            metric7,
            metric8,
        ],
    )
