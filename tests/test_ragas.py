import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    RagasMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    RAGASFaithfulnessMetric,
    ContextualRecallMetric,
    ConcisenessMetric,
    CorrectnessMetric,
    CoherenceMetric,
    MaliciousnessMetric,
)
from deepeval.metrics.ragas_metric import RAGASAnswerRelevancyMetric
from deepeval import assert_test, evaluate

query = "Who won the FIFA World Cup in 2018 and what was the score?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = [
    "The FIFA World Cup in 2018 was won by the French national football team.",
    "I am birdy",
    "I am a froggy",
    "The French defeated Croatia 4-2 in the final match to claim the championship.",
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


# @pytest.mark.skip(reason="openai is expensive")
def test_everything():
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        retrieval_context=context,
        context=context,
    )
    # metric1 = ContextualRelevancyMetric(model="gpt-4")
    # metric2 = RAGASFaithfulnessMetric(model="gpt-4")
    # metric3 = ContextualRecallMetric(model="gpt-4")
    # metric4 = ConcisenessMetric(model="gpt-4")
    # metric5 = CorrectnessMetric(model="gpt-4")
    # metric6 = CoherenceMetric(model="gpt-4")
    # metric7 = MaliciousnessMetric(model="gpt-4")
    # metric8 = RAGASAnswerRelevancyMetric(model="gpt-4")
    metric9 = ContextualPrecisionMetric(model="gpt-4")
    # metric10 = RagasMetric()
    assert_test(
        test_case,
        [
            # metric1,
            # metric2,
            # metric3,
            # metric4,
            # metric5,
            # metric6,
            # metric7,
            # metric8,
            metric9,
            # metric10,
        ],
    )
