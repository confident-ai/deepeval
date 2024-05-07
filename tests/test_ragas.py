import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas import (
    RagasMetric,
    RAGASContextualPrecisionMetric,
    RAGASContextualRelevancyMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualRecallMetric,
    RAGASAnswerRelevancyMetric,
)
from deepeval import assert_test
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

query = "Who won the FIFA World Cup in 2018 and what was the score?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = [
    "The FIFA World Cup in 2018 was won by the French national football team.",
    "I am birdy",
    "I am a froggy",
    "The French defeated Croatia 4-2 in the final FIFA match to claim the championship.",
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        retrieval_context=context,
        context=context,
    )
    metric1 = RAGASContextualRelevancyMetric()
    metric2 = RAGASFaithfulnessMetric()
    metric3 = RAGASContextualRecallMetric()
    metric8 = RAGASAnswerRelevancyMetric(embeddings=embeddings)
    metric9 = RAGASContextualPrecisionMetric()
    metric10 = RagasMetric(
        model=ChatOpenAI(model_name="gpt-3.5-turbo"), embeddings=embeddings
    )
    assert_test(
        test_case,
        [
            metric1,
            metric2,
            # metric3,
            # metric4,
            # metric5,
            # metric6,
            # metric7,
            # metric8,
            # metric9,
            metric10,
        ],
    )
