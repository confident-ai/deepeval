import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas import (
    RagasMetric,
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASContextualEntitiesRecall,
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
)
from deepeval import assert_test
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

query = "Who won the FIFA World Cup in 2018 and what was the score?"
output = (
    "The winners of the FIFA World Cup in 2018 were the French national football team. "
    "They defeated the Croatian national football team with a score of 4-2 in the final match. "
    "The final was held at Luzhniki Stadium in Moscow, Russia on July 15, 2018. "
    "The French team played exceptionally well, with goals scored by Mario Mandzukic (own goal), Antoine Griezmann, Paul Pogba, and Kylian Mbappe. "
    "Croatia fought valiantly, with goals by Ivan Perisic and Mario Mandzukic, but ultimately could not overcome France's lead. "
    "This victory marked France's second World Cup title, with their first win being in 1998 when they hosted the tournament."
)
expected_output = "French national football team"
context = [
    "The FIFA World Cup in 2018 was won by the French national football team.",
    "I am birdy.",
    "I am a froggy.",
    "The French defeated Croatia 4-2 in the final FIFA match to claim the championship.",
    "The final match took place on July 15, 2018, in Moscow, Russia at Luzhniki Stadium.",
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected_output,
        retrieval_context=context,
        context=context,
    )
    contextual_precision = RAGASContextualPrecisionMetric()
    context_recall = RAGASContextualRecallMetric()
    context_entities_recall = RAGASContextualEntitiesRecall()
    answer_relevancy = RAGASAnswerRelevancyMetric(embeddings=embeddings)
    faithfulness = RAGASFaithfulnessMetric()

    ragas = RagasMetric(
        model=ChatOpenAI(model_name="gpt-3.5-turbo"), embeddings=embeddings
    )
    assert_test(test_case, [ragas])
