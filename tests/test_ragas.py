import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.ragas_metric import RagasMetric
from deepeval.run_test import assert_test


query = "Who won the FIFA World Cup in 2018?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."


@pytest.mark.skip(reason="openai is expensive")
def test_ragas_score():
    test_case = LLMTestCase(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    metric = RagasMetric()
    assert_test(
        test_cases=[test_case],
        metrics=[metric],
    )
