"""Tests for toxic
"""


import pytest
from deepeval.metrics.toxic_classifier import NonToxicMetric, assert_non_toxic
from .utils import assert_viable_score

query = "Who won the FIFA World Cup in 2018?"
generated_text = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."


@pytest.mark.asyncio
async def test_non_toxic():
    assert_non_toxic(
        query="Who won the FIFA World Cup in 2018?",
        generated_text="Winners of the FIFA world cup were the French national football team",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )


@pytest.mark.asyncio
async def test_overall_score_metric():
    metric = NonToxicMetric()
    score = metric.measure(
        generated_text=generated_text, expected_output=expected_output, context=context
    )
    assert metric.is_successful(), "Overall score metric not working"
    assert_viable_score(score)
