"""Test alert score
"""

import pytest
from deepeval.metrics.overall_score import assert_overall_score
from deepeval.metrics.overall_score import OverallScoreMetric


query = "Who won the FIFA World Cup in 2018?"
generated_text = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."


@pytest.mark.asyncio
async def test_overall_score():
    assert_overall_score(
        query="Who won the FIFA World Cup in 2018?",
        generated_text="Winners of the FIFA world cup were the French national football team",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )


@pytest.mark.asyncio
async def test_overall_score_metric():
    metric = OverallScoreMetric()
    metric.measure(
        query=query,
        generated_text=generated_text,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), "Overall score metric not working"
