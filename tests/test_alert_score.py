"""Test alert score
"""

import pytest
from deepeval.metrics.alert_score import assert_alert_score


@pytest.mark.asyncio
async def test_alert_score():
    assert_alert_score(
        query="Who won the FIFA World Cup in 2018?",
        generated_text="Winners of the FIFA world cup were the French national football team",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )
