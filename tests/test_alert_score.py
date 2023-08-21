"""Test alert score
"""

import pytest
from deepeval.metrics.alert_score import assert_alert_score

@pytest.mark.asyncio
async def test_factual_consistency():
    assert_alert_score(
        generated_text="Who won the FIFA World Cup in 2018?",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )
