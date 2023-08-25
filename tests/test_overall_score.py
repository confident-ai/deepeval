"""Test alert score
"""

import pytest
from deepeval.api import Api
from deepeval.metrics.overall_score import assert_overall_score
from deepeval.metrics.overall_score import OverallScoreMetric
from .utils import assert_viable_score
import os

IMPLEMENTATION_NAME = "Fifa"
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME

query = "Who won the FIFA World Cup in 2018?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."

client = Api()

@pytest.mark.asyncio
async def test_overall_score():
    assert_overall_score(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )


@pytest.mark.asyncio
async def test_overall_score_metric():
    metric = OverallScoreMetric()
    score = metric.measure(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), "Overall score metric not working"
    assert_viable_score(score)


def test_implementation_inside_overall():
    imps = client.list_implementations()
    FOUND = False
    for imp in imps:
        if imp["name"] == IMPLEMENTATION_NAME:
            FOUND = True
    assert FOUND, f"Not found in {imps}"
