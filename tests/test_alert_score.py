"""Test alert score
"""
import os
from deepeval.metrics.alert_score import assert_alert_score
from deepeval.metrics.alert_score import AlertScoreMetric
from deepeval.client import Client
from .utils import assert_viable_score

IMPLEMENTATION_NAME = "Alert"
TEST_API_KEY = "u1s5aFlB6kRyVz/16CZuc7JOQ7e7sCw00N7nfeMZOrk="
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME
os.environ["CONFIDENT_AI_API_KEY"] = TEST_API_KEY

query = "Who won the FIFA World Cup in 2018?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."


def test_alert_score():
    assert_alert_score(
        query="Who won the FIFA World Cup in 2018?",
        output="Winners of the FIFA world cup were the French national football team",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )


def test_alert_score_metric():
    metric = AlertScoreMetric()
    score = metric.measure(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), "Overall score metric not working"
    assert_viable_score(score)


def test_implementation_inside_overall():
    client = Client(TEST_API_KEY)
    imps = client.list_implementations()
    FOUND = False
    for imp in imps:
        if imp["name"] == IMPLEMENTATION_NAME:
            FOUND = True
    assert FOUND, f"{IMPLEMENTATION_NAME} not found in {[x['name'] for x in imps]}"
