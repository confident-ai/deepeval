"""Test alert score
"""
import os
from deepeval.metrics.alert_score import assert_alert_score
from deepeval.metrics.alert_score import AlertScoreMetric
from .utils import assert_viable_score

IMPLEMENTATION_NAME = "Alert"
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME

query = "Who won the FIFA World Cup in 2018?"
generated_text = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."


def test_alert_score():
    assert_alert_score(
        query="Who won the FIFA World Cup in 2018?",
        generated_text="Winners of the FIFA world cup were the French national football team",
        expected_output="French national football team",
        context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    )


def test_alert_score_metric():
    metric = AlertScoreMetric()
    score = metric.measure(
        query=query,
        generated_text=generated_text,
        expected_output=expected_output,
        context=context,
    )
    assert metric.is_successful(), "Overall score metric not working"
    assert_viable_score(score)
