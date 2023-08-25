"""Test alert score
"""
import pytest
from deepeval.api import Api
from deepeval.metrics.overall_score import assert_overall_score
from deepeval.metrics.overall_score import OverallScoreMetric
from .utils import assert_viable_score
import os

IMPLEMENTATION_NAME = "Overall"
TEST_API_KEY = "u1s5aFlB6kRyVz/16CZuc7JOQ7e7sCw00N7nfeMZOrk="
os.environ["CONFIDENT_AI_API_KEY"] = TEST_API_KEY
os.environ["CONFIDENT_AI_IMP_NAME"] = IMPLEMENTATION_NAME

query = "Who won the FIFA World Cup in 2018?"
output = "Winners of the FIFA world cup were the French national football team"
expected_output = "French national football team"
context = "The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship."

client = Api(api_key=TEST_API_KEY)

metric = OverallScoreMetric()


@pytest.fixture
def score_1():
    return metric.measure(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )


@pytest.fixture
def score_2():
    return metric.measure(
        query=query,
        output=output,
        expected_output=expected_output,
        context="He doesn't know how to code",
    )


@pytest.fixture
def score_3():
    return metric.measure(
        query=query,
        output="Not relevant",
        expected_output=expected_output,
        context="He doesn't know how to code",
    )


@pytest.fixture
def score_4():
    return metric.measure(
        query=query,
        output="Not relevant",
        expected_output="STranger things",
        context="He doesn't know how to code",
    )


def test_overall_score():
    os.environ["CONFIDENT_AI_API_KEY"] = TEST_API_KEY
    assert_overall_score(
        query=query,
        output=output,
        expected_output=expected_output,
        context=context,
    )


def test_overall_score_worst_context(score_2, score_1):
    assert score_2 < score_1, "Worst context."


def test_overall_score_worst_output(score_3, score_2):
    assert score_3 < score_2, "Worst output and context."


def test_worst_expected_output(score_4, score_3):
    assert score_4 < score_3, "Worst lol"


def test_overall_score_metric():
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
    assert FOUND, f"{IMPLEMENTATION_NAME} not found in {[x['name'] for x in imps]}"
