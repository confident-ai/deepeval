"""Test alert score
"""
import os

import pytest
from deepeval.test_case import LLMTestCase
from deepeval.api import Api
from deepeval.metrics.overall_score import (
    OverallScoreMetric,
    assert_overall_score,
)
from deepeval.run_test import assert_test, run_test

from .utils import assert_viable_score

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


class TestOverallScore:
    metric = OverallScoreMetric()

    def test_overall_score(self):
        os.environ["CONFIDENT_AI_API_KEY"] = TEST_API_KEY
        assert_overall_score(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context,
        )

    def test_overall_score_worst_context(self):
        test_case = LLMTestCase(
            query=query,
            output=output,
            expected_output=expected_output,
            context="He doesn't know how to code",
        )
        test_case_2 = LLMTestCase(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context,
        )
        scores = run_test([test_case, test_case_2], metrics=[self.metric])
        assert scores[1].score > scores[0].score, "Failed the test"

    def test_overall_score_worst_output(self):
        test_case = LLMTestCase(
            query=query,
            output="Not relevant",
            expected_output=expected_output,
            context="He doesn't know how to code",
        )
        score_3 = self.metric.measure(test_case)
        test_case_2 = LLMTestCase(
            query=query,
            output=output,
            expected_output=expected_output,
            context="He doesn't know how to code",
        )
        scores = run_test([test_case, test_case_2], metrics=[self.metric])
        assert scores[0] > scores[1]

    def test_worst_expected_output(self):
        test_case = LLMTestCase(
            query=query,
            output="Not relevant",
            expected_output="STranger things",
            context="He doesn't know how to code",
        )
        score_4 = self.metric.measure(test_case)
        test_case_2 = LLMTestCase(
            query=query,
            output="Not relevant",
            expected_output=expected_output,
            context="He doesn't know how to code",
        )
        scores = run_test([test_case, test_case_2], metrics=[self.metric])
        assert scores[0] > scores[1]

    def test_overall_score_metric(self):
        test_case = LLMTestCase(
            query=query,
            output=output,
            expected_output=expected_output,
            context=context,
        )
        result = run_test([test_case], metrics=[self.metric])
        assert_viable_score(result.score)

    def test_overall_score_metric_no_query(self):
        test_case = LLMTestCase(
            output=output,
            expected_output=expected_output,
            context=context,
        )
        assert_test([test_case], metrics=[self.metric])

    def test_overall_score_metric_no_query_no_context(self):
        test_case = LLMTestCase(
            output=output,
            expected_output=expected_output,
        )
        result = run_test([test_case], metrics=[self.metric])
        assert result[0].success, "Overall score metric not working"
        assert_viable_score(result[0].score)

    def test_overall_score_metric_no_context_no_expected_output(self):
        test_case = LLMTestCase(
            query=query,
            output=output,
        )
        score = self.metric.measure(test_case)
        result = run_test([test_case], metrics=[self.metric])
        assert result[0].success, "Overall score metric not working"
        assert_viable_score(result[0].score)

    def test_implementation_inside_overall(self):
        imps = self.client.list_implementations()
        FOUND = False
        for imp in imps:
            if imp["name"] == IMPLEMENTATION_NAME:
                FOUND = True
        assert (
            FOUND
        ), f"{IMPLEMENTATION_NAME} not found in {[x['name'] for x in imps]}"
