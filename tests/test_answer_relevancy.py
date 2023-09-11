"""Tests for answer relevancy
"""
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.answer_relevancy import (
    AnswerRelevancyMetric,
    assert_answer_relevancy,
)
from deepeval.run_test import run_test

query = "What is Python?"
answer = "Python is a programming language?"


def test_answer_relevancy():
    assert_answer_relevancy(query, answer, minimum_score=0.5)


def test_answer_not_relevant():
    with pytest.raises(AssertionError):
        assert_answer_relevancy(query, "He is not your friend")


def test_query_answer_relevancy():
    scorer = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=answer)
    score = scorer.measure(test_case)
    assert scorer.is_successful(), f"Failed - {score}'"


def test_compare_answer_relevancy_2():
    scorer = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output="Programming lang")
    result = scorer.measure(test_case)
    test_case_2 = LLMTestCase(
        query=query, output="Python is a programming lang"
    )
    result_2 = scorer.measure(test_case_2)
    assert result_2 > result


def test_compare_answer_relevancy():
    metric = AnswerRelevancyMetric(minimum_score=0.5)
    query = "what is python"
    test_case = LLMTestCase(query=query, output="Programming lang")
    result = metric.measure(test_case)

    test_case_2 = LLMTestCase(
        query=query, output="Python is a programming lang"
    )
    result_2 = metric.measure(test_case_2)
    run_test([test_case, test_case_2], metrics=[metric])
    assert result_2 > result
