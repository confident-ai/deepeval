"""Tests for answer relevancy
"""
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.answer_relevancy import (
    AnswerRelevancyMetric,
    assert_answer_relevancy,
)
from deepeval.run_test import run_test, assert_test
from .utils import assert_viable_score

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
    assert_test(test_case, [scorer])


def test_compare_answer_relevancy_2():
    scorer = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output="Programming lang")
    test_case_2 = LLMTestCase(
        query=query, output="Python is a programming lang"
    )
    results = run_test([test_case, test_case_2], metrics=[scorer])
    assert results[1].score > results[0].score


def test_compare_answer_relevancy():
    metric = AnswerRelevancyMetric(minimum_score=0.5)
    query = "what is python"
    test_case = LLMTestCase(query=query, output="Programming lang")
    test_case_2 = LLMTestCase(
        query=query, output="Python is a programming lang"
    )
    result = run_test([test_case, test_case_2], metrics=[metric])
    assert result[1].score > result[0].score


def test_cross_encoder_answer_relevancy():
    scorer = AnswerRelevancyMetric(
        minimum_score=0.5, model_type="cross_encoder"
    )
    test_case = LLMTestCase(query=query, output=answer)
    score = assert_test(test_case, [scorer])
    assert_viable_score(score[0].score)
