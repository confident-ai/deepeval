"""Tests for answer relevancy
"""
import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from deepeval.evaluator import assert_test, run_test

query = "What is Python?"
answer = "Python is a programming language?"


def test_query_answer_relevancy():
    scorer = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(input=query, actual_output=answer)
    assert_test(test_case, [scorer])


def test_compare_answer_relevancy_2():
    scorer = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(input=query, actual_output="Programming lang")
    run_test(test_case, metrics=[scorer])


def test_compare_answer_relevancy():
    metric = AnswerRelevancyMetric(minimum_score=0.5)
    query = "what is python"
    test_case_2 = LLMTestCase(
        input=query, actual_output="Python is a programming lang"
    )
    run_test(test_case_2, metrics=[metric])


def test_cross_encoder_answer_relevancy():
    metric = AnswerRelevancyMetric(
        minimum_score=0.5, model_type="cross_encoder"
    )
    test_case = LLMTestCase(input=query, actual_output=answer)
    assert_test(test_case, [metric])
