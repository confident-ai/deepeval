"""Tests for answer relevancy
"""
from deepeval.test_case import LLMTestCase
from deepeval.run_test import run_test, assert_test
from deepeval.metrics.ranking_similarity import (
    RankingSimilarity,
    assert_ranking_similarity,
)

list1 = ["a", "b"]
list2 = ["b", "c"]
list3 = ["b", "a"]
input = "Placeholder Input"
actual_output = "Placeholder Output"


def test_assert_similar_ranking():
    assert_ranking_similarity(
        input=input,
        actual_output=actual_output,
        context=list1,
        retrieval_context=list2,
        minimum_score=0.4,
    )


def test_similar_ranking_2():
    scorer = RankingSimilarity(minimum_score=0.5)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=list1,
        retrieval_context=list2,
    )
    test_case_2 = LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=list1,
        retrieval_context=list3,
    )
    test_results = run_test([test_case, test_case_2], metrics=[scorer])
    assert (
        test_results[1].score > test_results[0].score
    ), "Similarity not working"


def test_similar_ranking_3():
    metric = RankingSimilarity(minimum_score=0.3)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=list1,
        retrieval_context=list2,
    )
    assert_test(test_cases=test_case, metrics=[metric])
