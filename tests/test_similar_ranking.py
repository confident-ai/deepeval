"""Tests for answer relevancy
"""
from deepeval.test_case import SearchTestCase
from deepeval.run_test import run_test, assert_test
from deepeval.metrics.ranking_similarity import (
    RankingSimilarity,
    assert_ranking_similarity,
)

list1 = ["a", "b"]
list2 = ["b", "c"]
list3 = ["b", "a"]

list_dict_1 = [{"text": "a"}, {"text": "b"}]
list_dict_2 = [{"text": "b"}, {"text": "c"}]


def test_assert_similar_ranking():
    assert_ranking_similarity(list1, list2, minimum_score=0.4)


def test_similar_ranking_2():
    scorer = RankingSimilarity(minimum_score=0.5)
    test_case = SearchTestCase(list1, list2)
    test_case_2 = SearchTestCase(list1, list3)
    test_results = run_test([test_case, test_case_2], metrics=[scorer])
    assert (
        test_results[1].score > test_results[0].score
    ), "Similarity not working"


def test_similar_ranking_3():
    metric = RankingSimilarity(minimum_score=0.3)
    test_case = SearchTestCase(list1, list2)
    assert_test(test_cases=test_case, metrics=[metric])
