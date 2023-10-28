"""Tests for metrics calculator
"""

from deepeval.metrics.scoring import Scorer

# Testing exact_match_score metric


def test_exact_equal_strings():
    target = (
        "Winners of the FIFA world cup were the French national football team"
    )
    prediction = (
        "Winners of the FIFA world cup were the French national football team"
    )
    assert Scorer.exact_match_score(target, prediction) == 1


def test_exact_match_score_unequal_strings():
    target = "Hello, World!"
    prediction = "Goodbye, World!"
    assert Scorer.exact_match_score(target, prediction) == 0


def test_exact_match_score_whitespace_difference():
    target = "Hello, World!"
    prediction = " Hello, World! "
    assert Scorer.exact_match_score(target, prediction) == 1


def test_exact_match_score_empty_prediction():
    target = "Hello, World!"
    prediction = ""
    assert Scorer.exact_match_score(target, prediction) == 0


def test_exact_match_score_empty_target():
    target = ""
    prediction = "Hello, World!"
    assert Scorer.exact_match_score(target, prediction) == 0
