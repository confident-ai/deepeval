"""Whitespace-only predictions must score like empty ones (#3186).

These scorers are pure string functions, so the tests run offline with no
API keys or extra dependencies.
"""

import pytest

from deepeval.scorer import Scorer

BLANK_PREDICTIONS = ["", " ", "\t", "\n", "  \n\t "]


@pytest.mark.parametrize("prediction", BLANK_PREDICTIONS)
@pytest.mark.parametrize("target", ["", "Paris"])
def test_exact_match_blank_prediction_scores_zero(target, prediction):
    assert Scorer.exact_match_score(target, prediction) == 0


@pytest.mark.parametrize("prediction", BLANK_PREDICTIONS)
@pytest.mark.parametrize("target", ["", "Paris"])
def test_quasi_exact_match_blank_prediction_scores_zero(target, prediction):
    assert Scorer.quasi_exact_match_score(target, prediction) == 0


@pytest.mark.parametrize("prediction", BLANK_PREDICTIONS)
@pytest.mark.parametrize("targets", [[""], ["Paris"], ["", "Paris"]])
def test_quasi_contains_blank_prediction_scores_zero(targets, prediction):
    assert Scorer.quasi_contains_score(targets, prediction) == 0


@pytest.mark.parametrize(
    "target, prediction, expected",
    [
        ("Paris", "Paris", 1),
        ("Paris", "  Paris\n", 1),
        ("Paris", "paris", 0),
        ("Paris", "London", 0),
    ],
)
def test_exact_match_real_predictions_unchanged(target, prediction, expected):
    assert Scorer.exact_match_score(target, prediction) == expected


@pytest.mark.parametrize(
    "target, prediction, expected",
    [
        ("Paris", "Paris", 1),
        ("Paris", "paris.", 1),
        ("The Eiffel Tower", "eiffel tower!", 1),
        ("Paris", "London", 0),
    ],
)
def test_quasi_exact_match_real_predictions_unchanged(
    target, prediction, expected
):
    assert Scorer.quasi_exact_match_score(target, prediction) == expected


@pytest.mark.parametrize(
    "targets, prediction, expected",
    [
        (["Paris", "London"], "London", 1),
        (["Paris", "London"], "LONDON.", 1),
        (["New York City"], "York", 0),
        (["Paris"], "Berlin", 0),
    ],
)
def test_quasi_contains_real_predictions_unchanged(
    targets, prediction, expected
):
    assert Scorer.quasi_contains_score(targets, prediction) == expected
