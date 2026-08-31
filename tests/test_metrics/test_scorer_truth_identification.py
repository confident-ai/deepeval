"""
Tests for `Scorer.truth_identification_score`.

The metric reports the percentage of correct answers a prediction identifies.
Previously a repeated index in either list was counted with multiplicity, so a
duplicate-laden prediction could silently return a percentage above 100
(e.g. target "1,2" vs prediction "1,1,1,2" returned 200). Now both lists are
de-duplicated before counting, capping the score at 100, while unique inputs
keep returning exactly the same values as before.
"""

from deepeval.scorer import Scorer

scorer = Scorer()


# --------------------------------------------------------------------------- #
# Default behavior is preserved for unique (non-duplicated) inputs
# --------------------------------------------------------------------------- #


def test_partial_identification_returns_percentage():
    assert scorer.truth_identification_score("1,2,3", "[1, 3]") == 67
    assert scorer.truth_identification_score("1,2,3", "1,3") == 67


def test_no_overlap_returns_zero():
    assert scorer.truth_identification_score("1,2", "3,4") == 0


def test_all_correct_returns_one_hundred():
    assert scorer.truth_identification_score("5,4,3,2,1", "[1,2,3,4,5]") == 100


def test_single_answer():
    assert scorer.truth_identification_score("2", "[2]") == 100
    assert scorer.truth_identification_score("2", "9") == 0


def test_extra_unique_predictions_do_not_penalize():
    # Identifying every correct answer is worth 100 even with extra guesses.
    assert scorer.truth_identification_score("1,2,3", "1,2,3,4,5") == 100


def test_bracket_and_whitespace_variants():
    assert scorer.truth_identification_score("1, 2", "[1,2]") == 100
    assert scorer.truth_identification_score("[1,2]", "1,2") == 100


def test_empty_inputs_return_zero():
    assert scorer.truth_identification_score("", "[1]") == 0
    assert scorer.truth_identification_score("1,2", "[]") == 0
    assert scorer.truth_identification_score("1,2", "") == 0


# --------------------------------------------------------------------------- #
# New behavior: duplicated indices cannot inflate the percentage past 100
# --------------------------------------------------------------------------- #


def test_duplicate_predictions_capped_at_one_hundred():
    # Previously this returned 200.
    assert scorer.truth_identification_score("1,2", "1,1,1,2") == 100


def test_duplicate_predictions_for_partial_overlap():
    # Previously this returned 133.
    assert scorer.truth_identification_score("1,3,4", "[1, 1, 3, 4]") == 100


def test_duplicate_targets_capped_at_one_hundred():
    # A repeated index in the target must not inflate the denominator either.
    assert scorer.truth_identification_score("1,1,2", "1,2") == 100


def test_duplicates_do_not_affect_a_wrong_prediction():
    # Duplicating a wrong guess still contributes nothing.
    assert scorer.truth_identification_score("1,2", "3,3,3") == 0
