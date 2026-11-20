from deepeval.scorer import Scorer


def test_truth_identification_ignores_duplicate_predictions():
    # 1 of 3 correct answers identified; a repeated index must not be re-counted.
    assert Scorer.truth_identification_score("1,2,3", "1,1,1") == 33


def test_truth_identification_never_exceeds_100_percent():
    assert Scorer.truth_identification_score("1,2,3", "1,2,3,3") == 100


def test_truth_identification_partial_recall_with_duplicate():
    # "[2, 5]" is the exact target format the benchmark template produces.
    assert Scorer.truth_identification_score("[2, 5]", "2, 2") == 50


def test_truth_identification_no_duplicates_unchanged():
    assert Scorer.truth_identification_score("1,2,3", "1,2") == 67
    assert Scorer.truth_identification_score("1,2,3", "1,2,3") == 100
    assert Scorer.truth_identification_score("1,2,3", "4,5,6") == 0
