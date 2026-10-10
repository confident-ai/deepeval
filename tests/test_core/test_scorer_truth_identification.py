"""Regression test: ``Scorer.truth_identification_score`` could return more
than 100.

The score is a percentage of ``target_list`` identified correctly, so it must
never exceed 100. Before this fix, ``correct_matches`` counted every
occurrence of a correct value in ``prediction_list`` rather than each
distinct value once, so a prediction that repeated a correct index inflated
the count past ``len(target_list)``.

This is reachable from ``TruthfulQABenchmark`` in MC2 mode: the model's
answer comes from ``ListOfNumbersSchema`` (a plain ``List[int]`` with no
uniqueness constraint), stringified and fed straight into this scorer, so a
model repeating an index in its structured output reaches this path
unfiltered.
"""

from deepeval.scorer import Scorer

scorer = Scorer()


def test_repeated_correct_index_does_not_exceed_100():
    # Two correct answers (1, 2); the model names index 1 three times and
    # never names 2. Only one distinct correct answer was identified.
    score = scorer.truth_identification_score(target="1,2", prediction="1,1,1")
    assert score <= 100
    assert score == 50


def test_all_correct_with_a_repeat_is_capped_at_100():
    # Both correct answers named, one of them twice.
    score = scorer.truth_identification_score(target="1,2", prediction="1,1,2")
    assert score == 100


def test_no_repeats_is_unaffected():
    score = scorer.truth_identification_score(target="1,2,3", prediction="1,2")
    assert score == round(2 / 3 * 100)
