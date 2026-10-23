"""Unit tests for GEval's logprob-weighted score.

Constructs synthetic ``ChatCompletion`` responses so no judge is called.
"""

import math
from types import SimpleNamespace

from deepeval.metrics.g_eval.utils import (
    calculate_weighted_summed_score,
    is_ambiguous_score_token,
)


def _token(token, alternatives):
    """Build a logprob entry whose alternatives are (token, linear prob)."""
    return SimpleNamespace(
        token=token,
        logprob=math.log([p for t, p in alternatives if t == token][0]),
        top_logprobs=[
            SimpleNamespace(token=t, logprob=math.log(p)) for t, p in alternatives
        ],
    )


def _literal(token):
    return _token(token, [(token, 1.0)])


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))]
    )


SCORE_TOKENS = [
    _literal('{"'),
    _literal("reason"),
    _literal('":'),
    _literal(' "'),
    _literal("perfect"),
    _literal('",'),
    _literal(' "'),
    _literal("score"),
    _literal('":'),
]


class TestIsAmbiguousScoreToken:
    def test_opening_of_ten_is_ambiguous(self):
        # "1" may be the score 1, or merely the opening of 10.
        assert is_ambiguous_score_token("1", 9) is True

    def test_token_equal_to_raw_score_is_not_ambiguous(self):
        # It denotes the score that was actually emitted.
        assert is_ambiguous_score_token("1", 1) is False

    def test_digits_that_cannot_open_a_longer_score_are_not_ambiguous(self):
        # No valid score lies in 20..29 given the 0..10 rubric bound.
        assert is_ambiguous_score_token("2", 9) is False
        assert is_ambiguous_score_token("9", 1) is False

    def test_leading_zero_is_not_ambiguous(self):
        assert is_ambiguous_score_token("0", 9) is False


class TestCalculateWeightedSummedScore:
    def test_single_digit_score_is_weighted(self):
        content = SCORE_TOKENS + [
            _token("7", [("7", 0.6), ("8", 0.3), ("6", 0.1)]),
            _literal("}"),
        ]
        result = calculate_weighted_summed_score(7, _response(content))
        assert abs(result - 7.2) < 1e-9

    def test_probability_below_one_percent_is_filtered_out(self):
        content = SCORE_TOKENS + [
            _token("5", [("5", 0.001), ("6", 0.999)]),
            _literal("}"),
        ]
        result = calculate_weighted_summed_score(5, _response(content))
        assert abs(result - 6.0) < 1e-9

    def test_digit_opening_a_multi_digit_score_falls_back_to_raw_score(self):
        """Regression test for issue #3517.

        With a tokenizer that splits digits, the candidate "1" really stands for
        10. Reading it as 1 drags the weighted score far below what the judge
        intended, so no weighting is applied instead.
        """
        content = SCORE_TOKENS + [
            _token("9", [("9", 0.812), ("1", 0.16), ("8", 0.028)]),
            _literal("}"),
        ]
        result = calculate_weighted_summed_score(9, _response(content))
        assert result == 9

    def test_score_split_across_tokens_falls_back_to_raw_score(self):
        """A score emitted as several tokens has no single matching token."""
        content = SCORE_TOKENS + [
            _token("1", [("1", 0.7), ("9", 0.3)]),
            _literal("0"),
            _literal("}"),
        ]
        result = calculate_weighted_summed_score(10, _response(content))
        assert result == 10

    def test_reasoning_containing_the_score_does_not_shadow_the_final_one(self):
        earlier = _literal("8")
        content = (
            SCORE_TOKENS[:2]
            + [earlier]
            + SCORE_TOKENS[2:]
            + [_token("7", [("7", 0.5), ("9", 0.5)]), _literal("}")]
        )
        result = calculate_weighted_summed_score(7, _response(content))
        assert abs(result - 8.0) < 1e-9
