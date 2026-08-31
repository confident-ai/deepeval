"""
Regression tests for the deterministic string scorers' handling of
whitespace-only predictions.

All three string scorers previously treated a whitespace-only prediction as
non-empty, so ``exact_match_score('', '   ')`` returned 1 while
``exact_match_score('', '')`` returned 0 — the same "no answer" input scored
differently depending on whether it contained spaces.

These tests are offline: no model, network, dataset download, or API key
required.
"""

import pytest

from deepeval.scorer.scorer import Scorer


class TestExactMatchScore:
    def test_default_behaviour_unchanged(self):
        assert Scorer.exact_match_score("hello", "hello") == 1
        assert Scorer.exact_match_score("hello", "world") == 0
        assert Scorer.exact_match_score("hello", "") == 0
        assert Scorer.exact_match_score("hello", "  hello  ") == 1
        assert Scorer.exact_match_score("  hello", "hello  ") == 1

    @pytest.mark.parametrize(
        "target,prediction",
        [
            ("", ""),
            ("", "   "),
            ("", "\n\t"),
            ("hello", "   "),
            ("hello", "\t"),
        ],
    )
    def test_whitespace_only_prediction_is_empty(self, target, prediction):
        # empty and whitespace-only predictions are treated alike (score 0)
        assert Scorer.exact_match_score(target, prediction) == 0


class TestQuasiExactMatchScore:
    def test_default_behaviour_unchanged(self):
        assert (
            Scorer.quasi_exact_match_score("Hello, World!", "hello world") == 1
        )
        assert Scorer.quasi_exact_match_score("hello world", "world") == 0
        assert Scorer.quasi_exact_match_score("x", "") == 0

    @pytest.mark.parametrize(
        "target,prediction",
        [
            ("", ""),
            ("", "   "),
            ("hello", " \t "),
        ],
    )
    def test_whitespace_only_prediction_is_empty(self, target, prediction):
        assert Scorer.quasi_exact_match_score(target, prediction) == 0


class TestQuasiContainsScore:
    def test_default_behaviour_unchanged(self):
        assert Scorer.quasi_contains_score(["hello world"], "hello  world") == 1
        assert Scorer.quasi_contains_score(["a", "b"], "a") == 1
        assert Scorer.quasi_contains_score(["hello"], "world") == 0
        assert Scorer.quasi_contains_score(["hello"], "") == 0

    @pytest.mark.parametrize(
        "targets,prediction",
        [
            ([""], ""),
            ([""], "   "),
            (["hello"], "   "),
            (["hello"], "\t"),
        ],
    )
    def test_whitespace_only_prediction_is_empty(self, targets, prediction):
        assert Scorer.quasi_contains_score(targets, prediction) == 0
