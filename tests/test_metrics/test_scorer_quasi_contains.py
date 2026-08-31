"""Tests for ``Scorer.quasi_contains_score`` with a bare-string target.

``quasi_contains_score(targets, prediction)`` declared ``targets: List[str]``
but never validated it. Strings are iterable, so a caller passing a bare string
(say, the expected answer) silently split it into **single characters**,
producing nonsense scores — an exact match scored ``0`` while a single-character
prediction scored ``1``.

These tests verify that:
  * a bare-string target with an exact (normalized) match scores ``1``;
  * a bare-string target no longer yields spurious character matches;
  * list targets keep their previous behaviour (no regression);
  * an empty prediction still scores ``0``;
  * normalization (case / extra whitespace) still applies.
"""

import pytest

from deepeval.scorer.scorer import Scorer


def test_bare_string_target_exact_match_scores_one():
    assert Scorer.quasi_contains_score("cat", "cat") == 1


def test_bare_string_target_no_spurious_character_match():
    # Previously "cat" was split into ["c", "a", "t"], so a single-character
    # prediction matched spuriously.
    assert Scorer.quasi_contains_score("cat", "c") == 0
    assert Scorer.quasi_contains_score("cat", "ca") == 0


def test_bare_string_target_normalized_match_scores_one():
    # normalize_text lower-cases and collapses whitespace.
    assert Scorer.quasi_contains_score("Cat", "cat") == 1
    assert Scorer.quasi_contains_score("cat", " cat ") == 1


def test_list_target_exact_match_scores_one():
    assert Scorer.quasi_contains_score(["cat"], "cat") == 1


def test_list_target_non_member_scores_zero():
    assert Scorer.quasi_contains_score(["cat", "dog"], "bird") == 0


def test_list_target_picks_a_member():
    assert Scorer.quasi_contains_score(["cat", "dog"], "dog") == 1


def test_empty_prediction_scores_zero():
    assert Scorer.quasi_contains_score("cat", "") == 0
    assert Scorer.quasi_contains_score(["cat"], "") == 0


def test_empty_targets_scores_zero():
    assert Scorer.quasi_contains_score([], "cat") == 0
