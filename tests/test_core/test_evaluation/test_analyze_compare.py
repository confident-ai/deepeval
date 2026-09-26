"""Tests for deepeval.evaluate.analyze_compare_results (#3161).

Two layers:

* **Unit** — the statistical primitives are anchored to hand-derivable
  binomial sums (so a refactor that silently breaks the math fails here), and
  the tie/edge/validation handling is checked explicitly.
* **End-to-end** — a deterministic "arena" produces a per-case winner list
  exactly like ``compare()`` would internally; we feed that into
  ``analyze_compare_results`` and assert the final significance verdict,
  effect size and human-readable sentence.

Run with: ``poetry run pytest tests/test_core/test_evaluation/test_analyze_compare.py -q``
"""

from __future__ import annotations

import math

import pytest

from deepeval.evaluate.analyze import (
    CompareSignificance,
    _normal_cdf,
    _two_sided_binomial_exact_p,
    _wilson_score_interval,
    analyze_compare_results,
)


# ---------------------------------------------------------------------------
# Statistical primitive correctness
# ---------------------------------------------------------------------------


class TestBinomialExactP:
    def test_known_tails(self):
        # These are hand-computed exact two-sided sign-test p-values:
        #   P(X>=k) for a fair coin, doubled and capped at 1.0.
        # 6/10 -> 386/1024*2 = 0.75390625
        assert _two_sided_binomial_exact_p(10, 6) == pytest.approx(
            0.75390625, abs=1e-9
        )
        # 9/10 -> 11/1024*2 = 0.021484375
        assert _two_sided_binomial_exact_p(10, 9) == pytest.approx(
            0.021484375, abs=1e-9
        )
        # 15/20 -> (sum C(20,15..20))/2^20 * 2 = 0.041389
        assert _two_sided_binomial_exact_p(20, 15) == pytest.approx(
            0.04138946533203125, abs=1e-9
        )

    def test_perfect_split_is_least_significant(self):
        # With every other round decided either way for the *same* count, a
        # fair-coin model says any outcome is plausible.
        assert _two_sided_binomial_exact_p(8, 4) == pytest.approx(1.0, abs=1e-9)

    def test_extreme_outcome_is_significant(self):
        # 9/9 decisive wins: P(X>=9) = C(9,9)/512 = 1/512, doubled = 0.00390625
        assert _two_sided_binomial_exact_p(9, 9) == pytest.approx(
            0.00390625, abs=1e-9
        )

    def test_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            _two_sided_binomial_exact_p(0, 0)  # n < 1
        with pytest.raises(ValueError):
            _two_sided_binomial_exact_p(5, 6)  # k > n


class TestNormalCdf:
    def test_symmetry_and_anchors(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5)
        assert _normal_cdf(1.9599639845400545) == pytest.approx(0.975, abs=1e-4)
        # Symmetric about zero.
        assert _normal_cdf(-1.0) == pytest.approx(1.0 - _normal_cdf(1.0))


class TestWilsonInterval:
    def test_bounds_and_pihat_containment(self):
        lo, hi = _wilson_score_interval(successes=7, total=10)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= 0.7 <= hi

    def test_non_degenerate_at_extremes(self):
        # Wald degenerates (zero width / out-of-range) at k=n; Wilson does
        # not. For a single decisive "win" the interval stays non-degenerate
        # and inside [0, 1].
        lo, hi = _wilson_score_interval(successes=1, total=1)
        assert lo > 0.0
        assert lo < hi
        assert hi <= 1.0

    def test_interval_shrinks_with_more_data(self):
        lo_big, hi_big = _wilson_score_interval(successes=100, total=200)
        lo_small, hi_small = _wilson_score_interval(successes=10, total=20)
        width_big = hi_big - lo_big
        width_small = hi_small - lo_small
        assert width_big < width_small

    def test_rejects_zero_total(self):
        with pytest.raises(ValueError):
            _wilson_score_interval(successes=0, total=0)


# ---------------------------------------------------------------------------
# analyze_compare_results behaviour
# ---------------------------------------------------------------------------


class TestAnalyzeTally:
    def test_basic_tally_drop(self):
        winners = [None, "A", "B", "A", None, "A", "B"]
        res = analyze_compare_results(winners, "A", "B")
        assert res.n_cases == 7
        assert res.n_ties == 2
        assert res.n_decided == 5
        assert res.n_wins_a == 3
        assert res.n_wins_b == 2

    def test_drop_vs_split_tallies(self):
        winners = ["A", None, "B", None]
        dropped = analyze_compare_results(
            winners, "A", "B", tie_strategy="drop"
        )
        split = analyze_compare_results(winners, "A", "B", tie_strategy="split")
        assert dropped.n_decided == 2
        assert split.n_decided == 4
        assert split.n_wins_a == 2.0
        assert split.n_wins_b == 2.0
        # Both read a balanced outcome.
        assert dropped.win_rate_a == pytest.approx(0.5)
        assert split.win_rate_a == pytest.approx(0.5)

    def test_fully_tied_is_inconclusive(self):
        res = analyze_compare_results([None, None, None], "A", "B")
        assert res.n_decided == 0
        assert res.significant is False
        assert res.p_value is None
        assert math.isnan(res.win_rate_a)


class TestAnalyzeInference:
    def test_extreme_win_is_significant(self):
        winners = ["B"] * 9 + [None]
        res = analyze_compare_results(winners, "A", "B")
        assert res.significant is True
        assert res.win_rate_a == pytest.approx(0.0)
        assert res.odds_ratio is None  # B swept every decided round
        assert (
            "B" in res.interpretation
            and "significantly better" in res.interpretation
        )

    def test_balanced_is_not_significant(self):
        winners = ["A", "B"] * 6
        res = analyze_compare_results(winners, "A", "B")
        assert res.significant is False
        assert res.p_value == pytest.approx(
            _two_sided_binomial_exact_p(12, 6), abs=1e-9
        )

    def test_effect_size_convention(self):
        # 3-to-1 odds should surface as log(3).
        winners = ["A"] * 9 + ["B"] * 3
        res = analyze_compare_results(winners, "A", "B")
        assert res.odds_ratio == pytest.approx(3.0)
        assert res.log_odds_ratio == pytest.approx(math.log(3.0))

    def test_alpha_respected(self):
        # 6/10 decisive wins is NOT significant at alpha=0.05 (p~0.75)...
        winners = ["A"] * 6 + ["B"] * 4
        assert analyze_compare_results(winners, "A", "B").significant is False
        # ...and treating every tie as a win (split) can swing it, which is
        # exactly the conservatism contrast we document.
        tied = ["A"] * 6 + [None] * 4
        assert (
            analyze_compare_results(
                tied, "A", "B", tie_strategy="drop"
            ).significant
            is True
        )
        assert (
            analyze_compare_results(
                tied, "A", "B", tie_strategy="split"
            ).significant
            is False
        )


class TestAnalyzeValidation:
    def test_empty_winners_rejected(self):
        with pytest.raises(ValueError):
            analyze_compare_results([], "A", "B")

    def test_same_names_rejected(self):
        with pytest.raises(ValueError):
            analyze_compare_results(["A"], "A", "A")

    def test_bad_tie_strategy_rejected(self):
        with pytest.raises(ValueError):
            analyze_compare_results(["A"], "A", "B", tie_strategy="bogus")

    def test_bad_alpha_rejected(self):
        with pytest.raises(ValueError):
            analyze_compare_results(["A"], "A", "B", alpha=1.0)

    def test_unknown_winner_token_treated_as_other_win(self):
        # A contestant outside (model_a, model_b) used to raise ValueError,
        # but that made the API non-composable over multi-way arenas where a
        # third contestant can legitimately win rounds. The new semantics
        # folds third-party wins into n_other_wins and counts them as
        # non-decisive ties for the analysed pair; callers can audit the
        # n_other_wins field to spot real typos.
        sig = analyze_compare_results(["A", "Ae", "B"], "A", "B")
        # "Ae" is a third-party win for this pair.
        assert sig.n_other_wins == 1
        assert sig.n_decided == 2
        assert sig.n_wins_a == 1
        assert sig.n_wins_b == 1
        # 1-1 split over 2 decided => not significant.
        assert sig.significant is False
        # A genuine typo can still be detected: if N_other_wins is large
        # relative to N_decided the user knows something is off.


# ---------------------------------------------------------------------------
# End-to-end: a deterministic arena that mirrors compare()'s internal output
# ---------------------------------------------------------------------------


def _fake_arena(n_cases: int, win_ratio_a: float) -> list:
    """Build the per-case winner list compare() would produce for a head-to-head
    where model_a wins roughly `win_ratio_a` of decisive rounds (ties ignored
    here). Winners are the real contestant names, exactly like compare()."""
    import random

    model_a, model_b = "RetrieverV2", "RetrieverV1"
    rng = random.Random(0)
    out = []
    for _ in range(n_cases):
        out.append(model_a if rng.random() < win_ratio_a else model_b)
    return out


class TestEndToEnd:
    def test_from_winners_to_verdict(self):
        winners = _fake_arena(n_cases=50, win_ratio_a=0.8)
        res = analyze_compare_results(winners, "RetrieverV2", "RetrieverV1")
        assert isinstance(res, CompareSignificance)
        assert res.n_decided == 50
        assert res.significant is True  # 40/50 should clear alpha=0.05
        assert res.win_rate_a == pytest.approx(res.n_wins_a / res.n_decided)
        assert "significantly better" in res.interpretation
        # Winner must be the higher win-rate side.
        assert res.interpretation.startswith("'RetrieverV2'")

    def test_from_winners_to_no_verdict(self):
        winners = _fake_arena(n_cases=30, win_ratio_a=0.52)
        res = analyze_compare_results(winners, "RetrieverV2", "RetrieverV1")
        assert res.significant is False
        assert "No significant difference" in res.interpretation

    def test_consistent_with_compare_win_counts(self):
        # The tally must reproduce the raw win counts compare() returns, so
        # the two APIs always agree on the same experiment.
        winners = ["A"] * 17 + ["B"] * 3 + [None] * 5
        res = analyze_compare_results(winners, "A", "B", tie_strategy="drop")
        assert res.n_decided == 20
        assert res.n_wins_a == 17
        assert res.n_wins_b == 3
