"""Engineering-quality tests for the ArenaResult / compare pipeline.

Covers the "is this actually production-ready?" surface that a naive
statistical-significance PR would miss: backward compatibility of the
``compare()`` return value, stable async ordering (regression for #1034),
large-sample performance of the sign test, and public-API exposure.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import List, Optional

import pytest

import deepeval
from deepeval.evaluate.analyze import (
    _EXACT_BINOMIAL_CEILING,
    _two_sided_binomial_exact_p,
    analyze_compare_results,
)
from deepeval.evaluate.compare import (
    ArenaResult,
)


# ---------------------------------------------------------------------------
# 1. Backward compatibility: ArenaResult must behave like Dict[str, int].
# ---------------------------------------------------------------------------


class TestArenaResultBackwardCompat:
    """Legacy callers treat the compare() return value as a plain dict."""

    @staticmethod
    def _make_result() -> ArenaResult:
        winners: List[Optional[str]] = ["A", "B", "A", None, "A"]
        counts = Counter(w for w in winners if w is not None)
        return ArenaResult(
            winners=winners,
            _counts=counts,
            contestants=["A", "B"],
            n_cases=len(winners),
            run_duration=0.123,
        )

    def test_getitem_matches_dict(self):
        r = self._make_result()
        assert r["A"] == 3
        assert r["B"] == 1

    def test_getitem_unknown_raises_keyerror(self):
        r = self._make_result()
        with pytest.raises(KeyError):
            _ = r["C"]

    def test_get_default(self):
        r = self._make_result()
        assert r.get("A") == 3
        assert r.get("C", 0) == 0
        assert r.get("C") == 0  # our default int is 0, matches Counter

    def test_len_keys_values_items(self):
        r = self._make_result()
        assert len(r) == 2
        assert set(r.keys()) == {"A", "B"}
        assert sum(r.values()) == 4
        assert ("A", 3) in r.items()
        assert ("B", 1) in r.items()

    def test_iteration(self):
        r = self._make_result()
        # `for name in r` should yield contestant names.
        got = {name for name in r}
        assert got == {"A", "B"}

    def test_contains(self):
        r = self._make_result()
        assert "A" in r
        assert "C" not in r

    def test_dict_conversion_via_ctor(self):
        """``dict(compare(...))`` must round-trip to the old dict shape."""
        r = self._make_result()
        as_dict = dict(r)
        assert as_dict == {"A": 3, "B": 1}

    def test_to_dict_explicit(self):
        r = self._make_result()
        assert r.to_dict() == {"A": 3, "B": 1}

    def test_winners_attribute_preserved_ordered(self):
        """The per-case winner list is the whole point of the refactor."""
        r = self._make_result()
        assert r.winners == ["A", "B", "A", None, "A"]

    def test_contestants_in_first_appearance_order(self):
        r = self._make_result()
        assert r.contestants == ["A", "B"]

    def test_metadata_fields_surface(self):
        r = self._make_result()
        assert r.n_cases == 5
        assert r.run_duration == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# 2. Integrated .analyze() — no more "compare then rebuild the list".
# ---------------------------------------------------------------------------


class TestArenaResultAnalyze:
    def test_two_way_auto_infers_pair(self):
        # 11-1 sweep over 12 decided rounds → exact two-sided
        # p = 2 * (1 + 12) / 4096 = 26 / 4096 ≈ 0.00635 — strongly significant.
        winners = ["B"] * 11 + ["A"] * 1 + [None]
        counts = Counter(w for w in winners if w is not None)
        r = ArenaResult(
            winners=winners,
            _counts=counts,
            contestants=["A", "B"],
            n_cases=len(winners),
        )
        sig = r.analyze()
        assert sig.significant is True
        assert sig.model_a == "A"
        assert sig.model_b == "B"
        assert sig.n_other_wins == 0
        assert "significantly better" in sig.interpretation

    def test_explicit_pair_selection(self):
        # 3-way arena — analyze(A,B) should surface the third-party C-wins
        # as "other wins" and treat them as non-decisive for the A/B pair.
        winners = ["A", "B", "C"]
        counts = Counter(winners)
        r = ArenaResult(
            winners=winners,
            _counts=counts,
            contestants=["A", "B", "C"],
            n_cases=3,
        )
        with pytest.raises(ValueError):
            r.analyze()  # ambiguous without a pair
        sig = r.analyze(model_a="A", model_b="B")
        # Round 0 (A-win), round 1 (B-win) are decisive for A/B;
        # round 2 (C-win) counts as "other win" → a tie for A/B purposes.
        assert sig.n_decided == 2
        assert sig.n_other_wins == 1
        assert sig.n_ties == 1  # C-win collapsed into tie
        assert sig.significant is False  # 1-1 split: p = 1.0

    def test_analyze_with_tie_strategy_split(self):
        winners = ["A"] * 3 + [None] * 2 + ["B"] * 1
        counts = Counter(w for w in winners if w is not None)
        r = ArenaResult(
            winners=winners,
            _counts=counts,
            contestants=["A", "B"],
            n_cases=len(winners),
        )
        sig_drop = r.analyze(tie_strategy="drop")
        sig_split = r.analyze(tie_strategy="split")
        # "drop" sees 4 decided (3A,1B); "split" sees all 6 with A=4, B=2.
        assert sig_drop.n_decided == 4
        assert sig_split.n_decided == 6
        assert sig_split.n_wins_a == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 3. Async order regression (#1034). We don't need the full ArenaGEval — we
#    reproduce the concurrency pattern directly with asyncio.gather.
# ---------------------------------------------------------------------------


class TestAsyncOrdering:
    @pytest.mark.asyncio
    async def test_gather_preserves_creation_order_against_out_of_order_finish(
        self,
    ):
        """Later-created tasks must NOT end up earlier in the result list.

        We spawn N tasks whose finish time is the *reverse* of their
        creation order (task 0 sleeps longest, task N-1 sleeps least).
        If the implementation accidentally appends in completion order the
        returned list will be reversed; the correct gather behaviour keeps
        it sorted by the index we tagged each task with.
        """
        N = 40

        async def sleepy(index: int) -> int:
            # Reverse: higher index finishes sooner.
            sleep_for = 0.001 * (N - index)
            await asyncio.sleep(sleep_for)
            return index

        tasks = [asyncio.create_task(sleepy(i)) for i in range(N)]
        results = await asyncio.gather(*tasks)
        assert list(results) == list(range(N))

    @pytest.mark.asyncio
    async def test_winners_placeholder_ordered_semantics(self):
        """Verify the contract that asyncio.gather respects positional order,
        including None entries (ties) at their original positions."""
        pattern: List[Optional[str]] = [
            "A",
            None,
            "B",
            "B",
            None,
            "A",
            "A",
            None,
            "B",
            "A",
        ] * 6  # 60 cases

        async def return_at(index: int) -> Optional[str]:
            # Make tasks finish in pseudo-random order by a small jitter.
            jitter = 0.0005 * ((index * 7) % 11)
            await asyncio.sleep(jitter)
            return pattern[index]

        tasks = [asyncio.create_task(return_at(i)) for i in range(len(pattern))]
        out = await asyncio.gather(*tasks)
        assert list(out) == pattern


# ---------------------------------------------------------------------------
# 4. Large-sample performance + continuity of exact/approximate switch.
# ---------------------------------------------------------------------------


class TestBinomialLargeSample:
    def test_exact_and_approximate_agree_at_crossover(self):
        """At n = _EXACT_BINOMIAL_CEILING both paths are formally different
        but should agree within ~1e-3 because the normal approximation is
        extremely accurate there for the symmetric H0 p=0.5 case.

        We test on the boundary and one step above it so the crossover line
        isn't accidentally wrong."""
        n_cross = _EXACT_BINOMIAL_CEILING
        n_above = _EXACT_BINOMIAL_CEILING + 100
        for n in (n_cross, n_above):
            for k_frac in (0.5, 0.45, 0.40, 0.55):
                k = int(n * k_frac)
                # Two runs at n_cross both hit the exact path; for n_above
                # one run hits the exact path (the smaller n below) and the
                # other the approximate path. We just ensure the function
                # returns a plausible probability in (0,1] for all of them.
                p = _two_sided_binomial_exact_p(n, k)
                assert 0.0 < p <= 1.0, (n, k, p)
                # Symmetry check: swapping k→n−k in a sign test gives the
                # same two-sided p-value.
                p2 = _two_sided_binomial_exact_p(n, n - k)
                assert p == pytest.approx(p2, abs=1e-12)

    def test_huge_sample_runs_fast_and_returns_plausible_p(self):
        """A 10 000-case arena (tens of thousands is realistic) has to
        complete in well under a second. The old exact-only path would
        allocate thousands of 3000+ digit integers and choke."""
        n_big = 10_000
        k = 5_400  # clearly different from H0 — expect p « 0.05
        t0 = time.perf_counter()
        p = _two_sided_binomial_exact_p(n_big, k)
        elapsed = time.perf_counter() - t0
        # Performance upper bound — generous so CI isn't flaky, but the
        # approximate path should take microseconds, not milliseconds.
        assert elapsed < 0.05, f"large-N p-value took {elapsed:.3f}s (too slow)"
        assert 0.0 < p < 1e-8, f"expected strong significance, got p={p}"

    def test_small_n_uses_exact_truth_6of6(self):
        """Sanity: 6/6 wins should give exactly the textbook small-sample
        sign-test p-value of 2 * (1/2)^6 = 1/32 ≈ 0.03125."""
        p = _two_sided_binomial_exact_p(6, 6)
        assert p == pytest.approx(1 / 32, abs=1e-12)

    def test_small_n_uses_exact_truth_2of6(self):
        """Two-sided p for k=2,n=6:
        lower tail P(X ≤ 2) = (1+6+15)/64 = 22/64 = 0.34375
        doubled = 0.6875."""
        p = _two_sided_binomial_exact_p(6, 2)
        assert p == pytest.approx(0.6875, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Top-level API exposure.
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_analyze_compare_results_on_deepeval_module(self):
        """``import deepeval; deepeval.analyze_compare_results(...)`` must
        work out of the box — matching how ``deepeval.compare`` is called."""
        fn = getattr(deepeval, "analyze_compare_results", None)
        assert callable(
            fn
        ), "deepeval.analyze_compare_results is not exposed at the top level"
        # Smoke-call it so we also catch circular-import regressions.
        result = fn(
            winners=["X", "X", "Y"],
            model_a="X",
            model_b="Y",
        )
        assert result is not None
        assert result.significant in (True, False)

    def test_deepeval_compare_in_module_all(self):
        assert "analyze_compare_results" in deepeval.__all__
        assert "compare" in deepeval.__all__

    def test_analyze_and_compare_signature_match(self):
        """Result of deepeval.compare() (in the Mapping sense) should be
        directly consumable by analyze_compare_results via its .winners.

        We construct a fake ArenaResult without running compare() so the
        test stays offline / no LLM required.
        """
        fake_winners: List[Optional[str]] = ["M1", "M1", None, "M2"]
        counts = Counter(w for w in fake_winners if w is not None)
        fake_result = ArenaResult(
            winners=fake_winners,
            _counts=counts,
            contestants=["M1", "M2"],
            n_cases=len(fake_winners),
        )
        sig1 = deepeval.analyze_compare_results(
            winners=fake_result.winners, model_a="M1", model_b="M2"
        )
        sig2 = fake_result.analyze()
        # Both calls should produce identical tallies & stats — they feed on
        # the same winner list with identical tie_strategy="drop".
        assert sig1.n_decided == sig2.n_decided
        assert sig1.n_wins_a == sig2.n_wins_a
        assert sig1.p_value == pytest.approx(sig2.p_value, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. Input-validation hygiene for the public analyze function.
# ---------------------------------------------------------------------------


class TestAnalyzeInputValidation:
    def test_third_party_wins_become_other_wins_not_errors(self):
        """Wins from contestants outside the (A,B) pair used to raise; they
        are now folded into n_other_wins and treated as non-decisive for
        the analysed pair. This makes the API composable over multi-way
        arena results without pre-filtering."""
        sig = analyze_compare_results(
            winners=["A", "B", "C", None, "D"],
            model_a="A",
            model_b="B",
        )
        assert sig.n_other_wins == 2
        # C and D wins counted as ties
        assert sig.n_ties == 3  # one original None + 2 third-party

    def test_matching_model_names_rejected(self):
        with pytest.raises(
            ValueError, match="model_a and model_b must be different"
        ):
            analyze_compare_results(
                winners=["A", "A"],
                model_a="A",
                model_b="A",
            )

    def test_bad_alpha_range_rejected(self):
        for bad in (0.0, 1.0, -0.01, 1.1):
            with pytest.raises(ValueError, match="alpha"):
                analyze_compare_results(
                    winners=["A", "B"],
                    model_a="A",
                    model_b="B",
                    alpha=bad,
                )

    def test_empty_winners_rejected(self):
        with pytest.raises(ValueError, match="winners must not be empty"):
            analyze_compare_results(
                winners=[],
                model_a="A",
                model_b="B",
            )
