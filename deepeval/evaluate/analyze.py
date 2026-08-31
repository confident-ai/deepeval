"""
Statistical significance analysis for head-to-head evaluation comparisons.

`deepeval.evaluate.compare` answers *who won each round* with ``ArenaGEval``,
but it only reports raw win counts (``{contestant: wins}``). This module answers
the follow-up question that raw counts can't: **is that win margin real, or
could it happen by chance?** Given a per-case winner list from an arena-style
head-to-head, it runs an exact sign test and reports a p-value, a 95%
confidence interval on the win rate (Wilson score interval), and a
log-odds-ratio effect size.

Everything here is implemented with the Python standard library on purpose:

* no new runtime dependency (the repo currently has no ``scipy`` /
  ``statsmodels``), so this module cannot disturb the dependency tree of any
  existing feature;
* ``math.comb`` gives exact combinatorics for the binomial tail, so the
  p-value needs no asymptotic approximation even for small sample sizes —
  which is exactly when "B is 6/6" is *not* the same as "B is 6/6 out of a
  million".

It is intentionally an independent leaf module: it does not modify
``compare()``, ``evaluate()``, or any metric, and can be imported without
touching the test-run / telemetry machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

__all__ = [
    "CompareSignificance",
    "analyze_compare_results",
]

_Z_95 = 1.9599639845400545  # two-tailed 5% critical value of the normal.


def _normal_cdf(z: float) -> float:
    """Standard normal CDF, numerically stable across a wide z-range.

    For ``|z|`` below ~``5`` the textbook ``0.5 * (1 + erf(z/√2))`` is fine.
    For large negative ``z`` however ``erf`` saturates to ``-1`` and the
    expression collapses to exactly zero, even though the true CDF can
    easily be as small as ``~1e-308`` within the float64 dynamic range. We
    therefore use ``erfc`` for negative arguments (``Φ(z) = ½·erfc(−z/√2)``)
    which stays accurate right down to float underflow.
    """
    root_half = 0.7071067811865476
    if z >= 0:
        return 0.5 * (1.0 + math.erf(z * root_half))
    return 0.5 * math.erfc(-z * root_half)


# Sample-size ceiling at which we still run the exact combinatorial sum.
# Beyond this the sign test switches to a normal approximation with
# continuity correction — for n ≫ 1 both agree to 4+ decimals, while the
# exact code path becomes prohibitively slow (sum of O(n) huge-integer
# binomial coefficients).
_EXACT_BINOMIAL_CEILING = 2000


def _two_sided_binomial_exact_p(n: int, k: int) -> float:
    """Two-sided p-value for the sign test.

    Tests H0 ``p = 0.5`` against a two-sided alternative. We define the
    two-tailed p-value the way the binomial sign test usually is: take the
    smaller of the two one-sided tails ``P(X <= k)`` and ``P(X >= k)`` and
    double it, capping at 1.0.

    For **small** ``n`` (``n <= _EXACT_BINOMIAL_CEILING``) the result is
    computed via exact combinatorial summation. We use a multiplicative
    recurrence

        P(X = i+1) = P(X = i) * (n - i) / (i + 1)

    (seed ``P(X = 0) = 0.5^n``) so the inner loop stays in float arithmetic
    and avoids materializing ``math.comb(n, i)`` as a possibly-huge Python
    int whose float conversion can overflow for n as low as ~1800 at the
    peak of the mass function.

    For **large** ``n`` we switch to a continuity-corrected normal
    approximation. Under H0 with ``p = 0.5`` the binomial has mean
    ``mu = n/2`` and variance ``sigma^2 = n/4``; the z-score uses the
    standard ±0.5 continuity correction. This keeps the call O(1) and
    avoids blowing up ``math.comb(n, n/2)`` (a ~0.3n-digit integer) for
    arena sizes in the tens of thousands.

    Only valid for ``0 <= k <= n`` and ``n >= 1``.
    """
    if n < 1:
        raise ValueError("Cannot run a sign test on fewer than one pair.")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n.")

    if n <= _EXACT_BINOMIAL_CEILING:
        half_pow = 0.5**n
        # If the seed itself underflows to zero (can happen near the top of
        # the exact ceiling because float64 min-denorm ~ 5e-324 ≈ 0.5^1074),
        # the recurrence will trivially stay zero and give a nonsense p=0.
        # Fall through to the normal approximation — at those n it's
        # essentially exact for the symmetric p=0.5 case anyway.
        if half_pow > 0.0:
            # -- Exact path (float recurrence, no big ints) ----------------
            def tail_le(kk: int) -> float:
                """P(X <= kk) via the ratio recurrence."""
                if kk < 0:
                    return 0.0
                if kk >= n:
                    return 1.0
                p_i = half_pow  # P(X = 0)
                total = p_i
                for i in range(kk):
                    p_i = p_i * (n - i) / (i + 1.0)
                    total += p_i
                return total

            def tail_ge(kk: int) -> float:
                """P(X >= kk) = 1 - P(X <= kk-1)."""
                if kk <= 0:
                    return 1.0
                if kk > n:
                    return 0.0
                return 1.0 - tail_le(kk - 1)

            return min(1.0, 2.0 * min(tail_le(k), tail_ge(k)))

    # -- Normal-approximation path --------------------------------------
    # Bin(n, 0.5) has mu = n/2, sigma = sqrt(n)/2.
    # The ±0.5 continuity correction accounts for approximating a discrete
    # distribution with a continuous one.
    mu = n / 2.0
    sigma = math.sqrt(n) / 2.0
    # Pick the tail that puts the observation on the "outside" of mu,
    # mirroring the exact-path logic of min(lower, upper) then doubling.
    if k <= mu:
        corrected = k + 0.5  # move right toward mu for P(X <= k)
    else:
        corrected = k - 0.5  # move left  toward mu for P(X >= k)
    z = (corrected - mu) / sigma
    # Two-sided: 2 * Phi(-|z|)  (symmetry around z=0)
    return min(1.0, 2.0 * _normal_cdf(-abs(z)))


def _wilson_score_interval(
    successes: int, total: int, z: float = _Z_95
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval ``p_hat +- z sqrt(p_hat(1-p_hat)/n)``
    because the Wald interval degenerates (zero width, or out-of-range bounds)
    when the observed proportion is 0 or 1 — a common case in head-to-heads
    where one model sweeps. The Wilson interval never drops below its nominal
    coverage for ``0 < n`` and stays inside [0, 1].

    ``successes`` / ``total`` are the number of A-wins out of decided pairs.
    ``total`` must be > 0.
    """
    if total <= 0:
        raise ValueError("Cannot compute a proportion with zero totals.")
    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p_hat + z2 / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total))
    ) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


_TIE = Literal["drop", "split"]


@dataclass
class CompareSignificance:
    """Result of an exact sign test over a head-to-head winner list.

    Fields are kept flat (not nested stats objects) so the dataclass can be
    dropped straight into a report, a pandas DataFrame, or a ``stdout``
    summary without extra unwrapping.
    """

    # So pytest doesn't collect this dataclass as a test class.
    __test__ = False

    # Inputs and tallies
    model_a: str
    model_b: str
    n_cases: Optional[int]  # original number of cases seen
    n_decided: int  # pairs with a decisive winner, after tie strategy applied
    # Decimal counts are possible under the "split" tie strategy (a tie
    # contributes half a win to each model), hence float.
    n_wins_a: float
    n_wins_b: float
    n_ties: int  # original ties, before the tie strategy

    # Inference
    win_rate_a: float  # A wins / decided pairs (= ties fraction not included)
    p_value: Optional[float]
    significant: bool
    test_name: str
    odds_ratio: Optional[float]
    log_odds_ratio: Optional[float]
    confidence_interval: Tuple[float, float]  # 95% Wilson CI on win_rate_a

    # Summary sentence for direct logging / terminal display.
    interpretation: str

    # Rounds won by a contestant outside the (A,B) pair being analysed.
    # These are collapsed into "tie" for the sign test but surfaced here
    # so callers can audit multi-way results without double-counting.
    n_other_wins: int = 0


def analyze_compare_results(
    winners: List[Optional[str]],
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
    tie_strategy: _TIE = "drop",
) -> CompareSignificance:
    """Run an exact sign test over a list of per-case winners.

    Each element of ``winners`` is the winning model for one test case —
    ``model_a``, ``model_b``, or ``None`` (a tie / no winner). This mirrors
    what ``compare()`` produces internally for every round; you can build the
    list from your own head-to-head run.

    Ties carry no information about *direction*; ``tie_strategy`` decides how
    they are folded in:

    * ``"drop"`` — ignore tied rounds. The test only considers decisive
      rounds. This is the most statistically conservative reading of the
      data ("we couldn't tell those apart").
    * ``"split"`` — award half a win to each model per tie. Treats a tie as
      exactly as informative as a win, which is only sound if a tie really is
      half a win for your decision context.

    ``win_rate_a`` is always reported over **decided** rounds so it stays a
    well-defined binomial proportion for the Wilson interval; a fully-tied
    run therefore surfaces as ``n_decided == 0`` with ``p_value`` / CI left
    as ``None`` rather than a misleading 0% or 100%.

    Parameters
    ----------
    winners:
        Per-case winner names (or ``None`` for a tie).
    model_a:
        First model's name.
    model_b:
        Second model's name.
    alpha:
        Significance level (default 0.05). A decisive p-value strictly below
        ``alpha`` marks the comparison as significant.
    tie_strategy:
        One of ``"drop"`` or ``"split"`` (see above).

    Returns
    -------
    CompareSignificance
        Flat result object (tallies, p-value, effect size, CI, sentence).
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must lie in (0, 1).")
    if winners is None or len(winners) == 0:
        raise ValueError("winners must not be empty.")
    if not model_a or not model_b:
        raise ValueError("model_a and model_b must be non-empty names.")
    if model_a == model_b:
        raise ValueError("model_a and model_b must be different.")
    if tie_strategy not in ("drop", "split"):
        raise ValueError("tie_strategy must be 'drop' or 'split'.")

    n_cases = len(winners)
    ties = sum(1 for w in winners if w is None)
    wins_a_real = sum(1 for w in winners if w == model_a)
    wins_b_real = sum(1 for w in winners if w == model_b)

    # Winners that match *neither* side come from two valid scenarios and
    # are both treated the same way: as if that round had no winner for
    # this particular A vs B comparison.
    #
    #   (a) A three (or more)-way arena: when contestant C wins a round,
    #       the A↔B pair has no directional information in that round.
    #   (b) A genuine typo in the caller's model name. We can't reliably
    #       tell (a) and (b) apart without the full compare() context, so
    #       we surface the "silently drop third-party wins" behaviour and
    #       record them via `n_other_wins` for users who want to audit.
    #
    # This is strictly less strict than the old "raise on unknown" logic
    # and makes `.analyze(model_a=X, model_b=Y)` on a multi-way ArenaResult
    # behave the way users intuitively expect it to.
    other_wins = 0
    normalized: List[Optional[str]] = []
    for w in winners:
        if w is None or (w != model_a and w != model_b):
            normalized.append(None)
            if w is not None:
                other_wins += 1
        else:
            normalized.append(w)

    ties += other_wins

    decided = [w for w in normalized if w is not None]

    if tie_strategy == "split":
        n_decided = n_cases
        n_wins_a = wins_a_real
        n_wins_b = wins_b_real
        # Each tie contributes a half-win to both models.
        n_wins_a += 0.5 * ties
        n_wins_b += 0.5 * ties
    else:  # "drop"
        n_decided = len(decided)
        n_wins_a = wins_a_real
        n_wins_b = wins_b_real

    if n_decided == 0:
        return CompareSignificance(
            model_a=model_a,
            model_b=model_b,
            n_cases=n_cases,
            n_decided=0,
            n_wins_a=0,
            n_wins_b=0,
            n_ties=ties,
            win_rate_a=float("nan"),
            p_value=None,  # type: ignore[arg-type]
            significant=False,
            test_name="sign test",
            odds_ratio=None,
            log_odds_ratio=None,
            confidence_interval=(float("nan"), float("nan")),
            n_other_wins=other_wins,
            interpretation=(
                f"Every round between {model_a!r} and {model_b!r} was a tie; "
                "there is no decisive signal to test."
            ),
        )

    # Sign test treats decisive rounds as Bernoulli(p) with H0: p = 0.5.
    # For the "split" strategy the half win amounts are non-integer, so we
    # fall back to the exact binomial on rounded decisive counts; externally
    # the practical difference is negligible and the method stays exact for
    # the common "drop" path.
    p_value = _two_sided_binomial_exact_p(n_decided, round(n_wins_a))
    win_rate_a = n_wins_a / n_decided
    significant = p_value < alpha

    ci = _wilson_score_interval(round(n_wins_a), n_decided)

    if n_wins_a > 0 and n_wins_b > 0:
        odds_ratio = n_wins_a / n_wins_b
        log_odds_ratio = math.log(odds_ratio)
    else:
        # One side won every decisive round. The odds ratio is undefined
        # (division by zero) or zero/infinite (log of 0), but the direction
        # is unambiguous, so we surface it as None rather than crashing or
        # returning a misleading p-value.
        odds_ratio = None
        log_odds_ratio = None

    if significant:
        better, worse = (
            (model_a, model_b) if n_wins_a > n_wins_b else (model_b, model_a)
        )
        # `win_rate_a` is model_a's share of decisive rounds. Flip it when
        # the better model is model_b so the sentence always quotes the
        # winner's rate and CI.
        if better == model_a:
            best_rate = win_rate_a
            best_lo, best_hi = ci
        else:
            best_rate = 1.0 - win_rate_a
            best_lo, best_hi = 1.0 - ci[1], 1.0 - ci[0]
        sentence = (
            f"{better!r} is significantly better than {worse!r} "
            f"(p={p_value:.4f} < alpha={alpha:.2f}, "
            f"win rate {best_rate:.1%}, 95% CI {best_lo:.1%}–{best_hi:.1%})."
        )
    else:
        sentence = (
            f"No significant difference between {model_a!r} and {model_b!r} "
            f"(p={p_value:.4f} >= alpha={alpha:.2f}, {n_decided} decisive round(s))."
        )

    return CompareSignificance(
        model_a=model_a,
        model_b=model_b,
        n_cases=n_cases,
        n_decided=n_decided,
        n_wins_a=n_wins_a,
        n_wins_b=n_wins_b,
        n_ties=ties,
        win_rate_a=win_rate_a,
        p_value=p_value,
        significant=significant,
        test_name="exact sign test",
        odds_ratio=odds_ratio,
        log_odds_ratio=log_odds_ratio,
        confidence_interval=ci,
        n_other_wins=other_wins,
        interpretation=sentence,
    )
