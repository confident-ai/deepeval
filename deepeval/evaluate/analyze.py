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
    """Standard normal CDF via the library ``math.erf``."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_binomial_exact_p(n: int, k: int) -> float:
    """Exact two-sided p-value for the sign test.

    Tests H0 ``p = 0.5`` against a two-sided alternative. We define the
    two-tailed p-value the way the binomial sign test usually is: take the
    smaller of the two one-sided tails ``P(X <= k)`` and ``P(X >= k)`` and
    double it, capping at 1.0. Using the exact binomial mass keeps the result
    valid at tiny ``n`` where a normal approximation would be misleading.

    Only valid for ``0 <= k <= n`` and ``n >= 1``.
    """
    if n < 1:
        raise ValueError("Cannot run a sign test on fewer than one pair.")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n.")

    def lower_tail(kk: int) -> float:
        return sum(math.comb(n, i) for i in range(kk + 1)) * (0.5**n)

    def upper_tail(kk: int) -> float:
        return sum(math.comb(n, i) for i in range(kk, n + 1)) * (0.5**n)

    return min(1.0, 2.0 * min(lower_tail(k), upper_tail(k)))


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

    # Reject tokens that match neither side. Silently treating them as
    # ties would let a typo'd contestant name (or feeding in compare()'s
    # output keyed by non-matching aliases) quietly zero out wins and flip
    # the verdict — the exact thing a significance check exists to prevent.
    unknown = {
        w for w in winners if w is not None and w != model_a and w != model_b
    }
    if unknown:
        raise ValueError(
            "winners contains names that are neither model_a nor model_b: "
            f"{sorted(unknown)!r}. Every non-tie winner must be one of the "
            "two compared models."
        )

    decided = [w for w in winners if w is not None]

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
        interpretation=sentence,
    )
