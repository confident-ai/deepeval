import random

import pytest

from deepeval.errors import DeepEvalError
from deepeval.optimization.policies.tie_breaker import (
    TieBreaker,
    pick_best_with_ties,
)


def test_pick_best_with_ties_single_candidate():
    """
    When there is only one candidate, it should be chosen and be the only tied id.
    """
    totals = {"p1": 0.42}
    parents_by_id = {"p1": None}
    rng = random.Random(123)

    chosen, tied, max_score = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng,
    )

    assert chosen == "p1"
    assert tied == ["p1"]
    assert max_score == pytest.approx(0.42)


def test_pick_best_with_ties_raises_on_empty_totals():
    """
    When there are no candidates, DeepEvalError should be raised.
    """
    totals = {}
    parents_by_id = {}
    rng = random.Random(123)

    with pytest.raises(DeepEvalError):
        pick_best_with_ties(
            totals,
            parents_by_id,
            random_state=rng,
        )


def test_pick_best_with_ties_prefers_child_when_tied():
    """
    When parent and child are tied and policy is PREFER_CHILD, the child should win.
    """
    totals = {
        "root": 0.8,
        "child": 0.8,
    }
    parents_by_id = {
        "root": None,
        "child": "root",
    }
    rng = random.Random(123)

    chosen, tied, max_score = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng,
        tie_tolerance=1e-9,
        policy=TieBreaker.PREFER_CHILD,
    )

    assert set(tied) == {"root", "child"}
    # child should be preferred over root
    assert chosen == "child"
    assert max_score == pytest.approx(0.8)


def test_pick_best_with_ties_prefers_root_when_tied():
    """
    When parent and child are tied and policy is PREFER_ROOT (default),
    the root should win.
    """
    totals = {
        "root": 0.8,
        "child": 0.8,
    }
    parents_by_id = {
        "root": None,
        "child": "root",
    }
    rng = random.Random(123)

    chosen, tied, max_score = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng,
        tie_tolerance=1e-9,
        policy=TieBreaker.PREFER_ROOT,
    )

    assert set(tied) == {"root", "child"}
    assert chosen == "root"
    assert max_score == pytest.approx(0.8)


def test_pick_best_with_ties_random_policy_is_deterministic_with_seed():
    """
    RANDOM policy should be deterministic when given the same Random instance seed.
    We don't care *which* id is chosen, only that the same seed produces the same choice.
    """
    totals = {"a": 1.0, "b": 1.0, "c": 1.0}
    parents_by_id = {k: None for k in totals.keys()}

    rng1 = random.Random(7)
    rng2 = random.Random(7)

    chosen1, tied1, max1 = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng1,
        tie_tolerance=1e-9,
        policy=TieBreaker.RANDOM,
    )
    chosen2, tied2, max2 = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng2,
        tie_tolerance=1e-9,
        policy=TieBreaker.RANDOM,
    )

    # All candidates are tied
    assert set(tied1) == set(tied2) == {"a", "b", "c"}
    # Deterministic with same seed
    assert chosen1 == chosen2
    assert max1 == pytest.approx(max2) == pytest.approx(1.0)


def test_pick_best_with_ties_respects_tie_tolerance():
    """
    tie_tolerance should control when two candidates are considered tied.
    """
    totals = {"a": 1.0, "b": 1.005}
    parents_by_id = {"a": None, "b": None}
    rng = random.Random(123)

    # With a small tolerance, only 'b' should be considered best.
    chosen_strict, tied_strict, _ = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng,
        tie_tolerance=1e-4,  # smaller than the gap 0.005
        policy=TieBreaker.PREFER_ROOT,
    )
    assert chosen_strict == "b"
    assert tied_strict == ["b"]

    # With a looser tolerance, both should be tied.
    rng = random.Random(123)
    chosen_loose, tied_loose, _ = pick_best_with_ties(
        totals,
        parents_by_id,
        random_state=rng,
        tie_tolerance=0.01,  # larger than the gap
        policy=TieBreaker.PREFER_ROOT,
    )
    assert set(tied_loose) == {"a", "b"}
    # PREFER_ROOT => first inserted root ('a') wins in this tie
    assert chosen_loose == "a"
