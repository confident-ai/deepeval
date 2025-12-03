import pytest

from deepeval.optimizer.algorithms import GEPA
from deepeval.optimizer.policies import TieBreaker


def test_gepa_defaults_sanity():
    """
    Basic sanity check on GEPA defaults.
    """
    gepa = GEPA()

    # Core iteration & minibatch defaults
    assert gepa.iterations == 5
    assert gepa.minibatch_size == 8

    # Pareto split
    assert gepa.pareto_size == 3

    # Tie handling
    assert gepa.tie_breaker == TieBreaker.PREFER_CHILD

    # Random seed default (should be set to time-based seed)
    assert isinstance(gepa.random_seed, int)


def test_gepa_random_seed_none_uses_time_based_seed():
    """
    If random_seed is None, GEPA should use a time-based seed.
    """
    gepa = GEPA(
        iterations=1,
        minibatch_size=1,
        pareto_size=1,
        random_seed=None,
    )
    assert isinstance(gepa.random_seed, int)
    # We don't know the exact value, but it should not be None
    # and it should not fall back to 0.
    assert gepa.random_seed is not None
    assert gepa.random_seed != 0


def test_gepa_random_seed_preserves_explicit_value():
    """
    When an explicit random_seed is provided (including 0),
    it should be preserved.
    """
    gepa = GEPA(random_seed=123)
    assert gepa.random_seed == 123

    gepa_zero = GEPA(random_seed=0)
    assert gepa_zero.random_seed == 0


def test_gepa_tie_breaker_defaults_and_alias():
    """
    GEPA should expose its tie breaker enum and default policy.
    """
    gepa = GEPA()

    # The alias is kept for user ergonomics.
    assert GEPA.TieBreaker is TieBreaker

    # Default tie breaker should be PREFER_CHILD.
    assert gepa.tie_breaker == TieBreaker.PREFER_CHILD


def test_gepa_accepts_non_default_tie_breaker():
    """
    Users should be able to select a non-default tie breaker policy.
    """
    gepa = GEPA(tie_breaker=TieBreaker.RANDOM)
    assert gepa.tie_breaker == TieBreaker.RANDOM


def test_gepa_field_bounds_validated():
    """
    GEPA should reject out of range values.
    """
    # iterations must be >= 1
    with pytest.raises(ValueError):
        GEPA(iterations=0)

    # pareto_size must be >= 1
    with pytest.raises(ValueError):
        GEPA(pareto_size=0)

    # minibatch_size must be >= 1
    with pytest.raises(ValueError):
        GEPA(minibatch_size=0)
