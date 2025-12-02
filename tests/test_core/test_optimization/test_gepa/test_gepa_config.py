import pytest
from pydantic import ValidationError

from deepeval.optimization.gepa.configs import GEPAConfig
from deepeval.optimization.policies.tie_breaker import TieBreaker


def test_gepa_config_defaults_sanity():
    """
    Basic sanity check on GEPAConfig defaults.
    """
    cfg = GEPAConfig()

    # Core iteration & minibatch defaults
    assert cfg.iterations == 5
    assert cfg.minibatch_size is None
    assert cfg.minibatch_min_size == 4
    assert cfg.minibatch_max_size == 32
    assert cfg.minibatch_ratio == pytest.approx(0.05)

    # Pareto split & acceptance
    assert cfg.pareto_size == 3
    assert cfg.min_delta == pytest.approx(0.0)

    # Tie handling
    assert cfg.tie_tolerance == pytest.approx(1e-9)
    assert cfg.tie_breaker == TieBreaker.PREFER_CHILD

    # Random seed default (no validator involved when not None)
    assert cfg.random_seed == 0

    # Prompt feedback / rewrite text length limit
    assert cfg.rewrite_instruction_max_chars == 4096


def test_gepa_config_random_seed_none_uses_time_based_seed():
    """
    If random_seed is None, validator should replace it with a non-zero int
    (time based seed), not the default 0.
    """
    cfg = GEPAConfig(
        iterations=1,
        minibatch_size=1,
        pareto_size=1,
        random_seed=None,
    )
    assert isinstance(cfg.random_seed, int)
    # We don't know the exact value, but it should not be None
    # and it should not fall back to 0.
    assert cfg.random_seed is not None
    assert cfg.random_seed != 0


def test_gepa_config_random_seed_preserves_explicit_value():
    """
    When an explicit random_seed is provided (including 0),
    the validator should not override it.
    """
    cfg = GEPAConfig(random_seed=123)
    assert cfg.random_seed == 123

    cfg_zero = GEPAConfig(random_seed=0)
    assert cfg_zero.random_seed == 0


def test_gepa_config_tie_breaker_defaults_and_alias():
    """
    GEPAConfig should expose its tie breaker enum and default policy.
    """
    cfg = GEPAConfig()

    # The alias is kept for user ergonomics.
    assert GEPAConfig.TieBreaker is TieBreaker

    # Default tie breaker should be PREFER_CHILD.
    assert cfg.tie_breaker == TieBreaker.PREFER_CHILD


def test_gepa_config_accepts_non_default_tie_breaker():
    """
    Users should be able to select a non-default tie breaker policy.
    """
    cfg = GEPAConfig(tie_breaker=TieBreaker.RANDOM)
    assert cfg.tie_breaker == TieBreaker.RANDOM


def test_gepa_config_field_bounds_validated():
    """
    Pydantic constraints should reject out of range values.
    """
    # iterations must be PositiveInt
    with pytest.raises(ValidationError):
        GEPAConfig(iterations=0)

    # pareto_size must be >= 1
    with pytest.raises(ValidationError):
        GEPAConfig(pareto_size=0)

    # minibatch_size, when provided, must be >= 1
    with pytest.raises(ValidationError):
        GEPAConfig(minibatch_size=0)

    # minibatch_ratio must be in (0, 1]
    with pytest.raises(ValidationError):
        GEPAConfig(minibatch_ratio=0.0)

    with pytest.raises(ValidationError):
        GEPAConfig(minibatch_ratio=1.5)

    # tie_tolerance and min_delta must be >= 0
    with pytest.raises(ValidationError):
        GEPAConfig(tie_tolerance=-1e-3)

    with pytest.raises(ValidationError):
        GEPAConfig(min_delta=-0.1)

    # rewrite_instruction_max_chars must be PositiveInt (>= 1)
    with pytest.raises(ValidationError):
        GEPAConfig(rewrite_instruction_max_chars=0)

    with pytest.raises(ValidationError):
        GEPAConfig(rewrite_instruction_max_chars=-10)
