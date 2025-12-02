from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepeval.optimization.copro.configs import COPROConfig


def test_copro_config_defaults_inherit_mipro_fields():
    cfg = COPROConfig()

    # Inherited MIPROConfig defaults
    assert cfg.iterations == 5
    assert cfg.minibatch_min_size == 4
    assert cfg.minibatch_max_size == 32
    assert cfg.minibatch_ratio == 0.05
    assert cfg.min_delta == 0.0
    assert cfg.exploration_probability == 0.2
    assert cfg.full_eval_every == 5
    assert isinstance(cfg.random_seed, int)

    # COPRO-specific defaults
    assert cfg.population_size == 4
    assert cfg.proposals_per_step == 4


@pytest.mark.parametrize("value", [0, -1])
def test_copro_config_rejects_non_positive_population_size(value: int):
    with pytest.raises(ValidationError):
        COPROConfig(population_size=value)


@pytest.mark.parametrize("value", [0, -3])
def test_copro_config_rejects_non_positive_proposals_per_step(value: int):
    with pytest.raises(ValidationError):
        COPROConfig(proposals_per_step=value)
