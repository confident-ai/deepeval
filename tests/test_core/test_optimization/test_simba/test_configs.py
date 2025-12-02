from __future__ import annotations

import pytest

from deepeval.optimization.copro.configs import COPROConfig
from deepeval.optimization.simba.configs import SIMBAConfig


def test_simba_config_inherits_copro_defaults():
    """
    SIMBAConfig should inherit all fields from COPROConfig / MIPROConfig
    and provide sensible defaults for its own demo-related fields.
    """
    cfg = SIMBAConfig()

    # Inherited from MIPRO / COPRO
    assert isinstance(cfg, COPROConfig)
    assert cfg.iterations == 5
    assert cfg.population_size == 4
    assert cfg.proposals_per_step == 4
    assert cfg.minibatch_min_size == 4
    assert cfg.minibatch_max_size == 32
    assert cfg.minibatch_ratio == 0.05
    assert cfg.rewrite_instruction_max_chars == 4096

    # SIMBA-specific defaults
    assert cfg.max_demos_per_proposal == 3
    assert cfg.demo_input_max_chars == 256


def test_simba_config_allows_zero_demos():
    """
    max_demos_per_proposal can be set to 0 to effectively disable
    APPEND_DEMO, leaving SIMBA in a rule-only configuration.
    """
    cfg = SIMBAConfig(max_demos_per_proposal=0)
    assert cfg.max_demos_per_proposal == 0


def test_simba_config_rejects_negative_demos():
    """
    max_demos_per_proposal is constrained to be >= 0.
    """
    with pytest.raises(ValueError):
        SIMBAConfig(max_demos_per_proposal=-1)
