from __future__ import annotations

import pytest

from deepeval.optimizer.algorithms import SIMBA


def test_simba_defaults():
    """
    SIMBA should have sensible defaults for all parameters.
    """
    algo = SIMBA()

    # Base defaults
    assert algo.iterations == 5
    assert algo.population_size == 4
    assert algo.proposals_per_step == 4
    assert algo.minibatch_size == 8

    # SIMBA-specific defaults
    assert algo.max_demos_per_proposal == 3


def test_simba_allows_zero_demos():
    """
    max_demos_per_proposal can be set to 0 to effectively disable
    APPEND_DEMO, leaving SIMBA in a rule-only configuration.
    """
    algo = SIMBA(max_demos_per_proposal=0)
    assert algo.max_demos_per_proposal == 0


def test_simba_rejects_negative_demos():
    """
    max_demos_per_proposal is constrained to be >= 0.
    """
    with pytest.raises(ValueError):
        SIMBA(max_demos_per_proposal=-1)
