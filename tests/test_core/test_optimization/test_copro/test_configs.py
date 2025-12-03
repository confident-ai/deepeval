from __future__ import annotations

import pytest

from deepeval.optimizer.algorithms import COPRO


def test_copro_defaults():
    algo = COPRO()

    # Base defaults
    assert algo.iterations == 5
    assert algo.minibatch_size == 8
    assert algo.exploration_probability == 0.2
    assert algo.full_eval_every == 5
    assert isinstance(algo.random_seed, int)

    # COPRO-specific defaults
    assert algo.population_size == 4
    assert algo.proposals_per_step == 4


@pytest.mark.parametrize("value", [0, -1])
def test_copro_rejects_non_positive_population_size(value: int):
    with pytest.raises(ValueError):
        COPRO(population_size=value)


@pytest.mark.parametrize("value", [0, -3])
def test_copro_rejects_non_positive_proposals_per_step(value: int):
    with pytest.raises(ValueError):
        COPRO(proposals_per_step=value)
