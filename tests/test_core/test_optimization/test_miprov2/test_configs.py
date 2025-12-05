import pytest

from deepeval.optimizer.algorithms import MIPROV2


def test_miprov2_defaults():
    algo = MIPROV2()

    assert algo.iterations == 5
    assert algo.minibatch_size == 8
    assert isinstance(algo.random_seed, int)
    assert algo.exploration_probability == 0.2
    assert algo.full_eval_every == 5


def test_miprov2_coerces_none_random_seed_to_int():
    algo = MIPROV2(random_seed=None)

    # Should replace None with an integer seed derived from time.time_ns()
    assert isinstance(algo.random_seed, int)


def test_miprov2_rejects_invalid_minibatch_size():
    # minibatch_size must be >= 1
    with pytest.raises(ValueError):
        MIPROV2(minibatch_size=0)


def test_miprov2_rejects_invalid_probabilities():
    # exploration_probability must be in [0, 1]
    with pytest.raises(ValueError):
        MIPROV2(exploration_probability=-0.1)

    with pytest.raises(ValueError):
        MIPROV2(exploration_probability=1.1)
