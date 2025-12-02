import pytest
from pydantic import ValidationError

from deepeval.optimization.miprov2.configs import MIPROConfig


def test_miproconfig_defaults():
    cfg = MIPROConfig()

    assert cfg.iterations == 5
    assert cfg.minibatch_size is None
    assert cfg.minibatch_min_size == 4
    assert cfg.minibatch_max_size == 32
    assert cfg.minibatch_ratio == 0.05
    assert cfg.random_seed == 0
    assert cfg.min_delta == 0.0
    assert cfg.exploration_probability == 0.2
    assert cfg.full_eval_every == 5
    assert cfg.rewrite_instruction_max_chars == 4096


def test_miproconfig_coerces_none_random_seed_to_int():
    cfg = MIPROConfig(random_seed=None)

    # Validator should replace None with an integer seed derived from time.time_ns()
    assert isinstance(cfg.random_seed, int)
    assert cfg.random_seed != 0  # very unlikely to conflict with the default


def test_miproconfig_rejects_invalid_minibatch_and_ratio():
    # minibatch_min_size must be >= 1
    with pytest.raises(ValidationError):
        MIPROConfig(minibatch_min_size=0)

    # minibatch_max_size must be >= 1
    with pytest.raises(ValidationError):
        MIPROConfig(minibatch_max_size=0)

    # minibatch_ratio must be > 0 and <= 1
    with pytest.raises(ValidationError):
        MIPROConfig(minibatch_ratio=0.0)

    with pytest.raises(ValidationError):
        MIPROConfig(minibatch_ratio=1.5)


def test_miproconfig_rejects_invalid_probabilities_and_min_delta():
    # exploration_probability must be in [0, 1]
    with pytest.raises(ValidationError):
        MIPROConfig(exploration_probability=-0.1)

    with pytest.raises(ValidationError):
        MIPROConfig(exploration_probability=1.1)

    # min_delta must be >= 0
    with pytest.raises(ValidationError):
        MIPROConfig(min_delta=-0.001)
