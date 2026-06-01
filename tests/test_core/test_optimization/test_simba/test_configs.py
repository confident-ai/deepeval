from __future__ import annotations

import random

from deepeval.optimizer.algorithms import SIMBA


def test_simba_defaults() -> None:
    algo = SIMBA()
    assert algo.iterations == 8
    assert algo.minibatch_size == 15
    assert algo.num_candidates == 4
    assert algo.num_samples == 3
    assert algo.minibatch_full_eval_steps == 4
    assert isinstance(algo.random_state, random.Random)
    assert isinstance(algo.seed, int)


def test_simba_accepts_explicit_random_state() -> None:
    r = random.Random(42)
    algo = SIMBA(random_state=r)
    assert algo.random_state is r
    assert isinstance(algo.seed, int)


def test_simba_int_random_state_sets_seed() -> None:
    algo = SIMBA(random_state=7)
    assert algo.seed == 7
    assert isinstance(algo.random_state, random.Random)


def test_simba_allows_minimal_hyperparameters() -> None:
    algo = SIMBA(
        iterations=1,
        minibatch_size=2,
        num_candidates=1,
        num_samples=2,
        minibatch_full_eval_steps=1,
        random_state=0,
    )
    assert algo.iterations == 1
    assert algo.num_candidates == 1
