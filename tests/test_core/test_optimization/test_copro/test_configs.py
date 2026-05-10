from __future__ import annotations

import random

from deepeval.optimizer.algorithms import COPRO


def test_copro_defaults() -> None:
    algo = COPRO()
    assert algo.depth == 4
    assert algo.breadth == 7
    assert algo.minibatch_size == 25
    assert isinstance(algo.random_state, random.Random)
    assert isinstance(algo.seed, int)


def test_copro_accepts_explicit_random_state() -> None:
    r = random.Random(123)
    algo = COPRO(random_state=r)
    assert algo.random_state is r
    assert isinstance(algo.seed, int)


def test_copro_int_random_state_sets_seed() -> None:
    algo = COPRO(random_state=99)
    assert algo.seed == 99
    assert isinstance(algo.random_state, random.Random)


def test_copro_allows_minimal_hyperparameters() -> None:
    algo = COPRO(depth=1, breadth=1, minibatch_size=1, random_state=0)
    assert algo.depth == 1
    assert algo.breadth == 1
    assert algo.minibatch_size == 1
