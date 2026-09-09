"""
Regression tests for benchmark ``n_problems`` / ``n_shots`` validation.

The bounds checks on these two parameters used to be upper-bound only, so a
non-positive value was accepted silently and then corrupted the run rather than
failing. They are offline: no model, network, dataset download, or API key
required.
"""

import pytest

from deepeval.benchmarks.arc.arc import ARC
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.bool_q.bool_q import BoolQ
from deepeval.benchmarks.gsm8k.gsm8k import GSM8K
from deepeval.benchmarks.lambada.lambada import LAMBADA

# (class, kwargs building a valid instance, full-size n_problems)
BENCHMARKS = [
    (BoolQ, {}, 3270),
    (LAMBADA, {}, 5153),
    (GSM8K, {}, 1319),
    (ARC, {"mode": ARCMode.EASY}, 2376),
]

IDS = [b[0].__name__ for b in BENCHMARKS]


# --------------------------------------------------------------------------- #
# Why a non-positive n_problems is not merely an odd argument
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_problems", [0, -1, -5])
def test_non_positive_n_problems_would_corrupt_scoring(n_problems):
    # evaluate() does `goldens[: self.n_problems]` and then divides by
    # `self.n_problems`. This pins down what those two lines do for a value an
    # upper-bound-only check lets through, and is the reason the guard rejects
    # it instead of passing it on.
    goldens = list(range(3270))
    selected = goldens[:n_problems]

    if n_problems == 0:
        assert selected == []
        with pytest.raises(ZeroDivisionError):
            _ = 3 / n_problems
    else:
        # Negative slices from the end: nearly the whole benchmark still runs,
        # burning the API budget the small n_problems was meant to cap...
        assert len(selected) == 3270 + n_problems
        assert len(selected) > 3000
        # ...and the reported accuracy comes out negative.
        assert 3 / n_problems < 0


# --------------------------------------------------------------------------- #
# n_problems
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
@pytest.mark.parametrize("n_problems", [0, -1, -5])
def test_non_positive_n_problems_is_rejected(
    benchmark, kwargs, full_size, n_problems
):
    with pytest.raises(ValueError, match="n_problems must be >= 1"):
        benchmark(n_problems=n_problems, **kwargs)


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
def test_n_problems_above_maximum_is_rejected(benchmark, kwargs, full_size):
    with pytest.raises(ValueError, match="n_problems"):
        benchmark(n_problems=full_size + 1, **kwargs)


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
def test_n_problems_boundaries_are_accepted(benchmark, kwargs, full_size):
    # 1 and the full size are both legitimate; the guard must not narrow the
    # range that already worked.
    assert benchmark(n_problems=1, **kwargs).n_problems == 1
    assert benchmark(n_problems=full_size, **kwargs).n_problems == full_size


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
def test_non_int_n_problems_is_rejected(benchmark, kwargs, full_size):
    # A float silently produces `TypeError: slice indices must be integers`
    # much later, inside evaluate(); bool is an int subclass and n_problems=True
    # would mean "1 problem", which is never what the caller meant.
    with pytest.raises(TypeError, match="n_problems must be an int"):
        benchmark(n_problems=2.5, **kwargs)
    with pytest.raises(TypeError, match="n_problems must be an int"):
        benchmark(n_problems=True, **kwargs)


# --------------------------------------------------------------------------- #
# n_shots
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
@pytest.mark.parametrize("n_shots", [-1, -3])
def test_negative_n_shots_is_rejected(benchmark, kwargs, full_size, n_shots):
    with pytest.raises(ValueError, match="n_shots must be >= 0"):
        benchmark(n_shots=n_shots, **kwargs)


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
def test_zero_n_shots_is_accepted(benchmark, kwargs, full_size):
    # 0 is a real configuration -- zero-shot -- and must stay allowed.
    assert benchmark(n_shots=0, **kwargs).n_shots == 0


@pytest.mark.parametrize("benchmark,kwargs,full_size", BENCHMARKS, ids=IDS)
def test_n_shots_above_maximum_is_rejected(benchmark, kwargs, full_size):
    with pytest.raises(ValueError, match="n_shots"):
        benchmark(n_shots=99, **kwargs)


# --------------------------------------------------------------------------- #
# ARC picks its n_problems ceiling from the mode
# --------------------------------------------------------------------------- #


def test_arc_challenge_uses_its_own_smaller_maximum():
    # ARC-Challenge has 1172 problems, fewer than ARC-Easy's 2376, and the
    # error message should name the mode that was actually validated.
    assert ARC(mode=ARCMode.CHALLENGE, n_problems=1172).n_problems == 1172
    with pytest.raises(ValueError, match="ARC-Challenge"):
        ARC(mode=ARCMode.CHALLENGE, n_problems=1173)


def test_arc_default_n_problems_matches_mode():
    assert ARC(mode=ARCMode.EASY).n_problems == 2376
    assert ARC(mode=ARCMode.CHALLENGE).n_problems == 1172
