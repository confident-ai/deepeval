import math
import pytest

from deepeval.evaluate.statistics import wilson_interval


def test_returns_none_for_zero_observations():
    assert wilson_interval(0, 0) is None


def test_all_failures_does_not_collapse_to_zero():
    # Wald would return (0.0, 0.0), claiming certainty from 10 samples.
    lower, upper = wilson_interval(0, 10)
    assert lower == 0.0
    assert upper == pytest.approx(0.2775, abs=1e-3)


def test_all_passes_does_not_collapse_to_one():
    # Wald would return (1.0, 1.0).
    lower, upper = wilson_interval(40, 40)
    assert lower == pytest.approx(0.9124, abs=1e-3)
    assert upper == 1.0


def test_known_value():
    lower, upper = wilson_interval(34, 40)
    assert lower == pytest.approx(0.7093, abs=1e-3)
    assert upper == pytest.approx(0.9294, abs=1e-3)


def test_bounds_always_within_unit_interval():
    for n in range(1, 60):
        for k in range(n + 1):
            lower, upper = wilson_interval(k, n)
            assert 0.0 <= lower <= upper <= 1.0


def test_interval_narrows_as_sample_grows():
    small = wilson_interval(17, 20)
    large = wilson_interval(170, 200)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_rejects_successes_greater_than_n():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=0.975)
