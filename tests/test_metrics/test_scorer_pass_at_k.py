"""
Tests for `Scorer.pass_at_k` argument validation.

`pass_at_k` computes the HumanEval pass@k score. Degenerate inputs (non-int,
n < 1, c outside [0, n], k < 1) previously returned silently wrong scores;
they must now raise an explicit TypeError/ValueError, while all valid inputs
keep returning exactly the same scores as before.
"""

import importlib.util

import pytest

from deepeval.scorer import Scorer

scorer = Scorer()

# numpy is an optional dependency (dev/integrations groups only), so the
# metric CI job installs the package without it. Argument validation must
# work regardless of numpy; only the score-computation assertions need it.
requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy is not installed in this environment",
)


# --------------------------------------------------------------------------- #
# Default behavior is preserved for valid inputs
# --------------------------------------------------------------------------- #


@requires_numpy
def test_pass_at_k_returns_previous_values_for_valid_inputs():
    # These are the values produced before the validation was added; the fix
    # must not change any of them.
    assert scorer.pass_at_k(200, 0, 1) == pytest.approx(0.0)
    assert scorer.pass_at_k(200, 200, 1) == pytest.approx(1.0)
    assert scorer.pass_at_k(200, 100, 1) == pytest.approx(0.5, abs=1e-9)
    assert scorer.pass_at_k(200, 100, 2) == pytest.approx(0.7512562814070352)


@requires_numpy
def test_pass_at_k_is_monotonic_in_c():
    # More correct samples can never lower the score.
    assert scorer.pass_at_k(200, 50, 1) >= scorer.pass_at_k(200, 20, 1)
    assert scorer.pass_at_k(200, 150, 1) >= scorer.pass_at_k(200, 50, 1)


@requires_numpy
def test_pass_at_k_is_monotonic_in_k():
    # A larger k can never lower the score.
    assert scorer.pass_at_k(200, 50, 2) >= scorer.pass_at_k(200, 50, 1)


@requires_numpy
def test_pass_at_k_perfect_score_is_exactly_one():
    assert scorer.pass_at_k(200, 200, 1) == 1.0


# --------------------------------------------------------------------------- #
# New validation: degenerate inputs raise instead of returning wrong scores
# --------------------------------------------------------------------------- #


def test_pass_at_k_rejects_zero_samples():
    with pytest.raises(ValueError, match="'n'.*at least 1"):
        scorer.pass_at_k(0, 0, 1)


def test_pass_at_k_rejects_negative_samples():
    with pytest.raises(ValueError, match="'n'.*at least 1"):
        scorer.pass_at_k(-1, 0, 1)


def test_pass_at_k_rejects_negative_correct_count():
    with pytest.raises(ValueError, match="'c'"):
        scorer.pass_at_k(200, -1, 1)


def test_pass_at_k_rejects_correct_count_greater_than_samples():
    with pytest.raises(ValueError, match="'c'"):
        scorer.pass_at_k(200, 250, 1)


def test_pass_at_k_rejects_zero_k():
    with pytest.raises(ValueError, match="'k'.*at least 1"):
        scorer.pass_at_k(200, 100, 0)


def test_pass_at_k_rejects_negative_k():
    with pytest.raises(ValueError, match="'k'.*at least 1"):
        scorer.pass_at_k(200, 100, -1)


def test_pass_at_k_rejects_non_integer_arguments():
    with pytest.raises(TypeError, match="'n'"):
        scorer.pass_at_k(200.0, 0, 1)
    with pytest.raises(TypeError, match="'c'"):
        scorer.pass_at_k(200, 0.0, 1)
    with pytest.raises(TypeError, match="'k'"):
        scorer.pass_at_k(200, 0, 1.0)


def test_pass_at_k_rejects_bool_arguments():
    # bool is a subclass of int but is not a valid sample count.
    with pytest.raises(TypeError, match="'n'"):
        scorer.pass_at_k(True, 0, 1)
