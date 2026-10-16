import math
from dataclasses import FrozenInstanceError

import pytest

from deepeval.metrics.community import CalibrationResult, calculate_calibration


def test_perfect_calibration_does_not_require_perfect_accuracy():
    result = calculate_calibration(
        [0.25] * 4 + [0.75] * 4, [1, 0, 0, 0, 1, 1, 1, 0]
    )

    assert isinstance(result, CalibrationResult)
    assert result.ece == 0.0
    assert result.brier == 0.1875


def test_maximal_miscalibration():
    result = calculate_calibration([0.0, 1.0], [1, 0])

    assert result.ece == 1.0
    assert result.brier == 1.0


def test_brier_hand_calculation():
    result = calculate_calibration([0.1, 0.4, 0.8], [0, 1, 1])

    # (0.01 + 0.36 + 0.04) / 3
    assert result.brier == pytest.approx(0.41 / 3)


@pytest.mark.parametrize("confidence,outcome", [(0.0, 0), (1.0, 1)])
def test_endpoints(confidence, outcome):
    result = calculate_calibration([confidence], [outcome])

    assert result.ece == 0.0
    assert result.brier == 0.0


def test_equal_width_bins_weight_by_sample_count_and_ignore_empty_bins():
    result = calculate_calibration([0.1, 0.1, 0.8], [0, 0, 1])

    assert result.ece == pytest.approx((2 / 3) * 0.1 + (1 / 3) * 0.2)


@pytest.mark.parametrize("n_bins", [2, 10, 39])
def test_boundary_and_adjacent_floats(n_bins):
    for index in range(1, n_bins):
        edge = index / n_bins
        below = math.nextafter(edge, 0.0)
        above = math.nextafter(edge, 1.0)

        # The edge belongs to the upper bin. Separate bins cannot cancel
        # opposite confidence errors (including the 31/39 regression).
        separate = calculate_calibration([below, edge], [0, 1], n_bins)
        assert separate.ece == pytest.approx((below + 1 - edge) / 2)

        together = calculate_calibration([edge, above], [0, 1], n_bins)
        assert together.ece == pytest.approx(abs((edge + above) / 2 - 0.5))


def test_one_bin_and_brier_independence_from_binning():
    confidences = [0.1, 0.4, 0.8]
    outcomes = [0, 1, 1]
    one_bin = calculate_calibration(confidences, outcomes, n_bins=1)
    ten_bins = calculate_calibration(confidences, outcomes)

    assert one_bin.ece == pytest.approx(abs(1.3 / 3 - 2 / 3))
    assert one_bin.brier == ten_bins.brier


def test_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        calculate_calibration([], [])


@pytest.mark.parametrize(
    "confidences,outcomes", [([0.5], []), ([], [0]), ([0.5], [0, 1])]
)
def test_mismatched_lengths(confidences, outcomes):
    with pytest.raises(ValueError, match="same length"):
        calculate_calibration(confidences, outcomes)


@pytest.mark.parametrize(
    "confidence", [-0.1, 1.1, math.nan, math.inf, -math.inf, 10**1000]
)
def test_invalid_confidence_values(confidence):
    with pytest.raises(ValueError, match=r"confidences\[0\]"):
        calculate_calibration([confidence], [0])


@pytest.mark.parametrize("confidence", ["0.5", None, True, False, 0.5 + 0j])
def test_invalid_confidence_types(confidence):
    with pytest.raises(TypeError, match=r"confidences\[0\]"):
        calculate_calibration([confidence], [0])


@pytest.mark.parametrize("outcome", [-1, 2, 0.5, math.nan, math.inf, -math.inf])
def test_invalid_outcome_values(outcome):
    with pytest.raises(ValueError, match=r"outcomes\[0\]"):
        calculate_calibration([0.5], [outcome])


@pytest.mark.parametrize("outcome", ["1", None, True, False, 1 + 0j])
def test_invalid_outcome_types(outcome):
    with pytest.raises(TypeError, match=r"outcomes\[0\]"):
        calculate_calibration([0.5], [outcome])


@pytest.mark.parametrize("n_bins", [0, -1])
def test_invalid_bin_count_values(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        calculate_calibration([0.5], [1], n_bins)


@pytest.mark.parametrize("n_bins", [True, False, 2.0, 1.5, "10", None])
def test_invalid_bin_count_types(n_bins):
    with pytest.raises(TypeError, match="n_bins"):
        calculate_calibration([0.5], [1], n_bins)


def test_numeric_endpoints_and_float_binary_outcomes():
    assert calculate_calibration((0, 1), (0.0, 1.0)) == CalibrationResult(
        0.0, 0.0
    )


def test_repeated_calls_and_input_order_do_not_leak_state():
    confidences = [0.1, 0.2, 0.8, 0.9]
    outcomes = [0, 1, 1, 0]
    first = calculate_calibration(confidences, outcomes)
    calculate_calibration([0.0, 1.0], [0, 1])

    assert calculate_calibration(confidences, outcomes) == first
    assert calculate_calibration(confidences[::-1], outcomes[::-1]) == first
    assert confidences == [0.1, 0.2, 0.8, 0.9]
    assert outcomes == [0, 1, 1, 0]
    with pytest.raises(FrozenInstanceError):
        first.ece = 1.0


def test_brier_matches_sklearn_when_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    confidences = [0.0, 0.1, 0.4, 0.8, 1.0]
    outcomes = [0, 1, 0, 1, 1]

    assert calculate_calibration(confidences, outcomes).brier == pytest.approx(
        sklearn_metrics.brier_score_loss(outcomes, confidences)
    )


def test_numpy_unsigned_inputs_when_available():
    numpy = pytest.importorskip("numpy")

    result = calculate_calibration(
        numpy.array([0, 1], dtype=numpy.uint64),
        numpy.array([1, 0], dtype=numpy.uint64),
        n_bins=numpy.int64(10),
    )

    assert result.ece == 1.0
    assert result.brier == 1.0
