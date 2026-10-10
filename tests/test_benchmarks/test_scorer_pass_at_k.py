import pytest

from deepeval.scorer.scorer import Scorer


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (200, 0, 1, 0.0),
        (200, 1, 1, 0.005),
        (200, 100, 1, 0.5),
        (200, 100, 200, 1.0),
    ],
)
def test_pass_at_k_valid_inputs_keep_existing_scores(n, c, k, expected):
    assert Scorer().pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (0, 0, 1),
        (-1, 0, 1),
        (200, -1, 1),
        (200, 201, 1),
        (200, 100, 0),
    ],
)
def test_pass_at_k_rejects_out_of_range_inputs(n, c, k):
    with pytest.raises(ValueError):
        Scorer().pass_at_k(n, c, k)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (200.0, 100, 1),
        (200, 100.0, 1),
        (200, 100, 1.0),
        (True, 0, 1),
    ],
)
def test_pass_at_k_rejects_non_integer_inputs(n, c, k):
    with pytest.raises(TypeError):
        Scorer().pass_at_k(n, c, k)
