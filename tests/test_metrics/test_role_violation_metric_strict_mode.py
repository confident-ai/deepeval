"""Regression tests for #2901: RoleViolationMetric's strict_mode always
succeeded because the threshold was initialized to 0 instead of 1.

RoleViolationMetric's score is binary (0.0 = violation detected, 1.0 = no
violation). With `self.threshold = 0 if strict_mode else threshold`, a
detected violation (score=0.0) satisfied `0.0 >= 0`, so strict_mode had no
effect. These tests use DummyModel and do not require OPENAI_API_KEY.
"""

from unittest.mock import patch

from deepeval.metrics import RoleViolationMetric
from tests.test_core.stubs import DummyModel


def make_metric(*, strict_mode: bool, threshold: float = 0.5):
    """Create RoleViolationMetric with DummyModel so no LLM calls are made."""
    with patch(
        "deepeval.metrics.role_violation.role_violation.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return RoleViolationMetric(
            role="helpful assistant",
            threshold=threshold,
            strict_mode=strict_mode,
        )


def test_strict_mode_threshold_is_one():
    metric = make_metric(strict_mode=True)
    assert metric.threshold == 1


def test_non_strict_mode_keeps_configured_threshold():
    metric = make_metric(strict_mode=False, threshold=0.5)
    assert metric.threshold == 0.5


def test_strict_mode_fails_on_role_violation():
    metric = make_metric(strict_mode=True)
    metric.score = 0.0  # a detected role violation always scores 0.0
    assert metric.is_successful() is False


def test_strict_mode_passes_on_no_violation():
    metric = make_metric(strict_mode=True)
    metric.score = 1.0  # no role violation detected
    assert metric.is_successful() is True
