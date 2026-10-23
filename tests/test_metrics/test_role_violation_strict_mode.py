"""Regression tests for RoleViolationMetric's strict-mode threshold.

In strict mode the metric must fail whenever a role violation is detected.
The bug was initializing ``threshold = 0`` in strict mode: the scoring is
binary (0.0 = violation, 1.0 = adherence), so a violating score of 0.0 still
satisfied ``0.0 >= 0`` and the metric always "succeeded". Strict mode now uses
a threshold of 1, so only a perfect adherence score passes.

These tests exercise the threshold and ``is_successful`` logic directly with a
stub model, so they run without an LLM provider API key.
"""

from deepeval.metrics import RoleViolationMetric
from deepeval.models.base_model import DeepEvalBaseLLM


class _StubLLM(DeepEvalBaseLLM):
    """Minimal DeepEvalBaseLLM that never calls out to a provider."""

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs) -> str:
        raise AssertionError("generate should not be called in this test")

    async def a_generate(self, *args, **kwargs) -> str:
        raise AssertionError("a_generate should not be called in this test")

    def get_model_name(self, *args, **kwargs) -> str:
        return "stub-llm"


def _metric(strict_mode: bool) -> RoleViolationMetric:
    return RoleViolationMetric(
        role="helpful assistant",
        model=_StubLLM(),
        strict_mode=strict_mode,
    )


def test_strict_mode_threshold_is_one():
    assert _metric(strict_mode=True).threshold == 1


def test_strict_mode_fails_on_violation():
    metric = _metric(strict_mode=True)
    # A detected role violation scores 0.0 (see ``_calculate_score``).
    metric.score = 0.0
    assert metric.is_successful() is False


def test_strict_mode_passes_on_full_adherence():
    metric = _metric(strict_mode=True)
    metric.score = 1.0
    assert metric.is_successful() is True


def test_non_strict_mode_uses_provided_threshold():
    metric = RoleViolationMetric(
        role="helpful assistant",
        model=_StubLLM(),
        threshold=0.5,
        strict_mode=False,
    )
    assert metric.threshold == 0.5
    # A violating score of 0.0 fails a 0.5 threshold either way, but a
    # partial/perfect score should still pass as before.
    metric.score = 0.0
    assert metric.is_successful() is False
    metric.score = 1.0
    assert metric.is_successful() is True
