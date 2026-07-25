"""is_successful() must recompute the verdict from score/threshold, not discard it.

TurnRelevancyMetric and TopicAdherenceMetric evaluated `self.score >= self.threshold`
as a bare expression, so the computed pass/fail was thrown away and the method returned
whatever `self.success` already held (the class default None, or a stale value from an
earlier run). Every other metric assigns `self.success = self.score >= self.threshold`.

These tests build the metric with a no-op judge, so they run without any API key and
never call a model: is_successful() is pure score/threshold arithmetic.
"""

import pytest

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import TurnRelevancyMetric, TopicAdherenceMetric


class _NoOpJudge(DeepEvalBaseLLM):
    """Satisfies construction without touching a provider; never invoked by is_successful()."""

    def __init__(self):
        super().__init__(model="noop")

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        return None

    async def a_generate(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "noop"


def _metrics():
    return [
        TurnRelevancyMetric(model=_NoOpJudge(), async_mode=False),
        TopicAdherenceMetric(
            model=_NoOpJudge(), relevant_topics=["x"], async_mode=False
        ),
    ]


@pytest.mark.parametrize("metric", _metrics())
def test_is_successful_recomputes_pass_when_never_measured(metric):
    # success left at its class default (None): a passing score must yield True.
    metric.threshold = 0.5
    metric.score = 0.9
    metric.error = None
    assert metric.is_successful() is True
    assert metric.success is True


@pytest.mark.parametrize("metric", _metrics())
def test_is_successful_recomputes_fail(metric):
    metric.threshold = 0.5
    metric.score = 0.1
    metric.error = None
    assert metric.is_successful() is False
    assert metric.success is False


@pytest.mark.parametrize("metric", _metrics())
def test_is_successful_overwrites_stale_success(metric):
    # a prior failing run left success=False; score/threshold now describe a pass.
    metric.threshold = 0.5
    metric.score = 0.9
    metric.error = None
    metric.success = False
    assert metric.is_successful() is True
    assert metric.success is True
