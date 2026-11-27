"""Regression tests for strict_mode scoring in multi-window turn metrics."""

from unittest.mock import AsyncMock, patch

import pytest

from deepeval.metrics import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
)
from deepeval.metrics.turn_contextual_precision.schema import (
    InteractionContextualPrecisionScore,
)
from deepeval.metrics.turn_contextual_recall.schema import (
    InteractionContextualRecallScore,
)
from deepeval.metrics.turn_contextual_relevancy.schema import (
    InteractionContextualRelevancyScore,
)
from deepeval.metrics.turn_faithfulness.schema import (
    InteractionFaithfulnessScore,
)
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_core.stubs import DummyModel

METRIC_SPECS = [
    (
        TurnContextualPrecisionMetric,
        "deepeval.metrics.turn_contextual_precision.turn_contextual_precision.initialize_model",
        InteractionContextualPrecisionScore,
        "_get_contextual_precision_scores",
        "_a_get_contextual_precision_scores",
        {"verdicts": []},
    ),
    (
        TurnContextualRecallMetric,
        "deepeval.metrics.turn_contextual_recall.turn_contextual_recall.initialize_model",
        InteractionContextualRecallScore,
        "_get_contextual_recall_scores",
        "_a_get_contextual_recall_scores",
        {"verdicts": []},
    ),
    (
        TurnContextualRelevancyMetric,
        "deepeval.metrics.turn_contextual_relevancy.turn_contextual_relevancy.initialize_model",
        InteractionContextualRelevancyScore,
        "_get_contextual_relevancy_scores",
        "_a_get_contextual_relevancy_scores",
        {"verdicts": []},
    ),
    (
        TurnFaithfulnessMetric,
        "deepeval.metrics.turn_faithfulness.turn_faithfulness.initialize_model",
        InteractionFaithfulnessScore,
        "_get_faithfulness_scores",
        "_a_get_faithfulness_scores",
        {"claims": [], "truths": [], "verdicts": []},
    ),
]


def _make_metric(metric_cls, init_path, *, strict_mode: bool, async_mode: bool):
    with patch(init_path, return_value=(DummyModel(), True)):
        return metric_cls(
            strict_mode=strict_mode,
            async_mode=async_mode,
            include_reason=False,
            window_size=1,
        )


def _two_window_test_case() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Question 1"),
            Turn(
                role="assistant",
                content="Answer 1",
                retrieval_context=["Context 1"],
            ),
            Turn(role="user", content="Question 2"),
            Turn(
                role="assistant",
                content="Answer 2",
                retrieval_context=["Context 2"],
            ),
        ],
        expected_outcome="Answer both questions accurately.",
    )


@pytest.mark.parametrize(
    "metric_cls,init_path,score_cls,_sync_hook,_async_hook,extra_fields",
    METRIC_SPECS,
)
def test_calculate_score_enforces_binary_strict_mode(
    metric_cls, init_path, score_cls, _sync_hook, _async_hook, extra_fields
):
    strict_metric = _make_metric(
        metric_cls, init_path, strict_mode=True, async_mode=False
    )
    mixed = [
        score_cls(score=1.0, reason="pass", **extra_fields),
        score_cls(score=0.0, reason="fail", **extra_fields),
    ]
    partial = [
        score_cls(score=1.0, reason="pass", **extra_fields),
        score_cls(score=0.5, reason="partial", **extra_fields),
    ]
    perfect = [
        score_cls(score=1.0, reason="pass", **extra_fields),
        score_cls(score=1.0, reason="pass", **extra_fields),
    ]

    assert strict_metric._calculate_score(mixed) == 0
    assert strict_metric._calculate_score(partial) == 0
    assert strict_metric._calculate_score(perfect) == 1.0

    non_strict_metric = _make_metric(
        metric_cls, init_path, strict_mode=False, async_mode=False
    )
    assert non_strict_metric._calculate_score(mixed) == 0.5


@pytest.mark.parametrize(
    "metric_cls,init_path,score_cls,sync_hook,_async_hook,extra_fields",
    METRIC_SPECS,
)
def test_sync_measure_strict_mode_returns_zero_when_any_window_fails(
    metric_cls, init_path, score_cls, sync_hook, _async_hook, extra_fields
):
    metric = _make_metric(
        metric_cls, init_path, strict_mode=True, async_mode=False
    )
    window_results = iter(
        [
            [score_cls(score=1.0, reason="window 1 passed", **extra_fields)],
            [score_cls(score=0.0, reason="window 2 failed", **extra_fields)],
        ]
    )

    with patch.object(
        metric,
        sync_hook,
        side_effect=lambda *args, **kwargs: next(window_results),
    ):
        score = metric.measure(_two_window_test_case(), _show_indicator=False)

    assert score == 0
    assert metric.score == 0
    assert metric.success is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_cls,init_path,score_cls,_sync_hook,async_hook,extra_fields",
    METRIC_SPECS,
)
async def test_async_measure_strict_mode_returns_zero_when_any_window_fails(
    metric_cls, init_path, score_cls, _sync_hook, async_hook, extra_fields
):
    metric = _make_metric(
        metric_cls, init_path, strict_mode=True, async_mode=True
    )
    window_results = iter(
        [
            [score_cls(score=1.0, reason="window 1 passed", **extra_fields)],
            [score_cls(score=0.0, reason="window 2 failed", **extra_fields)],
        ]
    )

    async def fake_async_scores(*args, **kwargs):
        return next(window_results)

    with patch.object(metric, async_hook, side_effect=fake_async_scores):
        score = await metric.a_measure(
            _two_window_test_case(), _show_indicator=False
        )

    assert score == 0
    assert metric.score == 0
    assert metric.success is False
