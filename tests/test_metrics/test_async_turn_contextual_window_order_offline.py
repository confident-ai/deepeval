"""Offline regression coverage for async window ordering in the contextual turn metrics.

Same completion-order bug as confident-ai/deepeval#3480 (TurnFaithfulnessMetric,
fixed in #3481): per-window coroutines appended scores in completion order, so a
later window finishing first shifted verbose labels and final-reason inputs.
"""

import asyncio
from contextlib import nullcontext
import os

os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

import pytest

from deepeval.metrics import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
)
from deepeval.metrics.turn_contextual_precision import (
    turn_contextual_precision as precision_module,
)
from deepeval.metrics.turn_contextual_precision.schema import (
    InteractionContextualPrecisionScore,
)
from deepeval.metrics.turn_contextual_recall import (
    turn_contextual_recall as recall_module,
)
from deepeval.metrics.turn_contextual_recall.schema import (
    InteractionContextualRecallScore,
)
from deepeval.metrics.turn_contextual_relevancy import (
    turn_contextual_relevancy as relevancy_module,
)
from deepeval.metrics.turn_contextual_relevancy.schema import (
    InteractionContextualRelevancyScore,
)
from deepeval.test_case import ConversationalTestCase, Turn


def make_case():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="first user"),
            Turn(
                role="assistant",
                content="first answer",
                retrieval_context=["first fact"],
            ),
            Turn(role="user", content="second user"),
            Turn(
                role="assistant",
                content="second answer",
                retrieval_context=["second fact"],
            ),
        ],
        # Required by the precision and recall metrics; ignored by relevancy.
        expected_outcome="expected outcome",
    )


METRICS = {
    "precision": (
        TurnContextualPrecisionMetric,
        precision_module,
        "_a_get_contextual_precision_scores",
        InteractionContextualPrecisionScore,
    ),
    "recall": (
        TurnContextualRecallMetric,
        recall_module,
        "_a_get_contextual_recall_scores",
        InteractionContextualRecallScore,
    ),
    "relevancy": (
        TurnContextualRelevancyMetric,
        relevancy_module,
        "_a_get_contextual_relevancy_scores",
        InteractionContextualRelevancyScore,
    ),
}


def make_metric(metric_cls):
    metric = metric_cls.__new__(metric_cls)
    metric.threshold = 0.5
    metric.model = None
    metric.using_native_model = False
    metric.evaluation_model = "offline stub"
    metric.window_size = 1
    metric.async_mode = True
    metric.include_reason = True
    metric.error = None
    return metric


def stub_module(monkeypatch, module):
    async def no_system_one(*args, **kwargs):
        return False

    monkeypatch.setattr(module, "a_run_system_one_eval", no_system_one)
    monkeypatch.setattr(module, "run_system_one_eval", lambda *a, **kw: False)
    monkeypatch.setattr(
        module,
        "metric_progress_indicator",
        lambda *args, **kwargs: nullcontext(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_name", list(METRICS))
@pytest.mark.parametrize(
    "first_delay,second_delay",
    [(0.02, 0), (0, 0.02)],
    ids=["first-finishes-last", "first-finishes-first"],
)
async def test_async_windows_keep_conversation_order(
    monkeypatch, metric_name, first_delay, second_delay
):
    metric_cls, module, scores_method, score_cls = METRICS[metric_name]
    metric = make_metric(metric_cls)
    stub_module(monkeypatch, module)

    async def fake_window_scores(turns_window, *args):
        is_first = turns_window[0].content == "first user"
        await asyncio.sleep(first_delay if is_first else second_delay)
        return [
            score_cls(
                score=0.2 if is_first else 0.8,
                reason="first" if is_first else "second",
                verdicts=[],
            )
        ]

    monkeypatch.setattr(metric, scores_method, fake_window_scores)

    observed_reason_order = []

    async def fake_final_reason(scores):
        observed_reason_order.extend(score.reason for score in scores)
        return "stub"

    monkeypatch.setattr(metric, "_a_generate_reason", fake_final_reason)

    score = await metric.a_measure(make_case(), _show_indicator=False)

    assert score == pytest.approx(0.5)
    assert observed_reason_order == ["first", "second"]
    assert "Reason: first" in metric.verbose_logs.split("Window 2")[0]
    assert "Reason: second" in metric.verbose_logs.split("Window 2")[1]
