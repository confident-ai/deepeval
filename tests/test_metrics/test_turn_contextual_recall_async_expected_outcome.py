"""Regression test for TurnContextualRecallMetric async scoring.

In async mode (the default), ``a_measure`` used to forward
``test_case.multimodal`` (a bool) into the ``expected_outcome`` slot of the
scoring helper, dropping the real ``expected_outcome`` text. Every verdict then
scored 0. The sync ``measure`` path was unaffected.

These tests use DummyModel and do not require OPENAI_API_KEY: the scoring helper
and reason generation are stubbed so no LLM call is made, and we assert directly
on the argument that ``a_measure`` forwards.
"""

import pytest
from unittest.mock import patch, AsyncMock

from deepeval.metrics import TurnContextualRecallMetric
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_core.stubs import DummyModel


def make_metric() -> TurnContextualRecallMetric:
    """Create the metric with DummyModel so construction makes no LLM calls."""
    with patch(
        "deepeval.metrics.turn_contextual_recall.turn_contextual_recall.initialize_model"
    ) as mock_init:
        mock_init.return_value = (DummyModel(), True)
        return TurnContextualRecallMetric(async_mode=True)


def make_test_case() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What if these shoes don't fit?"),
            Turn(
                role="assistant",
                content="We offer a 30-day full refund at no extra cost.",
                retrieval_context=[
                    "All customers are eligible for a 30 day full refund at no extra cost."
                ],
            ),
        ],
        expected_outcome="The chatbot must explain store policies like refunds.",
        chatbot_role="A helpful assistant",
    )


@pytest.mark.asyncio
async def test_a_measure_forwards_expected_outcome_not_multimodal_flag():
    """a_measure must pass expected_outcome (str), never the multimodal bool."""
    metric = make_metric()
    test_case = make_test_case()

    captured = {}

    async def fake_scores(window, expected_outcome, multimodal):
        captured["expected_outcome"] = expected_outcome
        captured["multimodal"] = multimodal
        return []

    with patch.object(
        metric, "_a_get_contextual_recall_scores", side_effect=fake_scores
    ), patch.object(
        metric, "_a_generate_reason", new=AsyncMock(return_value="")
    ):
        await metric.a_measure(
            test_case,
            _show_indicator=False,
            _log_metric_to_confident=False,
        )

    assert "expected_outcome" in captured, "scoring helper was never called"
    assert captured["expected_outcome"] == test_case.expected_outcome
    assert isinstance(captured["expected_outcome"], str)
    assert captured["multimodal"] is False
