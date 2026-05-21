import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deepeval.metrics import TurnContextualRecallMetric
from deepeval.test_case import (
    ConversationalTestCase,
    MLLMImage,
    MultiTurnParams,
    Turn,
)
from deepeval import evaluate

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestTurnContextualRecallMetric:
    """Tests for answer relevancy metric"""

    def test_normal_sync_metric_measure(self):
        convo_test_case = ConversationalTestCase(
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
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric(async_mode=False)
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is False

    def test_normal_async_metric_measure(self):
        convo_test_case = ConversationalTestCase(
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
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric()
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(
                    role="user",
                    content=f"What's shown in this image? {image}'",
                ),
                Turn(
                    role="assistant",
                    content=f"That's an image of a car",
                    retrieval_context=[f"Cars are great to look at {image}"],
                ),
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric()
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(
                    role="user",
                    content=f"What's shown in this image? {image}'",
                ),
                Turn(
                    role="assistant",
                    content=f"That's an image of a car",
                    retrieval_context=[f"Cars are great to look at {image}"],
                ),
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric(async_mode=False)
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(
                    role="user",
                    content=f"What's shown in this image? {image}'",
                ),
                Turn(
                    role="assistant",
                    content=f"That's an image of a car",
                    retrieval_context=[f"Cars are great to look at {image}"],
                ),
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        with pytest.raises(ValueError):
            metric = TurnContextualRecallMetric(model="gpt-3.5-turbo")
            metric.measure(convo_test_case)

    def test_normal_evaluate_method(self):
        convo_test_case = ConversationalTestCase(
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
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric()

        results = evaluate([convo_test_case], [metric])

        assert results is not None

    def test_multimodal_evaluate_method(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(
                    role="user",
                    content=f"What's shown in this image? {image}'",
                ),
                Turn(
                    role="assistant",
                    content=f"That's an image of a car",
                    retrieval_context=[f"Cars are great to look at {image}"],
                ),
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )
        metric = TurnContextualRecallMetric()

        results = evaluate([convo_test_case], [metric])

        assert results is not None


class TestTurnContextualRecallMetricUnit:
    """Unit tests that don't require an API key."""

    def test_async_expected_outcome_forwarded_not_multimodal(self):
        """Regression: a_measure must pass expected_outcome (str) to
        _a_get_contextual_recall_scores, not multimodal (bool).

        Before the fix, test_case.multimodal (always False) was passed as
        the expected_outcome arg, causing _a_generate_verdicts to receive an
        empty string and score every turn as 0.
        """
        from deepeval.metrics.turn_contextual_recall.schema import (
            InteractionContextualRecallScore,
        )

        convo_test_case = ConversationalTestCase(
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
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant",
        )

        captured = {}

        async def spy_get_scores(window, expected_outcome, multimodal):
            captured["expected_outcome"] = expected_outcome
            captured["multimodal"] = multimodal
            return [InteractionContextualRecallScore(score=1.0, reason="ok", verdicts=[])]

        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "mock-model"

        base = "deepeval.metrics.turn_contextual_recall.turn_contextual_recall"
        with (
            patch(f"{base}.initialize_model", return_value=(mock_model, True)),
            patch(f"{base}.check_conversational_test_case_params"),
        ):
            metric = TurnContextualRecallMetric()
            with (
                patch.object(metric, "_a_get_contextual_recall_scores", side_effect=spy_get_scores),
                patch.object(metric, "_a_generate_reason", new=AsyncMock(return_value="ok")),
            ):
                asyncio.run(metric.a_measure(convo_test_case, _show_indicator=False))

        assert captured["expected_outcome"] == convo_test_case.expected_outcome, (
            f"expected_outcome should be '{convo_test_case.expected_outcome}', "
            f"got {captured['expected_outcome']!r}"
        )
        assert captured["multimodal"] == convo_test_case.multimodal
