import os
from unittest.mock import MagicMock, patch
import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.metrics.hallucination.schema import HallucinationVerdict
from deepeval.test_case import LLMTestCase, MLLMImage, ToolCall
from deepeval import evaluate

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestHallucinationMetric:
    """Tests for answer relevancy metric"""

    def test_normal_sync_metric_measure(self):
        test_case = LLMTestCase(
            input="What if these shoes don't fit?",
            expected_output="We offer a 30-day full refund at no extra cost.",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = HallucinationMetric(async_mode=False)
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

    def test_normal_async_metric_measure(self):
        test_case = LLMTestCase(
            input="What if these shoes don't fit?",
            expected_output="We offer a 30-day full refund at no extra cost.",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = HallucinationMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=CAR)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a car",
            actual_output=f"That is a car.",
            retrieval_context=[f"Cars are great to look at {image}"],
            context=[f"Cars are great to look at {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = HallucinationMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=CAR)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a car",
            actual_output=f"That is a car.",
            retrieval_context=[f"Cars are great to look at {image}"],
            context=[f"Cars are great to look at {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = HallucinationMetric(async_mode=False)
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=CAR)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a car",
            actual_output=f"That is a car.",
            retrieval_context=[f"Cars are great to look at {image}"],
            context=[f"Cars are great to look at {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        with pytest.raises(ValueError):
            metric = HallucinationMetric(
                async_mode=False, model="gpt-3.5-turbo"
            )
            metric.measure(test_case)

    def test_normal_evaluate_method(self):
        test_case = LLMTestCase(
            input="What if these shoes don't fit?",
            expected_output="We offer a 30-day full refund at no extra cost.",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )

        metric = HallucinationMetric()

        results = evaluate([test_case], [metric])

        assert results is not None

    def test_multimodal_evaluate_method(self):
        image = MLLMImage(url=CAR)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a car",
            actual_output=f"That is a car.",
            retrieval_context=[f"Cars are great to look at {image}"],
            context=[f"Cars are great to look at {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )

        metric = HallucinationMetric()

        results = evaluate([test_case], [metric])

        assert results is not None


class TestHallucinationPenalizeAmbiguousClaims:
    """No-API-key unit tests for penalize_ambiguous_claims flag."""

    def _make_metric(self, penalize: bool) -> HallucinationMetric:
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "mock"
        with patch(
            "deepeval.metrics.hallucination.hallucination.initialize_model",
            return_value=(mock_model, True),
        ):
            return HallucinationMetric(penalize_ambiguous_claims=penalize)

    def test_idk_not_penalized_by_default(self):
        metric = self._make_metric(penalize=False)
        metric.verdicts = [
            HallucinationVerdict(verdict="yes", reason="aligned"),
            HallucinationVerdict(verdict="idk", reason="uncertain"),
        ]
        score = metric._calculate_score()
        assert score == 0.0

    def test_idk_penalized_when_flag_set(self):
        metric = self._make_metric(penalize=True)
        metric.verdicts = [
            HallucinationVerdict(verdict="yes", reason="aligned"),
            HallucinationVerdict(verdict="idk", reason="uncertain"),
        ]
        score = metric._calculate_score()
        assert score == 0.5

    def test_no_verdict_still_penalized(self):
        metric = self._make_metric(penalize=True)
        metric.verdicts = [
            HallucinationVerdict(verdict="no", reason="contradicts"),
            HallucinationVerdict(verdict="idk", reason="uncertain"),
        ]
        score = metric._calculate_score()
        assert score == 1.0
