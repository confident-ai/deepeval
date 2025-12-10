import os
import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, MLLMImage, ToolCall

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
BANANA = os.path.join(current_dir, "./images/Banana.jpg")


class TestAnswerRelevancyMetric:
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
        metric = AnswerRelevancyMetric(async_mode=False)
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
        metric = AnswerRelevancyMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=BANANA, local=True)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a banana",
            actual_output=f"That is a banana.",
            retrieval_context=[f"Bananas are good for health {image}"],
            context=[f"Bananas are good for health {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = AnswerRelevancyMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=BANANA, local=True)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a banana",
            actual_output=f"That is a banana.",
            retrieval_context=[f"Bananas are good for health {image}"],
            context=[f"Bananas are good for health {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        metric = AnswerRelevancyMetric(async_mode=False)
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=BANANA, local=True)
        test_case = LLMTestCase(
            input=f"What's shown in this image? {image}'",
            expected_output=f"That's an image of a banana",
            actual_output=f"That is a banana.",
            retrieval_context=[f"Bananas are good for health {image}"],
            context=[f"Bananas are good for health {image}"],
            tools_called=[
                ToolCall(name="ImageAnalysis"),
                ToolCall(name="ToolQuery"),
            ],
            expected_tools=[ToolCall(name="ImageAnalysis")],
        )
        with pytest.raises(ValueError):
            metric = AnswerRelevancyMetric(
                async_mode=False, model="gpt-3.5-turbo"
            )
            metric.measure(test_case)
