import os
import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, MLLMImage, ToolCall
from deepeval import evaluate
from deepeval.errors import DeepEvalError

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


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
        metric = AnswerRelevancyMetric()
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
        metric = AnswerRelevancyMetric(async_mode=False)
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
        with pytest.raises(DeepEvalError):
            metric = AnswerRelevancyMetric(async_mode=False, model="gpt-4.1")
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

        metric = AnswerRelevancyMetric()

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

        metric = AnswerRelevancyMetric()

        results = evaluate([test_case], [metric])

        assert results is not None
