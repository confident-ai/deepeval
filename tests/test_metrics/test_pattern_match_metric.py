import os
import pytest
from deepeval.metrics import PatternMatchMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestPatternMatchMetric:
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
        metric = PatternMatchMetric(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert test_case.multimodal is False

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

        metric = PatternMatchMetric(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

        results = evaluate([test_case], [metric])

        assert results is not None
