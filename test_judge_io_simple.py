"""Simple test to verify judge I/O capture works."""

from unittest.mock import Mock
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel


class MockSchema(BaseModel):
    reason: str
    score: float


class MockMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = 0.5
        self.model = Mock(spec=DeepEvalBaseLLM)
        self.using_native_model = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        from deepeval.metrics.utils import generate_with_schema_and_extract

        prompt = "This is a test prompt"
        mock_response = MockSchema(reason="Test reason", score=0.8)

        # Mock the model to return our mock response
        self.model.generate_with_schema = Mock(return_value=mock_response)

        result = generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MockSchema,
            extract_schema=lambda s: (s.reason, s.score),
            extract_json=lambda d: (d["reason"], d["score"]),
        )

        self.score = result[1]
        self.reason = result[0]
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError


def test_sync_judge_io_capture():
    """Test that sync judge calls capture prompts and responses."""
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")

    metric.measure(test_case, _show_indicator=False)

    # Verify judge_prompts and judge_responses are captured
    assert metric.judge_prompts is not None, "judge_prompts should not be None"
    assert len(metric.judge_prompts) == 1, f"Expected 1 prompt, got {len(metric.judge_prompts)}"
    assert metric.judge_prompts[0] == "This is a test prompt"

    assert metric.judge_responses is not None, "judge_responses should not be None"
    assert len(metric.judge_responses) == 1, f"Expected 1 response, got {len(metric.judge_responses)}"
    # The response should be stringified
    assert "Test reason" in metric.judge_responses[0]
    assert "0.8" in metric.judge_responses[0]
    
    print("✓ Sync judge I/O capture test passed")


def test_multiple_judge_calls():
    """Test that multiple judge calls are all captured."""
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")

    # Call measure twice to simulate multiple judge calls
    metric.measure(test_case, _show_indicator=False)
    metric.measure(test_case, _show_indicator=False)

    # Verify both calls are captured
    assert metric.judge_prompts is not None
    assert len(metric.judge_prompts) == 2, f"Expected 2 prompts, got {len(metric.judge_prompts)}"
    assert metric.judge_prompts[0] == "This is a test prompt"
    assert metric.judge_prompts[1] == "This is a test prompt"

    assert metric.judge_responses is not None
    assert len(metric.judge_responses) == 2, f"Expected 2 responses, got {len(metric.judge_responses)}"
    
    print("✓ Multiple judge calls test passed")


def test_metric_data_serialization():
    """Test that judge_prompts and judge_responses are included in MetricData."""
    from deepeval.evaluate.utils import create_metric_data
    
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")
    
    metric.measure(test_case, _show_indicator=False)
    
    # Create MetricData from the metric
    metric_data = create_metric_data(metric)
    
    # Verify the fields are present
    assert hasattr(metric_data, 'judge_prompts'), "MetricData should have judge_prompts field"
    assert hasattr(metric_data, 'judge_responses'), "MetricData should have judge_responses field"
    
    # Verify the values are correct
    assert metric_data.judge_prompts == metric.judge_prompts
    assert metric_data.judge_responses == metric.judge_responses
    
    print("✓ MetricData serialization test passed")


if __name__ == "__main__":
    test_sync_judge_io_capture()
    test_multiple_judge_calls()
    test_metric_data_serialization()
    print("\n✅ All tests passed!")
