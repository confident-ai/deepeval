"""Test that judge LLM prompts and responses are captured correctly."""

import pytest
from unittest.mock import Mock, AsyncMock
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
        from deepeval.metrics.utils import a_generate_with_schema_and_extract

        prompt = "This is an async test prompt"
        mock_response = MockSchema(reason="Async test reason", score=0.9)

        # Mock the model to return our mock response
        self.model.a_generate_with_schema = AsyncMock(return_value=mock_response)

        result = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=MockSchema,
            extract_schema=lambda s: (s.reason, s.score),
            extract_json=lambda d: (d["reason"], d["score"]),
        )

        self.score = result[1]
        self.reason = result[0]
        return self.score


def test_sync_judge_io_capture():
    """Test that sync judge calls capture prompts and responses."""
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")

    metric.measure(test_case, _show_indicator=False)

    # Verify judge_prompts and judge_responses are captured
    assert metric.judge_prompts is not None
    assert len(metric.judge_prompts) == 1
    assert metric.judge_prompts[0] == "This is a test prompt"

    assert metric.judge_responses is not None
    assert len(metric.judge_responses) == 1
    # The response should be stringified
    assert "Test reason" in metric.judge_responses[0]
    assert "0.8" in metric.judge_responses[0]


@pytest.mark.asyncio
async def test_async_judge_io_capture():
    """Test that async judge calls capture prompts and responses."""
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")

    await metric.a_measure(test_case, _show_indicator=False)

    # Verify judge_prompts and judge_responses are captured
    assert metric.judge_prompts is not None
    assert len(metric.judge_prompts) == 1
    assert metric.judge_prompts[0] == "This is an async test prompt"

    assert metric.judge_responses is not None
    assert len(metric.judge_responses) == 1
    # The response should be stringified
    assert "Async test reason" in metric.judge_responses[0]
    assert "0.9" in metric.judge_responses[0]


def test_multiple_judge_calls():
    """Test that multiple judge calls are all captured."""
    metric = MockMetric()
    test_case = LLMTestCase(input="test", actual_output="test output")

    # Call measure twice to simulate multiple judge calls
    metric.measure(test_case, _show_indicator=False)
    metric.measure(test_case, _show_indicator=False)

    # Verify both calls are captured
    assert metric.judge_prompts is not None
    assert len(metric.judge_prompts) == 2
    assert metric.judge_prompts[0] == "This is a test prompt"
    assert metric.judge_prompts[1] == "This is a test prompt"

    assert metric.judge_responses is not None
    assert len(metric.judge_responses) == 2


if __name__ == "__main__":
    # Run sync tests
    test_sync_judge_io_capture()
    print("✓ Sync judge I/O capture test passed")

    test_multiple_judge_calls()
    print("✓ Multiple judge calls test passed")

    # Run async test
    import asyncio

    asyncio.run(test_async_judge_io_capture())
    print("✓ Async judge I/O capture test passed")

    print("\n✅ All tests passed!")
