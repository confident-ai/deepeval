"""Integration tests for OpenRouterModel (requires OPENROUTER_API_KEY)"""

import os
import pytest
from pydantic import BaseModel
from deepeval.models.llms.openrouter_model import OpenRouterModel


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""
    name: str
    age: int


@pytest.mark.skipif(
    os.getenv("OPENROUTER_API_KEY") is None
    or not os.getenv("OPENROUTER_API_KEY").strip(),
    reason="OPENROUTER_API_KEY is not set",
)
class TestOpenRouterModelIntegration:
    """Integration tests that make real API calls"""

    def test_basic_generation(self):
        """Test basic text generation"""
        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
        )
        output, cost = model.generate("Say hello in one word.")
        assert isinstance(output, str)
        assert len(output) > 0
        assert cost > 0

    @pytest.mark.asyncio
    async def test_async_generation(self):
        """Test async text generation"""
        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
        )
        output, cost = await model.a_generate("Say hello in one word.")
        assert isinstance(output, str)
        assert len(output) > 0
        assert cost > 0

    def test_structured_outputs(self):
        """Test structured outputs with JSON Schema"""
        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
        )
        output, cost = model.generate(
            "Return a JSON object with name='Alice' and age=30",
            schema=SampleSchema
        )
        assert isinstance(output, SampleSchema)
        assert output.name == "Alice"
        assert output.age == 30
        assert cost > 0

    def test_different_models(self):
        """Test that different OpenRouter models work"""
        models_to_test = [
            "openai/gpt-4o-mini",
            # "anthropic/claude-sonnnet-4.5"
            # Add more models as needed
        ]

        for model_name in models_to_test:
            model = OpenRouterModel(model=model_name)
            output, cost = model.generate("Say 'test'")
            assert isinstance(output, str)
            assert len(output) > 0