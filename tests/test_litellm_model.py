import pytest
import os
from unittest.mock import patch
from pydantic import BaseModel
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase


class TestSchema(BaseModel):
    score: float
    reason: str


def test_litellm_model_initialization():
    """Test LiteLLM model initialization with different parameters."""
    # Test with OpenAI model
    model = LiteLLMModel(model="gpt-3.5-turbo")
    assert model.model_name == "gpt-3.5-turbo"
    assert model.temperature == 0

    # Test with Anthropic model
    model = LiteLLMModel(model="claude-3-opus", temperature=0.7)
    assert model.model_name == "claude-3-opus"
    assert model.temperature == 0.7

    # Test with Google model
    model = LiteLLMModel(model="gemini-pro")
    assert model.model_name == "gemini-pro"

    # Test with Mistral model
    model = LiteLLMModel(model="mistral-large")
    assert model.model_name == "mistral-large"

    # Test with custom endpoint
    model = LiteLLMModel(
        model="custom-model",
        api_base="https://custom-endpoint.com",
        api_key="test-key",
    )
    assert model.model_name == "custom-model"
    assert model.api_base == "https://custom-endpoint.com"
    assert model.api_key == "test-key"

    # Test invalid temperature
    with pytest.raises(ValueError):
        LiteLLMModel(model="gpt-3.5-turbo", temperature=-1)


def test_litellm_model_env_vars():
    """Test LiteLLM model initialization with environment variables."""
    # Test OpenAI API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        model = LiteLLMModel(model="gpt-3.5-turbo")
        assert model.api_key == "test-openai-key"

    # Test Anthropic API key
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"}):
        model = LiteLLMModel(model="claude-3-opus")
        assert model.api_key == "test-anthropic-key"

    # Test Google API key
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}):
        model = LiteLLMModel(model="gemini-pro")
        assert model.api_key == "test-google-key"

    # Test custom API base
    with patch.dict(os.environ, {"LITELLM_API_BASE": "https://test-base.com"}):
        model = LiteLLMModel(model="custom-model")
        assert model.api_base == "https://test-base.com"


def test_litellm_model_generation():
    """Test text generation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test basic generation
    response, cost = model.generate("What is 2+2?")
    assert isinstance(response, str)
    assert isinstance(cost, float)
    assert cost >= 0

    # Test generation with schema
    response, cost = model.generate(
        "Rate the following answer on a scale of 0-1: 'The capital of France is Paris'",
        schema=TestSchema,
    )
    assert isinstance(response, TestSchema)
    assert 0 <= response.score <= 1
    assert isinstance(response.reason, str)
    assert isinstance(cost, float)
    assert cost >= 0


@pytest.mark.asyncio
async def test_litellm_model_async_generation():
    """Test async text generation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test basic async generation
    response, cost = await model.a_generate("What is 2+2?")
    assert isinstance(response, str)
    assert isinstance(cost, float)
    assert cost >= 0

    # Test async generation with schema
    response, cost = await model.a_generate(
        "Rate the following answer on a scale of 0-1: 'The capital of France is Paris'",
        schema=TestSchema,
    )
    assert isinstance(response, TestSchema)
    assert 0 <= response.score <= 1
    assert isinstance(response.reason, str)
    assert isinstance(cost, float)
    assert cost >= 0


def test_litellm_model_error_handling():
    """Test error handling with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test invalid API key
    with pytest.raises(Exception):
        model = LiteLLMModel(model="gpt-3.5-turbo", api_key="invalid-key")
        model.generate("Test")

    # Test invalid model name
    with pytest.raises(Exception):
        model = LiteLLMModel(model="invalid-model")
        model.generate("Test")

    # Test invalid API base
    with pytest.raises(Exception):
        model = LiteLLMModel(
            model="gpt-3.5-turbo", api_base="https://invalid-base.com"
        )
        model.generate("Test")


def test_litellm_model_schema_validation():
    """Test schema validation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test invalid JSON response
    with pytest.raises(Exception):
        model.generate("Return invalid JSON", schema=TestSchema)

    # Test schema validation failure
    with pytest.raises(Exception):
        model.generate("Return a score greater than 1", schema=TestSchema)


def test_litellm_model_raw_response():
    """Test raw response generation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test raw response generation
    response, cost = model.generate_raw_response("What is 2+2?", top_logprobs=5)
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(cost, float)
    assert cost >= 0


@pytest.mark.asyncio
async def test_litellm_model_async_raw_response():
    """Test async raw response generation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test async raw response generation
    response, cost = await model.a_generate_raw_response(
        "What is 2+2?", top_logprobs=5
    )
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], "message")
    assert hasattr(response.choices[0].message, "content")
    assert isinstance(cost, float)
    assert cost >= 0


def test_litellm_model_samples():
    """Test sample generation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test sample generation
    samples, cost = model.generate_samples(
        "Give me a number between 1 and 10", n=3, temperature=0.7
    )
    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(isinstance(sample, str) for sample in samples)
    assert isinstance(cost, float)
    assert cost >= 0


def test_litellm_model_cost_calculation():
    """Test cost calculation with LiteLLM model."""
    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Test cost calculation
    response, cost = model.generate("What is 2+2?")
    assert isinstance(cost, float)
    assert cost >= 0

    # Test cost calculation with longer prompt
    response, cost = model.generate("Write a short story about a robot.")
    assert isinstance(cost, float)
    assert cost >= 0


def test_litellm_model_with_metrics():
    """Test LiteLLM model with DeepEval metrics."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    model = LiteLLMModel(model="gpt-3.5-turbo")

    # Create a test case
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        context="Geography question about France",
    )

    # Create a metric
    metric = GEval(
        name="test-eval",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
        threshold=0.5,
    )

    # Test metric evaluation
    score = metric.measure(test_case)
    assert isinstance(score, float)
    assert 0 <= score <= 1
