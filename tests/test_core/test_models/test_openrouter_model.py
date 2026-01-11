"""Tests for OpenRouterModel"""

from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
import pytest
import warnings

from deepeval.models.llms.openrouter_model import OpenRouterModel


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""

    field1: str
    field2: int


class TestOpenRouterModel:
    """Test suite for OpenRouterModel functionality"""

    def test_init_without_generation_kwargs(self, settings):
        """Test that OpenRouterModel initializes correctly without generation_kwargs"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        assert model.generation_kwargs == {}
        assert model.name == "openai/gpt-4o-mini"
        assert model.base_url == "https://openrouter.ai/api/v1"

    def test_init_with_generation_kwargs(self, settings):
        """Test that OpenRouterModel initializes correctly with generation_kwargs"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        generation_kwargs = {
            "max_tokens": 1000,
            "top_p": 0.9,
        }
        model = OpenRouterModel(
            model="openai/gpt-4o-mini", generation_kwargs=generation_kwargs
        )
        assert model.generation_kwargs == generation_kwargs

    def test_init_with_custom_pricing(self, settings):
        """Test that user-provided pricing is stored correctly"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
        )
        assert model.cost_per_input_token == 0.0001
        assert model.cost_per_output_token == 0.0002

    @patch("deepeval.models.llms.openrouter_model.AsyncOpenAI")
    def test_generate_with_generation_kwargs(
        self, mock_async_openai_class, settings
    ):
        """Test that generation_kwargs are passed to generate method"""
        # Setup mock
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20

        call_args = {}

        async def async_create(*args, **kwargs):
            call_args.update(kwargs)
            return mock_completion

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            generation_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )

        # Call generate
        output, cost = model.generate("test prompt")

        # Verify the completion was called with generation_kwargs
        assert call_args["model"] == "openai/gpt-4o-mini"
        assert call_args["messages"] == [
            {"role": "user", "content": "test prompt"}
        ]
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 1000
        assert call_args["top_p"] == 0.9
        assert output == "test response"

    @patch("deepeval.models.llms.openrouter_model.AsyncOpenAI")
    async def test_async_generate(self, mock_async_openai_class, settings):
        """Test async generation"""
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="async response"))]
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 25

        async def async_create(*args, **kwargs):
            return mock_completion

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        output, cost = await model.a_generate("async prompt")

        assert output == "async response"

    @patch("deepeval.models.llms.openrouter_model.AsyncOpenAI")
    def test_generate_with_structured_outputs(
        self, mock_async_openai_class, settings
    ):
        """Test structured outputs with OpenRouter's JSON Schema format"""
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
        # OpenRouter returns JSON string in message.content
        mock_completion.choices = [
            Mock(message=Mock(content='{"field1": "test", "field2": 42}'))
        ]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20

        call_args = {}

        async def async_create(*args, **kwargs):
            call_args.update(kwargs)
            return mock_completion

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        output, cost = model.generate("test prompt", schema=SampleSchema)

        # Verify response_format was set correctly
        assert "response_format" in call_args
        response_format = call_args["response_format"]
        assert response_format["type"] == "json_schema"
        assert "json_schema" in response_format
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "SampleSchema"

        # Verify output is validated against schema
        assert isinstance(output, SampleSchema)
        assert output.field1 == "test"
        assert output.field2 == 42

    @patch("deepeval.models.llms.openrouter_model.AsyncOpenAI")
    def test_generate_with_structured_outputs_fallback(
        self, mock_async_openai_class, settings
    ):
        """Test that structured outputs fall back to JSON parsing if native format fails"""
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client

        # First call (structured output) raises error
        # Second call (fallback) succeeds
        mock_completion_fallback = Mock()
        mock_completion_fallback.choices = [
            Mock(message=Mock(content='{"field1": "fallback", "field2": 99}'))
        ]
        mock_completion_fallback.usage.prompt_tokens = 10
        mock_completion_fallback.usage.completion_tokens = 20

        call_count = {"count": 0}

        async def async_create(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # First call fails (structured output not supported)
                raise Exception("Structured outputs not supported")
            # Second call succeeds (fallback)
            return mock_completion_fallback

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")

        # Should warn and fall back
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output, cost = model.generate("test prompt", schema=SampleSchema)

            # Verify warning was issued
            assert len(w) == 1
            assert "Structured outputs not supported" in str(w[0].message)

        # Verify fallback worked
        assert isinstance(output, SampleSchema)
        assert output.field1 == "fallback"
        assert output.field2 == 99

    def test_calculate_cost_with_user_pricing(self, settings):
        """Test cost calculation with user-provided pricing"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
        )

        cost = model.calculate_cost(input_tokens=100, output_tokens=50)
        expected_cost = (100 * 0.0001) + (50 * 0.0002)
        assert cost == expected_cost

    def test_calculate_cost_with_response_pricing(self, settings):
        """Test cost calculation extracting from API response"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")

        # Mock response with cost
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cost = 0.015

        cost = model.calculate_cost(
            input_tokens=100, output_tokens=50, response=mock_response
        )
        assert cost == 0.015

    def test_calculate_cost_fallback_to_zero(self, settings):
        """Test cost calculation falls back to 0 if no pricing available"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")

        # No pricing provided, no cost in response
        cost = model.calculate_cost(input_tokens=100, output_tokens=50)
        assert cost == 0.0

    @patch("deepeval.models.llms.openrouter_model.OpenAI")
    def test_client_kwargs_includes_custom_headers(
        self, mock_openai_class, settings
    ):
        """Test that custom headers passed via kwargs are included in client kwargs"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(
            model="openai/gpt-4o-mini",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "My App",
            },
        )

        _ = model.load_model(async_mode=False)

        # Verify client was called with headers
        call_kwargs = mock_openai_class.call_args[1]
        assert "default_headers" in call_kwargs
        assert (
            call_kwargs["default_headers"]["HTTP-Referer"]
            == "https://example.com"
        )
        assert call_kwargs["default_headers"]["X-Title"] == "My App"

    def test_default_model(self, settings):
        """Test that default model is used when none provided"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel()
        assert model.name == "openai/gpt-4o-mini"

    def test_dynamic_model_name(self, settings):
        """Test that any model string is accepted (dynamic model support)"""
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        # Test various model formats
        models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-opus",
            "meta-llama/llama-3.1-70b-instruct",
            "custom/provider-model",
        ]

        for model_name in models:
            model = OpenRouterModel(model=model_name)
            assert model.name == model_name

    @patch("deepeval.models.llms.openrouter_model.OpenAI")
    def test_generate_raw_response(self, mock_openai_class, settings):
        """Test generate_raw_response method"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="raw response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        completion, cost = model.generate_raw_response(
            "test prompt", top_logprobs=3
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
            logprobs=True,
            top_logprobs=3,
        )
        assert completion == mock_completion

    @patch("deepeval.models.llms.openrouter_model.OpenAI")
    def test_generate_samples(self, mock_openai_class, settings):
        """Test generate_samples method"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="sample1")),
            Mock(message=Mock(content="sample2")),
        ]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 40
        mock_client.chat.completions.create.return_value = mock_response

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        samples, cost = model.generate_samples(
            "test prompt", n=2, temperature=0.7
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "test prompt"}],
            n=2,
            temperature=0.7,
        )
        assert samples == ["sample1", "sample2"]

    def test_base_url_uses_settings_when_not_passed(self, settings):
        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"
            settings.OPENROUTER_BASE_URL = (
                "https://proxy.example.com/openrouter"
            )

        model = OpenRouterModel(model="openai/gpt-4o-mini")
        assert model.base_url == "https://proxy.example.com/openrouter"

    def test_init_rejects_negative_temperature(self, settings):
        from deepeval.errors import DeepEvalError

        with settings.edit(persist=False):
            settings.OPENROUTER_API_KEY = "test-key"

        with pytest.raises(DeepEvalError):
            OpenRouterModel(model="openai/gpt-4o-mini", temperature=-0.1)
