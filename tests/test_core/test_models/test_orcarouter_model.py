import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from deepeval.constants import ProviderSlug
from deepeval.models.llms.constants import DEFAULT_ORCAROUTER_MODEL
from deepeval.models.llms.orcarouter_model import OrcaRouterModel


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""

    field1: str
    field2: int


class TestOrcaRouterModel:
    """Test suite for OrcaRouterModel functionality"""

    def test_init_without_generation_kwargs(self, settings):
        """Test that OrcaRouterModel initializes correctly without generation_kwargs"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
        assert model.generation_kwargs == {}
        assert model.name == "openai/gpt-4o-mini"
        assert model.base_url == "https://api.orcarouter.ai/v1"

    def test_init_with_generation_kwargs(self, settings):
        """Test that OrcaRouterModel initializes correctly with generation_kwargs"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        generation_kwargs = {
            "max_tokens": 1000,
            "top_p": 0.9,
        }
        model = OrcaRouterModel(
            model="openai/gpt-4o-mini", generation_kwargs=generation_kwargs
        )
        assert model.generation_kwargs == generation_kwargs

    def test_init_with_custom_pricing(self, settings):
        """Test that user-provided pricing is stored correctly"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(
            model="openai/gpt-4o-mini",
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
        )
        assert model.cost_per_input_token == 0.0001
        assert model.cost_per_output_token == 0.0002

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
    def test_generate_with_generation_kwargs(
        self, mock_async_openai_class, settings
    ):
        """Test that generation_kwargs are passed to generate method"""
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(
            model="openai/gpt-4o-mini",
            generation_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )

        output, cost = model.generate("test prompt")

        assert call_args["model"] == "openai/gpt-4o-mini"
        assert call_args["messages"] == [
            {"role": "user", "content": "test prompt"}
        ]
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 1000
        assert call_args["top_p"] == 0.9
        assert output == "test response"

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
    def test_generate_forwards_extra_body_routing_preferences(
        self, mock_async_openai_class, settings
    ):
        """`extra_body` is how OrcaRouter routing preferences reach the gateway.

        It must survive as a nested dict rather than being flattened into the
        top-level request body, otherwise the gateway never sees the routing
        hint.
        """
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="routed"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20

        call_args = {}

        async def async_create(*args, **kwargs):
            call_args.update(kwargs)
            return mock_completion

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        extra_body = {
            "models": ["openai/gpt-4o-mini", "anthropic/claude-opus-4.8"],
            "route": "fallback",
        }
        model = OrcaRouterModel(
            model="orcarouter/auto",
            generation_kwargs={"extra_body": extra_body},
        )

        output, cost = model.generate("test prompt")

        assert call_args["model"] == "orcarouter/auto"
        assert call_args["extra_body"] == extra_body
        assert output == "routed"

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
        output, cost = await model.a_generate("async prompt")

        assert output == "async response"

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
    def test_generate_with_structured_outputs(
        self, mock_async_openai_class, settings
    ):
        """Test structured outputs with OrcaRouter's JSON Schema format"""
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
        output, cost = model.generate("test prompt", schema=SampleSchema)

        assert "response_format" in call_args
        response_format = call_args["response_format"]
        assert response_format["type"] == "json_schema"
        assert "json_schema" in response_format
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "SampleSchema"

        assert isinstance(output, SampleSchema)
        assert output.field1 == "test"
        assert output.field2 == 42

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
    def test_generate_with_structured_outputs_fallback(
        self, mock_async_openai_class, settings
    ):
        """Test that structured outputs fall back to JSON parsing if native format fails"""
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client

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
                raise Exception("Structured outputs not supported")
            return mock_completion_fallback

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output, cost = model.generate("test prompt", schema=SampleSchema)

            assert len(w) == 1
            assert "Structured outputs not supported" in str(w[0].message)

        assert isinstance(output, SampleSchema)
        assert output.field1 == "fallback"
        assert output.field2 == 99

    def test_calculate_cost_with_user_pricing(self, settings):
        """Test cost calculation with user-provided pricing"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")

        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.cost = 0.015

        cost = model.calculate_cost(
            input_tokens=100, output_tokens=50, response=mock_response
        )
        assert cost == 0.015

    def test_calculate_cost_when_cost_is_unknown_returns_none(self, settings):
        """Test cost calculation falls back to None if no pricing available"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")

        cost = model.calculate_cost(input_tokens=100, output_tokens=50)
        assert cost is None

    @patch("deepeval.models.llms.gateway_model.OpenAI")
    def test_client_kwargs_includes_custom_headers(
        self, mock_openai_class, settings
    ):
        """Test that custom headers passed via kwargs are included in client kwargs"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(
            model="openai/gpt-4o-mini",
            default_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "My App",
            },
        )

        _ = model.load_model(async_mode=False)

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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel()
        assert model.name == DEFAULT_ORCAROUTER_MODEL

    def test_dynamic_model_name(self, settings):
        """Test that any model string is accepted (dynamic model support)"""
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "orcarouter/auto",
        ]

        for model_name in models:
            model = OrcaRouterModel(model=model_name)
            assert model.name == model_name

    def test_model_name_from_settings_when_not_passed(self, settings):
        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"
            settings.ORCAROUTER_MODEL_NAME = "anthropic/claude-opus-4.8"

        model = OrcaRouterModel()
        assert model.name == "anthropic/claude-opus-4.8"

    @patch("deepeval.models.llms.gateway_model.OpenAI")
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
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

    @patch("deepeval.models.llms.gateway_model.OpenAI")
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
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
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
            settings.ORCAROUTER_API_KEY = "test-key"
            settings.ORCAROUTER_BASE_URL = (
                "https://proxy.example.com/orcarouter"
            )

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
        assert model.base_url == "https://proxy.example.com/orcarouter"

    def test_init_rejects_negative_temperature(self, settings):
        from deepeval.errors import DeepEvalError

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        with pytest.raises(DeepEvalError):
            OrcaRouterModel(model="openai/gpt-4o-mini", temperature=-0.1)

    def test_is_recognized_as_native_model(self, settings):
        """OrcaRouterModel must be flagged as a native deepeval model so that
        `initialize_model()` returns the same instance with using_native=True
        and metrics don't silently wrap it in a non-native adapter."""
        from deepeval.metrics.utils import initialize_model, is_native_model

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        model = OrcaRouterModel(model="openai/gpt-4o-mini")
        assert is_native_model(model)

        returned_model, using_native = initialize_model(model)
        assert using_native is True
        assert returned_model is model

    def test_provider_slug_has_a_retry_policy(self, settings):
        """Every provider slug must resolve to a retry policy, otherwise the
        centralized retry decorator silently degrades to no retries."""
        from deepeval.models.retry_policy import get_retry_policy_for

        assert OrcaRouterModel.PROVIDER_SLUG is ProviderSlug.ORCAROUTER
        assert get_retry_policy_for(ProviderSlug.ORCAROUTER.value) is not None

    def test_use_flag_selects_orcarouter(self, settings, monkeypatch):
        """Selecting OrcaRouter must route bare model strings through
        OrcaRouter rather than falling through to the `GPTModel` default at the
        end of `initialize_model`.

        The flag is set on `deepeval.metrics.utils.SETTINGS` rather than on the
        `settings` fixture because that module binds its own snapshot at import
        time (`SETTINGS = get_settings()`), while the fixture's sandbox resets
        the singleton to a new object. Every `should_use_*_model()` helper reads
        the snapshot, so that is the object this branch actually consults.
        """
        from deepeval.metrics import utils as metrics_utils

        with settings.edit(persist=False):
            settings.ORCAROUTER_API_KEY = "test-key"

        monkeypatch.setattr(
            metrics_utils.SETTINGS, "USE_ORCAROUTER_MODEL", True
        )

        model, using_native = metrics_utils.initialize_model(
            "anthropic/claude-opus-4.8"
        )
        assert isinstance(model, OrcaRouterModel)
        assert using_native is True
        assert model.name == "anthropic/claude-opus-4.8"

    def test_use_flag_accepts_yes_no_strings(self, settings):
        """Env vars arrive as strings; the USE_* flag must be coerced the same
        way the other provider switches are."""
        with settings.edit(persist=False):
            settings.USE_ORCAROUTER_MODEL = "YES"
        assert settings.USE_ORCAROUTER_MODEL is True

        with settings.edit(persist=False):
            settings.USE_ORCAROUTER_MODEL = "NO"
        assert settings.USE_ORCAROUTER_MODEL is False
