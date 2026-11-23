from unittest.mock import Mock, MagicMock, patch

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models.llms.constants import DEFAULT_ATLASCLOUD_MODEL
from deepeval.models.llms.atlascloud_model import AtlasCloudModel


class TestAtlasCloudModel:
    """Test suite for AtlasCloudModel functionality.

    AtlasCloudModel extends ``DeepEvalOpenAICompatibleModel`` exactly like
    ``OpenRouterModel`` does, so the shared transport, structured-output and
    cost-accounting logic is already covered by
    ``tests/test_core/test_models/test_openrouter_model.py``. These tests only
    exercise what this subclass actually owns: settings resolution, the
    provider-specific default/base-url, and the ``initialize_model()`` /
    ``is_native_model()`` wiring in ``deepeval/metrics/utils.py``.
    """

    def test_default_model(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        model = AtlasCloudModel()
        assert model.name == DEFAULT_ATLASCLOUD_MODEL
        assert model.base_url == "https://api.atlascloud.ai/v1"

    def test_dynamic_model_name(self, settings):
        """Atlas Cloud model ids are `vendor/model`, unvalidated, same as OpenRouter."""
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        for model_name in (
            "openai/gpt-4.1-mini",
            "deepseek-ai/deepseek-v3.2",
            "zai-org/GLM-4.6",
        ):
            model = AtlasCloudModel(model=model_name)
            assert model.name == model_name

    def test_init_with_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        generation_kwargs = {"max_tokens": 1000, "top_p": 0.9}
        model = AtlasCloudModel(
            model="openai/gpt-4.1-mini", generation_kwargs=generation_kwargs
        )
        assert model.generation_kwargs == generation_kwargs

    def test_init_with_custom_pricing(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        model = AtlasCloudModel(
            model="openai/gpt-4.1-mini",
            cost_per_input_token=0.0004,
            cost_per_output_token=0.0016,
        )
        assert model.cost_per_input_token == 0.0004
        assert model.cost_per_output_token == 0.0016

    def test_base_url_uses_settings_when_not_passed(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"
            settings.ATLASCLOUD_BASE_URL = (
                "https://proxy.example.com/atlascloud"
            )

        model = AtlasCloudModel(model="openai/gpt-4.1-mini")
        assert model.base_url == "https://proxy.example.com/atlascloud"

    def test_explicit_base_url_overrides_settings(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"
            settings.ATLASCLOUD_BASE_URL = (
                "https://proxy.example.com/atlascloud"
            )

        model = AtlasCloudModel(
            model="openai/gpt-4.1-mini", base_url="https://override.example.com"
        )
        assert model.base_url == "https://override.example.com"

    def test_init_rejects_negative_temperature(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        with pytest.raises(DeepEvalError):
            AtlasCloudModel(model="openai/gpt-4.1-mini", temperature=-0.1)

    def test_calculate_cost_with_user_pricing(self, settings):
        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        model = AtlasCloudModel(
            model="openai/gpt-4.1-mini",
            cost_per_input_token=0.0004,
            cost_per_output_token=0.0016,
        )
        cost = model.calculate_cost(input_tokens=1000, output_tokens=500)
        assert cost == (1000 * 0.0004) + (500 * 0.0016)

    def test_is_recognized_as_native_model(self, settings):
        """AtlasCloudModel must be flagged as a native deepeval model so that
        `initialize_model()` returns the same instance with using_native=True
        and metrics don't silently wrap it in a non-native adapter."""
        from deepeval.metrics.utils import initialize_model, is_native_model

        with settings.edit(persist=False):
            settings.ATLASCLOUD_API_KEY = "test-key"

        model = AtlasCloudModel(model="openai/gpt-4.1-mini")
        assert is_native_model(model)

        returned_model, using_native = initialize_model(model)
        assert using_native is True
        assert returned_model is model

    @patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
    def test_generate_with_generation_kwargs(
        self, mock_async_openai_class, settings
    ):
        """End-to-end through this class specifically: config resolved here
        (model id, temperature) reaches the actual `chat.completions.create`
        call, not just the base class in isolation."""
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
            settings.ATLASCLOUD_API_KEY = "test-key"

        model = AtlasCloudModel(
            model="openai/gpt-4.1-mini",
            generation_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )

        output, _cost = model.generate("test prompt")

        assert call_args["model"] == "openai/gpt-4.1-mini"
        assert call_args["messages"] == [
            {"role": "user", "content": "test prompt"}
        ]
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 1000
        assert call_args["top_p"] == 0.9
        assert output == "test response"
