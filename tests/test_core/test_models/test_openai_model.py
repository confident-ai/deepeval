"""Tests for GPTModel generation_kwargs parameter"""

import deepeval.models.llms.openai_model as openai_mod

from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, SecretStr
from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.openai_model import GPTModel
from tests.test_core.stubs import _RecordingClient


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""

    field1: str
    field2: int


class TestGPTModelCompletionKwargs:
    """Test suite for GPTModel generation_kwargs functionality"""

    def test_init_without_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(model="gpt-4o")
        assert model.generation_kwargs == {}
        assert model.model_name == "gpt-4o"

    def test_init_with_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        generation_kwargs = {
            "reasoning_effort": "high",
            "max_tokens": 2000,
            "seed": 42,
        }
        model = GPTModel(
            model="gpt-5-mini", generation_kwargs=generation_kwargs
        )
        assert model.generation_kwargs == generation_kwargs
        assert model.model_name == "gpt-5-mini"

    def test_init_with_both_client_and_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        generation_kwargs = {"reasoning_effort": "medium"}
        model = GPTModel(
            model="gpt-4o",
            timeout=30,  # client kwarg
            max_retries=5,  # client kwarg
            generation_kwargs=generation_kwargs,
        )
        assert model.generation_kwargs == generation_kwargs
        assert model.kwargs == {"timeout": 30, "max_retries": 5}

    @patch("deepeval.models.llms.openai_model.OpenAI")
    def test_generate_with_generation_kwargs(self, mock_openai_class, settings):
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(
            model="gpt-5",
            generation_kwargs={"reasoning_effort": "high", "seed": 123},
        )

        # Call generate
        output, cost = model.generate("test prompt")

        # Verify the completion was called with generation_kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-5",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=1,  # GPT-5 auto-sets to 1
            reasoning_effort="high",
            seed=123,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.openai_model.OpenAI")
    def test_generate_without_generation_kwargs(
        self, mock_openai_class, settings
    ):
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(model="gpt-4o")

        # Call generate without generation_kwargs
        output, cost = model.generate("test prompt")

        # Verify the completion was called without extra kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.openai_model.OpenAI")
    def test_generate_with_schema_and_generation_kwargs(
        self, mock_openai_class, settings
    ):
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_beta = Mock()
        mock_client.beta = mock_beta

        # Create a mock parsed response
        mock_parsed = SampleSchema(field1="test", field2=42)
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(parsed=mock_parsed))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_beta.chat.completions.parse.return_value = mock_completion

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(
            model="gpt-4o",  # Supports structured output
            generation_kwargs={"reasoning_effort": "low", "top_p": 0.9},
        )

        # Call generate with schema
        output, cost = model.generate("test prompt", SampleSchema)

        # Verify the parse method was called with generation_kwargs
        mock_beta.chat.completions.parse.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}],
            response_format=SampleSchema,
            temperature=0,
            reasoning_effort="low",
            top_p=0.9,
        )
        assert output == mock_parsed

    @patch("deepeval.models.llms.openai_model.AsyncOpenAI")
    async def test_async_generate_with_generation_kwargs(
        self, mock_async_openai_class, settings
    ):
        # Setup mock
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(message=Mock(content="async test response"))
        ]
        mock_completion.usage.prompt_tokens = 15
        mock_completion.usage.completion_tokens = 25

        # Create a mock that tracks the call arguments
        call_args = {}

        async def async_create(*args, **kwargs):
            call_args.update(kwargs)
            return mock_completion

        mock_client.chat.completions.create = async_create

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(
            model="gpt-5-nano",
            generation_kwargs={
                "reasoning_effort": "medium",
                "max_tokens": 1500,
            },
        )

        # Call async generate
        output, cost = await model.a_generate("async test prompt")

        # Verify the output
        assert output == "async test response"

        # Verify the completion was called with the correct parameters
        assert call_args["model"] == "gpt-5-nano"
        assert call_args["messages"] == [
            {"role": "user", "content": "async test prompt"}
        ]
        assert call_args["temperature"] == 1  # GPT-5-nano auto-sets to 1
        assert call_args["reasoning_effort"] == "medium"
        assert call_args["max_tokens"] == 1500

    @patch("deepeval.models.llms.openai_model.AsyncOpenAI")
    async def test_async_generate_with_schema_and_generation_kwargs(
        self, mock_async_openai_class, settings
    ):
        # Setup mock
        mock_client = MagicMock()
        mock_async_openai_class.return_value = mock_client
        mock_beta = MagicMock()
        mock_client.beta = mock_beta

        # Create a mock parsed response
        mock_parsed = SampleSchema(field1="async test", field2=99)
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(parsed=mock_parsed))]
        mock_completion.usage.prompt_tokens = 20
        mock_completion.usage.completion_tokens = 30

        # Track call arguments
        call_args = {}

        async def async_parse(*args, **kwargs):
            call_args.update(kwargs)
            return mock_completion

        mock_beta.chat.completions.parse = async_parse

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(
            model="gpt-4o",  # Supports structured output
            generation_kwargs={"reasoning_effort": "high", "seed": 42},
        )

        # Call async generate with schema
        output, cost = await model.a_generate("async test prompt", SampleSchema)

        # Verify the output
        assert output == mock_parsed

        # Verify the parse method was called with correct parameters
        assert call_args["model"] == "gpt-4o"
        assert call_args["messages"] == [
            {"role": "user", "content": "async test prompt"}
        ]
        assert call_args["response_format"] == SampleSchema
        assert call_args["temperature"] == 0
        assert call_args["reasoning_effort"] == "high"
        assert call_args["seed"] == 42

    @patch("deepeval.models.llms.openai_model.OpenAI")
    def test_generate_raw_response_with_generation_kwargs(
        self, mock_openai_class, settings
    ):
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        model = GPTModel(
            model="gpt-4o",
            generation_kwargs={
                "reasoning_effort": "high",
                "presence_penalty": 0.5,
            },
        )

        # Call generate_raw_response
        completion, cost = model.generate_raw_response(
            "test prompt", top_logprobs=3
        )

        # Verify the completion was called with both method params and generation_kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
            logprobs=True,
            top_logprobs=3,
            reasoning_effort="high",
            presence_penalty=0.5,
        )
        assert completion == mock_completion

    @patch("deepeval.models.llms.openai_model.OpenAI")
    def test_generate_samples_with_generation_kwargs(
        self, mock_openai_class, settings
    ):
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="sample1")),
            Mock(message=Mock(content="sample2")),
        ]
        mock_client.chat.completions.create.return_value = mock_response

        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(
            model="gpt-4o", generation_kwargs={"reasoning_effort": "low"}
        )

        # Call generate_samples
        samples = model.generate_samples("test prompt", n=2, temperature=0.7)

        # Verify the completion was called with generation_kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}],
            n=2,
            temperature=0.7,
            reasoning_effort="low",
        )
        assert samples == ["sample1", "sample2"]

    def test_backwards_compatibility(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        # This should work exactly as before
        model = GPTModel(
            model="gpt-4o", temperature=0.5, timeout=30  # client kwarg
        )
        assert model.model_name == "gpt-4o"
        assert model.temperature == 0.5
        assert model.kwargs == {"timeout": 30}
        assert model.generation_kwargs == {}

    def test_gpt5_auto_temperature_adjustment(self, settings):
        """Test that GPT-5 models automatically adjust temperature to 1"""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"

        # Test various GPT-5 models
        gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

        for model_name in gpt5_models:
            model = GPTModel(
                model=model_name,
                temperature=0,  # Should be auto-adjusted to 1
                generation_kwargs={"reasoning_effort": "high"},
            )
            assert (
                model.temperature == 1
            ), f"Temperature should be 1 for {model_name}"
            assert model.generation_kwargs == {"reasoning_effort": "high"}

    def test_empty_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-4o", generation_kwargs={})
        assert model.generation_kwargs == {}

    def test_none_generation_kwargs(self, settings):
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-4o", generation_kwargs=None)
        assert model.generation_kwargs == {}


##########################
# Test Secret Management #
##########################


def test_openai_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    # Put OPENAI_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # Stub the OpenAI SDK clients so we don't make any real calls
    monkeypatch.setattr(openai_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        openai_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )

    # Construct GPTModel with an explicit key
    model = GPTModel(
        model="gpt-4.1",
        _openai_api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    assert isinstance(api_key, str)
    assert api_key == "constructor-key"


##########################################
# Tests for Settings-based model/pricing #
##########################################


def test_openai_model_defaults_model_from_settings_when_no_ctor_model(settings):
    """
    When no `model` is provided, GPTModel should fall back to
    Settings.OPENAI_MODEL_NAME (instead of the legacy key file).
    """
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"
        settings.OPENAI_MODEL_NAME = "gpt-4o-mini"

    model = GPTModel()
    assert model.model_name == "gpt-4o-mini"


def test_openai_model_costs_defaults_from_settings_for_missing_pricing(
    settings,
):
    """
    When a model is missing from `model_pricing`, GPTModel should populate
    pricing from Settings.OPENAI_COST_PER_INPUT_TOKEN and
    Settings.OPENAI_COST_PER_OUTPUT_TOKEN instead of the legacy key file.
    """
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"
        settings.OPENAI_MODEL_NAME = "gpt-5-chat-latest"
        settings.OPENAI_COST_PER_INPUT_TOKEN = 0.123
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = 0.456

    # Ensure this model has no pricing so GPTModel must use Settings-based costs
    openai_mod.model_pricing.pop("gpt-5-chat-latest", None)

    model = GPTModel()  # Uses Settings.OPENAI_MODEL_NAME + Settings pricing
    assert model.model_name == "gpt-5-chat-latest"

    pricing = openai_mod.model_pricing["gpt-5-chat-latest"]
    assert pricing["input"] == 0.123
    assert pricing["output"] == 0.456
