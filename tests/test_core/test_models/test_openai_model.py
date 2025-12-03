import types
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, SecretStr

import deepeval.models.llms.openai_model as openai_mod
from deepeval.models.llms.openai_model import (
    GPTModel,
    valid_gpt_models,
    structured_outputs_models,
    json_mode_models,
)
from deepeval.config.settings import get_settings, reset_settings
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


class TinySchema(BaseModel):
    foo: int


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _ParsedMessage:
    def __init__(self, parsed):
        self.parsed = parsed  # for structured-output


class _ContentMessage:
    def __init__(self, content: str):
        self.content = content  # for json-mode


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, message, usage: _Usage):
        self.choices = [_Choice(message)]
        self.usage = usage


def _make_model(model_name: str) -> GPTModel:
    # Avoid pricing lookup errors for models missing in model_pricing
    return GPTModel(
        model=model_name,
        _openai_api_key="test",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    )


def _make_async_fake_client(mode: str, called: dict):
    """
    mode: "structured" or "json"
    called: dict to track which path was used, e.g. {"parse": 0, "create": 0}
    """

    # async create() should never be called in either async test
    class AsyncFakeChatCompletions:
        @staticmethod
        async def create(**kwargs):
            called["create"] += 1
            raise AssertionError(
                "create() should not be called in async path tests"
            )

    # async parse() for structured-output
    class AsyncStructuredParse:
        @staticmethod
        async def parse(**kwargs):
            called["parse"] += 1
            # Must pass schema as response_format
            assert (
                "response_format" in kwargs
            ), "response_format not provided to parse()"
            assert (
                kwargs["response_format"] is TinySchema
            ), "Expected schema class as response_format"
            msg = _ParsedMessage(TinySchema(foo=42))
            return _Response(msg, _Usage(prompt_tokens=10, completion_tokens=5))

    # async parse() for json-mode
    class AsyncJsonParse:
        @staticmethod
        async def parse(**kwargs):
            called["parse"] += 1
            assert kwargs.get("response_format") == {
                "type": "json_object"
            }, "Expected JSON mode response_format"
            msg = _ContentMessage(json.dumps({"foo": 7}))
            return _Response(msg, _Usage(prompt_tokens=8, completion_tokens=3))

    if mode == "structured":
        fake_beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=AsyncStructuredParse)
        )
    elif mode == "json":
        fake_beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=AsyncJsonParse)
        )
    else:
        raise ValueError("mode must be 'structured' or 'json'")

    fake_client = types.SimpleNamespace(
        beta=fake_beta,
        chat=types.SimpleNamespace(completions=AsyncFakeChatCompletions),
    )
    return fake_client


####################
# Capability flags #
####################


# Structured outputs: every model we list should return True
@pytest.mark.parametrize("model_name", structured_outputs_models)
def test_capabilities_structured_outputs_list(model_name):
    m = _make_model(model_name)
    assert m.supports_structured_outputs() is True


# JSON mode: every model we list should return True
@pytest.mark.parametrize("model_name", json_mode_models)
def test_capabilities_json_mode_list(model_name):
    m = _make_model(model_name)
    assert m.supports_json_mode() is True


# Negative checks: models not in structured list should be False
@pytest.mark.parametrize(
    "model_name",
    [m for m in valid_gpt_models if m not in structured_outputs_models],
)
def test_capabilities_structured_outputs_negative(model_name):
    m = _make_model(model_name)
    assert m.supports_structured_outputs() is False


# Negative checks: models NOT in json-mode list should be False
@pytest.mark.parametrize(
    "model_name",
    [m for m in valid_gpt_models if m not in json_mode_models],
)
def test_capabilities_json_mode_negative(model_name):
    m = _make_model(model_name)
    assert m.supports_json_mode() is False


######################
# Structured outputs #
######################


def test_structured_outputs_path_returns_pydantic(monkeypatch):
    called = {"parse": 0, "create": 0}

    class FakeChatCompletions:
        @staticmethod
        def create(**kwargs):
            called["create"] += 1
            # If this is ever called, it means we didn’t take the structured-output path.
            raise AssertionError(
                "create() should not be called for structured outputs path"
            )

    class FakeBetaCompletions:
        @staticmethod
        def parse(**kwargs):
            called["parse"] += 1
            # Assert we passed the schema as response_format
            assert (
                "response_format" in kwargs
            ), "response_format not provided to parse()"
            assert (
                kwargs["response_format"] is TinySchema
            ), "Expected schema class as response_format"
            msg = _ParsedMessage(TinySchema(foo=42))
            return _Response(msg, _Usage(prompt_tokens=10, completion_tokens=5))

    class FakeBeta:
        chat = types.SimpleNamespace(completions=FakeBetaCompletions)

    fake_client = types.SimpleNamespace(
        beta=FakeBeta,
        chat=types.SimpleNamespace(completions=FakeChatCompletions),
    )

    monkeypatch.setattr(
        GPTModel, "load_model", lambda self, async_mode=False: fake_client
    )

    model = GPTModel(model="gpt-4o", _openai_api_key="ignored")
    result, cost = model.generate("any prompt", schema=TinySchema)

    # Core behavior: Pydantic instance returned via .parse
    assert isinstance(result, TinySchema)
    assert result.foo == 42

    # Prove the intended path was taken
    assert called["parse"] == 1
    assert called["create"] == 0


@pytest.mark.asyncio
async def test_async_structured_outputs_path_returns_pydantic(monkeypatch):
    called = {"parse": 0, "create": 0}
    fake_client = _make_async_fake_client("structured", called)

    monkeypatch.setattr(
        GPTModel, "load_model", lambda self, async_mode=True: fake_client
    )

    model = GPTModel(model="gpt-4o", _openai_api_key="ignored")
    result, cost = await model.a_generate("any prompt", schema=TinySchema)

    assert isinstance(result, TinySchema)
    assert result.foo == 42

    # prove the intended path was taken
    assert called["parse"] == 1
    assert called["create"] == 0


#############
# JSON mode #
#############


def test_json_mode_path_returns_pydantic(monkeypatch):
    called = {"parse": 0, "create": 0}

    class FakeChatCompletions:
        @staticmethod
        def create(**kwargs):
            called["create"] += 1
            raise AssertionError("create() should not be called for JSON-mode")

    class FakeBetaCompletions:
        @staticmethod
        def parse(**kwargs):
            called["parse"] += 1
            assert kwargs.get("response_format") == {
                "type": "json_object"
            }, "Expected JSON mode response_format"
            msg = _ContentMessage(json.dumps({"foo": 7}))
            return _Response(msg, _Usage(prompt_tokens=8, completion_tokens=3))

    class FakeBeta:
        chat = types.SimpleNamespace(completions=FakeBetaCompletions)

    fake_client = types.SimpleNamespace(
        beta=FakeBeta,
        chat=types.SimpleNamespace(completions=FakeChatCompletions),
    )

    monkeypatch.setattr(
        GPTModel, "load_model", lambda self, async_mode=False: fake_client
    )

    model = GPTModel(
        model="gpt-4-turbo",
        _openai_api_key="ignored",
    )

    result, _ = model.generate("any prompt", schema=TinySchema)

    assert isinstance(result, TinySchema)
    assert result.foo == 7

    assert called["parse"] == 1
    assert called["create"] == 0


@pytest.mark.asyncio
async def test_async_json_mode_path_returns_pydantic(monkeypatch):
    called = {"parse": 0, "create": 0}
    fake_client = _make_async_fake_client("json", called)

    monkeypatch.setattr(
        GPTModel, "load_model", lambda self, async_mode=True: fake_client
    )

    model = GPTModel(model="gpt-4-turbo", _openai_api_key="ignored")
    result, cost = await model.a_generate("any prompt", schema=TinySchema)

    assert isinstance(result, TinySchema)
    assert result.foo == 7

    assert called["parse"] == 1
    assert called["create"] == 0
