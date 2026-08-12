import asyncio

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from deepeval.errors import DeepEvalError
from deepeval.models.llms.anthropic_model import AnthropicModel
from deepeval.config.settings import reset_settings, get_settings
from pydantic import BaseModel, SecretStr

from tests.test_core.stubs import _RecordingClient

########################################################
# Legacy keyword backwards compatibility behavior      #
########################################################


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_accepts_legacy_anthropic_api_key_keyword_and_uses_it(
    mock_require_dep,
    settings,
):
    """
    Using the legacy `_anthropic_api_key` keyword should:

    - Populate the canonical `api_key` (via SecretStr)
    - Result in the underlying client receiving the correct `api_key` value
    - Not forward `_anthropic_api_key` in model.kwargs
    """
    # Put ANTHROPIC_API_KEY into the process env so Settings sees it
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "env-secret-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Fake anthropic module returned by require_dependency
    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Construct AnthropicModel with the legacy key name
    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # The client should see a plain string API key coming from the legacy param
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"

    # And the legacy key should not be present in the model's kwargs
    assert "_anthropic_api_key" not in model.kwargs


##########################
# Test Secret Management #
##########################


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_explicit_key_over_settings_and_strips_secret(
    mock_require_dep,
    settings,
):
    """
    Added with fix for Issue: #2326
    """
    # Put ANTHROPIC_API_KEY into the process env so Settings sees it
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "env-secret-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Fake anthropic module returned by require_dependency
    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Construct AnthropicModel with an explicit key
    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # Before the fix for Issue #2326:
    #   api_key is the SecretStr from settings.ANTHROPIC_API_KEY, and this assertion FAILS.
    # After the fix:
    #   api_key is a plain str, equal to the explicit constructor key.
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_settings_key_when_no_explicit_key(
    mock_require_dep,
    settings,
):
    # Ensure env has a key
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "env-only-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    reset_settings(reload_dotenv=False)

    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Fake anthropic module returned by require_dependency
    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Stub Anthropic client to avoid real network and inspect kwargs
    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    client = model.model
    assert client.kwargs["api_key"] == "env-only-key"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_explicit_key_when_settings_missing(
    mock_require_dep,
    monkeypatch,
):
    # Make sure ANTHROPIC_API_KEY is not present
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    with settings.edit(persist=False):
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6
    assert settings.ANTHROPIC_API_KEY is None

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        api_key="explicit-key",
    )
    client = model.model
    assert client.kwargs["api_key"] == "explicit-key"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_no_key_configured(
    mock_require_dep,
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    with settings.edit(persist=False):
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    assert get_settings().ANTHROPIC_API_KEY is None

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Error should come from require_secret_api_key / DeepEvalError,
    # not from missing anthropic dependency.
    with pytest.raises(DeepEvalError, match="not configured"):
        AnthropicModel(model="claude-3-7-sonnet-latest")


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_explicit_key_empty(
    mock_require_dep,
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)

    settings = get_settings()
    with settings.edit(persist=False):
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(
            model="claude-3-7-sonnet-latest",
            api_key="",
        )


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_settings_key_empty(
    mock_require_dep,
    settings,
):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = ""
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6
    reset_settings(reload_dotenv=False)
    # pydantic will treat this as SecretStr(""), which is what we want to test
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)
    assert settings.ANTHROPIC_API_KEY.get_secret_value() == ""

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(model="claude-3-7-sonnet-latest")


##############################
# calculate_cost unit tests  #
##############################


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_calculate_cost_returns_correct_value(
    mock_require_dep, settings
):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 0.003
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 0.012

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    model.model_data.input_price = 0.003
    model.model_data.output_price = 0.012
    cost = model.calculate_cost(input_tokens=500, output_tokens=200)
    expected = 500 * 0.003 + 200 * 0.012
    assert cost == expected


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_calculate_cost_returns_none_when_prices_missing(
    mock_require_dep, settings
):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    model.model_data.input_price = None
    model.model_data.output_price = None

    cost = model.calculate_cost(input_tokens=500, output_tokens=200)
    assert cost is None


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_calculate_cost_with_zero_tokens(mock_require_dep, settings):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 0.003
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 0.012

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    cost = model.calculate_cost(input_tokens=0, output_tokens=0)
    assert cost == 0.0


########################################################
# max_tokens defaults and truncation reporting         #
########################################################


class _Verdict(BaseModel):
    reason: str


def _make_message(stop_reason, text):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=10, output_tokens=100),
    )


class _SyncMessagesClient(_RecordingClient):
    """Recording client whose messages.create returns a canned message."""

    response = _make_message("end_turn", '{"reason": "ok"}')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **create_kwargs):
        self.create_kwargs = create_kwargs
        return type(self).response


class _AsyncMessagesClient(_SyncMessagesClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async def _a_create(**create_kwargs):
            self.create_kwargs = create_kwargs
            return type(self).response

        self.messages = SimpleNamespace(create=_a_create)


def _fake_module(sync_cls, async_cls=None):
    return SimpleNamespace(
        Anthropic=sync_cls, AsyncAnthropic=async_cls or sync_cls
    )


def _configure_settings(settings):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_defaults_max_tokens_to_8192(
    mock_require_dep, settings
):
    """1024 starved thinking-by-default models (e.g. claude-opus-5), where
    max_tokens caps thinking + response text combined."""
    _configure_settings(settings)
    mock_require_dep.return_value = _fake_module(_RecordingClient)

    model = AnthropicModel(model="claude-opus-5")
    assert model._max_tokens == 8192


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_explicit_max_tokens_still_wins(
    mock_require_dep, settings
):
    _configure_settings(settings)
    mock_require_dep.return_value = _fake_module(_RecordingClient)

    model = AnthropicModel(model="claude-opus-5", max_tokens=512)
    assert model._max_tokens == 512

    model = AnthropicModel(
        model="claude-opus-5", generation_kwargs={"max_tokens": 2048}
    )
    assert model._max_tokens == 2048


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_raises_clear_error_when_truncated_with_schema(
    mock_require_dep, settings
):
    """A response cut off at max_tokens used to surface downstream as an
    invalid-JSON error; it should name max_tokens instead."""
    _configure_settings(settings)

    class _TruncatedClient(_SyncMessagesClient):
        response = _make_message("max_tokens", '{"reason": "truncat')

    mock_require_dep.return_value = _fake_module(_TruncatedClient)

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        model.generate("judge this", schema=_Verdict)


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_a_generate_raises_clear_error_when_truncated_with_schema(
    mock_require_dep, settings
):
    _configure_settings(settings)

    class _TruncatedAsyncClient(_AsyncMessagesClient):
        response = _make_message("max_tokens", '{"reason": "truncat')

    mock_require_dep.return_value = _fake_module(
        _RecordingClient, _TruncatedAsyncClient
    )

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        asyncio.run(model.a_generate("judge this", schema=_Verdict))


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_truncated_without_schema_returns_partial_text(
    mock_require_dep, settings
):
    """Plain-text callers may cap output deliberately; keep partial text."""
    _configure_settings(settings)

    class _TruncatedTextClient(_SyncMessagesClient):
        response = _make_message("max_tokens", "partial answer")

    mock_require_dep.return_value = _fake_module(_TruncatedTextClient)

    model = AnthropicModel(model="claude-opus-5")
    text, _ = model.generate("summarize")
    assert text == "partial answer"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_truncated_without_schema_and_no_text_raises(
    mock_require_dep, settings
):
    """All 8192 tokens spent on thinking: no text block at all."""
    _configure_settings(settings)

    class _EmptyTruncatedClient(_SyncMessagesClient):
        response = _make_message("max_tokens", None)

    mock_require_dep.return_value = _fake_module(_EmptyTruncatedClient)

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        model.generate("summarize")
