import pytest
from types import SimpleNamespace
from unittest.mock import patch

from deepeval.errors import DeepEvalError
from deepeval.models.llms.anthropic_model import (
    AnthropicModel,
    DEFAULT_MAX_TOKENS,
    DEFAULT_THINKING_MAX_TOKENS,
    MIN_THINKING_BUDGET_TOKENS,
)
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


###################################
# DEEPEVAL_MODEL_THINKING behavior #
###################################


class _Verdict(BaseModel):
    verdict: str


class _MessagesClient(_RecordingClient):
    """Records `messages.create` kwargs and replays canned content blocks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_kwargs = None
        self.blocks = [SimpleNamespace(type="text", text='{"verdict": "yes"}')]
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **create_kwargs):
        self.create_kwargs = create_kwargs
        return SimpleNamespace(
            content=self.blocks,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )


def _anthropic_model(mock_require_dep, settings, model, thinking=None):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6
        settings.DEEPEVAL_MODEL_THINKING = thinking

    client = _MessagesClient()
    mock_require_dep.return_value = SimpleNamespace(
        Anthropic=lambda *a, **kw: client,
        AsyncAnthropic=lambda *a, **kw: client,
    )
    return AnthropicModel(model=model), client


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_disables_thinking_by_default(mock_require_dep, settings):
    """Unset means off, so the judge's whole budget goes to the verdict."""
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5"
    )

    model.generate("prompt", schema=_Verdict)
    assert client.create_kwargs["thinking"] == {"type": "disabled"}


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_thinking_enabled_sends_budget(mock_require_dep, settings):
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5", thinking=True
    )

    model.generate("prompt", schema=_Verdict)
    thinking = client.create_kwargs["thinking"]
    assert thinking["type"] == "enabled"
    # Anthropic counts thinking against max_tokens, so the budget must leave
    # room for the response text.
    assert thinking["budget_tokens"] < client.create_kwargs["max_tokens"]
    assert thinking["budget_tokens"] >= MIN_THINKING_BUDGET_TOKENS


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_thinking_raises_when_budget_cannot_fit(
    mock_require_dep, settings
):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6
        settings.DEEPEVAL_MODEL_THINKING = True

    mock_require_dep.return_value = SimpleNamespace(
        Anthropic=_RecordingClient, AsyncAnthropic=_RecordingClient
    )

    with pytest.raises(DeepEvalError, match="max_tokens"):
        AnthropicModel(model="claude-opus-5", max_tokens=512)


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_omits_thinking_for_models_without_the_parameter(
    mock_require_dep, settings
):
    """Claude 3 rejects `thinking`, and claude-fable-5 always thinks."""
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-3-haiku", thinking=True
    )

    model.generate("prompt", schema=_Verdict)
    assert "thinking" not in client.create_kwargs


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_explicit_thinking_kwarg_wins(mock_require_dep, settings):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6

    client = _MessagesClient()
    mock_require_dep.return_value = SimpleNamespace(
        Anthropic=lambda *a, **kw: client,
        AsyncAnthropic=lambda *a, **kw: client,
    )
    caller_thinking = {"type": "enabled", "budget_tokens": 3000}
    model = AnthropicModel(
        model="claude-opus-5",
        generation_kwargs={"thinking": caller_thinking, "max_tokens": 6000},
    )

    model.generate("prompt", schema=_Verdict)
    assert client.create_kwargs["thinking"] == caller_thinking


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_thinking_enabled_raises_max_tokens_default(
    mock_require_dep, settings
):
    """Thinking needs headroom the 1024 default does not have."""
    thinking_model, _ = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5", thinking=True
    )
    plain_model, _ = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5"
    )

    assert plain_model._max_tokens == DEFAULT_MAX_TOKENS
    assert thinking_model._max_tokens == DEFAULT_THINKING_MAX_TOKENS


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_skips_leading_thinking_block(mock_require_dep, settings):
    """A thinking response puts its reasoning in content[0]."""
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5", thinking=True
    )
    client.blocks = [
        SimpleNamespace(type="thinking", thinking="reasoning..."),
        SimpleNamespace(type="text", text='{"verdict": "yes"}'),
    ]

    verdict, _ = model.generate("prompt", schema=_Verdict)
    assert verdict.verdict == "yes"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_raises_when_response_has_no_text_block(
    mock_require_dep, settings
):
    """All budget spent reasoning: no text to parse."""
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5", thinking=True
    )
    client.blocks = [SimpleNamespace(type="thinking", thinking="reasoning...")]

    with pytest.raises(DeepEvalError, match="no text block"):
        model.generate("prompt", schema=_Verdict)


@patch("deepeval.models.llms.anthropic_model.require_dependency")
async def test_anthropic_thinking_applies_to_a_generate(
    mock_require_dep, settings
):
    model, client = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5", thinking=True
    )

    async def _acreate(**create_kwargs):
        return client._create(**create_kwargs)

    client.messages = SimpleNamespace(create=_acreate)

    await model.a_generate("prompt", schema=_Verdict)
    assert client.create_kwargs["thinking"]["type"] == "enabled"
