"""Tests for GPTModel generation_kwargs parameter"""

import uuid as _uuid
import time as _time
import deepeval.models.llms.openai_model as openai_mod

from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, SecretStr
from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.openai_model import GPTModel
from deepeval.tracing.patchers import patch_openai_client
from deepeval.tracing.types import LlmSpan, TraceSpanStatus
from deepeval.models.llms.constants import OPENAI_MODELS_DATA
from tests.test_core.stubs import _RecordingClient


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_llm_span() -> LlmSpan:
    return LlmSpan(
        uuid=str(_uuid.uuid4()),
        status=TraceSpanStatus.IN_PROGRESS,
        trace_uuid=str(_uuid.uuid4()),
        start_time=_time.time(),
    )


def _make_usage(**fields):
    return SimpleNamespace(**fields)


def _make_completion(usage, content="hello"):
    choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


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
        assert model.name == "gpt-4o"

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
        assert model.name == "gpt-5-mini"

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
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test prompt"}],
                }
            ],
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
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test prompt"}],
                }
            ],
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
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test prompt"}],
                }
            ],
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
            {
                "role": "user",
                "content": [{"type": "text", "text": "async test prompt"}],
            }
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
            {
                "role": "user",
                "content": [{"type": "text", "text": "async test prompt"}],
            }
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
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test prompt"}],
                }
            ],
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
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test prompt"}],
                }
            ],
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
        assert model.name == "gpt-4o"
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


########################################################
# Test legacy keyword backwards compatability behavior #
########################################################


def test_openai_model_accepts_legacy_model_keyword_and_maps_to_model(
    settings,
):
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model`
    - It should not be forwarded through `model.kwargs`
    """
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"

    model = GPTModel(model="gpt-4o")

    # legacy keyword mapped to canonical parameter
    assert model.name == "gpt-4o"

    # legacy key should not be forwarded to the client kwargs
    assert "model" not in model.kwargs


def test_openai_model_accepts_legacy_openai_api_key_keyword_and_uses_it(
    monkeypatch,
):
    """
    Using the legacy `_openai_api_key` keyword should:
    - Populate the canonical `api_key` (via SecretStr)
    - Result in the underlying client receiving the correct `api_key` value
    - Not forward `_openai_api_key` in model.kwargs
    """
    # Put OPENAI_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # Stub the OpenAI SDK clients so we don't make any real calls
    monkeypatch.setattr(openai_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        openai_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )

    # Construct GPTModel with the legacy key name
    model = GPTModel(
        model="gpt-4.1",
        api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # The client should see a plain string API key coming from the legacy param
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"

    # And the legacy key should not be present in the model's kwargs
    assert "_openai_api_key" not in model.kwargs


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
        api_key="constructor-key",
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
    assert model.name == "gpt-4o-mini"


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
        settings.OPENAI_MODEL_NAME = "model-not-yet-in-our-registry"  # <- A model not in our registry will not have pricing
        settings.OPENAI_COST_PER_INPUT_TOKEN = 0.123
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = 0.456

    model = GPTModel()  # Uses Settings.OPENAI_MODEL_NAME + Settings pricing
    assert model.name == "model-not-yet-in-our-registry"
    assert model.model_data.input_price == 0.123
    assert model.model_data.output_price == 0.456


#############################################################
# Tests for fix: token counts and cost for gpt-5.x in LLM  #
# spans (fixes #2531)                                        #
#############################################################


class TestGPTModelUpdateLlmSpanTokenFields:
    """
    Unit-tests for GPTModel._update_llm_span_from_completion.

    Verifies that both the classic (prompt_tokens/completion_tokens) and
    the newer Responses-API (input_tokens/output_tokens) field names are read
    correctly, and that cost fields are populated for known models like gpt-5.2.
    No real OpenAI calls are made — update_llm_span is patched at the module level.
    """

    @patch("deepeval.models.llms.openai_model.update_llm_span")
    @patch("deepeval.models.llms.openai_model.update_current_span")
    def test_classic_prompt_tokens_read(self, _mock_span, mock_llm, settings):
        """prompt_tokens / completion_tokens (classic chat-completions style) must be read."""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-4.1")
        completion = _make_completion(_make_usage(prompt_tokens=10, completion_tokens=20))
        model._update_llm_span_from_completion(completion)
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 10
        assert kw["output_token_count"] == 20

    @patch("deepeval.models.llms.openai_model.update_llm_span")
    @patch("deepeval.models.llms.openai_model.update_current_span")
    def test_new_input_tokens_read_for_gpt52(self, _mock_span, mock_llm, settings):
        """input_tokens / output_tokens (Responses API / gpt-5.x style) must be read."""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-5.2")
        completion = _make_completion(_make_usage(input_tokens=15, output_tokens=30))
        model._update_llm_span_from_completion(completion)
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 15
        assert kw["output_token_count"] == 30

    @patch("deepeval.models.llms.openai_model.update_llm_span")
    @patch("deepeval.models.llms.openai_model.update_current_span")
    def test_cost_fields_non_none_for_gpt52(self, _mock_span, mock_llm, settings):
        """cost_per_input_token and cost_per_output_token must be non-None for gpt-5.2."""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-5.2")
        completion = _make_completion(_make_usage(input_tokens=5, output_tokens=10))
        model._update_llm_span_from_completion(completion)
        kw = mock_llm.call_args.kwargs
        assert kw["cost_per_input_token"] is not None
        assert kw["cost_per_output_token"] is not None

    @patch("deepeval.models.llms.openai_model.update_llm_span")
    @patch("deepeval.models.llms.openai_model.update_current_span")
    def test_gpt41_no_regression(self, _mock_span, mock_llm, settings):
        """gpt-4.1 with classic field names must still produce correct counts and costs."""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-4.1")
        completion = _make_completion(_make_usage(prompt_tokens=7, completion_tokens=14))
        model._update_llm_span_from_completion(completion)
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 7
        assert kw["output_token_count"] == 14
        assert kw["cost_per_input_token"] is not None
        assert kw["cost_per_output_token"] is not None

    @patch("deepeval.models.llms.openai_model.update_llm_span")
    @patch("deepeval.models.llms.openai_model.update_current_span")
    def test_zero_prompt_tokens_not_overwritten_by_fallback(
        self, _mock_span, mock_llm, settings
    ):
        """prompt_tokens=0 must be preserved, not replaced by input_tokens fallback."""
        with settings.edit(persist=False):
            settings.OPENAI_API_KEY = "test-key"
        model = GPTModel(model="gpt-4.1")
        completion = _make_completion(
            _make_usage(prompt_tokens=0, completion_tokens=0, input_tokens=99, output_tokens=99)
        )
        model._update_llm_span_from_completion(completion)
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 0
        assert kw["output_token_count"] == 0


class TestPatchOpenaiClientTokenCounts:
    """
    Unit-tests for the patch_openai_client() patcher.

    Verifies that the wrapped chat.completions.create method reads both
    token-field naming conventions and populates cost fields from
    OPENAI_MODELS_DATA for all known models, including gpt-5.2.
    """

    def _make_fake_client(self, completion):
        chat_completions = SimpleNamespace(create=Mock(return_value=completion))
        chat = SimpleNamespace(completions=chat_completions)
        beta_completions = SimpleNamespace(parse=Mock(return_value=completion))
        beta_chat = SimpleNamespace(completions=beta_completions)
        return SimpleNamespace(chat=chat, beta=SimpleNamespace(chat=beta_chat))

    @patch("deepeval.tracing.patchers.update_llm_span")
    @patch("deepeval.tracing.patchers.update_current_span")
    @patch("deepeval.tracing.patchers.current_span_context")
    def test_new_field_names_in_patcher_for_gpt52(
        self, mock_ctx, _mock_span, mock_llm
    ):
        """input_tokens / output_tokens must be read by the patcher for gpt-5.2."""
        mock_ctx.get.return_value = _make_llm_span()
        completion = _make_completion(_make_usage(input_tokens=12, output_tokens=24))
        client = self._make_fake_client(completion)
        patch_openai_client(client)
        client.chat.completions.create(
            model="gpt-5.2", messages=[{"role": "user", "content": "hi"}]
        )
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 12
        assert kw["output_token_count"] == 24

    @patch("deepeval.tracing.patchers.update_llm_span")
    @patch("deepeval.tracing.patchers.update_current_span")
    @patch("deepeval.tracing.patchers.current_span_context")
    def test_cost_populated_in_patcher_for_gpt52(
        self, mock_ctx, _mock_span, mock_llm
    ):
        """patch_openai_client must populate cost fields from OPENAI_MODELS_DATA for gpt-5.2."""
        mock_ctx.get.return_value = _make_llm_span()
        completion = _make_completion(_make_usage(input_tokens=5, output_tokens=10))
        client = self._make_fake_client(completion)
        patch_openai_client(client)
        client.chat.completions.create(
            model="gpt-5.2", messages=[{"role": "user", "content": "hi"}]
        )
        kw = mock_llm.call_args.kwargs
        expected = OPENAI_MODELS_DATA.get("gpt-5.2")
        assert kw["cost_per_input_token"] == expected.input_price
        assert kw["cost_per_output_token"] == expected.output_price

    @patch("deepeval.tracing.patchers.update_llm_span")
    @patch("deepeval.tracing.patchers.update_current_span")
    @patch("deepeval.tracing.patchers.current_span_context")
    def test_classic_field_names_no_regression_in_patcher(
        self, mock_ctx, _mock_span, mock_llm
    ):
        """gpt-4.1 with prompt_tokens/completion_tokens must still work via patcher."""
        mock_ctx.get.return_value = _make_llm_span()
        completion = _make_completion(_make_usage(prompt_tokens=8, completion_tokens=16))
        client = self._make_fake_client(completion)
        patch_openai_client(client)
        client.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}]
        )
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 8
        assert kw["output_token_count"] == 16
        assert kw["cost_per_input_token"] is not None
        assert kw["cost_per_output_token"] is not None

    @patch("deepeval.tracing.patchers.update_llm_span")
    @patch("deepeval.tracing.patchers.update_current_span")
    @patch("deepeval.tracing.patchers.current_span_context")
    def test_patcher_unknown_model_does_not_crash(
        self, mock_ctx, _mock_span, mock_llm
    ):
        """Unknown model must not crash -- cost fields should be None."""
        mock_ctx.get.return_value = _make_llm_span()
        completion = _make_completion(_make_usage(prompt_tokens=5, completion_tokens=10))
        client = self._make_fake_client(completion)
        patch_openai_client(client)
        client.chat.completions.create(
            model="ft:gpt-4o:my-org:custom:id", messages=[{"role": "user", "content": "hi"}]
        )
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 5
        assert kw["output_token_count"] == 10
        assert kw["cost_per_input_token"] is None
        assert kw["cost_per_output_token"] is None

    @patch("deepeval.tracing.patchers.update_llm_span")
    @patch("deepeval.tracing.patchers.update_current_span")
    @patch("deepeval.tracing.patchers.current_span_context")
    def test_patcher_zero_prompt_tokens_not_overwritten(
        self, mock_ctx, _mock_span, mock_llm
    ):
        """prompt_tokens=0 must be preserved, not replaced by input_tokens fallback."""
        mock_ctx.get.return_value = _make_llm_span()
        completion = _make_completion(
            _make_usage(prompt_tokens=0, completion_tokens=0, input_tokens=99, output_tokens=99)
        )
        client = self._make_fake_client(completion)
        patch_openai_client(client)
        client.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "hi"}]
        )
        kw = mock_llm.call_args.kwargs
        assert kw["input_token_count"] == 0
        assert kw["output_token_count"] == 0


##############################
# calculate_cost unit tests  #
##############################


def test_openai_calculate_cost_returns_correct_value(settings):
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"
        settings.OPENAI_COST_PER_INPUT_TOKEN = 0.005
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = 0.015

    model = GPTModel(model="model-not-in-registry")

    cost = model.calculate_cost(input_tokens=100, output_tokens=50)
    expected = 100 * 0.005 + 50 * 0.015
    assert cost == expected


def test_openai_calculate_cost_returns_none_when_prices_missing(settings):
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"

    model = GPTModel(model="model-not-in-registry")
    assert model.model_data.input_price is None
    assert model.model_data.output_price is None

    cost = model.calculate_cost(input_tokens=100, output_tokens=50)
    assert cost is None


def test_openai_calculate_cost_with_zero_tokens(settings):
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"
        settings.OPENAI_COST_PER_INPUT_TOKEN = 0.005
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = 0.015

    model = GPTModel(model="model-not-in-registry")

    cost = model.calculate_cost(input_tokens=0, output_tokens=0)
    assert cost == 0.0
