"""Every judge whose registry entry or request handling changed, driven through
the real ``generate`` / ``a_generate`` path against a fake provider client.

Expected prices are typed from the providers' pricing pages rather than read
from the registry, so a registry edit that drifts from the published price
fails here.
"""

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from deepeval.models.llms.anthropic_model import AnthropicModel
from deepeval.models.llms.gemini_model import GeminiModel
from deepeval.models.llms.openai_model import OpenAIModel
from tests.test_core.stubs import _make_fake_genai_module


class _Verdict(BaseModel):
    verdict: str
    reason: str


class _Verdicts(BaseModel):
    verdicts: List[_Verdict]


_PARSED = _Verdicts(verdicts=[_Verdict(verdict="yes", reason="r")])
_JSON = _PARSED.model_dump_json()

INPUT_TOKENS = 1_000
OUTPUT_TOKENS = 200
THINKING_TOKENS = 3_000

sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


async def _call(model, is_async, prompt="prompt", schema=_Verdicts):
    if is_async:
        return await model.a_generate(prompt, schema)
    return model.generate(prompt, schema)


def _expected_cost(input_price, output_price, output_tokens=OUTPUT_TOKENS):
    return (INPUT_TOKENS * input_price + output_tokens * output_price) / 1e6


##########
# OpenAI #
##########

OPENAI_JUDGES = [
    ("gpt-5.6-sol", 4.00, 20.00),
    ("gpt-5.6-terra", 2.00, 12.00),
    ("gpt-5.6-luna", 0.20, 1.20),
    ("gpt-6-astra", 10.00, 50.00),
    ("gpt-6-sol", 2.00, 10.00),
    ("gpt-6-luna", 0.10, 0.50),
]


def _openai_client(record):
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=_PARSED))],
        usage=SimpleNamespace(
            prompt_tokens=INPUT_TOKENS, completion_tokens=OUTPUT_TOKENS
        ),
    )

    def parse(**kwargs):
        record.append(("parse", kwargs))
        return completion

    async def aparse(**kwargs):
        return parse(**kwargs)

    def create(**kwargs):
        record.append(("create", kwargs))
        return completion

    async def acreate(**kwargs):
        return create(**kwargs)

    sync = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    async_ = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=aparse))
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=acreate)),
    )
    return sync, async_


@sync_and_async
@pytest.mark.parametrize("model_name, input_price, output_price", OPENAI_JUDGES)
async def test_openai_judge_request(
    model_name, input_price, output_price, is_async, settings
):
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "test-key"
        settings.TEMPERATURE = None
        settings.OPENAI_COST_PER_INPUT_TOKEN = None
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = None
    record = []
    sync_client, async_client = _openai_client(record)

    with patch(
        "deepeval.models.llms.openai_model.OpenAI",
        lambda **kw: sync_client,
    ), patch(
        "deepeval.models.llms.openai_model.AsyncOpenAI",
        lambda **kw: async_client,
    ):
        model = OpenAIModel(model=model_name, temperature=0)
        output, cost = await _call(model, is_async)

    assert [kind for kind, _ in record] == ["parse"]
    kwargs = record[0][1]
    assert kwargs["model"] == model_name
    assert kwargs["response_format"] is _Verdicts
    assert "temperature" not in kwargs
    assert output == _PARSED
    assert cost == pytest.approx(_expected_cost(input_price, output_price))
    assert (cost.input_tokens, cost.output_tokens) == (
        INPUT_TOKENS,
        OUTPUT_TOKENS,
    )


##########
# Gemini #
##########

GEMINI_PRICED_JUDGES = [
    ("gemini-3.8-flash", 0.75, 3.75),
    ("gemini-3.7-flash", 0.75, 3.75),
    ("gemini-3.6-flash", 0.75, 3.75),
    ("gemini-3.5-flash", 1.50, 9.00),
    ("gemini-3.5-flash-lite", 0.30, 2.50),
    ("gemini-3.1-flash-lite", 0.25, 1.50),
    ("gemini-3.1-pro-preview", 2.00, 12.00),
    ("gemini-3-flash-preview", 0.50, 3.00),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
]


def _gemini_model(mock_require_dep, settings, model_name, response):
    genai = _make_fake_genai_module()
    genai.types.GenerateContentConfig = lambda **kwargs: kwargs
    mock_require_dep.return_value = genai

    client = MagicMock()
    client.models.generate_content.return_value = response

    async def agenerate(**kwargs):
        return client.models.generate_content(**kwargs)

    client.aio.models.generate_content = agenerate

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"
        settings.TEMPERATURE = None
        settings.GEMINI_COST_PER_INPUT_TOKEN = None
        settings.GEMINI_COST_PER_OUTPUT_TOKEN = None
    model = GeminiModel(model=model_name)
    model.load_model = lambda *a, **kw: client
    return model, client


@sync_and_async
@pytest.mark.parametrize(
    "model_name, input_price, output_price", GEMINI_PRICED_JUDGES
)
@patch("deepeval.models.llms.gemini_model.require_dependency")
async def test_gemini_judge_request(
    mock_require_dep, model_name, input_price, output_price, is_async, settings
):
    response = SimpleNamespace(
        parsed=_PARSED,
        usage_metadata=SimpleNamespace(
            prompt_token_count=INPUT_TOKENS,
            candidates_token_count=OUTPUT_TOKENS,
            thoughts_token_count=THINKING_TOKENS,
        ),
    )
    model, client = _gemini_model(
        mock_require_dep, settings, model_name, response
    )

    output, cost = await _call(model, is_async)

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config["response_schema"] is _Verdicts
    assert config["response_mime_type"] == "application/json"
    if model_name.startswith("gemini-3"):
        assert "temperature" not in config
        assert model.temperature == 1.0
    else:
        assert config["temperature"] == 0.0
    assert output == _PARSED
    billed_output = OUTPUT_TOKENS + THINKING_TOKENS
    assert cost == pytest.approx(
        _expected_cost(input_price, output_price, billed_output)
    )
    assert cost.output_tokens == billed_output


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_cost_without_thinking_tokens(mock_require_dep, settings):
    response = SimpleNamespace(
        parsed=_PARSED,
        usage_metadata=SimpleNamespace(
            prompt_token_count=INPUT_TOKENS,
            candidates_token_count=OUTPUT_TOKENS,
            thoughts_token_count=None,
        ),
    )
    model, _ = _gemini_model(
        mock_require_dep, settings, "gemini-3.8-flash", response
    )

    _, cost = model.generate("prompt", _Verdicts)

    assert cost == pytest.approx(_expected_cost(0.75, 3.75))
    assert cost.output_tokens == OUTPUT_TOKENS


#############
# Anthropic #
#############

anthropic = pytest.importorskip("anthropic")

ANTHROPIC_STRUCTURED_JUDGES = [
    ("claude-fable-5-1", 10.00, 50.00),
    ("claude-fable-5", 10.00, 50.00),
    ("claude-opus-5-5", 4.00, 20.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-sonnet-5", 2.00, 10.00),
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-opus-4-7", 5.00, 25.00),
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-opus-4-1-20250805", 15.00, 75.00),
]

ANTHROPIC_LEGACY_JUDGES = [
    ("claude-3-5-haiku", 0.80, 4.00, 8192),
    ("claude-3-7-sonnet-latest", 3.00, 15.00, 8192),
    ("claude-sonnet-4", 3.00, 15.00, 8192),
    ("claude-opus-4", 15.00, 75.00, 8192),
    ("claude-3-haiku", 0.25, 1.25, 4096),
    ("claude-3-opus", 15.00, 75.00, 4096),
]


class _Messages:
    def __init__(self, reject_output_config=False):
        self.calls = []
        self.reject_output_config = reject_output_config

    def _respond(self, kwargs):
        self.calls.append(kwargs)
        if self.reject_output_config and "output_config" in kwargs:
            raise _BadRequest("output_config not supported")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=_JSON)],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS
            ),
        )


class _BadRequest(Exception):
    pass


def _anthropic_model(mock_require_dep, settings, model_name, messages):
    async def acreate(**kwargs):
        return messages._respond(kwargs)

    sync_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: messages._respond(kw))
    )
    async_client = SimpleNamespace(messages=SimpleNamespace(create=acreate))
    mock_require_dep.return_value = SimpleNamespace(
        Anthropic=lambda **kw: sync_client,
        AsyncAnthropic=lambda **kw: async_client,
        BadRequestError=_BadRequest,
        transform_schema=anthropic.transform_schema,
    )
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.DEEPEVAL_MODEL_THINKING = None
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = None
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = None
    return AnthropicModel(model=model_name)


@sync_and_async
@pytest.mark.parametrize(
    "model_name, input_price, output_price", ANTHROPIC_STRUCTURED_JUDGES
)
@patch("deepeval.models.llms.anthropic_model.require_dependency")
async def test_anthropic_structured_judge_request(
    mock_require_dep, model_name, input_price, output_price, is_async, settings
):
    messages = _Messages()
    model = _anthropic_model(mock_require_dep, settings, model_name, messages)

    output, cost = await _call(model, is_async)

    (kwargs,) = messages.calls
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == anthropic.transform_schema(_Verdicts)
    assert fmt["schema"]["additionalProperties"] is False
    assert kwargs["max_tokens"] == 8192
    assert output == _PARSED
    assert cost == pytest.approx(_expected_cost(input_price, output_price))


@sync_and_async
@pytest.mark.parametrize(
    "model_name, input_price, output_price, max_tokens",
    ANTHROPIC_LEGACY_JUDGES,
)
@patch("deepeval.models.llms.anthropic_model.require_dependency")
async def test_anthropic_legacy_judge_request(
    mock_require_dep,
    model_name,
    input_price,
    output_price,
    max_tokens,
    is_async,
    settings,
):
    messages = _Messages()
    model = _anthropic_model(mock_require_dep, settings, model_name, messages)

    output, cost = await _call(model, is_async)

    (kwargs,) = messages.calls
    assert "output_config" not in kwargs
    assert kwargs["max_tokens"] == max_tokens
    assert output == _PARSED
    assert cost == pytest.approx(_expected_cost(input_price, output_price))


@sync_and_async
@patch("deepeval.models.llms.anthropic_model.require_dependency")
async def test_anthropic_rejected_schema_falls_back_once(
    mock_require_dep, is_async, settings
):
    messages = _Messages(reject_output_config=True)
    model = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5-5", messages
    )

    first, _ = await _call(model, is_async)
    second, _ = await _call(model, is_async)

    assert first == second == _PARSED
    assert ["output_config" in c for c in messages.calls] == [
        True,
        False,
        False,
    ]


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_other_bad_requests_still_raise_without_schema(
    mock_require_dep, settings
):
    messages = _Messages()
    model = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5-5", messages
    )

    def reject(kwargs):
        raise _BadRequest("prompt too long")

    messages._respond = reject

    with pytest.raises(_BadRequest):
        model.generate("prompt")


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_unrelated_bad_request_keeps_structured_outputs(
    mock_require_dep, settings
):
    messages = _Messages()
    model = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5-5", messages
    )

    def reject(kwargs):
        messages.calls.append(kwargs)
        raise _BadRequest("Your credit balance is too low")

    messages._respond = reject

    with pytest.raises(_BadRequest, match="credit balance"):
        model.generate("prompt", _Verdicts)
    assert len(messages.calls) == 1
    assert model._structured_outputs_rejected is False


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_merges_caller_output_config(mock_require_dep, settings):
    messages = _Messages()
    mock_model = _anthropic_model(
        mock_require_dep, settings, "claude-opus-5-5", messages
    )
    mock_model.generation_kwargs = {"output_config": {"effort": "low"}}

    mock_model.generate("prompt", _Verdicts)

    output_config = messages.calls[0]["output_config"]
    assert output_config["effort"] == "low"
    assert output_config["format"]["type"] == "json_schema"
