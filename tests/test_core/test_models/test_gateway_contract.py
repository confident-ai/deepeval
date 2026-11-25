"""Contract of the gateway models as metrics and subclasses rely on it:
log probabilities with a clean fallback, cost, provider parameters, string
routing, and the hooks a subclass may override."""

import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from deepeval.errors import DeepEvalError, LogProbsUnsupportedError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import no_log_prob_support
from deepeval.metrics.utils import initialize_model, is_native_model
from deepeval.models import (
    DeepEvalBaseLLM,
    GatewayModel,
    LiteLLMModel,
    OpenAIModel,
)
from deepeval.models.llms.gateway_model import MAX_TOP_LOGPROBS
from deepeval.models.utils import EvaluationCost, infer_provider
from deepeval.test_case import LLMTestCase, SingleTurnParams


class BadRequestError(Exception):
    status_code = 400


class RateLimitError(Exception):
    status_code = 429


def _response(content="hello", logprobs=None, hidden_cost=None):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), logprobs=logprobs
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    if hidden_cost is not None:
        response._hidden_params = {"response_cost": hidden_cost}
    return response


def _score_logprobs(token="8"):
    candidates = [
        SimpleNamespace(token="8", logprob=-0.223),  # ~0.8
        SimpleNamespace(token="6", logprob=-1.609),  # ~0.2
    ]
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                token=token, logprob=-0.223, top_logprobs=candidates
            )
        ]
    )


@pytest.fixture
def litellm(monkeypatch):
    """A stand-in `litellm` module recording every completion call."""
    calls = []
    fake = types.SimpleNamespace(calls=calls, responses=[], errors=[])

    def completion(**params):
        calls.append(params)
        if fake.errors:
            error = fake.errors.pop(0)
            if error is not None:
                raise error
        return fake.responses.pop(0) if fake.responses else _response()

    async def acompletion(**params):
        return completion(**params)

    def get_llm_provider(model, api_base=None, **_):
        provider, _, name = model.partition("/")
        return (name or provider), (provider if name else "openai"), None, None

    fake.completion = completion
    fake.acompletion = acompletion
    fake.get_llm_provider = get_llm_provider
    fake.supported = {"openai": ["temperature", "logprobs"], "anthropic": []}
    fake.get_supported_openai_params = (
        lambda model, custom_llm_provider: fake.supported.get(
            custom_llm_provider
        )
    )
    fake.completion_cost = lambda completion_response: 0.5
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setitem(
        sys.modules,
        "litellm.types.router",
        types.SimpleNamespace(GenericLiteLLMParams=dict),
    )
    return fake


@pytest.fixture(autouse=True)
def _no_provider_flags(monkeypatch, settings):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "openai-key"
        settings.ANTHROPIC_API_KEY = "anthropic-key"
        settings.GOOGLE_API_KEY = "google-key"
        settings.LITELLM_API_KEY = None
        settings.LITELLM_PROXY_API_KEY = None
        settings.LITELLM_API_BASE = None
        settings.LITELLM_PROXY_API_BASE = None
        settings.TEMPERATURE = None
        settings.DEEPEVAL_LLM = None
        for flag in (
            "USE_OPENAI_MODEL",
            "USE_ANTHROPIC_MODEL",
            "USE_GEMINI_MODEL",
            "USE_LITELLM",
            "USE_PORTKEY_MODEL",
            "USE_LOCAL_MODEL",
            "USE_AZURE_OPENAI",
            "USE_MOONSHOT_MODEL",
            "USE_GROK_MODEL",
            "USE_DEEPSEEK_MODEL",
            "USE_OPENROUTER_MODEL",
            "USE_AWS_BEDROCK_MODEL",
        ):
            setattr(settings, flag, None)
        settings.LOCAL_MODEL_API_KEY = None
    monkeypatch.setattr("deepeval.metrics.utils.models.SETTINGS", settings)
    monkeypatch.setattr(
        "deepeval.metrics.utils.models.KEY_FILE_HANDLER.fetch_data",
        lambda *_a, **_k: None,
    )


##############################
# log probabilities          #
##############################


def test_base_llm_has_no_log_probs_by_default():
    class Custom(DeepEvalBaseLLM):
        def load_model(self):
            return None

        def generate(self, prompt):
            return "x"

        async def a_generate(self, prompt):
            return "x"

        def get_model_name(self):
            return "custom"

    with pytest.raises(AttributeError) as raised:
        Custom().generate_raw_response("p", top_logprobs=5)
    assert isinstance(raised.value, LogProbsUnsupportedError)
    # "Unknown" is still attempted, as before.
    assert no_log_prob_support(Custom()) is False


def test_raw_response_requests_capped_logprobs(litellm):
    model = GatewayModel("openai/gpt-x")
    response, cost = model.generate_raw_response("p", top_logprobs=50)
    assert litellm.calls[-1]["logprobs"] is True
    assert litellm.calls[-1]["top_logprobs"] == MAX_TOP_LOGPROBS
    assert response.choices[0].message.content == "hello"
    assert cost == 0.5


def test_provider_rejection_becomes_unsupported_and_is_remembered(litellm):
    model = GatewayModel("anthropic/claude-x")
    litellm.errors = [
        BadRequestError("anthropic does not support parameters: ['logprobs']")
    ]
    with pytest.raises(LogProbsUnsupportedError):
        model.generate_raw_response("p", top_logprobs=5)
    assert model.supports_log_probs() is False
    assert no_log_prob_support(model) is True

    # The second attempt does not reach the provider at all.
    calls = len(litellm.calls)
    with pytest.raises(LogProbsUnsupportedError):
        model.generate_raw_response("p", top_logprobs=5)
    assert len(litellm.calls) == calls


def test_other_failures_are_not_mistaken_for_missing_logprobs(litellm):
    model = GatewayModel("openai/gpt-x")
    litellm.errors = [RateLimitError("slow down")]
    with pytest.raises(RateLimitError):
        model.generate_raw_response("p", top_logprobs=5)
    assert model.supports_log_probs() is True


def test_supports_log_probs_can_be_forced(litellm):
    assert GatewayModel("openai/gpt-x").supports_log_probs() is True
    assert GatewayModel("anthropic/claude-x").supports_log_probs() is None
    off = GatewayModel("openai/gpt-x", supports_log_probs=False)
    with pytest.raises(LogProbsUnsupportedError):
        off.generate_raw_response("p")
    assert litellm.calls == []


@pytest.mark.asyncio
async def test_async_raw_response_matches_sync(litellm):
    model = GatewayModel("anthropic/claude-x")
    litellm.errors = [BadRequestError("logprobs is not supported")]
    with pytest.raises(LogProbsUnsupportedError):
        await model.a_generate_raw_response("p", top_logprobs=5)
    assert model.supports_log_probs() is False


def _g_eval(model):
    return GEval(
        name="Quality",
        evaluation_steps=["Judge the answer."],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=model,
        async_mode=False,
    )


def test_g_eval_weights_score_with_logprobs(litellm):
    model = GatewayModel("openai/gpt-x")
    litellm.responses = [
        _response('{"score": 8, "reason": "ok"}', logprobs=_score_logprobs())
    ]
    metric = _g_eval(model)
    metric.measure(LLMTestCase(input="q", actual_output="a"))
    # 0.8 * 8 + 0.2 * 6 = 7.6 on a 0-10 scale.
    assert metric.score == pytest.approx(0.76, abs=0.01)
    assert metric.evaluation_cost == pytest.approx(0.5)


def test_g_eval_falls_back_to_raw_score_without_logprobs(litellm):
    model = GatewayModel("anthropic/claude-x")
    litellm.errors = [BadRequestError("does not support parameters: logprobs")]
    litellm.responses = [_response('{"score": 8, "reason": "ok"}')]
    metric = _g_eval(model)
    metric.measure(LLMTestCase(input="q", actual_output="a"))
    assert metric.score == pytest.approx(0.8)
    assert "logprobs" not in litellm.calls[-1]


##############################
# cost                       #
##############################


def test_cost_precedence(litellm):
    priced = GatewayModel(
        "openai/gpt-x", cost_per_input_token=1.0, cost_per_output_token=2.0
    )
    assert priced._response_cost(_response(hidden_cost=9.0)) == 20.0

    reported = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.3)
    )
    model = GatewayModel("openai/gpt-x")
    assert model._response_cost(reported) == 0.3
    assert model._response_cost(_response(hidden_cost=0.2)) == 0.2

    estimated = model._response_cost(_response())
    assert estimated == 0.5
    assert isinstance(estimated, EvaluationCost)
    assert (estimated.input_tokens, estimated.output_tokens) == (10, 5)


def test_cost_is_none_when_it_cannot_be_priced(litellm):
    def unpriced(completion_response):
        raise ValueError("model not in price map")

    litellm.completion_cost = unpriced
    assert GatewayModel("openai/gpt-x")._response_cost(_response()) is None


##############################
# provider parameters        #
##############################


def test_provider_params_reach_every_call(litellm):
    model = GatewayModel(
        "azure/my-deployment",
        api_key="k",
        base_url="https://example.azure.com",
        api_version="2026-01-01",
        aws_region_name="us-east-1",
        vertex_project="p",
        extra_headers={"x": "y"},
        reasoning_effort="high",
        system_prompt="Be strict.",
    )
    model.generate("p")
    model.generate_raw_response("p", top_logprobs=5)
    model.generate_samples("p", n=2, temperature=1.0)
    assert len(litellm.calls) == 3
    for call in litellm.calls:
        assert call["model"] == "azure/my-deployment"
        assert call["api_key"] == "k"
        assert call["api_base"] == "https://example.azure.com"
        assert call["api_version"] == "2026-01-01"
        assert call["aws_region_name"] == "us-east-1"
        assert call["vertex_project"] == "p"
        assert call["extra_headers"] == {"x": "y"}
        assert call["reasoning_effort"] == "high"
        assert call["messages"][0] == {
            "role": "system",
            "content": "Be strict.",
        }


def test_nothing_is_injected_by_default(litellm):
    GatewayModel("anthropic/claude-x").generate("p")
    assert set(litellm.calls[-1]) == {"model", "messages", "api_key"}
    assert len(litellm.calls[-1]["messages"]) == 1


@pytest.mark.parametrize(
    "model, expected",
    [
        ("openai/gpt-x", "openai-key"),
        ("gpt-x", "openai-key"),
        ("anthropic/claude-x", "anthropic-key"),
        ("claude-x", "anthropic-key"),
        ("gemini/gemini-x", "google-key"),
        ("bedrock/some-model", None),
    ],
)
def test_only_the_called_providers_key_is_sent(litellm, model, expected):
    resolved = LiteLLMModel(model=model).api_key
    assert (resolved.get_secret_value() if resolved else None) == expected


def test_custom_endpoint_keeps_the_previous_key_order(litellm):
    model = LiteLLMModel(model="claude-x", base_url="http://proxy")
    assert model.api_key.get_secret_value() == "openai-key"


##############################
# structured output          #
##############################


class Verdict(BaseModel):
    verdict: str


def test_schema_refusal_falls_back_to_parsing_json(litellm):
    model = GatewayModel("anthropic/claude-x")
    litellm.errors = [BadRequestError("response_format is not supported")]
    litellm.responses = [_response('{"verdict": "yes"}')]
    result, _ = model.generate("p", schema=Verdict)
    assert result == Verdict(verdict="yes")
    assert "response_format" in litellm.calls[0]
    assert "response_format" not in litellm.calls[1]

    # Remembered: the next call goes straight to plain generation.
    litellm.responses = [_response('{"verdict": "no"}')]
    model.generate("p", schema=Verdict)
    assert "response_format" not in litellm.calls[2]
    assert len(litellm.calls) == 3


def test_schema_call_does_not_swallow_other_failures(litellm):
    model = GatewayModel("openai/gpt-x")
    litellm.errors = [RateLimitError("slow down")]
    with pytest.raises(RateLimitError):
        model.generate("p", schema=Verdict)
    assert len(litellm.calls) == 1


##############################
# subclass hooks             #
##############################


def test_subclass_hooks_and_instance_patching_survive(litellm):
    class Tenant(LiteLLMModel):
        def _completion_params(self, content):
            params = super()._completion_params(content)
            params["api_key"] = "tenant-key"
            return params

        def _response_cost(self, response):
            cost = super()._response_cost(response)
            self.seen_cost = cost
            return cost

        def generate_raw_response(self, *args, **kwargs):
            raise AttributeError("log_probs unsupported.")

    model = Tenant(model="openai/gpt-x")
    assert is_native_model(model)
    assert initialize_model(model) == (model, True)

    original = model.generate
    seen = []

    def guarded(*args, **kwargs):
        seen.append(args)
        return original(*args, **kwargs)

    model.generate = guarded

    litellm.responses = [_response('{"score": 8, "reason": "ok"}')]
    metric = _g_eval(model)
    metric.measure(LLMTestCase(input="q", actual_output="a"))

    assert metric.score == pytest.approx(0.8)
    assert seen, "instance-level generate patch was bypassed"
    assert litellm.calls[-1]["api_key"] == "tenant-key"
    assert model.seen_cost == 0.5
    assert metric.evaluation_cost == pytest.approx(0.5)


##############################
# string routing             #
##############################


@pytest.mark.parametrize(
    "name, cls, resolved",
    [
        (None, OpenAIModel, None),
        ("gpt-4.1", OpenAIModel, "gpt-4.1"),
        ("openai/gpt-4.1", OpenAIModel, "gpt-4.1"),
        ("ft:gpt-4o:org::id", OpenAIModel, "ft:gpt-4o:org::id"),
        ("brand-new-model", OpenAIModel, "brand-new-model"),
        ("claude-opus-5", GatewayModel, "anthropic/claude-opus-5"),
        ("anthropic/claude-opus-5", GatewayModel, "anthropic/claude-opus-5"),
        ("bedrock/some-model", GatewayModel, "bedrock/some-model"),
    ],
)
def test_string_routing(litellm, name, cls, resolved):
    model, native = initialize_model(name)
    assert type(model) is cls
    assert native is True
    if resolved is not None:
        assert model.name == resolved


def test_custom_openai_base_url_keeps_every_name_on_openai(
    litellm, monkeypatch
):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy/v1")
    model, _ = initialize_model("anthropic/claude-opus-5")
    assert type(model) is OpenAIModel
    assert model.name == "anthropic/claude-opus-5"


def test_provider_flag_keeps_its_own_routing(litellm, settings):
    with settings.edit(persist=False):
        settings.USE_LITELLM = True
    model, _ = initialize_model("anthropic/claude-opus-5")
    assert type(model) is LiteLLMModel


def test_default_model_setting(litellm, settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_LLM = "anthropic/claude-opus-5"
    model, _ = initialize_model(None)
    assert type(model) is GatewayModel
    assert model.name == "anthropic/claude-opus-5"


def test_missing_gateway_dependency_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(DeepEvalError, match="pip install litellm"):
        GatewayModel("anthropic/claude-x").generate("p")


def test_infer_provider():
    assert infer_provider("anthropic/claude-x") == "anthropic"
    assert infer_provider("gemini-3-flash") == "gemini"
    assert infer_provider("my-finetune") is None
    assert infer_provider(None) is None
