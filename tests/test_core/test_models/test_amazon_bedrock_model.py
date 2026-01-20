import copy
import pytest
from typing import Any, Dict, Optional

from tests.test_core.stubs import _RecordingClient
from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel


class RecordingBedrockClient(_RecordingClient):
    def __init__(self, response, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response = response

    async def converse(self, **kwargs):
        return self._response


def _mk_model(gen_kwargs: Optional[Dict[str, Any]]):
    # bypass __init__, set only needed attributes for tests
    m = AmazonBedrockModel.__new__(AmazonBedrockModel)
    m.generation_kwargs = gen_kwargs or {}
    return m


def test_get_converse_request_body_contains_temperature_and_kwargs():
    gen_kwargs = {
        "maxTokens": 1234,
        "stopSequences": ["END", "STOP"],
        "temperature": 0.7,
    }
    model = _mk_model(gen_kwargs)
    body = model.get_converse_request_body("hello")

    assert body["messages"][0]["content"][0]["text"] == "hello"
    inf_cfg = body["inferenceConfig"]
    assert inf_cfg["temperature"] == 0.7
    assert inf_cfg["maxTokens"] == 1234
    assert inf_cfg["stopSequences"] == ["END", "STOP"]


def test_generation_kwargs_not_mutated():
    original = {"maxTokens": 500, "stopSequences": ["END"]}
    snapshot = copy.deepcopy(original)

    model = _mk_model(original)
    _ = model.get_converse_request_body("hi")

    assert original == snapshot, "generation_kwargs should not be mutated"


@pytest.mark.parametrize(
    "gen_kwargs",
    [
        {},
        {"maxTokens": 1000},
        {"stopSequences": ["STOP"]},
        {"temperature": 0.5},
        {
            "maxTokens": 1000,
            "stopSequences": ["STOP"],
            "topP": 0.5,
            "temperature": 0.5,
        },
    ],
)
def test_various_generation_kwargs_passed_through(gen_kwargs):
    model = _mk_model(gen_kwargs)
    body = model.get_converse_request_body("prompt")
    inf_cfg = body["inferenceConfig"]

    for key, value in gen_kwargs.items():
        assert key in inf_cfg
        assert inf_cfg[key] == value


def test_get_model_name_returns_name():
    model = _mk_model({})
    model.name = "my-model"
    assert model.get_model_name() == "my-model"


@pytest.mark.asyncio
async def test_bedrock_a_generate_skips_reasoning_content_and_reads_text_block(
    monkeypatch,
):
    m = AmazonBedrockModel.__new__(AmazonBedrockModel)
    m.generation_kwargs = {}
    m.kwargs = {}
    m.region = "us-east-1"
    m.name = "openai.gpt-oss-safeguard-20b"

    # model_data must exist because calculate_cost reads it
    # BUT prices are None in registry anyway, so cost will be None.
    class _MD:
        input_price = None
        output_price = None

    m.model_data = _MD()

    response = {
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "reasoning..."}
                        }
                    },
                    {
                        "text": '{"statements":["The capital of France is Paris."]}'
                    },
                ]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }

    async def _ensure_client():
        return RecordingBedrockClient(response)

    monkeypatch.setattr(m, "_ensure_client", _ensure_client)

    out, cost = await m.a_generate("prompt", schema=None)
    assert out == '{"statements":["The capital of France is Paris."]}'
    assert cost is None


@pytest.mark.asyncio
async def test_bedrock_a_generate_reads_text_block_when_first(monkeypatch):
    """
    if Bedrock returns a plain text block (no reasoningContent),
    we should still extract it correctly.
    """
    m = AmazonBedrockModel.__new__(AmazonBedrockModel)
    m.generation_kwargs = {}
    m.kwargs = {}
    m.region = "us-east-1"
    m.name = "openai.gpt-oss-safeguard-20b"

    class _MD:
        input_price = None
        output_price = None

    m.model_data = _MD()

    response = {
        "output": {
            "message": {
                "content": [
                    {"text": '{"statements":["hello"]}'},
                ]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }

    async def _ensure_client():
        return RecordingBedrockClient(response)

    monkeypatch.setattr(m, "_ensure_client", _ensure_client)

    out, cost = await m.a_generate("prompt", schema=None)
    assert out == '{"statements":["hello"]}'
    assert cost is None
