import copy
from typing import Any, Dict, Optional

import pytest

from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel
from unittest.mock import MagicMock, AsyncMock


def _mk_model(gen_kwargs: Optional[Dict[str, Any]]):
    # bypass __init__, set only needed attributes for tests
    m = AmazonBedrockModel.__new__(AmazonBedrockModel)
    m.model_id = "bedrock-model"
    m.input_token_cost = 0
    m.output_token_cost = 0
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


def test_get_model_name_returns_model_id():
    model = _mk_model({})
    model.model_id = "my-model"
    assert model.get_model_name() == "my-model"


@pytest.mark.asyncio
async def test_a_generate_handles_non_text_messages_correctly():
    """
    Ensures a_generate() correctly picks the first item containing 'text' key.
    Required since Bedrock responses may include non-text content objects such as
    reasoning messages.
    """

    mock_client = MagicMock()
    mock_client.converse = AsyncMock(
        return_value={
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "thinking..."}
                            }
                        },
                        {"text": "The answer."},
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }
    )

    async def mock_ensure_client(self):
        return mock_client

    model = _mk_model({})
    model._ensure_client = mock_ensure_client.__get__(model)

    async def async_noop_close(self):
        pass

    model.close = async_noop_close.__get__(model)
    message, _cost = await model.a_generate("question")

    assert message == "The answer."
    mock_client.converse.assert_called_once_with(
        modelId="bedrock-model",
        messages=[{"role": "user", "content": [{"text": "question"}]}],
        inferenceConfig={},
    )
