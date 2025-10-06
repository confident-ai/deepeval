import copy
from typing import Any, Dict, Optional

import pytest

from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel


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


def test_get_model_name_returns_model_id():
    model = _mk_model({})
    model.model_id = "my-model"
    assert model.get_model_name() == "my-model"
