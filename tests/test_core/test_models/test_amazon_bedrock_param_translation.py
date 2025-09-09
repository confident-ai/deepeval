import copy
from typing import Any, Dict, Optional

import pytest

from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel


def _mk_model(gen_kwargs: Optional[Dict[str, Any]], temperature: float = 0.5):
    # avoid network/client setup: bypass __init__ and set only what we need
    m = AmazonBedrockModel.__new__(AmazonBedrockModel)
    m.temperature = temperature
    m.generation_kwargs = gen_kwargs or {}
    return m


def test_snake_case_is_translated_and_removed_max_tokens_top_p():
    """Ensure that `max_tokens` and `top_p` are translated into
    `maxTokens` and `topP` in the request payload, and that the
    original snake_case keys are removed. Also verifies that the
    message body contains the prompt text unchanged.
    """
    model = _mk_model({"max_tokens": 1234, "top_p": 0.42})
    body = model.get_converse_request_body("hi")

    # minimal smoke-check on message body
    assert body["messages"][0]["content"][0]["text"] == "hi"

    cfg = body["inferenceConfig"]
    # canonical keys exist
    assert cfg["maxTokens"] == 1234
    assert cfg["topP"] == 0.42
    # Snake-case keys must NOT be present
    assert "max_tokens" not in cfg
    assert "top_p" not in cfg


def test_stop_sequences_and_top_k_aliases_are_translated_and_snake_removed():
    """Verify that `stop_sequences` -> `stopSequences` and `top_k` -> `topK`
    are properly translated in the payload, and that the snake_case
    keys are not present.
    """
    model = _mk_model(
        {
            "stop_sequences": ["END", "STOP"],
            "top_k": 7,
        }
    )
    body = model.get_converse_request_body("hi")
    cfg = body["inferenceConfig"]

    assert cfg["stopSequences"] == ["END", "STOP"]
    assert "stop_sequences" not in cfg

    assert cfg["topK"] == 7
    assert "top_k" not in cfg


@pytest.mark.parametrize(
    "gen_kwargs,expected_max_tokens",
    [
        ({}, 1000),  # default
        ({"max_tokens": 2222}, 2222),  # snake provided
        ({"maxTokens": 3333}, 3333),  # camel provided
        ({"max_tokens": 1111, "maxTokens": 4444}, 4444),  # camel should win
    ],
)
def test_max_tokens_defaults_and_precedence(gen_kwargs, expected_max_tokens):
    """Check that `maxTokens` defaults to 1000 if not supplied,
    and that precedence rules are correct:
    - snake_case overrides default,
    - camelCase overrides snake_case,
    - no duplicate keys remain.
    """
    model = _mk_model(gen_kwargs)
    cfg = model.get_converse_request_body("hi")["inferenceConfig"]
    assert cfg["maxTokens"] == expected_max_tokens
    # ensure no snake duplicate
    assert "max_tokens" not in cfg


@pytest.mark.parametrize(
    "gen_kwargs,expected_top_p",
    [
        ({}, 0),  # default
        ({"top_p": 0.2}, 0.2),
        ({"topP": 0.3}, 0.3),
        ({"top_p": 0.1, "topP": 0.9}, 0.9),  # camel should win
    ],
)
def test_top_p_defaults_and_precedence(gen_kwargs, expected_top_p):
    """Check that `topP` defaults to 0 if not supplied,
    and that precedence rules are correct:
    - snake_case overrides default,
    - camelCase overrides snake_case,
    - no duplicate keys remain.
    """
    model = _mk_model(gen_kwargs)
    cfg = model.get_converse_request_body("hi")["inferenceConfig"]
    assert cfg["topP"] == expected_top_p
    assert "top_p" not in cfg


@pytest.mark.parametrize(
    "gen_kwargs,expected_stop_sequences",
    [
        ({"stop_sequences": ["END"]}, ["END"]),
        ({"stopSequences": ["A", "B"]}, ["A", "B"]),
        # camel should win if both present
        ({"stop_sequences": ["X"], "stopSequences": ["C", "D"]}, ["C", "D"]),
    ],
)
def test_stop_sequences_precedence(gen_kwargs, expected_stop_sequences):
    """Validate that `stopSequences` precedence is handled correctly:
    - snake_case is translated,
    - camelCase is passed through,
    - if both forms are provided, camelCase wins,
    - no snake_case remains in the payload.
    """
    model = _mk_model(gen_kwargs)
    cfg = model.get_converse_request_body("hi")["inferenceConfig"]
    assert cfg["stopSequences"] == expected_stop_sequences
    assert "stop_sequences" not in cfg


@pytest.mark.parametrize(
    "gen_kwargs,expected_top_k",
    [
        ({"top_k": 5}, 5),
        ({"topK": 9}, 9),
        # camel should win if both present
        ({"top_k": 3, "topK": 8}, 8),
    ],
)
def test_top_k_precedence(gen_kwargs, expected_top_k):
    """Validate that `topK` precedence is handled correctly:
    - snake_case is translated,
    - camelCase is passed through,
    - if both forms are provided, camelCase wins,
    - no snake_case remains in the payload.
    """
    model = _mk_model(gen_kwargs)
    cfg = model.get_converse_request_body("hi")["inferenceConfig"]
    assert cfg["topK"] == expected_top_k
    assert "top_k" not in cfg


def test_generation_kwargs_is_not_mutated():
    """Ensure that the original `generation_kwargs` dictionary provided
    by the user is not mutated when building the request body.
    """
    original = {"max_tokens": 500, "stop_sequences": ["END"]}
    snapshot = copy.deepcopy(original)

    model = _mk_model(original)
    _ = model.get_converse_request_body("hi")

    assert (
        original == snapshot
    ), "get_converse_request_body must not mutate input dict"


def test_temperature_source_of_truth_is_documented_behavior():
    """Check that `temperature` from `generation_kwargs` overrides the
    model's default temperature. This documents the current precedence
    rule so it is explicit and test enforced.
    """
    # kwargs override the model default today
    model = _mk_model({"temperature": 0.9}, temperature=0.2)
    cfg = model.get_converse_request_body("hi")["inferenceConfig"]
    assert cfg["temperature"] == 0.9
