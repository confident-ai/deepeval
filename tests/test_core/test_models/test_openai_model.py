import types
import json
import pytest
from pydantic import BaseModel

from deepeval.models.llms.openai_model import (
    GPTModel,
    valid_gpt_models,
    structured_outputs_models,
    json_mode_models,
)


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
