import pytest
import inspect
import asyncio
from deepteam.test_case import RTTurn
from deepteam.red_teamer.utils import (
    resolve_model_callback,
    wrap_model_callback,
)
from deepeval.models import GPTModel
from deepteam.vulnerabilities import Bias
from deepteam import red_team


def custom_model_callback(input: str, turns=None):
    return f"Cannot respond to {input}"


async def custom_async_model_callback(input: str, turns=None):
    return f"Cannot respond to {input}"


def custom_invalid_model_callback(input: str, turns=None):
    return {"res": f"Cannot respond to {input}"}


async def custom_invalid_async_model_callback(input: str, turns=None):
    return {"res": f"Cannot respond to {input}"}


class TestModelCallbackVariations:

    def test_openai_valid_sync_model(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        model_callback = resolve_model_callback(OPENAI_MODEL, False)
        assert not inspect.iscoroutinefunction(model_callback)
        assert model_callback("Hello") is not None

    def test_openai_valid_async_model(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        model_callback = resolve_model_callback(OPENAI_MODEL, True)
        assert inspect.iscoroutinefunction(model_callback)
        assert asyncio.run(model_callback("Hello")) is not None

    def test_deepeval_base_llm_sync_model(self):
        model = GPTModel(model="gpt-3.5-turbo")
        model_callback = resolve_model_callback(model, False)
        assert not inspect.iscoroutinefunction(model_callback)
        assert model_callback("Hello") is not None

    def test_deepeval_base_llm_async_model(self):
        model = GPTModel(model="gpt-3.5-turbo")
        model_callback = resolve_model_callback(model, True)
        assert inspect.iscoroutinefunction(model_callback)
        assert asyncio.run(model_callback("Hello")) is not None

    def test_sync_wrapper_returns_valid_model_callback(self):
        model_callback = wrap_model_callback(custom_model_callback, False)
        response = model_callback("Hello this is a test")

        assert inspect.iscoroutinefunction(model_callback) is False
        assert isinstance(response, RTTurn)
        assert response.role == "assistant"

    def test_async_wrapper_returns_valid_model_callback(self):
        model_callback = wrap_model_callback(custom_async_model_callback, True)
        response = asyncio.run(model_callback("Hello this is a test"))

        assert inspect.iscoroutinefunction(model_callback) is True
        assert isinstance(response, RTTurn)
        assert response.role == "assistant"

    def test_sync_wrapper_raises_error_for_invalid_response(self):
        model_callback = wrap_model_callback(
            custom_invalid_model_callback, False
        )
        with pytest.raises(TypeError):
            model_callback("Hello this is a test")

    def test_async_wrapper_raises_error_for_invalid_response(self):
        model_callback = wrap_model_callback(
            custom_invalid_async_model_callback, True
        )
        with pytest.raises(TypeError):
            asyncio.run(model_callback("Hello this is a test"))

    def test_simple_red_teaming_async(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=True,
            ignore_errors=False,
        )
        assert risk_assessment is not None

    def test_simple_red_teaming_sync(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=False,
            ignore_errors=False,
        )
        assert risk_assessment is not None
