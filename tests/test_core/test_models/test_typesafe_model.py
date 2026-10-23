"""Jev's own `max_tokens_exceeded` rejection surfaces as the same
`SystemOneContextLimitError` the local pre-check raises."""

import asyncio

import pytest

typesafe_sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from deepeval.errors import ContextWindowExceededError
from deepeval.models import TypeSafeModel
from deepeval.models.system_one.limits import SystemOneContextLimitError
from deepeval.models.system_one.schema import NoulQuestion

QUESTIONS = {"completed": NoulQuestion(instructions="Was the task done?")}


def _jev_error(error_type: str) -> Exception:
    return typesafe_sdk.TypeSafeBadRequestError(
        400, {"detail": {"error_type": error_type}}, httpx2.Headers()
    )


class _RejectingClient:
    def __init__(self, error: Exception):
        self.error = error

    def system_one(self, *args, **kwargs):
        raise self.error


class _AsyncRejectingClient(_RejectingClient):
    async def system_one(self, *args, **kwargs):
        raise self.error


def _decide(error: Exception, async_mode: bool):
    model = TypeSafeModel(api_key="test-key")
    client_cls = _AsyncRejectingClient if async_mode else _RejectingClient
    model.load_model = lambda async_mode=False: client_cls(error)
    if async_mode:
        return asyncio.run(model.a_decide({"trace": "small"}, QUESTIONS))
    return model.decide({"trace": "small"}, QUESTIONS)


@pytest.mark.parametrize("async_mode", [False, True])
def test_max_tokens_exceeded_raises_context_limit(async_mode):
    error = _jev_error("max_tokens_exceeded")

    with pytest.raises(SystemOneContextLimitError) as info:
        _decide(error, async_mode)

    assert isinstance(info.value, ContextWindowExceededError)
    assert info.value.limit_tokens is None
    assert info.value.__cause__ is error


@pytest.mark.parametrize("async_mode", [False, True])
def test_other_rejections_propagate_unchanged(async_mode):
    error = _jev_error("invalid_question")

    with pytest.raises(typesafe_sdk.TypeSafeBadRequestError) as info:
        _decide(error, async_mode)

    assert info.value is error
