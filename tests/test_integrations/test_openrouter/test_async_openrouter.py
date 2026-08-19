import os

import pytest

from deepeval.openai import AsyncOpenAI
from deepeval.openrouter import OpenRouter
from deepeval.prompt import Prompt
from deepeval.tracing import LlmSpanContext, trace

from tests.test_integrations.test_openrouter.conftest import trace_test

MODEL = "openai/gpt-4o-mini"

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is required to run OpenRouter tests.",
)

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"


# Built per test rather than at module scope: each test gets its own event loop,
# and an async HTTP client outlives only the loop it was created on.
def openrouter_client() -> OpenRouter:
    """The official SDK. One client serves both `chat.send` and `send_async`."""
    return OpenRouter()


def openai_client() -> AsyncOpenAI:
    """The same gateway via the OpenAI SDK, which `deepeval.openai` recognizes
    from the base URL."""
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


@trace_test("test_async_chat_send_without_trace.json")
async def test_async_chat_send_without_trace():
    await openrouter_client().chat.send_async(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


@trace_test("test_async_chat_send_with_trace.json")
async def test_async_chat_send_with_trace():
    with trace(
        llm_span_context=LlmSpanContext(
            prompt=prompt,
            metric_collection="test_collection_1",
        ),
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
        thread_id="test_thread_id_1",
    ):
        await openrouter_client().chat.send_async(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )


@trace_test("test_async_openai_sdk_route.json")
async def test_async_openai_sdk_route():
    await openai_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


async def generate_all_json_dumps():
    await test_async_chat_send_without_trace()
    await test_async_chat_send_with_trace()
    await test_async_openai_sdk_route()
