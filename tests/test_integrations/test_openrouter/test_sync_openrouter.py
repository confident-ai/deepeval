import os

import pytest

from deepeval.openai import OpenAI
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


# Built inside the tests, not at module scope: `OpenAI(api_key=None)` raises on
# construction, which would break collection before `pytestmark` could skip.
def openrouter_client() -> OpenRouter:
    """The official SDK. Importing OpenRouter from deepeval installs the patches."""
    return OpenRouter()


def openai_client() -> OpenAI:
    """The same gateway reached through the OpenAI SDK — `deepeval.openai`
    recognizes OpenRouter's base URL, so these spans record provider
    "OpenRouter" while the integration stays "OpenAI"."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


@trace_test("test_sync_chat_send_without_trace.json")
def test_sync_chat_send_without_trace():
    openrouter_client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


@trace_test("test_sync_chat_send_with_trace.json")
def test_sync_chat_send_with_trace():
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
        openrouter_client().chat.send(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )


@trace_test("test_sync_openai_sdk_route.json")
def test_sync_openai_sdk_route():
    openai_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


def generate_all_json_dumps():
    test_sync_chat_send_without_trace()
    test_sync_chat_send_with_trace()
    test_sync_openai_sdk_route()
