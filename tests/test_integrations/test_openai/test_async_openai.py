import asyncio
import pytest
from openai import AsyncOpenAI
from deepeval.tracing import trace
from deepeval.prompt import Prompt
from deepeval.tracing.trace_context import LlmSpanContext
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
import os


client = AsyncOpenAI()

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

_current_dir = os.path.dirname(os.path.abspath(__file__))


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_async_openai_without_trace.json")
)
async def test_async_openai_without_trace():
    await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


@pytest.mark.skip
async def test_async_openai_with_trace():
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
        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )


@pytest.mark.skip
async def test_async_response_create_without_trace():
    await client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant. Always generate a string response.",
        input="Hello, how are you?",
    )


@assert_trace_json(
    json_path=os.path.join(
        _current_dir, "test_async_response_create_with_trace.json"
    )
)
async def test_async_response_create_with_trace():
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
        await client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant. Always generate a string response.",
            input="Hello, how are you?",
        )


async def generate_all_json_dumps():
    await test_async_openai_without_trace()
    # await test_async_openai_with_trace()
    # await test_async_response_create_without_trace()
    await test_async_response_create_with_trace()
