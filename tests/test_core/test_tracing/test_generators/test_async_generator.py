import pytest
import asyncio
from deepeval.tracing import observe, update_llm_span
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="llm", model="gpt-4-turbo")
async def async_streaming_llm(prompt: str):
    tokens = ["Async", " ", "response", "!"]
    for token in tokens:
        await asyncio.sleep(0.01)
        yield token
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=len(tokens),
    )


@observe()
async def async_streaming_processor(data: str):
    chunks = data.split()
    for chunk in chunks:
        await asyncio.sleep(0.01)
        yield f"<{chunk}>"


@observe()
async def async_streaming_with_nested(data: str):
    yield "Async Start"
    result = await async_helper(data)
    yield result
    yield "Async End"


@observe()
async def async_helper(data: str) -> str:
    await asyncio.sleep(0.01)
    return f"Async Processed: {data}"


@observe(type="llm", model="async-streaming-model")
async def async_streaming_with_updates(prompt: str):
    tokens = prompt.split()
    total_tokens = 0
    for token in tokens:
        await asyncio.sleep(0.005)
        yield token
        total_tokens += 1
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=total_tokens,
    )


@observe()
async def async_streaming_with_error(data: str):
    yield "First"
    await asyncio.sleep(0.01)
    yield "Second"
    if data == "error":
        raise ValueError("Async simulated error")
    yield "Third"


@observe()
async def async_streaming_concurrent(data: str):
    async def fetch_chunk(chunk: str) -> str:
        await asyncio.sleep(0.01)
        return f"Fetched: {chunk}"

    chunks = data.split()
    for chunk in chunks:
        result = await fetch_chunk(chunk)
        yield result


class TestAsyncGenerator:

    @trace_test("generators/async_streaming_llm_schema.json")
    @pytest.mark.asyncio
    async def test_async_streaming_llm(self):
        result = []
        async for token in async_streaming_llm("Test async prompt"):
            result.append(token)

    @trace_test("generators/async_streaming_processor_schema.json")
    @pytest.mark.asyncio
    async def test_async_streaming_processor(self):
        result = []
        async for chunk in async_streaming_processor("alpha beta gamma"):
            result.append(chunk)

    @trace_test("generators/async_streaming_nested_schema.json")
    @pytest.mark.asyncio
    async def test_async_streaming_with_nested(self):
        result = []
        async for item in async_streaming_with_nested("test"):
            result.append(item)

    @trace_test("generators/async_streaming_updates_schema.json")
    @pytest.mark.asyncio
    async def test_async_streaming_with_updates(self):
        result = []
        async for token in async_streaming_with_updates("one two three"):
            result.append(token)

    @pytest.mark.asyncio
    async def test_async_streaming_error_handling(self):
        gen = async_streaming_with_error("error")
        results = []
        with pytest.raises(ValueError, match="Async simulated error"):
            async for token in gen:
                results.append(token)
        assert results == ["First", "Second"]

    @trace_test("generators/async_streaming_concurrent_schema.json")
    @pytest.mark.asyncio
    async def test_async_streaming_concurrent(self):
        result = []
        async for item in async_streaming_concurrent("a b c"):
            result.append(item)
