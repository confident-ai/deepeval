import pytest
import asyncio
from deepeval.tracing import observe, update_llm_span
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import TraceSpanStatus
from tests.test_core.test_tracing.conftest import trace_test


@pytest.mark.parametrize("resume_method", ["__anext__", "athrow"])
@pytest.mark.asyncio
async def test_async_generator_cancellation_closes_trace(resume_method):
    waiting = asyncio.Event()
    captured = {}

    @observe()
    async def stream():
        try:
            yield "first"
        except ValueError:
            pass
        waiting.set()
        await asyncio.Event().wait()
        yield "unreachable"

    # Retain the iterator: cancellation must clean up without relying on GC.
    generator = stream()

    async def consume():
        assert await generator.__anext__() == "first"
        captured["span"] = current_span_context.get()
        captured["trace"] = current_trace_context.get()
        try:
            if resume_method == "athrow":
                await generator.athrow(ValueError("resume"))
            else:
                await generator.__anext__()
        except asyncio.CancelledError:
            captured["context_after_cancel"] = (
                current_span_context.get(),
                current_trace_context.get(),
            )
            raise

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel("stream cancelled")
        with pytest.raises(asyncio.CancelledError, match="stream cancelled"):
            await task

        span = captured["span"]
        trace = captured["trace"]
        assert span.end_time is not None
        assert span.status == TraceSpanStatus.ERRORED
        assert span.error == "stream cancelled"
        assert trace.end_time is not None
        assert trace.status == TraceSpanStatus.ERRORED
        assert not trace_manager.active_spans
        assert not trace_manager.active_traces
        assert captured["context_after_cancel"] == (None, None)

        await generator.aclose()
        assert span.status == TraceSpanStatus.ERRORED
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await generator.aclose()


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
