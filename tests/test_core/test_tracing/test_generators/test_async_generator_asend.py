import asyncio

import pytest

from deepeval.tracing import observe
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import TraceSpanStatus


@pytest.mark.asyncio
async def test_asend_forwards_values_and_finishes_trace():
    @observe()
    async def stream():
        value = yield "ready"
        yield {"received": value}

    generator = stream()
    try:
        first = generator.asend(None)
        assert not trace_manager.active_spans
        assert await first == "ready"
        span = current_span_context.get()
        trace = current_trace_context.get()

        assert await generator.asend("continue") == {"received": "continue"}
        assert current_span_context.get() is span
        with pytest.raises(StopAsyncIteration):
            await generator.asend(None)

        assert span.end_time is not None
        assert span.status == TraceSpanStatus.SUCCESS
        assert trace.end_time is not None
        assert not trace_manager.active_spans
        assert not trace_manager.active_traces
        assert current_span_context.get() is None
        assert current_trace_context.get() is None
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_asend_rejects_initial_value_without_consuming_generator():
    @observe()
    async def stream():
        yield "ready"

    generator = stream()
    try:
        with pytest.raises(TypeError, match="non-None value"):
            await generator.asend("too early")
        assert not trace_manager.active_spans
        assert not trace_manager.active_traces

        assert await generator.asend(None) == "ready"
        assert current_span_context.get() is not None
    finally:
        await generator.aclose()

    assert not trace_manager.active_spans
    assert not trace_manager.active_traces


@pytest.mark.asyncio
async def test_asend_records_and_propagates_error():
    error = ValueError("invalid command")

    @observe()
    async def stream():
        yield "ready"
        raise error

    generator = stream()
    try:
        assert await generator.__anext__() == "ready"
        span = current_span_context.get()
        with pytest.raises(ValueError) as exc:
            await generator.asend("continue")
        assert exc.value is error
        assert span.status == TraceSpanStatus.ERRORED
        assert span.error == "invalid command"
        assert span.end_time is not None
        assert not trace_manager.active_spans
        assert not trace_manager.active_traces
        assert current_span_context.get() is None
        assert current_trace_context.get() is None
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_asend_cancellation_finishes_trace_before_iterator_is_released():
    waiting = asyncio.Event()
    captured = {}

    @observe()
    async def stream():
        yield "ready"
        waiting.set()
        await asyncio.Event().wait()

    generator = stream()

    async def consume():
        assert await generator.__anext__() == "ready"
        captured["span"] = current_span_context.get()
        try:
            await generator.asend("continue")
        finally:
            captured["context"] = (
                current_span_context.get(),
                current_trace_context.get(),
            )

    task = asyncio.create_task(consume())
    wait_task = asyncio.create_task(waiting.wait())
    try:
        # Waiting for either condition also makes a missing asend fail promptly.
        done, _ = await asyncio.wait(
            [task, wait_task],
            timeout=5,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            await task
        assert waiting.is_set()
        task.cancel("stream cancelled")
        with pytest.raises(asyncio.CancelledError):
            await task

        span = captured["span"]
        assert span.status == TraceSpanStatus.ERRORED
        assert span.end_time is not None
        assert not trace_manager.active_spans
        assert not trace_manager.active_traces
        assert captured["context"] == (None, None)
    finally:
        task.cancel()
        wait_task.cancel()
        await asyncio.gather(task, wait_task, return_exceptions=True)
        await generator.aclose()
