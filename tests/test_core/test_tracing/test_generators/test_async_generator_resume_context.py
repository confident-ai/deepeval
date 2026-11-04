"""Resume observed async generators in their own tracing context."""

import asyncio
from contextvars import Context

import pytest

from deepeval.tracing import observe
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import TraceSpanStatus


@observe(type="tool")
def record_child_span(spans):
    spans.append(current_span_context.get())
    return "child result"


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_method", ["anext", "athrow"])
async def test_interleaved_async_generator_restores_its_span(
    resume_method, completed_traces
):
    children = []

    @observe()
    async def stream():
        try:
            yield "first"
        except ValueError:
            pass
        yield record_child_span(children)

    first = stream()
    second = stream()
    try:
        assert await first.__anext__() == "first"
        first_span = current_span_context.get()
        first_trace = current_trace_context.get()
        assert await second.__anext__() == "first"
        second_span = current_span_context.get()
        assert second_span is not first_span
        # Preserve the existing consumer-between-yields parenting semantics.
        assert second_span.parent_uuid == first_span.uuid

        if resume_method == "anext":
            result = await first.__anext__()
        else:
            result = await first.athrow(ValueError("resume first"))

        assert result == "child result"
        assert children[0].parent_uuid == first_span.uuid
        assert children[0].trace_uuid == first_trace.uuid
        assert children[0] in first_span.children
        assert children[0] not in second_span.children
        assert current_span_context.get() is first_span
        assert current_trace_context.get() is first_trace
    finally:
        await second.aclose()
        await first.aclose()

    assert len(completed_traces) == 1
    assert completed_traces[0].root_spans == [first_span]
    assert first_span.end_time is not None
    assert second_span.end_time is not None
    assert children[0].end_time is not None
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces
    assert current_span_context.get() is None
    assert current_trace_context.get() is None


@pytest.mark.asyncio
async def test_async_generator_restores_trace_in_another_task(completed_traces):
    children = []
    generator_contexts = []

    @observe()
    async def stream():
        generator_contexts.append(
            (current_span_context.get(), current_trace_context.get())
        )
        yield "first"
        generator_contexts.append(
            (current_span_context.get(), current_trace_context.get())
        )
        yield record_child_span(children)

    gen = stream()

    async def in_fresh_context(awaitable):
        return await Context().run(asyncio.create_task, awaitable)

    try:
        assert await in_fresh_context(gen.__anext__()) == "first"
        assert await in_fresh_context(gen.__anext__()) == "child result"
        first_span, first_trace = generator_contexts[0]
        resumed_span, resumed_trace = generator_contexts[1]
        assert resumed_span is first_span
        assert resumed_trace is first_trace
        assert children[0].parent_uuid == first_span.uuid
        assert children[0].trace_uuid == first_trace.uuid
        with pytest.raises(StopAsyncIteration):
            await in_fresh_context(gen.__anext__())
    finally:
        await in_fresh_context(gen.aclose())

    assert len(completed_traces) == 1
    assert completed_traces[0].root_spans == [first_span]
    assert first_span.children == children
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces
    assert current_span_context.get() is None
    assert current_trace_context.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_method", ["exhaust", "close", "error"])
async def test_finished_async_generator_does_not_reactivate_context(
    finish_method, completed_traces
):
    @observe()
    async def stream():
        yield "first"
        if finish_method == "error":
            raise ValueError("stream failed")

    gen = stream()
    assert await gen.__anext__() == "first"
    span = current_span_context.get()
    if finish_method == "close":
        await gen.aclose()
    elif finish_method == "error":
        with pytest.raises(ValueError, match="stream failed"):
            await gen.__anext__()
        assert span.status == TraceSpanStatus.ERRORED
    else:
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    assert span.end_time is not None
    assert len(completed_traces) == 1
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces
    with Observer("agent", "unrelated_consumer"):
        caller_span = current_span_context.get()
        caller_trace = current_trace_context.get()
        for _ in range(2):
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()
            assert await gen.athrow(ValueError("already finished")) is None
            await gen.aclose()
            assert current_span_context.get() is caller_span
            assert current_trace_context.get() is caller_trace


@pytest.mark.asyncio
async def test_closing_unstarted_async_generator_does_not_create_span(
    completed_traces,
):
    @observe()
    async def stream():
        yield "unused"

    gen = stream()
    await gen.aclose()
    assert not completed_traces
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces
    assert current_span_context.get() is None
    assert current_trace_context.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_method", ["anext", "athrow"])
async def test_async_generator_preserves_nested_context_across_yield(
    resume_method, completed_traces
):
    children = []
    nested_spans = []

    @observe()
    async def stream():
        with Observer("agent", "nested_context"):
            nested_spans.append(current_span_context.get())
            try:
                yield "first"
            except ValueError:
                pass
            yield record_child_span(children)

    gen = stream()
    try:
        assert await gen.__anext__() == "first"
        if resume_method == "anext":
            assert await gen.__anext__() == "child result"
        else:
            assert await gen.athrow(ValueError("resume")) == "child result"
        assert children[0].parent_uuid == nested_spans[0].uuid
        assert children[0] in nested_spans[0].children
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    finally:
        await gen.aclose()

    assert len(completed_traces) == 1
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_method", ["anext", "athrow"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("fresh_context", [False, True])
async def test_async_generator_does_not_restore_externally_closed_inner(
    resume_method, depth, fresh_context, completed_traces
):
    children = []
    inner_generators = []
    parent_spans = []

    @observe()
    async def inner(level):
        if level > 1:
            child = inner(level - 1)
            inner_generators.append(child)
            yield await child.__anext__()
            await child.aclose()
        else:
            yield "inner"

    @observe()
    async def outer():
        with Observer("agent", "middle_context"):
            parent_spans.append(current_span_context.get())
            child = inner(depth)
            inner_generators.append(child)
            try:
                yield await child.__anext__()
            except ValueError:
                pass
            yield record_child_span(children)
            await child.aclose()

    gen = outer()
    try:
        assert await gen.__anext__() == "inner"
        inner_span = current_span_context.get()
        for child in reversed(inner_generators):
            await child.aclose()
        assert inner_span.end_time is not None
        resume = (
            gen.__anext__()
            if resume_method == "anext"
            else gen.athrow(ValueError("resume"))
        )
        if fresh_context:
            result = await Context().run(asyncio.create_task, resume)
        else:
            result = await resume
        assert result == "child result"
        assert children[0].parent_uuid == parent_spans[0].uuid
        assert children[0] in parent_spans[0].children
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    finally:
        for child in reversed(inner_generators):
            await child.aclose()
        await gen.aclose()

    assert len(completed_traces) == 1
    assert len(completed_traces[0].root_spans) == 1
    assert not trace_manager.active_spans
    assert not trace_manager.active_traces


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_method", ["exhaust", "close", "error"])
async def test_finished_async_generator_releases_saved_context(finish_method):
    import gc
    import weakref

    @observe()
    async def stream():
        yield "first"

    gen = stream()
    assert await gen.__anext__() == "first"
    span_ref = weakref.ref(current_span_context.get())
    trace_ref = weakref.ref(current_trace_context.get())
    if finish_method == "close":
        await gen.aclose()
    elif finish_method == "error":
        with pytest.raises(ValueError, match="stream failed"):
            await gen.athrow(ValueError("stream failed"))
    else:
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
    gc.collect()
    assert span_ref() is None
    assert trace_ref() is None

    with Observer("agent", "unrelated_consumer"):
        caller_span_ref = weakref.ref(current_span_context.get())
        caller_trace_ref = weakref.ref(current_trace_context.get())
        assert await gen.athrow(ValueError("already closed")) is None
    gc.collect()
    assert caller_span_ref() is None
    assert caller_trace_ref() is None
