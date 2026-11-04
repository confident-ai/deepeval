"""OpenAI streaming trace regressions; no network or API key required."""

from types import SimpleNamespace as NS

import pytest

from deepeval.openai import patch
from deepeval.tracing.context import current_span_context, current_trace_context


class SyncStream:
    def __init__(self, events, error=None):
        self.events = iter(events)
        self.error = error
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.error:
                raise self.error
            raise

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncStream:
    def __init__(self, events, error=None):
        self.stream = SyncStream(events, error)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.stream)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.stream.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


@pytest.fixture
def captured(monkeypatch):
    observers = []
    updates = []

    class Observer:
        def __init__(self, *args, **kwargs):
            self.exited = False
            self.error = None
            observers.append(self)

        def __enter__(self):
            current_span_context.set(NS(uuid="stream-span"))
            current_trace_context.set(NS(uuid="stream-trace"))
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited = True
            self.error = exc

    monkeypatch.setattr(patch, "Observer", Observer)
    monkeypatch.setattr(
        patch,
        "_update_all_attributes",
        lambda _, output, *rest: updates.append(output),
    )
    return observers, updates


def chat_chunk(content=None, usage=None):
    return NS(
        choices=(
            [NS(delta=NS(content=content, tool_calls=None))] if content else []
        ),
        usage=usage,
    )


def test_sync_chat_stream_keeps_span_open_until_exhausted(captured):
    observers, updates = captured
    chunks = [
        chat_chunk("Hello"),
        chat_chunk(" world"),
        chat_chunk(usage=NS(prompt_tokens=3, completion_tokens=2)),
    ]
    raw = SyncStream(chunks)
    wrapped = patch._patch_sync_openai_client_method(lambda **_: raw, True)

    stream = wrapped(
        stream=True, model="test", messages=[{"role": "user", "content": "Hi"}]
    )
    assert not observers[0].exited
    assert current_span_context.get() is None
    assert list(stream) == chunks
    assert observers[0].exited
    assert updates[0].output == "Hello world"
    assert (updates[0].prompt_tokens, updates[0].completion_tokens) == (3, 2)


def test_sync_stream_close_preserves_partial_output(captured):
    observers, updates = captured
    raw = SyncStream([chat_chunk("partial")])
    parent_span = NS(uuid="parent-span")
    parent_trace = NS(uuid="parent-trace")
    span_token = current_span_context.set(parent_span)
    trace_token = current_trace_context.set(parent_trace)
    try:
        stream = patch._patch_sync_openai_client_method(lambda **_: raw, True)(
            stream=True, model="test", messages=[]
        )
        assert current_span_context.get() is parent_span
        assert next(stream) is not None
        stream.close()
        assert current_span_context.get() is parent_span
        assert current_trace_context.get() is parent_trace
        assert raw.closed and observers[0].exited
        assert updates[0].output == "partial"
    finally:
        current_span_context.reset(span_token)
        current_trace_context.reset(trace_token)


def test_sync_stream_context_manager_closes_on_early_exit(captured):
    observers, updates = captured
    raw = SyncStream([chat_chunk("partial"), chat_chunk("unused")])
    stream = patch._patch_sync_openai_client_method(lambda **_: raw, True)(
        stream=True, model="test", messages=[]
    )
    with stream as received:
        assert received is stream
        assert next(received) is not None
    assert raw.closed and observers[0].exited
    assert updates[0].output == "partial"


def test_sync_response_stream_without_usage_keeps_output(captured):
    observers, updates = captured
    response = NS(output_text="Final answer", usage=None, output=[])
    events = [
        NS(type="response.output_text.delta", delta="Final "),
        NS(type="response.completed", response=response),
    ]
    raw = SyncStream(events)
    stream = patch._patch_sync_openai_client_method(lambda **_: raw, False)(
        stream=True, model="test", input="question"
    )
    assert list(stream) == events
    assert observers[0].exited
    assert updates[0].output == "Final answer"
    assert updates[0].prompt_tokens is None


def test_sync_chat_stream_collects_tool_call_deltas(captured):
    observers, updates = captured
    calls = [
        NS(index=0, function=NS(name="weather", arguments='{"city":')),
        NS(index=0, function=NS(name=None, arguments='"Paris"}')),
    ]
    chunks = [
        NS(choices=[NS(delta=NS(content=None, tool_calls=[call]))], usage=None)
        for call in calls
    ]
    stream = patch._patch_sync_openai_client_method(
        lambda **_: SyncStream(chunks), True
    )(
        stream=True,
        model="test",
        messages=[],
        tools=[
            {"function": {"name": "weather", "description": "Look up weather"}}
        ],
    )
    assert list(stream) == chunks
    assert observers[0].exited
    assert updates[0].tools_called[0].name == "weather"
    assert updates[0].tools_called[0].input_parameters == {"city": "Paris"}


def test_raw_streaming_response_retains_sdk_response_object(captured):
    observers, _ = captured
    raw_response = NS(status_code=200)
    wrapped = patch._patch_sync_openai_client_method(
        lambda **_: raw_response, True
    )
    assert wrapped(stream=True, model="test", messages=[]) is raw_response
    assert observers[0].exited


@pytest.mark.asyncio
async def test_async_raw_streaming_response_retains_sdk_response_object(
    captured,
):
    observers, _ = captured
    raw_response = NS(status_code=200)

    async def original(**_):
        return raw_response

    wrapped = patch._patch_async_openai_client_method(original, False)
    assert (
        await wrapped(stream=True, model="test", input="question")
        is raw_response
    )
    assert observers[0].exited


@pytest.mark.asyncio
async def test_async_response_stream_captures_completed_response(captured):
    observers, updates = captured
    response = NS(
        output_text="The answer",
        usage=NS(input_tokens=8, output_tokens=3),
        output=[],
    )
    events = [
        NS(type="response.output_text.delta", delta="The "),
        NS(type="response.completed", response=response),
    ]
    raw = AsyncStream(events)

    async def original(**_):
        return raw

    wrapped = patch._patch_async_openai_client_method(original, False)
    stream = await wrapped(stream=True, model="test", input="question")
    assert not observers[0].exited
    seen = [event async for event in stream]
    assert seen == events
    assert observers[0].exited
    assert updates[0].output == "The answer"
    assert (updates[0].prompt_tokens, updates[0].completion_tokens) == (8, 3)


@pytest.mark.asyncio
async def test_async_chat_stream_collects_text_and_usage(captured):
    observers, updates = captured
    chunks = [
        chat_chunk("Hello"),
        chat_chunk(usage=NS(prompt_tokens=4, completion_tokens=1)),
    ]
    raw = AsyncStream(chunks)

    async def original(**_):
        return raw

    stream = await patch._patch_async_openai_client_method(original, True)(
        stream=True, model="test", messages=[]
    )
    assert [chunk async for chunk in stream] == chunks
    assert observers[0].exited
    assert updates[0].output == "Hello"
    assert (updates[0].prompt_tokens, updates[0].completion_tokens) == (4, 1)


@pytest.mark.asyncio
async def test_async_stream_error_closes_span(captured):
    observers, updates = captured
    error = RuntimeError("stream interrupted")
    raw = AsyncStream(
        [NS(type="response.output_text.delta", delta="partial")], error
    )

    async def original(**_):
        return raw

    parent_span = NS(uuid="parent-span")
    span_token = current_span_context.set(parent_span)
    try:
        stream = await patch._patch_async_openai_client_method(original, False)(
            stream=True, model="test", input="question"
        )
        assert current_span_context.get() is parent_span
        assert (await stream.__anext__()).delta == "partial"
        with pytest.raises(RuntimeError, match="stream interrupted"):
            await stream.__anext__()
        assert current_span_context.get() is parent_span
        assert observers[0].error is error
        assert updates[0].output == "partial"
    finally:
        current_span_context.reset(span_token)


@pytest.mark.asyncio
async def test_async_stream_close_preserves_partial_output(captured):
    observers, updates = captured
    raw = AsyncStream([NS(type="response.output_text.delta", delta="partial")])

    async def original(**_):
        return raw

    stream = await patch._patch_async_openai_client_method(original, False)(
        stream=True, model="test", input="question"
    )
    await stream.__anext__()
    await stream.close()
    assert raw.stream.closed and observers[0].exited
    assert updates[0].output == "partial"
