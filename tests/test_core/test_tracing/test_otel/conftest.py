import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import EvalMode, EvalSession

from .helpers import MEASURED

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini-2024-07-18",
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}
CHOICES = {
    "tool_call": {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"city": "Macau"}',
                    },
                }
            ],
        },
    },
    "text": {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "sunny"},
    },
}


@pytest.fixture
def openai_client():
    """Build an OpenAI client whose every call returns the named choice."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    def build(choice, asynchronous=False):
        body = {**COMPLETION, "choices": [CHOICES[choice]]}
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json=body)
        )
        if asynchronous:
            return openai.AsyncOpenAI(
                api_key="test",
                http_client=httpx.AsyncClient(transport=transport),
            )
        return openai.OpenAI(
            api_key="test", http_client=httpx.Client(transport=transport)
        )

    return build


@pytest.fixture
def provider():
    return TracerProvider(shutdown_on_exit=False)


@pytest.fixture
def exporter(provider):
    """A fresh confident-trace that exports to memory, started by deepeval."""
    pytest.importorskip("confident_trace")
    from deepeval.tracing.otel.setup import (
        ensure_confident_trace,
        shutdown_confident_trace,
    )

    exporter = InMemorySpanExporter()
    # Importing an integration elsewhere starts confident-trace for good.
    shutdown_confident_trace()
    ensure_confident_trace(exporter=exporter, tracer_provider=provider)
    MEASURED.clear()
    yield exporter
    shutdown_confident_trace()
    trace_manager.clear_traces()
    trace_manager.eval_session = EvalSession()


@pytest.fixture
def tracer(provider, exporter):
    return provider.get_tracer("test")


@pytest.fixture
def posted(monkeypatch):
    traces = []
    monkeypatch.setattr(trace_manager, "post_trace", traces.append)
    return traces


@pytest.fixture
def async_evaluation():
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    return trace_manager.eval_session
