import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import deepeval
from deepeval.tracing import a_flush_traces, flush_traces, observe
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
    shutdown_confident_trace,
)
from deepeval.tracing.tracing import trace_manager


@pytest.fixture
def exporter():
    """confident-trace exporting into memory."""
    exporter = InMemorySpanExporter()
    shutdown_confident_trace()
    ensure_confident_trace(exporter=exporter)
    yield exporter
    shutdown_confident_trace()


@observe(name="answer")
def answer(question):
    return question.upper()


class TestFlush:
    """Flushing waits for confident-trace to export every ended span."""

    def test_returns_true_when_nothing_is_running(self):
        shutdown_confident_trace()
        assert trace_manager.flush(timeout=1.0) is True

    def test_exports_ended_spans(self, exporter):
        answer("q")
        assert trace_manager.flush(timeout=10.0) is True
        assert [span.name for span in exporter.get_finished_spans()] == [
            "answer"
        ]

    async def test_a_flush_exports_ended_spans(self, exporter):
        answer("q")
        assert await trace_manager.a_flush(timeout=10.0) is True
        assert [span.name for span in exporter.get_finished_spans()] == [
            "answer"
        ]


class TestPublicApi:
    """flush_traces / a_flush_traces delegate to the global trace manager."""

    def test_exported_from_deepeval(self):
        assert deepeval.flush_traces is flush_traces
        assert deepeval.a_flush_traces is a_flush_traces

    def test_flush_traces_forwards_timeout(self, monkeypatch):
        recorded = {}

        def fake_flush(timeout):
            recorded["timeout"] = timeout
            return True

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.flush",
            fake_flush,
            raising=True,
        )
        assert flush_traces(timeout=12.5) is True
        assert recorded["timeout"] == 12.5

    async def test_a_flush_traces_forwards_timeout(self, monkeypatch):
        recorded = {}

        async def fake_a_flush(timeout):
            recorded["timeout"] = timeout
            return False

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.a_flush",
            fake_a_flush,
            raising=True,
        )
        assert await a_flush_traces(timeout=7.5) is False
        assert recorded["timeout"] == 7.5

    def test_default_timeout_is_thirty_seconds(self):
        import inspect

        assert (
            inspect.signature(flush_traces).parameters["timeout"].default
            == 30.0
        )
        assert (
            inspect.signature(a_flush_traces).parameters["timeout"].default
            == 30.0
        )
