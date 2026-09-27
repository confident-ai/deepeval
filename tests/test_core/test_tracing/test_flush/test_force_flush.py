import pytest

from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.otel.exporter import ConfidentSpanExporter


@pytest.fixture
def exporter() -> ConfidentSpanExporter:
    """Exporter without __init__ so no telemetry or clock bridge is touched."""
    return ConfidentSpanExporter.__new__(ConfidentSpanExporter)


class TestConfidentSpanExporterForceFlush:
    """force_flush waits on the trace worker instead of always returning True."""

    def test_converts_millis_to_seconds(self, exporter, monkeypatch):
        recorded = {}

        def fake_flush(timeout):
            recorded["timeout"] = timeout
            return True

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.flush",
            fake_flush,
            raising=True,
        )
        assert exporter.force_flush(timeout_millis=2500) is True
        assert recorded["timeout"] == 2.5

    def test_uses_thirty_second_default(self, exporter, monkeypatch):
        recorded = {}

        def fake_flush(timeout):
            recorded["timeout"] = timeout
            return True

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.flush",
            fake_flush,
            raising=True,
        )
        exporter.force_flush()
        assert recorded["timeout"] == 30.0

    def test_propagates_timeout_failure(self, exporter, monkeypatch):
        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.flush",
            lambda timeout: False,
            raising=True,
        )
        assert exporter.force_flush(timeout_millis=100) is False

    def test_returns_true_when_nothing_is_outstanding(self, exporter):
        assert exporter.force_flush(timeout_millis=1000) is True


class _FakeProcessor:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls = []

    def force_flush(self, timeout_millis):
        self.calls.append(timeout_millis)
        return self.result


@pytest.fixture
def processor():
    """ContextAwareSpanProcessor with both transports stubbed out.

    Built via __new__ because __init__ requires the OTLP HTTP exporter, which
    is only installed for the integrations test group.
    """
    instance = ContextAwareSpanProcessor.__new__(ContextAwareSpanProcessor)
    instance._rest_processor = _FakeProcessor()
    instance._rest_exporter = _FakeProcessor()
    instance._otlp_processor = _FakeProcessor()
    return instance


class TestContextAwareProcessorForceFlush:
    """The REST exporter is drained even though SimpleSpanProcessor is a no-op."""

    def test_forwards_to_rest_exporter(self, processor):
        assert processor.force_flush(timeout_millis=5000) is True
        assert processor._rest_exporter.calls == [5000]

    def test_forwards_to_both_transports(self, processor):
        processor.force_flush(timeout_millis=5000)
        assert processor._rest_processor.calls == [5000]
        assert processor._otlp_processor.calls == [5000]

    def test_returns_false_when_rest_exporter_times_out(self, processor):
        processor._rest_exporter.result = False
        assert processor.force_flush(timeout_millis=5000) is False

    def test_returns_false_when_otlp_times_out(self, processor):
        processor._otlp_processor.result = False
        assert processor.force_flush(timeout_millis=5000) is False

    def test_rest_exporter_failure_does_not_block_otlp(self, processor):
        def raise_error(timeout_millis):
            raise RuntimeError("boom")

        processor._rest_exporter.force_flush = raise_error
        assert processor.force_flush(timeout_millis=5000) is False
        assert processor._otlp_processor.calls == [5000]
