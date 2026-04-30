import json
from unittest.mock import MagicMock

from deepeval.integrations.pydantic_ai.instrumentator import SpanInterceptor
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.trace_context import trace


def _make_mock_span():
    """Return a mock span that records set_attribute calls."""
    span = MagicMock()
    span.attributes = {}
    span.set_attribute.side_effect = lambda k, v: span.attributes.__setitem__(
        k, v
    )
    span.get_span_context.return_value = MagicMock(trace_id=0)
    span.parent = None
    return span


def _make_settings(**kwargs):
    """Return a minimal mock ConfidentInstrumentationSettings."""
    settings = MagicMock()
    settings.thread_id = kwargs.get("thread_id", None)
    settings.name = kwargs.get("name", None)
    settings.metadata = kwargs.get("metadata", None)
    settings.user_id = kwargs.get("user_id", None)
    settings.tags = kwargs.get("tags", None)
    settings.metric_collection = kwargs.get("metric_collection", None)
    settings.environment = kwargs.get("environment", None)
    settings.trace_metric_collection = kwargs.get(
        "trace_metric_collection", None
    )
    settings.confident_prompt = kwargs.get("confident_prompt", None)
    settings.llm_metric_collection = kwargs.get("llm_metric_collection", None)
    settings.agent_metric_collection = kwargs.get(
        "agent_metric_collection", None
    )
    settings.tool_metric_collection_map = kwargs.get(
        "tool_metric_collection_map", {}
    )
    settings.is_test_mode = False
    return settings


class TestSpanInterceptorOnStart:
    def test_uses_settings_when_no_trace_context(self):
        """Falls back to self.settings when current_trace_context is None."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings(
                thread_id="settings-thread",
                name="settings-name",
                metadata={"source": "settings"},
            )
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)

            assert (
                span.attributes.get("confident.trace.thread_id")
                == "settings-thread"
            )
            assert (
                span.attributes.get("confident.trace.name") == "settings-name"
            )
            assert json.loads(span.attributes["confident.trace.metadata"]) == {
                "source": "settings"
            }
        finally:
            current_trace_context.reset(token)

    def test_prefers_trace_context_over_settings_for_scalars(self):
        """thread_id and name from current_trace_context override self.settings."""
        settings = _make_settings(
            thread_id="settings-thread",
            name="settings-name",
            metadata={"settings_key": "settings_val"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(
            thread_id="ctx-thread",
            name="ctx-name",
            metadata={"ctx_key": "ctx_val"},
        ):
            interceptor.on_start(span, None)

        assert span.attributes.get("confident.trace.thread_id") == "ctx-thread"
        assert span.attributes.get("confident.trace.name") == "ctx-name"

    def test_metadata_is_merged(self):
        """metadata from settings and current_trace_context are merged; context wins on conflict."""
        settings = _make_settings(
            metadata={"base_key": "base_val", "shared_key": "from_settings"},
        )
        interceptor = SpanInterceptor(settings)
        span = _make_mock_span()

        with trace(metadata={"ctx_key": "ctx_val", "shared_key": "from_ctx"}):
            interceptor.on_start(span, None)

        result = json.loads(span.attributes["confident.trace.metadata"])
        assert result["base_key"] == "base_val"
        assert result["ctx_key"] == "ctx_val"
        assert result["shared_key"] == "from_ctx"  # context wins

    def test_trace_context_is_populated_inside_block(self):
        """current_trace_context is set inside the with trace() block."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            with trace(thread_id="ctx-thread", name="ctx-name"):
                assert current_trace_context.get() is not None
                interceptor.on_start(span, None)

            assert (
                span.attributes.get("confident.trace.thread_id") == "ctx-thread"
            )
            assert span.attributes.get("confident.trace.name") == "ctx-name"
        finally:
            current_trace_context.reset(token)

    def test_no_attributes_set_when_all_none(self):
        """No trace attributes are stamped when both context and settings are empty."""
        token = current_trace_context.set(None)
        try:
            settings = _make_settings()
            interceptor = SpanInterceptor(settings)
            span = _make_mock_span()

            interceptor.on_start(span, None)

            assert "confident.trace.thread_id" not in span.attributes
            assert "confident.trace.name" not in span.attributes
            assert "confident.trace.metadata" not in span.attributes
        finally:
            current_trace_context.reset(token)
