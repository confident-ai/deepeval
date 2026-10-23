"""Unit tests for the patched ``@tool`` decorator
(``deepeval.integrations.langchain.patch.tool``).

The wrapper the patch installs around the user function used to write
``current_span.metrics`` / ``current_span.metric_collection``
unconditionally, which broke two cases:

1. With no active deepeval span — a direct ``tool.invoke(...)`` in a
   unit test, or an agent run without the ``CallbackHandler``
   attached — ``current_span_context.get()`` is None and the tool
   call itself crashed with ``AttributeError: 'NoneType' object has
   no attribute 'metrics'``.
2. With the handler attached, a ``@tool()`` decorated without
   metrics wrote ``metrics=None`` over metrics staged on the tool
   span via ``with next_tool_span(metrics=[...])`` — the handler
   applies pending values at ``on_tool_start``, the wrapper then
   wiped them.

These tests pin both regressions, plus the intended behavior the fix
must keep: decorator-supplied metrics / metric collections still land
on the tool span. Everything is exercised through the public runnable
surface with no LLM dependency.
"""

from typing import List
from unittest.mock import MagicMock

from deepeval.integrations.langchain import CallbackHandler, tool
from deepeval.metrics import BaseMetric
from deepeval.tracing import next_tool_span, trace_manager
from deepeval.tracing.types import ToolSpan


class _RecordingCallbackHandler(CallbackHandler):
    """Capture the tool span ref the moment it's created so tests can
    inspect it after the run (``trace_manager.remove_span(...)`` clears
    the active-spans map at span end)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_spans: List[ToolSpan] = []

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        res = super().on_tool_start(
            serialized, input_str, run_id=run_id, **kwargs
        )
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.tool_spans.append(span)
        return res


def _fake_metric() -> BaseMetric:
    """A throwaway metric stand-in. The handler only stores it on the
    span — it never runs ``measure(...)`` here — so a ``MagicMock``
    typed as ``BaseMetric`` is enough to assert the wiring."""
    return MagicMock(spec=BaseMetric)


@tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestPatchedToolWithoutActiveSpan:
    def test_invoke_without_active_span_does_not_crash(self):
        """``langchain_core.tools.tool`` supports bare invocation, so
        the patched decorator must too: with no CallbackHandler and no
        active span the tool runs and returns its result instead of
        raising AttributeError on the missing span."""
        assert add.invoke({"a": 1, "b": 2}) == 3


class TestPatchedToolWithActiveSpan:
    def test_decorator_metrics_land_on_tool_span(self):
        """Intended behavior to preserve: metrics / metric_collection
        passed to the decorator attach to the tool span the handler
        opens around the call."""
        callback = _RecordingCallbackHandler()
        metric = _fake_metric()

        @tool(metrics=[metric], metric_collection="decorator_collection")
        def shout(text: str) -> str:
            """Return the text uppercased."""
            return text.upper()

        shout.invoke({"text": "hi"}, config={"callbacks": [callback]})

        assert len(callback.tool_spans) == 1
        assert callback.tool_spans[0].metrics == [metric]
        assert (
            callback.tool_spans[0].metric_collection == "decorator_collection"
        )

    def test_tool_without_metrics_keeps_staged_metrics(self):
        """A ``@tool()`` decorated without metrics must not wipe
        metrics / metric_collection staged on the tool span via
        ``with next_tool_span(...)``: the handler applies the pending
        values at ``on_tool_start``, the wrapper runs after."""
        callback = _RecordingCallbackHandler()
        metric = _fake_metric()

        @tool()
        def echo(text: str) -> str:
            """Echo the text back."""
            return text

        with next_tool_span(
            metrics=[metric], metric_collection="staged_collection"
        ):
            echo.invoke({"text": "hi"}, config={"callbacks": [callback]})

        assert len(callback.tool_spans) == 1
        assert callback.tool_spans[0].metrics == [metric]
        assert callback.tool_spans[0].metric_collection == "staged_collection"
