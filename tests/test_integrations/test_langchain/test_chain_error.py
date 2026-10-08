"""Regression tests for chain-error finalization in the LangChain
``CallbackHandler``.

LangChain dispatches ``on_chain_error`` from the ``except BaseException``
branch of ``Runnable.invoke`` — the same place it dispatches
``on_tool_error`` / ``on_llm_error``. The handler defined ``on_chain_start``
/ ``on_chain_end`` but no ``on_chain_error``, so an erroring chain only ran
the inherited protocol no-op: the span stayed in
``trace_manager.active_spans`` with status ``SUCCESS``, ``span.error`` was
never set, and because the root chain span has ``parent_uuid is None`` the
``exit_current_context`` call that runs ``trace_manager.end_trace`` (and
therefore ``post_trace``) for a bare run was never reached. Leaked nested
chain spans additionally kept ``other_active_spans`` non-empty, so an
enclosing trace never ended either.

These tests drive the real runnable surface with ``RunnableLambda`` so the
callback ordering exercised is LangChain's, not the test's.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing import trace_manager
from deepeval.tracing.types import TraceSpanStatus


@pytest.fixture(autouse=True)
def _clear_traces_between_tests():
    trace_manager.clear_traces()
    yield


class _RecordingCallbackHandler(CallbackHandler):
    """Capture the span object at ``on_chain_start`` time.

    ``exit_current_context`` drops the span from ``active_spans``, so the
    reference has to be taken before the run fails in order to inspect the
    finalization state afterwards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_spans = []

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        result = super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.chain_spans.append(span)
        return result


def _failing_runnable(message: str, name: str = "chain") -> RunnableLambda:
    def _raise(_input):
        raise ValueError(message)

    return RunnableLambda(_raise, name=name)


def _capture_posted_traces(monkeypatch):
    posted = []
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda trace: posted.append(trace)
    )
    return posted


class TestChainErrorFinalization:
    def test_erroring_root_chain_finalizes_span_and_posts_trace(
        self, monkeypatch
    ):
        posted = _capture_posted_traces(monkeypatch)
        callback = _RecordingCallbackHandler()

        with pytest.raises(ValueError, match="pinecone exploded"):
            _failing_runnable("pinecone exploded").invoke(
                {"question": "hi"}, config={"callbacks": [callback]}
            )

        assert len(callback.chain_spans) == 1
        span = callback.chain_spans[0]

        assert span.status == TraceSpanStatus.ERRORED
        assert span.error == "pinecone exploded"
        assert span.end_time is not None
        assert trace_manager.get_span_by_uuid(span.uuid) is None
        assert trace_manager.get_trace_by_uuid(span.trace_uuid) is None
        assert [trace.uuid for trace in posted] == [span.trace_uuid]
        assert posted[0].status == TraceSpanStatus.ERRORED

    def test_erroring_nested_chain_ends_the_enclosing_trace(self, monkeypatch):
        posted = _capture_posted_traces(monkeypatch)
        callback = _RecordingCallbackHandler()

        inner = _failing_runnable("inner blew up", name="inner")
        outer = RunnableLambda(
            lambda _input: inner.invoke(_input), name="outer"
        )

        with pytest.raises(ValueError, match="inner blew up"):
            outer.invoke({"question": "hi"}, config={"callbacks": [callback]})

        assert [span.name for span in callback.chain_spans] == [
            "outer",
            "inner",
        ]
        outer_span, inner_span = callback.chain_spans

        assert outer_span.trace_uuid == inner_span.trace_uuid
        assert inner_span.parent_uuid == outer_span.uuid
        for span in (outer_span, inner_span):
            assert span.status == TraceSpanStatus.ERRORED
            assert span.error == "inner blew up"
            assert trace_manager.get_span_by_uuid(span.uuid) is None
        assert trace_manager.get_trace_by_uuid(outer_span.trace_uuid) is None
        assert [trace.uuid for trace in posted] == [outer_span.trace_uuid]

    def test_successful_chain_still_finalizes_span(self, monkeypatch):
        posted = _capture_posted_traces(monkeypatch)
        callback = _RecordingCallbackHandler()

        runnable = RunnableLambda(lambda _input: "pong", name="chain")
        assert (
            runnable.invoke(
                {"question": "hi"}, config={"callbacks": [callback]}
            )
            == "pong"
        )

        assert len(callback.chain_spans) == 1
        span = callback.chain_spans[0]
        assert span.status == TraceSpanStatus.SUCCESS
        assert span.error is None
        assert span.end_time is not None
        assert trace_manager.get_span_by_uuid(span.uuid) is None
        assert [trace.uuid for trace in posted] == [span.trace_uuid]
