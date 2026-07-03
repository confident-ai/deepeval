"""Unit tests for ``with next_*_span(...)`` support exercised through
LangGraph ``StateGraph`` execution.

LangGraph reuses the LangChain ``CallbackHandler`` (one shared
codepath), so the underlying ``pop_pending_for(...)`` +
``apply_pending_to_span(...)`` plumbing is the same as in
``test_langchain/test_next_span.py``. What's distinct here is the
LangGraph orchestration surface: nodes scheduled across asyncio tasks,
multi-node graphs that fire the LLM callback more than once per
``ainvoke``, and the conditional-edge / multi-step flow where the
"first LLM span only" one-shot rule is the surprising behavior users
need a regression guard for.
"""

from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import BaseMetric
from deepeval.tracing import (
    next_llm_span,
    next_span,
    next_tool_span,
    trace_manager,
)
from deepeval.tracing.types import LlmSpan, ToolSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCallbackHandler(CallbackHandler):
    """Capture span object refs at start so tests can assert against
    them after ``graph.ainvoke(...)`` (the trace ends and
    ``trace_manager.active_spans`` clears, but span objects stay
    attached to the trace tree)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_spans: List[LlmSpan] = []
        self.tool_spans: List[ToolSpan] = []

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        res = super().on_chat_model_start(
            serialized, messages, run_id=run_id, **kwargs
        )
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.llm_spans.append(span)
        return res

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        res = super().on_llm_start(serialized, prompts, run_id=run_id, **kwargs)
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.llm_spans.append(span)
        return res


class _State(TypedDict, total=False):
    prompt: str
    output: str


def _fake_metric(name: str = "fake") -> BaseMetric:
    metric = MagicMock(spec=BaseMetric)
    metric.__name__ = name
    return metric


def _build_single_llm_graph(llm: FakeListLLM):
    """Smallest meaningful graph: START → llm node → END. The node
    invokes ``llm`` so the handler sees one chain call + one LLM call
    per ``graph.ainvoke``."""

    async def node(state: _State, config=None) -> dict:
        out = await llm.ainvoke(state["prompt"], config=config)
        return {"output": out}

    builder = StateGraph(_State)
    builder.add_node("llm", node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()


def _build_two_llm_graph(llm: FakeListLLM):
    """Two LLM nodes back-to-back so we can pin down the "first LLM
    span only" one-shot semantics that bites ``create_agent`` /
    multi-step graphs in real workloads."""

    async def first(state: _State, config=None) -> dict:
        out = await llm.ainvoke(state["prompt"], config=config)
        return {"output": out}

    async def second(state: _State, config=None) -> dict:
        out = await llm.ainvoke(state["output"], config=config)
        return {"output": out}

    builder = StateGraph(_State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# next_llm_span via StateGraph nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
class TestNextLlmSpanInStateGraph:
    async def test_metric_collection_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])
        graph = _build_single_llm_graph(llm)

        with next_llm_span(metric_collection="graph_llm_v1"):
            await graph.ainvoke(
                {"prompt": "ping"}, config={"callbacks": [callback]}
            )

        assert len(callback.llm_spans) == 1
        assert callback.llm_spans[0].metric_collection == "graph_llm_v1"

    async def test_metrics_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])
        graph = _build_single_llm_graph(llm)
        metric = _fake_metric()

        with next_llm_span(metrics=[metric]):
            await graph.ainvoke(
                {"prompt": "ping"}, config={"callbacks": [callback]}
            )

        assert callback.llm_spans[0].metrics == [metric]

    async def test_metadata_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])
        graph = _build_single_llm_graph(llm)

        with next_llm_span(metadata={"node": "llm"}):
            await graph.ainvoke(
                {"prompt": "ping"}, config={"callbacks": [callback]}
            )

        assert callback.llm_spans[0].metadata == {"node": "llm"}

    async def test_only_first_llm_span_in_multi_node_graph(self):
        """The "create_agent gotcha" — a graph that opens two LLM spans
        in one ``ainvoke`` only stamps the FIRST one. This is what the
        docs caution-block warns about for ``StateGraph`` /
        ``create_agent`` loops; pin it down so a future change to drain
        order doesn't silently flip the contract."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong-1", "pong-2"])
        graph = _build_two_llm_graph(llm)

        with next_llm_span(metric_collection="only-first-node"):
            await graph.ainvoke(
                {"prompt": "ping"}, config={"callbacks": [callback]}
            )

        assert len(callback.llm_spans) == 2
        assert callback.llm_spans[0].metric_collection == "only-first-node"
        assert callback.llm_spans[1].metric_collection is None

    async def test_unconsumed_payload_does_not_leak_across_invocations(
        self,
    ):
        """Token-based reset: a ``with`` that never opens an LLM span
        (because we don't invoke the graph) doesn't pollute the next
        graph invocation."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])
        graph = _build_single_llm_graph(llm)

        with next_llm_span(metric_collection="leaked"):
            pass  # no ainvoke → nothing pops

        with next_llm_span(metric_collection="fresh"):
            await graph.ainvoke(
                {"prompt": "ping"}, config={"callbacks": [callback]}
            )

        assert callback.llm_spans[0].metric_collection == "fresh"


# ---------------------------------------------------------------------------
# Cross-type isolation in graph context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_next_tool_span_does_not_leak_to_llm_span_in_graph():
    """The handler pops only the slot matching the span type it's
    opening; staging a tool default and then opening an LLM span
    leaves the LLM span clean."""
    callback = _RecordingCallbackHandler()
    llm = FakeListLLM(responses=["pong"])
    graph = _build_single_llm_graph(llm)

    with next_tool_span(metric_collection="tool-only"):
        await graph.ainvoke(
            {"prompt": "ping"}, config={"callbacks": [callback]}
        )

    assert callback.llm_spans[0].metric_collection is None


# ---------------------------------------------------------------------------
# Base ``next_span`` slot via StateGraph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_base_next_span_lands_on_first_llm_span_in_graph():
    """``next_span(...)`` is "next of any type" — base slot also
    plumbs through the handler's ``pop_pending_for(...)`` merge for
    LLM spans inside a ``StateGraph`` node."""
    callback = _RecordingCallbackHandler()
    llm = FakeListLLM(responses=["pong"])
    graph = _build_single_llm_graph(llm)

    with next_span(metric_collection="from_base_in_graph"):
        await graph.ainvoke(
            {"prompt": "ping"}, config={"callbacks": [callback]}
        )

    assert callback.llm_spans[0].metric_collection == "from_base_in_graph"


# ---------------------------------------------------------------------------
# Sync StateGraph: typically users go async, but the same wiring must
# hold under ``graph.invoke(...)`` since the handler is the same code
# path.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
def test_next_llm_span_in_sync_state_graph():
    callback = _RecordingCallbackHandler()
    llm = FakeListLLM(responses=["pong"])

    def node(state: _State, config=None) -> dict:
        out = llm.invoke(state["prompt"], config=config)
        return {"output": out}

    builder = StateGraph(_State)
    builder.add_node("llm", node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    graph = builder.compile()

    with next_llm_span(metric_collection="sync_graph_v1"):
        graph.invoke({"prompt": "ping"}, config={"callbacks": [callback]})

    assert len(callback.llm_spans) == 1
    assert callback.llm_spans[0].metric_collection == "sync_graph_v1"
