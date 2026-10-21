"""Async LangGraph and LangChain runs keep their trace trees intact.

Nodes run across asyncio tasks, in parallel, nested, failing, and inside
``@observe``; every case must still produce one correctly parented trace.
"""

import asyncio
from typing import List, Optional

import pytest
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing import observe, trace_manager
from deepeval.tracing.types import LlmSpan, TraceSpanStatus
from tests.test_integrations.utils import (
    collecting_traces,
    confident_trace_instrumentation,
    walk_spans,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.filterwarnings(
        "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
    ),
]


class RaisingLLM(LLM):
    """Minimal LLM that always raises."""

    @property
    def _llm_type(self) -> str:
        return "raising-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        raise RuntimeError("llm-boom")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        raise RuntimeError("llm-boom")


class State(TypedDict, total=False):
    prompt: str
    output: str


@pytest.fixture
def traces():
    with confident_trace_instrumentation("langchain", "langgraph"):
        with collecting_traces() as collected:
            yield collected


def only_trace(traces):
    (collected,) = traces
    # Nothing stays open once the run is over.
    assert trace_manager.active_spans == {}
    assert trace_manager.active_traces == {}
    return collected


def spans_named(collected, name):
    return [
        span for span in walk_spans(collected.root_spans) if span.name == name
    ]


def llm_spans(collected):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, LlmSpan)
    ]


def single_llm_graph(llm):
    async def node(state: State, config=None) -> dict:
        return {"output": await llm.ainvoke(state["prompt"], config=config)}

    builder = StateGraph(State)
    builder.add_node("llm", node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()


async def test_async_graph_produces_one_trace(traces):
    graph = single_llm_graph(FakeListLLM(responses=["pong"]))

    result = await graph.ainvoke(
        {"prompt": "ping"}, config={"callbacks": [CallbackHandler()]}
    )

    assert result["output"] == "pong"
    collected = only_trace(traces)
    (llm,) = llm_spans(collected)
    (node,) = spans_named(collected, "llm")
    assert llm.parent_uuid == node.uuid


async def test_nested_async_calls_are_parented(traces):
    llm = FakeListLLM(responses=["pong"])

    async def outer(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    assert await RunnableLambda(outer).ainvoke("unused") == "pong"

    collected = only_trace(traces)
    (root,) = collected.root_spans
    (llm,) = llm_spans(collected)
    assert llm.parent_uuid == root.uuid


async def test_llm_errors_error_their_span_and_the_trace(traces):
    llm = RaisingLLM()

    async def outer(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    with pytest.raises(RuntimeError, match="llm-boom"):
        await RunnableLambda(outer).ainvoke("unused")

    collected = only_trace(traces)
    (llm_span,) = llm_spans(collected)
    assert llm_span.status == TraceSpanStatus.ERRORED
    assert collected.status == TraceSpanStatus.ERRORED


async def test_chain_errors_error_the_trace(traces):
    async def outer(_input, config=None):
        raise RuntimeError("chain-boom")

    with pytest.raises(RuntimeError, match="chain-boom"):
        await RunnableLambda(outer).ainvoke("unused")

    collected = only_trace(traces)
    (root,) = collected.root_spans
    assert root.status == TraceSpanStatus.ERRORED


async def test_parallel_llm_calls_share_their_parent(traces):
    llm = FakeListLLM(responses=["pong", "pong"])

    async def outer(_input, config=None):
        first, second = await asyncio.gather(
            llm.ainvoke("ping1", config=config),
            llm.ainvoke("ping2", config=config),
        )
        return first + second

    assert await RunnableLambda(outer).ainvoke("unused") == "pongpong"

    collected = only_trace(traces)
    (root,) = collected.root_spans
    assert [span.parent_uuid for span in llm_spans(collected)] == [
        root.uuid,
        root.uuid,
    ]


async def test_chain_inside_chain_then_llm_nests(traces):
    llm = FakeListLLM(responses=["pong"])

    async def inner(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    inner_runnable = RunnableLambda(inner)

    async def outer(_input, config=None):
        return await inner_runnable.ainvoke("unused-inner", config=config)

    assert await RunnableLambda(outer).ainvoke("unused") == "pong"

    collected = only_trace(traces)
    (root,) = collected.root_spans
    (nested,) = root.children
    (llm,) = llm_spans(collected)
    assert (nested.name, llm.parent_uuid) == ("inner", nested.uuid)


async def test_an_observed_async_graph_nests_under_observe(traces):
    graph = single_llm_graph(FakeListLLM(responses=["pong"]))

    @observe(type="custom", name="observed_endpoint")
    async def observed_run():
        return await graph.ainvoke(
            {"prompt": "ping"}, config={"callbacks": [CallbackHandler()]}
        )

    result = await asyncio.create_task(observed_run())

    assert result["output"] == "pong"
    collected = only_trace(traces)
    (endpoint,) = collected.root_spans
    assert endpoint.name == "observed_endpoint"
    (graph_root,) = endpoint.children
    assert graph_root.name == "LangGraph"
    assert len(llm_spans(collected)) == 1
