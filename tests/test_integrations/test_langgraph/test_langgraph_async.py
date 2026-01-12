import asyncio
import pytest
from typing import Any, List, Optional, Tuple
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing import trace_manager


class RaisingLLM(LLM):
    """Minimal LLM that always raises to trigger on_llm_error reliably."""

    @property
    def _llm_type(self) -> str:
        return "raising-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("boom")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("boom")


class RecordingCallbackHandler(CallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_runs: List[Tuple[str, Optional[str]]] = []
        self.llm_runs: List[Tuple[str, Optional[str]]] = []
        self.events: List[Tuple[str, str]] = []  # maps event name to run_id

        # mapping of langchain run_id -> DeepEval span.parent_uuid so we can validate parentage
        self.span_parents_start = {}
        self.span_parents_end = {}
        self.span_parents_error = {}

    def _record_parent_if_present(self, run_id: str, target: dict):
        span = trace_manager.get_span_by_uuid(run_id)
        if span is not None:
            target[run_id] = span.parent_uuid

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        rid = str(run_id)
        self.chain_runs.append(
            (rid, str(parent_run_id) if parent_run_id else None)
        )
        self.events.append(("chain_start", rid))

        res = super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._record_parent_if_present(rid, self.span_parents_start)
        return res

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        rid = str(run_id)
        self.events.append(("chain_end", rid))

        # Observe parent before super() exits/removes the span
        self._record_parent_if_present(rid, self.span_parents_end)
        res = super().on_chain_end(
            outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

        # After end, span should be removed from active store
        assert trace_manager.get_span_by_uuid(rid) is None
        return res

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        rid = str(run_id)
        self.events.append(("chain_error", rid))

        self._record_parent_if_present(rid, self.span_parents_error)
        res = super().on_chain_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

        assert trace_manager.get_span_by_uuid(rid) is None
        return res

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        rid = str(run_id)
        self.llm_runs.append(
            (rid, str(parent_run_id) if parent_run_id else None)
        )
        self.events.append(("llm_start", rid))

        res = super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._record_parent_if_present(rid, self.span_parents_start)
        return res

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        rid = str(run_id)
        self.events.append(("llm_end", rid))

        self._record_parent_if_present(rid, self.span_parents_end)
        res = super().on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

        assert trace_manager.get_span_by_uuid(rid) is None
        return res

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        rid = str(run_id)
        self.events.append(("llm_error", rid))

        self._record_parent_if_present(rid, self.span_parents_error)
        res = super().on_llm_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

        assert trace_manager.get_span_by_uuid(rid) is None
        return res


class State(TypedDict, total=False):
    prompt: str
    output: str


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_langgraph_async_callback_does_not_print_span_mismatch(capsys):
    """LangGraph async execution should not break the DeepEval span context stack:
    we should not print 'Current span in context does not match the span being exited'.
    """
    llm = FakeListLLM(responses=["pong"])

    async def node(state: State, config=None) -> dict:
        out = await llm.ainvoke(state["prompt"], config=config)
        return {"output": out}

    builder = StateGraph(State)
    builder.add_node("llm", node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    graph = builder.compile()

    callback = CallbackHandler(metric_collection="test_langgraph_async")

    result = await graph.ainvoke(
        {"prompt": "ping"},
        config={"callbacks": [callback]},
    )

    assert result["output"] == "pong"

    out = (
        capsys.readouterr().out
    )  # captures everything printed to stdout so far
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_nested_async_calls_are_parented_correctly_by_ids(capsys):
    """A chain that calls an LLM should report parentage consistently:
    LangChain passes parent_run_id=<chain run_id>, and DeepEval records span.parent_uuid=<chain run_id>.
    """
    llm = FakeListLLM(responses=["pong"])
    callback = RecordingCallbackHandler(
        metric_collection="test_nested_async_ids"
    )

    async def outer(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    result = await RunnableLambda(outer).ainvoke(
        "unused",
        config={"callbacks": [callback]},
    )
    assert result == "pong"

    # Symptom guard (stack mismatch)
    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    # assert that LangChain callback inputs report the expected parent_run_id relationship
    assert callback.chain_runs
    assert callback.llm_runs
    outer_run_id, _ = callback.chain_runs[0]
    llm_run_id, llm_parent = callback.llm_runs[0]
    assert (
        llm_parent == outer_run_id
    ), f"Expected LLM parent={outer_run_id}, got {llm_parent}"

    # assert that DeepEval spans created in trace_manager have the expected parent_uuid relationship
    assert (
        outer_run_id in callback.span_parents_start
    ), "Expected to observe root span in trace_manager during on_chain_start"
    assert (
        llm_run_id in callback.span_parents_start
    ), "Expected to observe llm span in trace_manager during on_llm_start"
    assert (
        callback.span_parents_start[llm_run_id] == outer_run_id
    ), f"Expected llm span.parent_uuid={outer_run_id}, got {callback.span_parents_start[llm_run_id]}"


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_llm_error_path_tracks_correct_ids_and_cleans_up(capsys):
    """If the LLM raises, we should report the error without corrupting the span stack:
    no span-mismatch print, an llm_error event is recorded, and the LLM span is removed.
    """
    llm = RaisingLLM()
    callback = RecordingCallbackHandler(
        metric_collection="test_llm_error_cleanup"
    )

    async def outer(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    with pytest.raises(RuntimeError, match="boom"):
        await RunnableLambda(outer).ainvoke(
            "unused", config={"callbacks": [callback]}
        )

    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    assert callback.llm_runs
    llm_run_id, _ = callback.llm_runs[0]

    # Span existed at start and was observed, and was cleaned on error.
    assert llm_run_id in callback.span_parents_start
    assert ("llm_error", llm_run_id) in callback.events


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_chain_error_path_cleans_up_and_no_mismatch(capsys):
    """If the outer chain raises, we should report the error without corrupting the span stack:
    no span-mismatch print, a chain_error event is recorded, and the chain span is removed.
    """

    callback = RecordingCallbackHandler(
        metric_collection="test_chain_error_cleanup"
    )

    async def outer(_input, config=None):
        raise RuntimeError("chain-boom")

    with pytest.raises(RuntimeError, match="chain-boom"):
        await RunnableLambda(outer).ainvoke(
            "unused", config={"callbacks": [callback]}
        )

    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    assert callback.chain_runs
    chain_run_id, _ = callback.chain_runs[0]
    assert ("chain_error", chain_run_id) in callback.events
    assert chain_run_id in callback.span_parents_start


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_parallel_llm_calls_under_same_parent_are_parented_correctly(
    capsys,
):
    """Two concurrent LLM calls inside one chain should share the same parent:
    LangChain passes parent_run_id=<chain run_id> for both, and DeepEval records span.parent_uuid=<chain run_id> for both.
    """

    llm = FakeListLLM(responses=["pong", "pong"])
    callback = RecordingCallbackHandler(
        metric_collection="test_parallel_llm_calls"
    )

    async def outer(_input, config=None):
        a, b = await asyncio.gather(
            llm.ainvoke("ping1", config=config),
            llm.ainvoke("ping2", config=config),
        )
        return a + b

    result = await RunnableLambda(outer).ainvoke(
        "unused",
        config={"callbacks": [callback]},
    )
    assert result == "pongpong"

    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    assert callback.chain_runs
    outer_run_id, _ = callback.chain_runs[0]

    assert (
        len(callback.llm_runs) >= 2
    ), f"Expected >=2 llm runs, got {len(callback.llm_runs)}"

    # Each llm call should be parented to the outer chain run
    for llm_run_id, llm_parent in callback.llm_runs[:2]:
        assert (
            llm_parent == outer_run_id
        ), f"Expected LLM parent={outer_run_id}, got {llm_parent}"
        assert llm_run_id in callback.span_parents_start
        assert callback.span_parents_start[llm_run_id] == outer_run_id


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_chain_inside_chain_then_llm_is_parented_correctly(capsys):
    """For nested chains, the LLM run/span should be parented to the inner chain (root -> nested -> llm)."""
    llm = FakeListLLM(responses=["pong"])
    callback = RecordingCallbackHandler(
        metric_collection="test_chain_chain_llm"
    )

    async def inner(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    inner_runnable = RunnableLambda(inner)

    async def outer(_input, config=None):
        # nested chain call
        return await inner_runnable.ainvoke("unused-inner", config=config)

    result = await RunnableLambda(outer).ainvoke(
        "unused-outer",
        config={"callbacks": [callback]},
    )
    assert result == "pong"

    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    # Identify root chain (no parent) and nested chain (parent == root)
    assert (
        len(callback.chain_runs) >= 2
    ), f"Expected >=2 chain runs, got {len(callback.chain_runs)}"

    root_chain_ids = [
        run_id for run_id, parent in callback.chain_runs if parent is None
    ]
    assert root_chain_ids, "Expected a root chain run (parent_run_id=None)"
    root_chain_id = root_chain_ids[0]

    nested_chain_ids = [
        run_id
        for run_id, parent in callback.chain_runs
        if parent == root_chain_id
    ]
    assert (
        nested_chain_ids
    ), "Expected a nested chain run parented to the root chain"
    nested_chain_id = nested_chain_ids[0]

    assert callback.llm_runs, "Expected at least one llm run"
    llm_run_id, llm_parent = callback.llm_runs[0]

    # In this structure, the LLM call should be parented to the nested chain run.
    assert (
        llm_parent == nested_chain_id
    ), f"Expected LLM parent={nested_chain_id}, got {llm_parent}"

    # DeepEval span parentage captured during starts should match as well
    assert llm_run_id in callback.span_parents_start
    assert (
        callback.span_parents_start[llm_run_id] == nested_chain_id
    ), f"Expected llm span.parent_uuid={nested_chain_id}, got {callback.span_parents_start[llm_run_id]}"


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_nested_chain_chain_llm_end_order_and_parentage(capsys):
    """For nested chains, parentage should be root -> nested -> llm, and completion should be recorded:
    the LLM and both chains should emit *_end events, and DeepEval should record span.parent_uuid consistent with that parentage.
    """
    llm = FakeListLLM(responses=["pong"])
    callback = RecordingCallbackHandler(
        metric_collection="test_nested_end_order"
    )

    async def inner(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    inner_runnable = RunnableLambda(inner)

    async def outer(_input, config=None):
        return await inner_runnable.ainvoke("unused-inner", config=config)

    result = await RunnableLambda(outer).ainvoke(
        "unused-outer", config={"callbacks": [callback]}
    )
    assert result == "pong"

    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )

    # Parentage: root chain -> nested chain -> llm
    root_chain_ids = [
        rid for rid, parent in callback.chain_runs if parent is None
    ]
    assert root_chain_ids
    root_chain_id = root_chain_ids[0]

    nested_chain_ids = [
        rid for rid, parent in callback.chain_runs if parent == root_chain_id
    ]
    assert nested_chain_ids
    nested_chain_id = nested_chain_ids[0]

    assert callback.llm_runs
    llm_run_id, llm_parent = callback.llm_runs[0]
    assert llm_parent == nested_chain_id
    assert callback.span_parents_start[llm_run_id] == nested_chain_id

    # End events happened and cleanup assertions in handler already enforced span removal
    assert ("llm_end", llm_run_id) in callback.events
    assert ("chain_end", nested_chain_id) in callback.events
    assert ("chain_end", root_chain_id) in callback.events
