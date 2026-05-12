"""Unit tests for ``with next_*_span(...)`` support in the LangChain
``CallbackHandler``.

The handler was wired to call ``pop_pending_for(span_type)`` +
``apply_pending_to_span(...)`` at the start of every span it opens —
``on_chat_model_start`` / ``on_llm_start`` (llm), ``on_tool_start``
(tool), ``on_retriever_start`` (retriever) — so users can stage
metric collections, metrics, metadata, etc. on the next span the
handler creates without baking them into ``with_config(metadata=...)``.

These tests pin down the contracts that surface flips would silently
break (one-shot consumption, cross-type isolation, override of the
metadata path), exercised through the public LangChain runnable
surface with ``FakeListLLM`` so no API key / network call is needed.
"""

import asyncio
from typing import Any, List, Optional, Type
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import BaseMetric
from deepeval.tracing import (
    next_llm_span,
    next_retriever_span,
    next_span,
    next_tool_span,
    trace_manager,
)
from deepeval.tracing.types import LlmSpan, RetrieverSpan, ToolSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCallbackHandler(CallbackHandler):
    """Capture span object refs the moment they're created so tests can
    inspect them after the trace has ended.

    ``trace_manager.remove_span(...)`` clears the active-spans map at
    span end but the span object itself stays parented in the trace
    tree, so we take the reference at start (after super() applied
    pending) and assert against it post-run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_spans: List[LlmSpan] = []
        self.tool_spans: List[ToolSpan] = []
        self.retriever_spans: List[RetrieverSpan] = []

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

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        res = super().on_tool_start(
            serialized, input_str, run_id=run_id, **kwargs
        )
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.tool_spans.append(span)
        return res

    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        res = super().on_retriever_start(
            serialized, query, run_id=run_id, **kwargs
        )
        span = trace_manager.get_span_by_uuid(str(run_id))
        if span is not None:
            self.retriever_spans.append(span)
        return res


class _EchoToolInput(BaseModel):
    text: str


class _EchoTool(BaseTool):
    """Minimal tool that drives ``on_tool_start`` / ``on_tool_end`` with
    no LLM dependency."""

    name: str = "echo"
    description: str = "Echoes the input back."
    args_schema: Type[BaseModel] = _EchoToolInput

    def _run(self, text: str, **_kwargs: Any) -> str:
        return text


class _StaticRetriever(BaseRetriever):
    """Retriever returning a fixed list of docs — drives
    ``on_retriever_start`` / ``on_retriever_end`` deterministically.

    We deliberately do NOT plumb metadata through ``with_config(...)``
    on the retriever in tests below so the staged value from
    ``next_retriever_span(...)`` isn't masked by a metadata fallback.
    """

    docs: List[Document] = [Document(page_content="hello")]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return list(self.docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return list(self.docs)


def _fake_metric(name: str = "fake") -> BaseMetric:
    """A throwaway metric stand-in. The handler only stores it on the
    span — it never runs ``measure(...)`` here — so a ``MagicMock``
    typed as ``BaseMetric`` is enough to assert the wiring."""
    metric = MagicMock(spec=BaseMetric)
    metric.__name__ = name
    return metric


# ---------------------------------------------------------------------------
# next_llm_span → on_chat_model_start / on_llm_start
# ---------------------------------------------------------------------------


class TestNextLlmSpanWiring:
    """``with next_llm_span(...)`` stages defaults that get drained by
    the FIRST LLM span the handler opens inside the scope. Verifies the
    handler's ``pop_pending_for("llm")`` + ``apply_pending_to_span(...)``
    plumbing for both ``on_llm_start`` (string-prompt LLMs like
    ``FakeListLLM``) and ``on_chat_model_start`` (chat models)."""

    def test_metric_collection_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_llm_span(metric_collection="llm_quality_v1"):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert len(callback.llm_spans) == 1
        assert callback.llm_spans[0].metric_collection == "llm_quality_v1"

    def test_metrics_list_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])
        metric = _fake_metric()

        with next_llm_span(metrics=[metric]):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metrics == [metric]

    def test_metadata_lands_on_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_llm_span(metadata={"trace_phase": "warmup"}):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metadata == {"trace_phase": "warmup"}

    def test_one_shot_only_first_llm_span_consumes(self):
        """One-shot semantics: a SECOND ``llm.invoke(...)`` inside the
        same ``with`` block does NOT inherit the staged value. This is
        the "gotcha" the docs call out for ``create_agent`` /
        ``StateGraph`` loops where the tool-call retry creates a second
        LLM span — and is exactly what should happen given
        ``pop_pending_for`` drains the slot."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong-1", "pong-2"])

        with next_llm_span(metric_collection="only-first"):
            llm.invoke("ping-1", config={"callbacks": [callback]})
            llm.invoke("ping-2", config={"callbacks": [callback]})

        assert len(callback.llm_spans) == 2
        assert callback.llm_spans[0].metric_collection == "only-first"
        assert callback.llm_spans[1].metric_collection is None

    def test_unconsumed_payload_does_not_leak_to_next_with(self):
        """Token-based reset on scope exit: a payload that nobody
        popped must NOT carry into a subsequent ``with`` block."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_llm_span(metric_collection="leaked"):
            pass  # no LLM call → nothing pops

        with next_llm_span(metric_collection="fresh"):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metric_collection == "fresh"

    def test_outside_with_block_no_staging(self):
        """Sanity floor: an LLM call outside any ``next_llm_span(...)``
        leaves ``metric_collection`` / ``metrics`` / ``metadata`` at
        their natural defaults (None, since no metadata baseline is
        provided either)."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        llm.invoke("ping", config={"callbacks": [callback]})

        span = callback.llm_spans[0]
        assert span.metric_collection is None
        assert span.metrics is None
        # metadata is left untouched (no metadata baseline → None).
        assert span.metadata is None

    def test_overrides_with_config_metadata_metric_collection(self):
        """``apply_pending_to_span(...)`` runs AFTER the metadata
        baseline is read in ``on_llm_start`` (see comment in
        ``callback.py``: "more specific wins"). So a staged
        ``next_llm_span(metric_collection=...)`` MUST override
        ``with_config(metadata={"metric_collection": ...})``."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"]).with_config(
            metadata={"metric_collection": "from_metadata"}
        )

        with next_llm_span(metric_collection="from_next_span"):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metric_collection == "from_next_span"

    def test_does_not_override_metadata_when_only_metric_collection_staged(
        self,
    ):
        """Negative guard for the override path: only fields PRESENT in
        the pending payload should overwrite. ``metadata`` is left
        alone when the staging block doesn't pass it."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"]).with_config(
            metadata={
                "metric_collection": "from_metadata",
                "extra_key": "preserved",
            }
        )

        with next_llm_span(metric_collection="staged"):
            llm.invoke("ping", config={"callbacks": [callback]})

        # metric_collection got overridden, but the metadata-driven
        # baseline (which the handler does NOT copy onto span.metadata
        # in on_llm_start) is unaffected — span.metadata stays None
        # because the staging block didn't pass metadata either.
        assert callback.llm_spans[0].metric_collection == "staged"
        assert callback.llm_spans[0].metadata is None


# ---------------------------------------------------------------------------
# next_tool_span → on_tool_start
# ---------------------------------------------------------------------------


class TestNextToolSpanWiring:
    def test_metric_collection_lands_on_tool_span(self):
        callback = _RecordingCallbackHandler()
        tool = _EchoTool()

        with next_tool_span(metric_collection="tool_quality_v1"):
            tool.invoke({"text": "hi"}, config={"callbacks": [callback]})

        assert len(callback.tool_spans) == 1
        assert callback.tool_spans[0].metric_collection == "tool_quality_v1"

    def test_metadata_lands_on_tool_span(self):
        callback = _RecordingCallbackHandler()
        tool = _EchoTool()

        with next_tool_span(metadata={"layer": "outer"}):
            tool.invoke({"text": "hi"}, config={"callbacks": [callback]})

        assert callback.tool_spans[0].metadata == {"layer": "outer"}

    def test_one_shot_only_first_tool_span_consumes(self):
        callback = _RecordingCallbackHandler()
        tool = _EchoTool()

        with next_tool_span(metric_collection="only-first-tool"):
            tool.invoke({"text": "hi-1"}, config={"callbacks": [callback]})
            tool.invoke({"text": "hi-2"}, config={"callbacks": [callback]})

        assert len(callback.tool_spans) == 2
        assert callback.tool_spans[0].metric_collection == "only-first-tool"
        assert callback.tool_spans[1].metric_collection is None


# ---------------------------------------------------------------------------
# next_retriever_span → on_retriever_start
# ---------------------------------------------------------------------------


class TestNextRetrieverSpanWiring:
    def test_metric_collection_lands_on_retriever_span(self):
        callback = _RecordingCallbackHandler()
        retriever = _StaticRetriever()

        with next_retriever_span(metric_collection="retriever_quality_v1"):
            retriever.invoke("query", config={"callbacks": [callback]})

        assert len(callback.retriever_spans) == 1
        assert (
            callback.retriever_spans[0].metric_collection
            == "retriever_quality_v1"
        )

    def test_top_k_and_embedder_land_on_retriever_span(self):
        """Retriever-specific kwargs flow through
        ``apply_pending_to_span(...)`` because the popped dict is
        setattr'd onto a ``RetrieverSpan`` placeholder which declares
        ``embedder`` / ``top_k`` / ``chunk_size``."""
        callback = _RecordingCallbackHandler()
        retriever = _StaticRetriever()

        with next_retriever_span(top_k=5, embedder="text-embedding-3-small"):
            retriever.invoke("query", config={"callbacks": [callback]})

        span = callback.retriever_spans[0]
        assert span.top_k == 5
        assert span.embedder == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Cross-type isolation between typed slots
# ---------------------------------------------------------------------------


class TestCrossTypeIsolation:
    """Each typed slot is independent. The handler pops only the slot
    matching the span it's about to open, so staging one type never
    leaks onto a different span type opened in the same scope."""

    def test_next_tool_span_does_not_leak_to_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_tool_span(metric_collection="tool-only"):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metric_collection is None

    def test_next_llm_span_does_not_leak_to_tool_span(self):
        callback = _RecordingCallbackHandler()
        tool = _EchoTool()

        with next_llm_span(metric_collection="llm-only"):
            tool.invoke({"text": "hi"}, config={"callbacks": [callback]})

        assert callback.tool_spans[0].metric_collection is None


# ---------------------------------------------------------------------------
# Base ``next_span(...)`` slot
# ---------------------------------------------------------------------------


class TestNextSpanBaseSlotWiring:
    """``next_span(...)`` sets defaults for the FIRST span of any type.
    Verifies the base slot also flows through the handler's
    ``pop_pending_for(...)`` call (which merges base + typed slots
    before applying)."""

    def test_base_slot_lands_on_first_llm_span(self):
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_span(metric_collection="from_base"):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metric_collection == "from_base"

    def test_typed_slot_overrides_base_slot_on_overlap(self):
        """When both ``next_span`` and ``next_llm_span`` set the same
        key, the typed slot wins (more specific > base)."""
        callback = _RecordingCallbackHandler()
        llm = FakeListLLM(responses=["pong"])

        with next_span(metric_collection="from_base"), next_llm_span(
            metric_collection="from_typed"
        ):
            llm.invoke("ping", config={"callbacks": [callback]})

        assert callback.llm_spans[0].metric_collection == "from_typed"


# ---------------------------------------------------------------------------
# Async path — the handler's pop happens inside the same async task
# as the runnable, so ``ainvoke`` must behave like ``invoke``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_next_llm_span_lands_on_async_llm_call():
    """``await llm.ainvoke(...)`` exercises the async callback path. The
    pending slot still pops because ``with next_llm_span(...)`` propagates
    via contextvars into the async task created by ``ainvoke``."""
    callback = _RecordingCallbackHandler()
    llm = FakeListLLM(responses=["pong"])

    with next_llm_span(metric_collection="async_v1"):
        await llm.ainvoke("ping", config={"callbacks": [callback]})

    assert len(callback.llm_spans) == 1
    assert callback.llm_spans[0].metric_collection == "async_v1"


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_next_llm_span_lands_inside_runnable_lambda_async():
    """Stage outside, invoke a ``RunnableLambda`` that calls the LLM
    inside its async body — verifies the ContextVar carries through
    LangChain's task-spawning machinery to the LLM callback."""
    callback = _RecordingCallbackHandler()
    llm = FakeListLLM(responses=["pong"])

    async def call_llm(_input, config=None):
        return await llm.ainvoke("ping", config=config)

    with next_llm_span(metric_collection="lambda_async_v1"):
        await RunnableLambda(call_llm).ainvoke(
            "unused", config={"callbacks": [callback]}
        )

    assert callback.llm_spans[0].metric_collection == "lambda_async_v1"
