"""deepeval's LangChain and LangGraph integration, built on confident-trace.

confident-trace traces every LangChain run once deepeval starts it with the
``langchain`` and ``langgraph`` instrumentations. ``CallbackHandler`` adds what
only deepeval knows: the trace's settings and metrics, and the metrics, metric
collection and prompt that a model's or retriever's ``metadata`` names.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import (
    Staged,
    stage_next_span,
    stage_next_span_type,
    stage_next_trace,
    unstage,
)
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.setup import ensure_confident_trace

try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    from deepeval.integrations.langchain.patch import tool  # noqa: F401

    langchain_installed = True
except ImportError:
    langchain_installed = False

INSTRUMENTATIONS = ("langchain", "langgraph")
# Keys in a model's ``metadata`` that configure its LLM span.
_LLM_METADATA_KEYS = ("metrics", "metric_collection", "prompt")


def is_langchain_installed():
    if not langchain_installed:
        raise ImportError(
            "LangChain is not installed. Please install it with `pip install langchain`."
        )


if langchain_installed:
    with capture_tracing_integration(Integration.LANGCHAIN):
        ensure_confident_trace(INSTRUMENTATIONS)


class CallbackHandler(BaseCallbackHandler):
    # Runs before confident-trace's own handler, in the same context, so what
    # it stages at a run's start reaches the span that handler opens.
    run_inline = True

    def __init__(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ):
        is_langchain_installed()
        with capture_tracing_integration(Integration.LANGCHAIN):
            ensure_confident_trace(INSTRUMENTATIONS)
        self._trace_fields = {
            "name": name,
            "tags": tags,
            "metadata": metadata,
            "thread_id": thread_id,
            "user_id": user_id,
            "metrics": metrics,
            "metric_collection": metric_collection,
            "test_case_id": test_case_id,
            "turn_id": turn_id,
        }
        self._staged: Dict[UUID, List[Staged]] = {}

        super().__init__()

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        # The run the user invoked is the agent; ``next_agent_span`` lands on it.
        self._start(
            run_id, parent_run_id, "agent" if parent_run_id is None else None
        )

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start(run_id, parent_run_id, "llm", _llm_fields(kwargs))

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start(run_id, parent_run_id, "llm", _llm_fields(kwargs))

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs
    ):
        self._start(run_id, parent_run_id)

    def on_retriever_start(
        self, serialized, query, *, run_id, parent_run_id=None, **kwargs
    ):
        metadata = kwargs.get("metadata") or {}
        self._start(
            run_id,
            parent_run_id,
            "retriever",
            {
                "embedder": metadata.get("ls_embedding_provider"),
                "metric_collection": metadata.get("metric_collection"),
            },
        )

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._end(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end(run_id)

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._end(run_id)

    def _start(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        span_type: Optional[str] = None,
        span_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        staged = []
        if parent_run_id is None:
            staged.append(stage_next_trace(self._trace_fields))
        if span_type == "agent":
            staged.append(stage_next_span_type(span_type))
        if span_fields:
            staged.append(stage_next_span(span_type, span_fields))
        if staged:
            self._staged[run_id] = staged

    def _end(self, run_id: UUID) -> None:
        for staged in reversed(self._staged.pop(run_id, [])):
            unstage(staged)


def _llm_fields(callback_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    metadata = callback_kwargs.get("metadata") or {}
    return {key: metadata.get(key) for key in _LLM_METADATA_KEYS}
