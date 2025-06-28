from typing import Any, Optional
from uuid import UUID
from time import perf_counter
from deepeval.tracing.attributes import LlmAttributes, RetrieverAttributes

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    langchain_installed = True
except:
    langchain_installed = False


def is_langchain_installed():
    if not langchain_installed:
        raise ImportError(
            "LangChain is not installed. Please install it with `pip install langchain`."
        )


# ASSUMPTIONS:
# cycle for a single invoke call
# one trace per cycle

from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    TraceSpanStatus,
    ToolSpan,
)
from deepeval.telemetry import capture_tracing_integration


class CallbackHandler(BaseCallbackHandler):

    def __init__(self):
        capture_tracing_integration("langchain")
        is_langchain_installed()
        super().__init__()

    active_trace_id: Optional[str] = None

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:

        if parent_run_id is None:
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

        base_span = BaseSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name="langchain_chain_span_" + str(run_id),
            input=inputs,
        )
        trace_manager.add_span(base_span)
        trace_manager.add_span_to_trace(base_span)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        base_span = trace_manager.get_span_by_uuid(str(run_id))

        if base_span is None:
            return

        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS
        base_span.output = outputs
        trace_manager.remove_span(str(run_id))

        if parent_run_id is None:
            current_trace = trace_manager.get_trace_by_uuid(
                self.active_trace_id
            )
            if current_trace is not None:
                current_trace.input = base_span.input
                current_trace.output = base_span.output
            trace_manager.end_trace(self.active_trace_id)
            self.active_trace_id = None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

        # prepare input
        input = "\n".join(prompts) if prompts else ""

        llm_span = LlmSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name="langchain_llm_span_" + str(run_id),
            # TODO: why model is coming unknown?
            model=serialized.get("model_name", "unknown"),
            attributes=LlmAttributes(input=input, output=""),
        )
        trace_manager.add_span(llm_span)
        trace_manager.add_span_to_trace(llm_span)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        llm_span = trace_manager.get_span_by_uuid(str(run_id))
        if llm_span is None:
            return

        # prepare output
        response_str = ""
        for generation in response.generations:
            for gen in generation:
                response_str += gen.text + "\n"

        llm_span.end_time = perf_counter()
        llm_span.status = TraceSpanStatus.SUCCESS
        llm_span.set_attributes(
            LlmAttributes(input=llm_span.attributes.input, output=response_str)
        )
        trace_manager.remove_span(str(run_id))

        if parent_run_id is None:
            current_trace = trace_manager.get_trace_by_uuid(
                self.active_trace_id
            )
            if current_trace is not None:
                current_trace.input = llm_span.input
                current_trace.output = llm_span.output
            trace_manager.end_trace(self.active_trace_id)
            self.active_trace_id = None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:

        if parent_run_id is None:
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

        tool_span = ToolSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name="langchain_tool_span_" + str(run_id),
            input=input_str,
        )
        trace_manager.add_span(tool_span)
        trace_manager.add_span_to_trace(tool_span)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        tool_span = trace_manager.get_span_by_uuid(str(run_id))

        if tool_span is None:
            return

        tool_span.end_time = perf_counter()
        tool_span.status = TraceSpanStatus.SUCCESS
        tool_span.output = output
        trace_manager.remove_span(str(run_id))

        if parent_run_id is None:
            current_trace = trace_manager.get_trace_by_uuid(
                self.active_trace_id
            )
            if current_trace is not None:
                current_trace.input = tool_span.input
                current_trace.output = tool_span.output
            trace_manager.end_trace(self.active_trace_id)
            self.active_trace_id = None

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:

        if parent_run_id is None:
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

        retriever_span = RetrieverSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name="langchain_retriever_span_" + str(run_id),
            embedder=metadata.get("ls_embedding_provider", "unknown"),
        )
        retriever_span.set_attributes(
            RetrieverAttributes(embedding_input=query, retrieval_context=[])
        )
        trace_manager.add_span(retriever_span)
        trace_manager.add_span_to_trace(retriever_span)

    def on_retriever_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        retriever_span = trace_manager.get_span_by_uuid(str(run_id))

        if retriever_span is None:
            return

        # prepare output
        output_list = []
        if isinstance(output, list):
            for item in output:
                output_list.append(str(item))
        else:
            output_list.append(str(output))

        retriever_span.end_time = perf_counter()
        retriever_span.status = TraceSpanStatus.SUCCESS
        retriever_span.set_attributes(
            RetrieverAttributes(
                embedding_input=retriever_span.attributes.embedding_input,
                retrieval_context=output_list,
            )
        )
        trace_manager.remove_span(str(run_id))

        if parent_run_id is None:
            current_trace = trace_manager.get_trace_by_uuid(
                self.active_trace_id
            )
            if current_trace is not None:
                current_trace.input = retriever_span.input
                current_trace.output = retriever_span.output
            trace_manager.end_trace(self.active_trace_id)
            self.active_trace_id = None
