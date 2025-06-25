from typing import Any, Dict, Optional
from uuid import UUID
from time import perf_counter

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from deepeval.tracing import trace_manager
from deepeval.tracing.attributes import LlmAttributes
from deepeval.tracing.types import BaseSpan, LlmSpan, RetrieverSpan, TraceSpanStatus, ToolSpan

class CallbackHandler(BaseCallbackHandler):

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
            output=None,
            error=None,
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
        
        # get the span
        base_span = trace_manager.get_span_by_uuid(str(run_id))
        
        if base_span is None:
            return
        
        # update the end time
        base_span.end_time = perf_counter()
        
        # update the status
        base_span.status = TraceSpanStatus.SUCCESS
        
        # update the attributes
        base_span.output = outputs

        # remove the span
        trace_manager.remove_span(str(run_id))

        # TODO: figure out how to add children to the span

        # assuming that the chain with no parent is the root span
        if parent_run_id is None:

            # update the trace input and output
            current_trace = trace_manager.get_trace_by_uuid(self.active_trace_id)
            if current_trace is not None:
                current_trace.input = base_span.input
                current_trace.output = base_span.output

            # end the trace
            trace_manager.end_trace(self.active_trace_id)

            # reset the active trace id
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
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid
            
        prompt_text = "\n".join(prompts) if prompts else ""
        self.llm_span_dict[str(run_id)] = LlmSpan(
            name="langchain_llm_span" + str(run_id),
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=None,
            start_time=perf_counter(),
            model=serialized.get("model_name", "unknown"),
            attributes=LlmAttributes(input=prompt_text, output=""),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        llm_span = self.llm_span_dict.get(str(run_id))
        
        if llm_span is None:
            return
        # prepare response str
        response_str = ""
        for generation in response.generations:
            for gen in generation:
                response_str += gen.text + "\n"

        if llm_span:
            llm_span.end_time = perf_counter()
            llm_span.status = TraceSpanStatus.SUCCESS
            llm_span.attributes.output = response_str
            self.llm_span_dict[str(run_id)] = llm_span

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
        
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid

        self.tool_span_dict[str(run_id)] = ToolSpan(
            name="langchain_tool_span" + str(run_id),
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=None,
            start_time=perf_counter(),
            input=input_str
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        
        tool_span = self.tool_span_dict.get(str(run_id))
        
        if tool_span is None:
            return
        
        tool_span.end_time = perf_counter()
        tool_span.status = TraceSpanStatus.SUCCESS
        tool_span.output = output
        self.tool_span_dict[str(run_id)] = tool_span

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
        
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid
        
        self.retriever_span_dict[str(run_id)] = RetrieverSpan(
            name="langchain_retriever_span" + str(run_id),
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=None,
            start_time=perf_counter(),
            embedder=metadata.get("ls_embedding_provider", "unknown"),
            input=query, 
        )

    def on_retriever_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        
        retriever_span = self.retriever_span_dict.get(str(run_id))
        
        if retriever_span is None:
            return
        
        output_list = []
        if isinstance(output, list):
            for item in output:
                output_list.append(str(item))
        else:
            output_list.append(str(output))
        
        retriever_span.end_time = perf_counter()
        retriever_span.status = TraceSpanStatus.SUCCESS
        retriever_span.output = output_list
        self.retriever_span_dict[str(run_id)] = retriever_span