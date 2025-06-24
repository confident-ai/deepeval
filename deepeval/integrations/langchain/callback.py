from typing import Any, Dict, Optional
from uuid import UUID
from time import perf_counter

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from deepeval.tracing import trace_manager
from deepeval.tracing.attributes import LlmAttributes
from deepeval.tracing.types import LlmSpan, TraceSpanStatus

class CallbackHandler(BaseCallbackHandler):

    active_trace_id: Optional[str] = None
    llm_span_dict: Dict[str, LlmSpan] = {}

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

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            for span in self.llm_span_dict.values():
                trace_manager.add_span_to_trace(span)

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