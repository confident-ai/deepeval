from typing import Any, Optional
from uuid import UUID
from time import perf_counter
from deepeval.tracing.attributes import LlmAttributes, RetrieverAttributes

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.outputs import ChatGeneration

    # contains langchain imports
    from deepeval.integrations.langchain.utils import (
        parse_prompts_to_messages,
        convert_chat_generation_to_string,
        prepare_dict,
        extract_token_usage,
        extract_name,
    )

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
        capture_tracing_integration(
            "deepeval.integrations.langchain.callback.CallbackHandler"
        )
        is_langchain_installed()
        super().__init__()

    active_trace_id: Optional[str] = None

    def check_active_trace_id(self):
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid

    def add_span_to_trace(self, span: BaseSpan):
        trace_manager.add_span(span)
        trace_manager.add_span_to_trace(span)

    def end_span(self, span: BaseSpan):
        span.end_time = perf_counter()
        span.status = TraceSpanStatus.SUCCESS
        trace_manager.remove_span(str(span.uuid))

    def end_trace(self, span: BaseSpan):
        current_trace = trace_manager.get_trace_by_uuid(self.active_trace_id)
        if current_trace is not None:
            current_trace.input = span.input
            current_trace.output = span.output
        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None

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

        self.check_active_trace_id()
        base_span = BaseSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=inputs,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )

        self.add_span_to_trace(base_span)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:

        base_span = trace_manager.get_span_by_uuid(str(run_id))
        if base_span is None:
            return

        base_span.output = outputs
        self.end_span(base_span)

        if parent_run_id is None:
            self.end_trace(base_span)

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

        self.check_active_trace_id()

        # extract input
        input_messages = parse_prompts_to_messages(prompts, **kwargs)

        llm_span = LlmSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            attributes=LlmAttributes(input=input_messages, output=""),
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )

        self.add_span_to_trace(llm_span)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        llm_span = trace_manager.get_span_by_uuid(str(run_id))
        if llm_span is None:
            return

        output_str = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for generation in response.generations:
            for gen in generation:
                if isinstance(gen, ChatGeneration):
                    input_tokens, output_tokens = extract_token_usage(
                        gen.message.response_metadata
                    )
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    # set model for any generation
                    if llm_span.model is None or llm_span.model == "unknown":
                        llm_span.model = gen.message.response_metadata.get(
                            "model_name", "unknown"
                        )

                    output_str += convert_chat_generation_to_string(gen) + "\n"

        llm_span.set_attributes(
            LlmAttributes(
                input=llm_span.attributes.input,
                output=output_str,
                input_token_count=total_input_tokens,
                output_token_count=total_output_tokens,
            )
        )

        self.end_span(llm_span)
        if parent_run_id is None:
            self.end_trace(llm_span)

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

        self.check_active_trace_id()

        tool_span = ToolSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=input_str,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )
        self.add_span_to_trace(tool_span)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:

        tool_span = trace_manager.get_span_by_uuid(str(run_id))
        if tool_span is None:
            return

        tool_span.output = output

        self.end_span(tool_span)

        if parent_run_id is None:
            self.end_trace(tool_span)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:

        self.check_active_trace_id()

        retriever_span = RetrieverSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            embedder=metadata.get("ls_embedding_provider", "unknown"),
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )
        retriever_span.set_attributes(
            RetrieverAttributes(embedding_input=query, retrieval_context=[])
        )

        self.add_span_to_trace(retriever_span)

    def on_retriever_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
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

        retriever_span.set_attributes(
            RetrieverAttributes(
                embedding_input=retriever_span.attributes.embedding_input,
                retrieval_context=output_list,
            )
        )

        self.end_span(retriever_span)

        if parent_run_id is None:
            self.end_trace(retriever_span)
