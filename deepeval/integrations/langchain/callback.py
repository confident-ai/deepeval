from typing import Any, Optional, List, Dict
from uuid import UUID
from time import perf_counter
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import (
    LlmOutput,
    LlmToolCall,
    TraceAttributes,
)
from deepeval.metrics import BaseMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager
import uuid

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.outputs import ChatGeneration
    from langchain_core.messages import AIMessage

    # contains langchain imports
    from deepeval.integrations.langchain.utils import (
        parse_prompts_to_messages,
        prepare_dict,
        extract_name,
        safe_extract_model_name,
        safe_extract_token_usage,
        enter_current_context,
        exit_current_context,
        exit_current_trace_context,
    )

    langchain_installed = True
except:
    langchain_installed = False


def is_langchain_installed():
    if not langchain_installed:
        raise ImportError(
            "LangChain is not installed. Please install it with `pip install langchain`."
        )

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

    def __init__(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        is_langchain_installed()
        with capture_tracing_integration("langchain.callback.CallbackHandler"):
            trace = trace_manager.start_new_trace()
            
            self.trace_uuid = trace.uuid
            
            trace.name = name
            trace.tags = tags
            trace.metadata = metadata
            trace.thread_id = thread_id
            trace.user_id = user_id
            
            current_trace_context.set(trace)
            super().__init__()

    def __del__(self):
        exit_current_trace_context(self.trace_uuid)

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
        uuid_str = str(run_id)
        input_messages = parse_prompts_to_messages(prompts, **kwargs)
        model = safe_extract_model_name(metadata, **kwargs)

        llm_span = enter_current_context(
            uuid_str=uuid_str,
            span_type="llm",
            func_name=extract_name(serialized, **kwargs),
        )

        llm_span.input = input_messages
        llm_span.model = model
        

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)

        output = ""
        total_input_tokens = 0
        total_output_tokens = 0
        model = None

        for generation in response.generations:
            for gen in generation:
                if isinstance(gen, ChatGeneration):
                    if gen.message.response_metadata and isinstance(
                        gen.message.response_metadata, dict
                    ):
                        # extract model name from response_metadata
                        model = gen.message.response_metadata.get("model_name")

                        # extract input and output token
                        input_tokens, output_tokens = safe_extract_token_usage(
                            gen.message.response_metadata
                        )
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens

                    if isinstance(gen.message, AIMessage):
                        ai_message = gen.message
                        tool_calls = []
                        for tool_call in ai_message.tool_calls:
                            tool_calls.append(
                                LlmToolCall(
                                    name=tool_call["name"],
                                    args=tool_call["args"],
                                    id=tool_call["id"],
                                )
                            )
                        output = LlmOutput(
                            role="AI",
                            content=ai_message.content,
                            tool_calls=tool_calls,
                        )

        llm_span.model = model if model else llm_span.model
        llm_span.input = llm_span.input
        llm_span.output = output
        llm_span.input_token_count = total_input_tokens if total_input_tokens > 0 else None
        llm_span.output_token_count = total_output_tokens if total_output_tokens > 0 else None

        exit_current_context(uuid_str=uuid_str)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        uuid_str = str(run_id)
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
        llm_span.status = TraceSpanStatus.ERRORED
        llm_span.error = str(error)
        exit_current_context(uuid_str=uuid_str)

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
        uuid_str = str(run_id)

        tool_span = enter_current_context(
            uuid_str=uuid_str,
            span_type="tool",
            func_name=extract_name(serialized, **kwargs), # ignored when setting the input
        )
        tool_span.input = inputs

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
    
        uuid_str = str(run_id)
        tool_span: ToolSpan = trace_manager.get_span_by_uuid(uuid_str)
        tool_span.output = output
        exit_current_context(str(run_id))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        tool_span: ToolSpan = trace_manager.get_span_by_uuid(uuid_str)
        tool_span.status = TraceSpanStatus.ERRORED
        tool_span.error = str(error)
        exit_current_context(uuid_str=uuid_str)

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
        uuid_str = str(run_id)
        retriever_span = enter_current_context(
            uuid_str=uuid_str,
            span_type="retriever",
            func_name=extract_name(serialized, **kwargs),
            observe_kwargs={
                "embedder": metadata.get("ls_embedding_provider", "unknown"),
            },
        )
        retriever_span.input = query

    def on_retriever_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        retriever_span: RetrieverSpan = trace_manager.get_span_by_uuid(uuid_str)
        
        # prepare output
        output_list = []
        if isinstance(output, list):
            for item in output:
                output_list.append(str(item))
        else:
            output_list.append(str(output))
        
        retriever_span.output = output_list
        exit_current_context(uuid_str=uuid_str)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        retriever_span: RetrieverSpan = trace_manager.get_span_by_uuid(uuid_str)
        retriever_span.status = TraceSpanStatus.ERRORED
        retriever_span.error = str(error)
        exit_current_context(uuid_str=uuid_str)