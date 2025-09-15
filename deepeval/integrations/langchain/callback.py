from typing import Any, Optional, List, Dict
from uuid import UUID
from time import perf_counter
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

CUSTOM_SPAN_NAME_LIST = ["LangGraph"]

class CallbackHandler(BaseCallbackHandler):

    def __init__(self):
        super().__init__()

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

        _name = extract_name(serialized, **kwargs)
        if _name in CUSTOM_SPAN_NAME_LIST:

            enter_current_context(
                uuid_str=str(run_id),
                span_type="custom", 
                func_name=_name,
            )

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
        
        exit_current_context(uuid_str=str(run_id))

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

        input_messages = parse_prompts_to_messages(prompts, **kwargs)
        model = safe_extract_model_name(metadata, **kwargs)

        llm_span = enter_current_context(
            uuid_str=str(run_id),
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
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(str(run_id))

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

        exit_current_context(
            uuid_str=str(run_id)
        )
