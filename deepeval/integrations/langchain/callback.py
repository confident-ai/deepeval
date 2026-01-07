import logging

from typing import Any, Optional, List, Dict
from uuid import UUID
from time import perf_counter

from deepeval.tracing.context import current_trace_context
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.types import (
    LlmOutput,
    LlmToolCall,
    LlmSpan,
    RetrieverSpan,
    TraceSpanStatus,
    ToolSpan,
)
from deepeval.metrics import BaseMetric
from deepeval.tracing.utils import (
    bind_trace_and_span,
    prepare_tool_call_input_parameters,
)
from deepeval.tracing import trace_manager
from deepeval.telemetry import capture_tracing_integration


logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.outputs import ChatGeneration
    from langchain_core.messages import AIMessage

    # contains langchain imports
    from deepeval.integrations.langchain.utils import (
        parse_prompts_to_messages,
        extract_name,
        safe_extract_model_name,
        safe_extract_token_usage,
        enter_current_context,
        exit_current_context,
    )

    langchain_installed = True
except ImportError:
    langchain_installed = False


def is_langchain_installed():
    if not langchain_installed:
        raise ImportError(
            "LangChain is not installed. Please install it with `pip install langchain`."
        )


class CallbackHandler(BaseCallbackHandler):

    def __init__(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
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
            self.metrics = metrics
            self.metric_collection = metric_collection
            current_trace_context.set(trace)
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
        if parent_run_id is None:
            uuid_str = str(run_id)
            with self._ctx(parent_run_id=parent_run_id):
                base_span = enter_current_context(
                    uuid_str=uuid_str,
                    span_type="custom",
                    func_name=extract_name(serialized, **kwargs),
                )
                base_span.input = inputs
                trace = current_trace_context.get()
                if trace is not None:
                    trace.input = inputs
                base_span.metrics = self.metrics
                base_span.metric_collection = self.metric_collection

    def on_chain_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        uuid_str = str(run_id)
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            base_span = trace_manager.get_span_by_uuid(uuid_str)
            if base_span is None:
                logger.debug(
                    "Missing span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return
            base_span.output = output
            trace = current_trace_context.get()
            if trace is not None:
                trace.output = output
            exit_current_context(uuid_str=uuid_str)

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
        metadata = dict(metadata or {})
        metrics = metadata.get("metrics")
        metric_collection = metadata.get("metric_collection")
        prompt = metadata.get("prompt")
        model = safe_extract_model_name(metadata, **kwargs)
        with self._ctx(parent_run_id=parent_run_id):
            llm_span = enter_current_context(
                uuid_str=uuid_str,
                span_type="llm",
                func_name=serialized.get("name"),
            )

            llm_span.input = input_messages
            llm_span.model = model
            llm_span.metrics = metrics
            llm_span.metric_collection = metric_collection
            llm_span.prompt = prompt

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            llm_span: Optional[LlmSpan] = trace_manager.get_span_by_uuid(
                uuid_str
            )
            if llm_span is None:
                logger.debug(
                    "Missing LLM span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return

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
                            model = gen.message.response_metadata.get(
                                "model_name"
                            )

                            # extract input and output token
                            input_tokens, output_tokens = (
                                safe_extract_token_usage(
                                    gen.message.response_metadata
                                )
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
            llm_span.input_token_count = (
                total_input_tokens if total_input_tokens > 0 else None
            )
            llm_span.output_token_count = (
                total_output_tokens if total_output_tokens > 0 else None
            )

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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
            if llm_span is None:
                logger.debug(
                    "Missing LLM span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return
            llm_span.status = TraceSpanStatus.ERRORED
            llm_span.error = str(error)
            exit_current_context(uuid_str=uuid_str)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        uuid_str = str(run_id)
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
            if llm_span is None:
                logger.debug(
                    "Missing LLM span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return
            if llm_span.token_intervals is None:
                llm_span.token_intervals = {perf_counter(): token}
            else:
                llm_span.token_intervals[perf_counter()] = token

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

        with self._ctx(parent_run_id=parent_run_id):
            tool_span = enter_current_context(
                uuid_str=uuid_str,
                span_type="tool",
                func_name=extract_name(
                    serialized, **kwargs
                ),  # ignored when setting the input
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            tool_span: ToolSpan = trace_manager.get_span_by_uuid(uuid_str)
            if tool_span is None:
                logger.debug(
                    "Missing tool span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return
            tool_span.output = output

            # set the tools called in the parent span as well as on the trace level
            tool_call = ToolCall(
                name=tool_span.name,
                description=tool_span.description,
                output=output,
                input_parameters=prepare_tool_call_input_parameters(
                    tool_span.input
                ),
            )
            parent_span = (
                trace_manager.get_span_by_uuid(str(parent_run_id))
                if parent_run_id is not None
                else None
            )
            if parent_span is not None:
                if parent_span.tools_called is None:
                    parent_span.tools_called = []
                parent_span.tools_called.append(tool_call)

            trace = current_trace_context.get()
            if trace:
                if trace.tools_called is None:
                    trace.tools_called = []

                trace.tools_called.append(tool_call)
            exit_current_context(uuid_str=uuid_str)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,  # un-logged kwargs
    ) -> Any:
        uuid_str = str(run_id)
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            tool_span: ToolSpan = trace_manager.get_span_by_uuid(uuid_str)
            if tool_span is None:
                logger.debug(
                    "Missing tool span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return

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
        metadata = metadata or {}
        with self._ctx(parent_run_id=parent_run_id):
            retriever_span = enter_current_context(
                uuid_str=uuid_str,
                span_type="retriever",
                func_name=extract_name(serialized, **kwargs),
                observe_kwargs={
                    "embedder": metadata.get(
                        "ls_embedding_provider", "unknown"
                    ),
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            retriever_span: RetrieverSpan = trace_manager.get_span_by_uuid(
                uuid_str
            )
            if retriever_span is None:
                logger.debug(
                    "Missing retriever span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            retriever_span: RetrieverSpan = trace_manager.get_span_by_uuid(
                uuid_str
            )
            if retriever_span is None:
                logger.debug(
                    "Missing retriever span for run_id=%s parent=%s",
                    run_id,
                    parent_run_id,
                )
                return

            retriever_span.status = TraceSpanStatus.ERRORED
            retriever_span.error = str(error)
            exit_current_context(uuid_str=uuid_str)

    def _ctx(
        self,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
    ):
        return bind_trace_and_span(
            trace_uuid=self.trace_uuid,
            span_uuid=str(run_id) if run_id is not None else None,
            parent_uuid=(
                str(parent_run_id) if parent_run_id is not None else None
            ),
        )
