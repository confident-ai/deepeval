from typing import Any, Optional, List, Dict
from uuid import UUID
from time import perf_counter
from contextlib import contextmanager

from deepeval.tracing.context import current_trace_context, current_span_context
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.types import (
    LlmOutput,
    LlmToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.tracing import trace_manager
from deepeval.tracing.utils import prepare_tool_call_input_parameters
from deepeval.tracing.types import (
    LlmSpan,
    RetrieverSpan,
    TraceSpanStatus,
    ToolSpan,
)
from deepeval.telemetry import capture_tracing_integration

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
    from deepeval.integrations.langchain.patch import tool  # noqa: F401

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
            # Do not create or set a trace in __init__.
            # CallbackHandler instances are often constructed outside the async Task
            # that actually runs LangGraph/LangChain. Creating a trace here can
            # corrupt ContextVars and break observe wrapped async execution
            self._trace = None
            self.trace_uuid = None

            # Lazily captured fallback parent when callbacks execute.
            self._parent_span = None

            # Stash trace metadata to apply once we know which trace we are using.
            self._trace_init_fields: Dict[str, Any] = {
                "name": name,
                "tags": tags,
                "metadata": metadata,
                "thread_id": thread_id,
                "user_id": user_id,
            }

            # Map LangChain run_id -> our span uuid for parent span restoration
            self._run_id_to_span_uuid: Dict[str, str] = {}

            # Only set trace metadata if values are provided
            self.metrics = metrics
            self.metric_collection = metric_collection
            super().__init__()

    def _ensure_trace(self):
        """
        Ensure there's an active trace in ContextVars for this callback invocation.
        This is done lazily during actual callback execution to avoid context
        corruption when the handler is constructed outside the async task/context.
        """
        # Prefer current context trace if it is active.
        ctx_trace = current_trace_context.get()
        if ctx_trace and ctx_trace.uuid in trace_manager.active_traces:
            trace = ctx_trace
        else:
            # Otherwise, restore our stored trace if still active.
            if self._trace and self._trace.uuid in trace_manager.active_traces:
                trace = self._trace
                current_trace_context.set(trace)
            else:
                # Otherwise, create a fresh trace now (in the right context).
                trace = trace_manager.start_new_trace()
                current_trace_context.set(trace)
                self._trace = trace

        # Keep a copy for quick access.
        self.trace_uuid = trace.uuid

        # Apply stashed metadata once.
        fields = getattr(self, "_trace_init_fields", None) or {}
        if fields:
            if fields.get("name") is not None:
                trace.name = fields["name"]
            if fields.get("tags") is not None:
                trace.tags = fields["tags"]
            if fields.get("metadata") is not None:
                trace.metadata = fields["metadata"]
            if fields.get("thread_id") is not None:
                trace.thread_id = fields["thread_id"]
            if fields.get("user_id") is not None:
                trace.user_id = fields["user_id"]
            # prevent re-applying on every callback
            self._trace_init_fields = {}

        # Lazily capture the observe parent span if present.
        if self._parent_span is None:
            self._parent_span = current_span_context.get()

        return trace

    @contextmanager
    def _ctx(self, run_id: UUID, parent_run_id: Optional[UUID] = None):
        """
        Context manager to restore trace and span context for callbacks running
        in different async tasks. In async LangChain/LangGraph execution, ContextVar
        values don't propagate across task boundaries, so we explicitly restore them.

        IMPORTANT: parent_run_id from LangChain is the source of truth for hierarchy.
        We ALWAYS use it to set the correct parent span, not just when context is lost.
        """
        span_token = None

        try:
            # Ensure we have a valid trace in this execution context.
            # May start a trace here, or restore a stored one, or reuse an @observe trace.
            self._ensure_trace()

            # Set parent span based on LangChain's parent_run_id (source of truth for hierarchy)
            # Priority order:
            # 1. Parent span from run_id mapping (LangChain's parent_run_id)
            # 2. Parent span captured at init (from @observe wrapper)
            # 3. Keep existing context

            target_parent_span = None

            # First, try to find parent from LangChain's parent_run_id
            if parent_run_id is not None:
                parent_run_id_str = str(parent_run_id)
                if parent_run_id_str in self._run_id_to_span_uuid:
                    parent_span_uuid = self._run_id_to_span_uuid[
                        parent_run_id_str
                    ]
                    target_parent_span = trace_manager.get_span_by_uuid(
                        parent_span_uuid
                    )

            # Fall back to the span captured at init (from @observe wrapper)
            if target_parent_span is None and self._parent_span:
                if trace_manager.get_span_by_uuid(self._parent_span.uuid):
                    target_parent_span = self._parent_span

            # Set the parent span context if we found one and it's different from current
            current_span = current_span_context.get()
            if target_parent_span and (
                current_span is None
                or current_span.uuid != target_parent_span.uuid
            ):
                span_token = current_span_context.set(target_parent_span)

            yield

        finally:
            if span_token is not None:
                current_span_context.reset(span_token)

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
        # Create spans for all chains to establish proper parent-child hierarchy
        # This is important for LangGraph where there are nested chains
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            uuid_str = str(run_id)
            base_span = enter_current_context(
                uuid_str=uuid_str,
                span_type="custom",
                func_name=extract_name(serialized, **kwargs),
            )
            # Register this run_id -> span mapping for child callbacks
            self._run_id_to_span_uuid[str(run_id)] = uuid_str

            base_span.input = inputs

            # Only set trace-level input/metrics for root chain
            if parent_run_id is None:
                trace = trace_manager.get_trace_by_uuid(base_span.trace_uuid)
                if trace:
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
        base_span = trace_manager.get_span_by_uuid(uuid_str)
        if base_span:
            with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
                base_span.output = output
                # Only set trace-level output for root chain
                if parent_run_id is None:
                    trace = trace_manager.get_trace_by_uuid(
                        base_span.trace_uuid
                    )
                    if trace:
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            uuid_str = str(run_id)
            input_messages = parse_prompts_to_messages(prompts, **kwargs)
            model = safe_extract_model_name(metadata, **kwargs)

            llm_span: LlmSpan = enter_current_context(
                uuid_str=uuid_str,
                span_type="llm",
                func_name=extract_name(serialized, **kwargs),
            )
            # Register this run_id -> span mapping for child callbacks
            self._run_id_to_span_uuid[str(run_id)] = uuid_str

            llm_span.input = input_messages
            llm_span.model = model
            metrics = metadata.pop("metrics", None)
            metric_collection = metadata.pop("metric_collection", None)
            prompt = metadata.pop("prompt", None)
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
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
        if llm_span is None:
            return

        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
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
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
        if llm_span is None:
            return
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
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
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(uuid_str)
        if llm_span is None:
            return
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            uuid_str = str(run_id)

            tool_span = enter_current_context(
                uuid_str=uuid_str,
                span_type="tool",
                func_name=extract_name(
                    serialized, **kwargs
                ),  # ignored when setting the input
            )
            # Register this run_id -> span mapping for child callbacks
            self._run_id_to_span_uuid[str(run_id)] = uuid_str
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
        if tool_span is None:
            return

        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            tool_span.output = output
            exit_current_context(uuid_str=uuid_str)

            # set the tools called in the parent span as well as on the trace level
            tool_call = ToolCall(
                name=tool_span.name,
                description=tool_span.description,
                output=output,
                input_parameters=prepare_tool_call_input_parameters(
                    tool_span.input
                ),
            )

            # Use span's stored trace_uuid and parent_uuid for reliable lookup
            # These are always available regardless of context state
            if tool_span.parent_uuid:
                parent_span = trace_manager.get_span_by_uuid(
                    tool_span.parent_uuid
                )
                if parent_span:
                    if parent_span.tools_called is None:
                        parent_span.tools_called = []
                    parent_span.tools_called.append(tool_call)

            if tool_span.trace_uuid:
                trace = trace_manager.get_trace_by_uuid(tool_span.trace_uuid)
                if trace:
                    if trace.tools_called is None:
                        trace.tools_called = []
                    trace.tools_called.append(tool_call)

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
        if tool_span is None:
            return
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
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
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            uuid_str = str(run_id)
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
            # Register this run_id -> span mapping for child callbacks
            self._run_id_to_span_uuid[str(run_id)] = uuid_str
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
        if retriever_span is None:
            return

        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
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
        if retriever_span is None:
            return
        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            retriever_span.status = TraceSpanStatus.ERRORED
            retriever_span.error = str(error)
            exit_current_context(uuid_str=uuid_str)
