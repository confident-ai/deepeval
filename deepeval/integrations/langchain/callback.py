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

    active_trace_id: Optional[str] = None
    metrics: List[BaseMetric] = []
    metric_collection: Optional[str] = None

    def __init__(
        self,
        metrics: List[BaseMetric] = [],
        metric_collection: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        is_langchain_installed()
        with capture_tracing_integration("langchain.callback.CallbackHandler"):
            self.metrics = metrics
            self.metric_collection = metric_collection
            self.trace_attributes = TraceAttributes(
                name=name,
                tags=tags,
                metadata=metadata,
                thread_id=thread_id,
                user_id=user_id,
            )
            super().__init__()

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

        ######## Conditions to add metric_collection to span ########
        if (
            self.metric_collection and span.parent_uuid is None
        ):  # if span is a root span
            span.metric_collection = self.metric_collection

        ######## Conditions to add metrics to span ########
        if self.metrics and span.parent_uuid is None:  # if span is a root span

            # prepare test_case for task_completion metric
            for metric in self.metrics:
                if isinstance(metric, TaskCompletionMetric):
                    self.prepare_span_metric_test_case(metric, span)

    def end_trace(self, span: BaseSpan):
        current_trace = trace_manager.get_trace_by_uuid(self.active_trace_id)

        ######## Conditions send the trace for evaluation ########
        if self.metrics:
            trace_manager.evaluating = (
                True  # to avoid posting the trace to the server
            )
            trace_manager.evaluation_loop = (
                True  # to avoid traces being evaluated twice
            )
            trace_manager.integration_traces_to_evaluate.append(current_trace)

        if current_trace is not None:
            current_trace.input = span.input
            current_trace.output = span.output

        # set trace attributes
        if self.trace_attributes:
            if self.trace_attributes.name:
                current_trace.name = self.trace_attributes.name
            if self.trace_attributes.tags:
                current_trace.tags = self.trace_attributes.tags
            if self.trace_attributes.metadata:
                current_trace.metadata = self.trace_attributes.metadata
            if self.trace_attributes.thread_id:
                current_trace.thread_id = self.trace_attributes.thread_id
            if self.trace_attributes.user_id:
                current_trace.user_id = self.trace_attributes.user_id

        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None

    def prepare_span_metric_test_case(
        self, metric: TaskCompletionMetric, span: BaseSpan
    ):
        task_completion_metric = TaskCompletionMetric(
            threshold=metric.threshold,
            model=metric.model,
            include_reason=metric.include_reason,
            async_mode=metric.async_mode,
            strict_mode=metric.strict_mode,
            verbose_mode=metric.verbose_mode,
        )
        task_completion_metric.evaluation_cost = 0
        _llm_test_case = LLMTestCase(input="None", actual_output="None")
        _llm_test_case._trace_dict = trace_manager.create_nested_spans_dict(
            span
        )
        task, _ = task_completion_metric._extract_task_and_outcome(
            _llm_test_case
        )
        task_completion_metric.task = task
        span.metrics = [task_completion_metric]

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
            status=TraceSpanStatus.ERRORED,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=inputs,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
            # fallback for on_end callback
            end_time=perf_counter(),
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

        # extract model name
        model = safe_extract_model_name(metadata, **kwargs)

        llm_span = LlmSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.ERRORED,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=input_messages,
            output="",
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
            model=model,
            # fallback for on_end callback
            end_time=perf_counter(),
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
        llm_span: LlmSpan = trace_manager.get_span_by_uuid(str(run_id))
        if llm_span is None:
            return

        if not isinstance(llm_span, LlmSpan):
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
        llm_span.input_token_count = (
            total_input_tokens if total_input_tokens > 0 else None
        )
        llm_span.output_token_count = (
            total_output_tokens if total_output_tokens > 0 else None
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
            status=TraceSpanStatus.ERRORED,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=input_str,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
            # fallback for on_end callback
            end_time=perf_counter(),
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
            status=TraceSpanStatus.ERRORED,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            embedder=metadata.get("ls_embedding_provider", "unknown"),
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
            # fallback for on_end callback
            end_time=perf_counter(),
        )
        retriever_span.input = query
        retriever_span.retrieval_context = []

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

        retriever_span.input = retriever_span.input
        retriever_span.retrieval_context = output_list

        self.end_span(retriever_span)

        if parent_run_id is None:
            self.end_trace(retriever_span)

    ################## on_error callbacks ###############

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        base_span = trace_manager.get_span_by_uuid(str(run_id))
        if base_span is None:
            return

        base_span.end_time = perf_counter()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        llm_span = trace_manager.get_span_by_uuid(str(run_id))
        if llm_span is None:
            return

        llm_span.end_time = perf_counter()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        tool_span = trace_manager.get_span_by_uuid(str(run_id))
        if tool_span is None:
            return

        tool_span.end_time = perf_counter()

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        retriever_span = trace_manager.get_span_by_uuid(str(run_id))
        if retriever_span is None:
            return

        retriever_span.end_time = perf_counter()
