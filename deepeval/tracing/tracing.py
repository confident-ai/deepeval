from typing import Any, Dict, List, Literal, Optional, Set, Union, Callable
from time import perf_counter
import threading
import functools
import inspect
import asyncio
import random
import atexit
import queue
import uuid
import os
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress


from deepeval.constants import (
    CONFIDENT_TRACE_VERBOSE,
    CONFIDENT_TRACE_FLUSH,
    CONFIDENT_SAMPLE_RATE,
    CONFIDENT_TRACE_ENVIRONMENT,
)
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.metrics import BaseMetric
from deepeval.tracing.api import (
    BaseApiSpan,
    SpanApiType,
    TraceSpanTestCase,
    TraceApi,
)
from deepeval.telemetry import capture_send_trace
from deepeval.tracing.attributes import (
    AgentAttributes,
    Attributes,
    LlmAttributes,
    RetrieverAttributes,
    ToolAttributes,
)
from deepeval.tracing.patchers import patch_openai_client
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    SpanType,
    ToolSpan,
    Trace,
    TraceSpanStatus,
    TraceWorkerStatus,
)
from deepeval.tracing.utils import (
    Environment,
    make_json_serializable,
    perf_counter_to_datetime,
    to_zod_compatible_iso,
    validate_environment,
    validate_sampling_rate,
)
from deepeval.feedback.utils import convert_feedback_to_api_feedback
from deepeval.utils import dataclass_to_dict, is_confident
from deepeval.tracing.context import current_span_context, current_trace_context


class TraceManager:
    def __init__(self):
        self.traces: List[Trace] = []
        self.active_traces: Dict[str, Trace] = {}  # Map of trace_uuid to Trace
        self.active_spans: Dict[str, BaseSpan] = (
            {}
        )  # Map of span_uuid to BaseSpan

        # Initialize queue and worker thread for trace posting
        self._trace_queue = queue.Queue()
        self._worker_thread = None
        self._min_interval = 0.2  # Minimum time between API calls (seconds)
        self._last_post_time = 0
        self._in_flight_tasks: Set[asyncio.Task[Any]] = set()
        self._daemon = (
            False if os.getenv(CONFIDENT_TRACE_FLUSH) == "YES" else True
        )
        self.evaluating = False

        # trace manager attributes
        self.confident_api_key = None
        self.custom_mask_fn: Optional[Callable] = None
        self.environment = os.environ.get(
            CONFIDENT_TRACE_ENVIRONMENT, Environment.DEVELOPMENT.value
        )
        validate_environment(self.environment)

        self.sampling_rate = os.environ.get(CONFIDENT_SAMPLE_RATE, 1)
        validate_sampling_rate(self.sampling_rate)
        self.openai_client = None

        # Register an exit handler to warn about unprocessed traces
        atexit.register(self._warn_on_exit)

    def _warn_on_exit(self):
        queue_size = self._trace_queue.qsize()
        in_flight = len(self._in_flight_tasks)
        remaining_tasks = queue_size + in_flight
        if os.getenv(CONFIDENT_TRACE_FLUSH) != "YES" and remaining_tasks > 0:
            self._print_trace_status(
                message=f"WARNING: Exiting with {queue_size + in_flight} abaonded trace(s).",
                trace_worker_status=TraceWorkerStatus.WARNING,
                description=f"Set {CONFIDENT_TRACE_FLUSH}=YES as an environment variable to flush remaining traces to Confident AI.",
            )

    def mask(self, data: Any):
        if self.custom_mask_fn is not None:
            self.custom_mask_fn(data)
        else:
            return data

    def configure(
        self,
        mask: Optional[Callable] = None,
        environment: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        confident_api_key: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        if mask is not None:
            self.custom_mask_fn = mask
        if environment is not None:
            validate_environment(environment)
            self.environment = environment
        if sampling_rate is not None:
            validate_sampling_rate(sampling_rate)
            self.sampling_rate = sampling_rate
        if confident_api_key is not None:
            self.confident_api_key = confident_api_key
        if openai_client is not None:
            self.openai_client = openai_client
            patch_openai_client(openai_client)

    def start_new_trace(
        self,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> Trace:
        """Start a new trace and set it as the current trace."""
        trace_uuid = str(uuid.uuid4())
        new_trace = Trace(
            uuid=trace_uuid,
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=perf_counter(),
            end_time=None,
            confident_api_key=self.confident_api_key,
            metric_collection=metric_collection,
            metrics=metrics,
        )
        self.active_traces[trace_uuid] = new_trace
        self.traces.append(new_trace)
        return new_trace

    def end_trace(self, trace_uuid: str):
        """End a specific trace by its UUID."""

        if trace_uuid in self.active_traces:
            trace = self.active_traces[trace_uuid]
            trace.end_time = perf_counter()

            # Default to SUCCESS for completed traces
            # This assumes that if a trace completes, it was successful overall
            # Users can manually set the status to ERROR if needed
            if trace.status == TraceSpanStatus.IN_PROGRESS:
                trace.status = TraceSpanStatus.SUCCESS

            # Post the trace to the server before removing it
            if not self.evaluating:
                self.post_trace(trace)
            else:
                # print(f"Ending trace: {trace.root_spans}")
                self.environment = Environment.TESTING
                trace.root_spans = [trace.root_spans[0].children[0]]
                for root_span in trace.root_spans:
                    root_span.parent_uuid = None

            # Remove from active traces
            del self.active_traces[trace_uuid]

    def set_trace_status(self, trace_uuid: str, status: TraceSpanStatus):
        """Manually set the status of a trace."""
        if trace_uuid in self.active_traces:
            trace = self.active_traces[trace_uuid]
            trace.status = status

    def add_span(self, span: BaseSpan):
        """Add a span to the active spans dictionary."""
        self.active_spans[span.uuid] = span

    def remove_span(self, span_uuid: str):
        """Remove a span from the active spans dictionary."""
        if span_uuid in self.active_spans:
            del self.active_spans[span_uuid]

    def add_span_to_trace(self, span: BaseSpan):
        """Add a span to its trace."""
        trace_uuid = span.trace_uuid
        if trace_uuid not in self.active_traces:
            raise ValueError(
                f"Trace with UUID {trace_uuid} does not exist. A span must have a valid trace."
            )

        trace = self.active_traces[trace_uuid]

        # If this is a root span (no parent), add it to the trace's root_spans
        if not span.parent_uuid:
            trace.root_spans.append(span)
        else:
            # This is a child span, find its parent and add it to the parent's children
            parent_span = self.get_span_by_uuid(span.parent_uuid)
            if parent_span:
                parent_span.children.append(span)
            else:
                raise ValueError(
                    f"Parent span with UUID {span.parent_uuid} does not exist."
                )

    def get_trace_by_uuid(self, trace_uuid: str) -> Optional[Trace]:
        """Get a trace by its UUID."""
        return self.active_traces.get(trace_uuid)

    def get_span_by_uuid(self, span_uuid: str) -> Optional[BaseSpan]:
        """Get a span by its UUID."""
        return self.active_spans.get(span_uuid)

    def get_all_traces(self) -> List[Trace]:
        """Get all traces."""
        return self.traces

    def clear_traces(self):
        """Clear all traces."""
        self.traces = []
        self.active_traces = {}
        self.active_spans = {}

    def get_trace_dict(self, trace: Trace) -> Dict:
        """Convert a trace to a dictionary."""
        return dataclass_to_dict(trace)

    def get_all_traces_dict(self) -> List[Dict]:
        """Get all traces as dictionaries."""
        return [self.get_trace_dict(trace) for trace in self.traces]

    def _print_trace_status(
        self,
        trace_worker_status: TraceWorkerStatus,
        message: str,
        description: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        if (
            os.getenv(CONFIDENT_TRACE_VERBOSE) != "NO"
            and self.evaluating is False
        ):
            console = Console()
            message_prefix = "[dim][Confident AI Trace Log][/dim]"
            if trace_worker_status == TraceWorkerStatus.SUCCESS:
                message = f"[green]{message}[/green]"
            elif trace_worker_status == TraceWorkerStatus.FAILURE:
                message = f"[red]{message}[/red]"
            elif trace_worker_status == TraceWorkerStatus.WARNING:
                message = f"[yellow]{message}[/yellow]"

            env_text = f"[{environment}]" if environment else ""

            if description:
                console.print(
                    message_prefix,
                    env_text,
                    message + ":",
                    description,
                    f"\nTo disable dev logging, set {CONFIDENT_TRACE_VERBOSE}=NO as an environment variable.",
                )
            else:
                console.print(message_prefix, env_text, message)

    def _should_sample_trace(self) -> bool:
        random_number = random.random()
        if random_number > self.sampling_rate:
            rate_str = f"{self.sampling_rate:.2f}"
            self._print_trace_status(
                message=f"Skipped posting trace due to sampling rate ({rate_str})",
                trace_worker_status=TraceWorkerStatus.SUCCESS,
            )
            return False

        return True

    def _ensure_worker_thread_running(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._process_trace_queue,
                daemon=self._daemon,
            )
            self._worker_thread.start()

    def post_trace_api(self, trace_api: TraceApi) -> Optional[str]:
        if not is_confident() and self.confident_api_key is None:
            self._print_trace_status(
                message="No Confident AI API key found. Skipping trace posting.",
                trace_worker_status=TraceWorkerStatus.FAILURE,
            )
            return None

        if not self._should_sample_trace():
            return None

        self._ensure_worker_thread_running()
        self._trace_queue.put(trace_api)

        return "ok"

    def post_trace(self, trace: Trace) -> Optional[str]:
        if not is_confident() and self.confident_api_key is None:
            self._print_trace_status(
                message="No Confident AI API key found. Skipping trace posting.",
                trace_worker_status=TraceWorkerStatus.FAILURE,
            )
            return None

        if not self._should_sample_trace():
            return None

        # Add the trace to the queue
        self._trace_queue.put(trace)

        # Start the worker thread if it's not already running
        self._ensure_worker_thread_running()

        return "ok"

    def _process_trace_queue(self):
        """Worker thread function that processes the trace queue"""
        import threading

        main_thr = threading.main_thread()

        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # buffer for payloads that need to be sent after main exits
        remaining_trace_request_bodies: List[Dict[str, Any]] = []

        async def _a_send_trace(trace_obj):
            nonlocal remaining_trace_request_bodies
            try:
                # Build API object & payload
                if isinstance(trace_obj, TraceApi):
                    trace_api = trace_obj
                else:
                    trace_api = self.create_trace_api(trace_obj)

                try:
                    body = trace_api.model_dump(
                        by_alias=True,
                        exclude_none=True,
                    )
                except AttributeError:
                    # Pydantic version below 2.0
                    body = trace_api.dict(by_alias=True, exclude_none=True)
                # If the main thread is still alive, send now
                body = make_json_serializable(body)
                if main_thr.is_alive():
                    api = Api(api_key=self.confident_api_key)
                    response = await api.a_send_request(
                        method=HttpMethods.POST,
                        endpoint=Endpoints.TRACING_ENDPOINT,
                        body=body,
                    )
                    queue_size = self._trace_queue.qsize()
                    in_flight = len(self._in_flight_tasks)
                    status = f"({queue_size} trace{'s' if queue_size!=1 else ''} remaining in queue, {in_flight} in flight)"
                    self._print_trace_status(
                        trace_worker_status=TraceWorkerStatus.SUCCESS,
                        message=f"Successfully posted trace {status}",
                        description=response["link"],
                        environment=self.environment,
                    )
                elif os.getenv(CONFIDENT_TRACE_FLUSH) == "YES":
                    # Main thread gone → to be flushed
                    remaining_trace_request_bodies.append(body)

            except Exception as e:
                queue_size = self._trace_queue.qsize()
                in_flight = len(self._in_flight_tasks)
                status = f"({queue_size} trace{'s' if queue_size!=1 else ''} remaining in queue, {in_flight} in flight)"
                self._print_trace_status(
                    trace_worker_status=TraceWorkerStatus.FAILURE,
                    message=f"Error posting trace {status}",
                    description=str(e),
                )
            finally:
                task = asyncio.current_task()
                if task:
                    self._in_flight_tasks.discard(task)

        async def async_worker():
            # Continue while user code is running or work remains
            while (
                main_thr.is_alive()
                or not self._trace_queue.empty()
                or self._in_flight_tasks
            ):
                try:
                    trace = self._trace_queue.get(block=True, timeout=1.0)

                    # rate-limit
                    now = perf_counter()
                    elapsed = now - self._last_post_time
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    self._last_post_time = perf_counter()

                    # schedule async send
                    task = asyncio.create_task(_a_send_trace(trace))
                    self._in_flight_tasks.add(task)
                    self._trace_queue.task_done()

                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                except Exception as e:
                    self._print_trace_status(
                        message="Error in worker",
                        trace_worker_status=TraceWorkerStatus.FAILURE,
                        description=str(e),
                    )
                    await asyncio.sleep(1.0)

        try:
            loop.run_until_complete(async_worker())
        finally:
            # Drain any pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self.flush_traces(remaining_trace_request_bodies)
            loop.close()

    def flush_traces(
        self, remaining_trace_request_bodies: List[Dict[str, Any]]
    ):
        self._print_trace_status(
            TraceWorkerStatus.WARNING,
            message=f"Flushing {len(remaining_trace_request_bodies)} remaining trace(s)",
        )
        for body in remaining_trace_request_bodies:
            with capture_send_trace():
                try:
                    api = Api(api_key=self.confident_api_key)
                    resp = api.send_request(
                        method=HttpMethods.POST,
                        endpoint=Endpoints.TRACING_ENDPOINT,
                        body=body,
                    )
                    qs = self._trace_queue.qsize()
                    self._print_trace_status(
                        trace_worker_status=TraceWorkerStatus.SUCCESS,
                        message=f"Successfully posted trace ({qs} traces remaining in queue, 1 in flight)",
                        description=resp["link"],
                        environment=self.environment,
                    )
                except Exception as e:
                    qs = self._trace_queue.qsize()
                    self._print_trace_status(
                        trace_worker_status=TraceWorkerStatus.FAILURE,
                        message="Error flushing remaining trace(s)",
                        description=str(e),
                    )

    def create_trace_api(self, trace: Trace) -> TraceApi:
        # Initialize empty lists for each span type
        base_spans = []
        agent_spans = []
        llm_spans = []
        retriever_spans = []
        tool_spans = []

        # Process all spans in the trace
        def process_spans(spans):
            for span in spans:
                # Convert BaseSpan to BaseApiSpan
                api_span = self._convert_span_to_api_span(span)

                # Categorize spans by type
                if isinstance(span, AgentSpan):
                    agent_spans.append(api_span)
                elif isinstance(span, LlmSpan):
                    llm_spans.append(api_span)
                elif isinstance(span, RetrieverSpan):
                    retriever_spans.append(api_span)
                elif isinstance(span, ToolSpan):
                    tool_spans.append(api_span)
                else:
                    base_spans.append(api_span)

                # Process children recursively
                if span.children:
                    process_spans(span.children)

        # Start processing from root spans
        process_spans(trace.root_spans)

        # Convert perf_counter values to ISO 8601 strings
        start_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(trace.start_time))
            if trace.start_time
            else None
        )
        end_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(trace.end_time))
            if trace.end_time
            else None
        )

        trace_test_case = (
            TraceSpanTestCase(
                input=trace.llm_test_case.input,
                actualOutput=trace.llm_test_case.actual_output,
                expectedOutput=trace.llm_test_case.expected_output,
                retrievalContext=trace.llm_test_case.retrieval_context,
                context=trace.llm_test_case.context,
                toolsCalled=trace.llm_test_case.tools_called,
                expectedTools=trace.llm_test_case.expected_tools,
            )
            if trace.llm_test_case
            else None
        )

        return TraceApi(
            uuid=trace.uuid,
            baseSpans=base_spans,
            agentSpans=agent_spans,
            llmSpans=llm_spans,
            retrieverSpans=retriever_spans,
            toolSpans=tool_spans,
            startTime=start_time,
            endTime=end_time,
            metadata=trace.metadata,
            tags=trace.tags,
            environment=self.environment,
            threadId=trace.thread_id,
            userId=trace.user_id,
            input=trace.input,
            output=trace.output,
            feedback=convert_feedback_to_api_feedback(
                trace.feedback, trace_uuid=trace.uuid
            ),
            traceTestCase=trace_test_case,
            metricCollection=(
                trace.metric_collection if trace.llm_test_case else None
            ),
        )

    def _convert_span_to_api_span(self, span: BaseSpan) -> BaseApiSpan:
        # Determine span type
        if isinstance(span, AgentSpan):
            span_type = SpanApiType.AGENT
        elif isinstance(span, LlmSpan):
            span_type = SpanApiType.LLM
        elif isinstance(span, RetrieverSpan):
            span_type = SpanApiType.RETRIEVER
        elif isinstance(span, ToolSpan):
            span_type = SpanApiType.TOOL
        else:
            span_type = SpanApiType.BASE

        # Initialize input and output fields
        input_data = None
        output_data = None

        if isinstance(span, RetrieverSpan):
            # For RetrieverSpan, input is embeddingInput, output is retrievalContext
            if span.attributes:
                input_data = span.attributes.embedding_input
                output_data = span.attributes.retrieval_context

        elif isinstance(span, LlmSpan):
            # For LlmSpan, input is attributes.input, output is attributes.output
            if span.attributes:
                input_data = span.attributes.input
                output_data = span.attributes.output
        else:
            # For BaseSpan, Agent, or Tool types, use the standard logic
            input_data = span.input
            output_data = span.output

        # Convert perf_counter values to ISO 8601 strings
        start_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(span.start_time))
            if span.start_time
            else None
        )
        end_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(span.end_time))
            if span.end_time
            else None
        )

        span_test_case = (
            TraceSpanTestCase(
                input=span.llm_test_case.input,
                actualOutput=span.llm_test_case.actual_output,
                expectedOutput=span.llm_test_case.expected_output,
                retrievalContext=span.llm_test_case.retrieval_context,
                context=span.llm_test_case.context,
                toolsCalled=span.llm_test_case.tools_called,
                expectedTools=span.llm_test_case.expected_tools,
            )
            if span.llm_test_case
            else None
        )

        # Create the base API span
        api_span = BaseApiSpan(
            uuid=span.uuid,
            name=span.name,
            status=span.status.value,
            type=span_type,
            traceUuid=span.trace_uuid,
            parentUuid=span.parent_uuid,
            startTime=start_time,
            endTime=end_time,
            input=input_data,
            output=output_data,
            metadata=span.metadata,
            error=span.error,
            spanTestCase=span_test_case,
            metricCollection=span.metric_collection,
            feedback=convert_feedback_to_api_feedback(
                span.feedback, span_uuid=span.uuid
            ),
        )

        # Add type-specific attributes
        if isinstance(span, AgentSpan):
            api_span.available_tools = span.available_tools
            api_span.agent_handoffs = span.agent_handoffs
        elif isinstance(span, ToolSpan):
            api_span.description = span.description
        elif isinstance(span, RetrieverSpan):
            api_span.embedder = span.embedder
            if span.attributes:
                api_span.top_k = span.attributes.top_k
                api_span.chunk_size = span.attributes.chunk_size
        elif isinstance(span, LlmSpan):
            api_span.model = span.model
            api_span.cost_per_input_token = span.cost_per_input_token
            api_span.cost_per_output_token = span.cost_per_output_token
            if span.attributes:
                api_span.input_token_count = span.attributes.input_token_count
                api_span.output_token_count = span.attributes.output_token_count

        return api_span


trace_manager = TraceManager()

########################################################
### Observer #############################################
########################################################


class Observer:
    def __init__(
        self,
        span_type: Union[
            Literal["agent", "llm", "retriever", "tool"], str, None
        ],
        func_name: str,
        metrics: Optional[Union[List[str], List[BaseMetric]]] = None,
        metric_collection: Optional[str] = None,
        _progress: Optional[Progress] = None,
        _pbar_callback_id: Optional[int] = None,
        **kwargs,
    ):
        self.start_time: float
        self.end_time: float
        self.status: TraceSpanStatus
        self.error: Optional[str] = None
        self.attributes: Optional[Attributes] = None
        self.uuid: str = str(uuid.uuid4())
        # Initialize trace_uuid and parent_uuid as None, they will be set in __enter__
        self.trace_uuid: Optional[str] = None
        self.parent_uuid: Optional[str] = None

        # Separate observe kwargs and function kwargs
        self.observe_kwargs = kwargs.get("observe_kwargs", {})
        self.function_kwargs = kwargs.get("function_kwargs", {})
        self.result = None

        self.name: str = self.observe_kwargs.get("name", func_name)
        self.metrics = metrics
        self.metric_collection = metric_collection
        self.span_type: SpanType | str = (
            self.name if span_type is None else span_type
        )
        self._progress = _progress
        self._pbar_callback_id = _pbar_callback_id

    def __enter__(self):
        """Enter the tracer context, creating a new span and setting up parent-child relationships."""
        self.start_time = perf_counter()

        # Get the current span from the context
        parent_span = current_span_context.get()

        # Determine trace_uuid and parent_uuid before creating the span instance
        if parent_span:
            self.parent_uuid = parent_span.uuid
            self.trace_uuid = parent_span.trace_uuid
        else:
            current_trace = current_trace_context.get()
            if current_trace:
                self.trace_uuid = current_trace.uuid
            else:
                trace = trace_manager.start_new_trace(
                    metric_collection=self.metric_collection,
                    metrics=self.metrics,
                )
                self.trace_uuid = trace.uuid
                current_trace_context.set(trace)

        # Now create the span instance with the correct trace_uuid and parent_uuid
        span_instance = self.create_span_instance()

        # Add the span to active spans and to its trace
        trace_manager.add_span(span_instance)
        trace_manager.add_span_to_trace(span_instance)

        # Set this span as the current span in the context
        current_span_context.set(span_instance)

        if (
            parent_span
            and parent_span.progress is not None
            and parent_span.pbar_callback_id is not None
        ):
            self._progress = parent_span.progress
            self._pbar_callback_id = parent_span.pbar_callback_id

        if self._progress is not None and self._pbar_callback_id is not None:
            span_instance.progress = self._progress
            span_instance.pbar_callback_id = self._pbar_callback_id

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the tracer context, updating the span status and handling trace completion."""
        end_time = perf_counter()
        # Get the current span from the context instead of looking it up by UUID
        current_span = current_span_context.get()
        # print(self.uuid)
        # print("Exit span: ", current_span)
        # print("\n" * 10)
        if not current_span or current_span.uuid != self.uuid:
            print(
                f"Error: Current span in context does not match the span being exited. Expected UUID: {self.uuid}, Got: {current_span.uuid if current_span else 'None'}"
            )
            return

        current_span.end_time = end_time
        if exc_type is not None:
            current_span.status = TraceSpanStatus.ERRORED
            current_span.error = str(exc_val)
        else:
            current_span.status = TraceSpanStatus.SUCCESS

        self.update_span_attributes(current_span)
        if current_span.input is None:
            current_span.input = self.function_kwargs
        if current_span.output is None:
            current_span.output = self.result

        trace_manager.remove_span(self.uuid)
        if current_span.parent_uuid:
            parent_span = trace_manager.get_span_by_uuid(
                current_span.parent_uuid
            )
            if parent_span:
                current_span_context.set(parent_span)
            else:
                current_span_context.set(None)
        else:
            current_trace = current_trace_context.get()
            if current_trace.input is None:
                current_trace.input = self.function_kwargs
            if current_trace.output is None:
                current_trace.output = self.result
            if current_trace and current_trace.uuid == current_span.trace_uuid:
                other_active_spans = [
                    span
                    for span in trace_manager.active_spans.values()
                    if span.trace_uuid == current_span.trace_uuid
                ]

                if not other_active_spans:
                    trace_manager.end_trace(current_span.trace_uuid)
                    current_trace_context.set(None)

            current_span_context.set(None)

        if self._progress is not None and self._pbar_callback_id is not None:
            self._progress.update(self._pbar_callback_id, advance=1)

    def create_span_instance(self):
        """Create a span instance based on the span type."""

        span_kwargs = {
            "uuid": self.uuid,
            "trace_uuid": self.trace_uuid,
            "parent_uuid": self.parent_uuid,
            "start_time": self.start_time,
            "end_time": None,
            "status": TraceSpanStatus.SUCCESS,
            "children": [],
            "name": self.name,
            # "metadata": None,
            "input": None,
            "output": None,
            "metrics": self.metrics,
            "metric_collection": self.metric_collection,
        }

        if self.span_type == SpanType.AGENT.value:
            available_tools = self.observe_kwargs.get("available_tools", [])
            agent_handoffs = self.observe_kwargs.get("agent_handoffs", [])

            return AgentSpan(
                **span_kwargs,
                attributes=None,
                available_tools=available_tools,
                agent_handoffs=agent_handoffs,
            )
        elif self.span_type == SpanType.LLM.value:
            model = self.observe_kwargs.get("model", None)
            if model is None and not trace_manager.openai_client:
                raise ValueError(
                    "Either provide a model in observe or configure an openai_client in trace_manager. For more information on openai_client, see https://documentation.confident-ai.com/llm-tracing/integrations/openai"
                )
            return LlmSpan(**span_kwargs, attributes=None, model=model)
        elif self.span_type == SpanType.RETRIEVER.value:
            embedder = self.observe_kwargs.get("embedder", None)
            if embedder is None:
                raise ValueError("embedder is required for RetrieverSpan")

            return RetrieverSpan(
                **span_kwargs, attributes=None, embedder=embedder
            )

        elif self.span_type == SpanType.TOOL.value:
            return ToolSpan(
                **span_kwargs, attributes=None, **self.observe_kwargs
            )
        else:
            return BaseSpan(**span_kwargs)

    def update_span_attributes(self, current_span: BaseSpan):
        """Update the span instance with execution results."""

        if isinstance(current_span, AgentSpan):
            if current_span and isinstance(
                current_span.attributes, AgentAttributes
            ):
                current_span.input = trace_manager.mask(
                    current_span.attributes.input
                )
                current_span.output = trace_manager.mask(
                    current_span.attributes.output
                )
            else:
                current_span.input = trace_manager.mask(self.function_kwargs)
                current_span.output = trace_manager.mask(self.result)

        elif isinstance(current_span, LlmSpan):
            if current_span and isinstance(
                current_span.attributes, LlmAttributes
            ):
                current_span.input = trace_manager.mask(
                    current_span.attributes.input
                )
                current_span.output = trace_manager.mask(
                    current_span.attributes.output
                )

        elif isinstance(current_span, RetrieverSpan):
            if current_span and isinstance(
                current_span.attributes, RetrieverAttributes
            ):
                current_span.input = trace_manager.mask(
                    current_span.attributes.embedding_input
                )
                current_span.output = trace_manager.mask(
                    current_span.attributes.retrieval_context
                )

        elif isinstance(current_span, ToolSpan):
            if current_span and isinstance(
                current_span.attributes, ToolAttributes
            ):
                current_span.input = trace_manager.mask(
                    current_span.attributes.input_parameters
                )
                current_span.output = trace_manager.mask(
                    current_span.attributes.output
                )
            else:
                current_span.input = trace_manager.mask(self.function_kwargs)
                current_span.output = trace_manager.mask(self.result)
        else:
            current_span.input = trace_manager.mask(self.function_kwargs)
            current_span.output = trace_manager.mask(self.result)


########################################################
### Decorator ##########################################
########################################################


def observe(
    _func: Optional[Callable] = None,
    *,
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
    type: Optional[
        Union[Literal["agent", "llm", "retriever", "tool"], str]
    ] = None,
    **observe_kwargs,
):
    """
    Decorator to trace a function as a span.

    Args:
        span_type: The type of span to create (AGENT, LLM, RETRIEVER, TOOL, or custom string)
        **observe_kwargs: Additional arguments to pass to the Observer

    Returns:
        A decorator function that wraps the original function with a Observer
    """

    def decorator(func):
        func_name = func.__name__  # Get func_name outside wrappers

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **func_kwargs):
                # func_name = func.__name__ # Removed from here
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **func_kwargs)
                bound_args.apply_defaults()

                # Construct complete kwargs dictionary & pass all kwargs with consistent naming
                complete_kwargs = dict(bound_args.arguments)
                observer_kwargs = {
                    "observe_kwargs": observe_kwargs,
                    "function_kwargs": complete_kwargs,  # Now contains all args mapped to their names
                }
                with Observer(
                    type,
                    metrics=metrics,
                    metric_collection=metric_collection,
                    func_name=func_name,
                    **observer_kwargs,
                ) as observer:
                    # Call the original function
                    result = await func(*args, **func_kwargs)
                    # Capture the result
                    observer.result = result
                    return result

            # Set the marker attribute on the wrapper
            setattr(async_wrapper, "_is_deepeval_observed", True)
            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                # func_name = func.__name__ # Removed from here
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **func_kwargs)
                bound_args.apply_defaults()
                complete_kwargs = dict(bound_args.arguments)
                observer_kwargs = {
                    "observe_kwargs": observe_kwargs,
                    "function_kwargs": complete_kwargs,  # Now contains all args mapped to their names
                }
                with Observer(
                    type,
                    metrics=metrics,
                    func_name=func_name,
                    **observer_kwargs,
                ) as observer:
                    # Call the original function
                    result = func(*args, **func_kwargs)
                    # Capture the result
                    observer.result = result
                    return result

            # Set the marker attribute on the wrapper
            setattr(wrapper, "_is_deepeval_observed", True)
            return wrapper

    if _func is not None and callable(_func):
        return decorator(_func)

    return decorator
