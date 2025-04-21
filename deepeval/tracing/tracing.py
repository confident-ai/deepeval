from contextvars import ContextVar
from datetime import datetime, time, timezone
from enum import Enum
from typing import Any, List, Set, Union, Optional, Dict, Literal
from time import perf_counter, sleep
from pydantic import BaseModel, Field
import inspect
import uuid
from rich.console import Console
import random
import threading
import queue
import asyncio

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.test_case import LLMTestCase
from deepeval.utils import dataclass_to_dict, is_confident
from deepeval.prompt import Prompt
from deepeval.tracing.api import TraceApi, BaseApiSpan, SpanApiType


def to_zod_compatible_iso(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def perf_counter_to_datetime(perf_counter_value: float) -> datetime:
    """
    Convert a perf_counter value to a datetime object.

    Args:
        perf_counter_value: A float value from perf_counter()

    Returns:
        A datetime object representing the current time
    """
    # Get the current time
    current_time = datetime.now(timezone.utc)
    # Calculate the time difference in seconds
    time_diff = current_time.timestamp() - perf_counter()
    # Convert perf_counter value to a real timestamp
    timestamp = time_diff + perf_counter_value
    # Return as a datetime object
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


class SpanType(Enum):
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


class TraceSpanStatus(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


class AgentAttributes(BaseModel):
    # input
    input: Union[str, Dict, list]
    # output
    output: Union[str, Dict, list]


class LlmAttributes(BaseModel):
    # input
    input: str
    # output
    output: str
    prompt: Optional[Prompt] = None

    # Optional variables
    input_token_count: Optional[int] = Field(
        None, serialization_alias="inputTokenCount"
    )
    output_token_count: Optional[int] = Field(
        None, serialization_alias="outputTokenCount"
    )

    model_config = {"arbitrary_types_allowed": True}


class RetrieverAttributes(BaseModel):
    # input
    embedding_input: str = Field(serialization_alias="embeddingInput")
    # output
    retrieval_context: List[str] = Field(serialization_alias="retrievalContext")

    # Optional variables
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    chunk_size: Optional[int] = Field(None, serialization_alias="chunkSize")


# Don't have to call this manually, will be taken as input and output of function
# Can be overridden by user
class ToolAttributes(BaseModel):
    # input
    input_parameters: Optional[Dict[str, Any]] = Field(
        None, serialization_alias="inputParameters"
    )
    # output
    output: Optional[Any] = None


########################################################
### Trace Types #######################################
########################################################


class BaseSpan(BaseModel):
    uuid: str
    status: TraceSpanStatus
    children: List["BaseSpan"]
    trace_uuid: str = Field(serialization_alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, serialization_alias="parentUuid")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    name: Optional[str] = None
    # metadata: Optional[Dict] = None
    input: Optional[Union[str, Dict, list]] = None
    output: Optional[Union[str, Dict, list]] = None
    error: Optional[str] = None

    llm_test_case: Optional[LLMTestCase] = None
    metrics: Optional[List[str]] = None


class AgentSpan(BaseSpan):
    name: str
    available_tools: List[str] = []
    agent_handoffs: List[str] = []
    attributes: Optional[AgentAttributes] = None

    def set_attributes(self, attributes: AgentAttributes):
        self.attributes = attributes


class LlmSpan(BaseSpan):
    model: str
    attributes: Optional[LlmAttributes] = None
    cost_per_input_token: Optional[float] = Field(
        None, serialization_alias="costPerInputToken"
    )
    cost_per_output_token: Optional[float] = Field(
        None, serialization_alias="costPerOutputToken"
    )

    def set_attributes(self, attributes: LlmAttributes):
        self.attributes = attributes


class RetrieverSpan(BaseSpan):
    embedder: str
    attributes: Optional[RetrieverAttributes] = None

    def set_attributes(self, attributes: RetrieverAttributes):
        self.attributes = attributes


class ToolSpan(BaseSpan):
    name: str  # Required name for ToolSpan
    attributes: Optional[ToolAttributes] = None
    description: Optional[str] = None

    def set_attributes(self, attributes: ToolAttributes):
        self.attributes = attributes


Attributes = Union[
    AgentAttributes, LlmAttributes, RetrieverAttributes, ToolAttributes
]


class Trace(BaseModel):
    uuid: str = Field(serialization_alias="uuid")
    status: TraceSpanStatus
    root_spans: List[BaseSpan] = Field(serialization_alias="rootSpans")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    # metadata: Optional[Dict] = None


# Create a context variable to track the current span
current_span_context: ContextVar[Optional[BaseSpan]] = ContextVar(
    "current_span", default=None
)

# Create a context variable to track the current trace
current_trace_context: ContextVar[Optional[Trace]] = ContextVar(
    "current_trace", default=None
)


# Simple stack implementation for traces and spans
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

    def start_new_trace(self) -> Trace:
        """Start a new trace and set it as the current trace."""
        trace_uuid = str(uuid.uuid4())
        new_trace = Trace(
            uuid=trace_uuid,
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=perf_counter(),
            end_time=None,
            # metadata=None,
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
            self.post_trace(trace)

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

    def post_trace(self, trace: Trace) -> Optional[str]:
        if not is_confident():
            return None

        # Add the trace to the queue
        self._trace_queue.put(trace)

        # Start the worker thread if it's not already running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._process_trace_queue,
                daemon=True,  # Make it a daemon so it won't block program exit
            )
            self._worker_thread.start()

        return "ok"

    def _process_trace_queue(self):
        """Worker thread function that processes the trace queue with throttling."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Define the async worker function
            async def async_worker():
                while True:
                    try:
                        # Clean up completed tasks
                        self._clean_finished_tasks()

                        # Get a trace from the queue
                        trace = self._trace_queue.get(block=True, timeout=1.0)

                        # Apply rate limiting
                        now = perf_counter()
                        time_since_last = now - self._last_post_time
                        if time_since_last < self._min_interval:
                            await asyncio.sleep(
                                self._min_interval - time_since_last
                            )

                        # Update the last post time BEFORE making the API call
                        self._last_post_time = perf_counter()

                        # Create the API request but don't await it
                        async def _a_send_trace(trace_obj):
                            try:
                                trace_api = self.create_trace_api(trace_obj)

                                try:
                                    body = trace_api.model_dump(
                                        by_alias=True, exclude_none=True
                                    )
                                except AttributeError:
                                    # Pydantic version below 2.0
                                    body = trace_api.dict(
                                        by_alias=True, exclude_none=True
                                    )

                                # Send the request without blocking the worker
                                api = Api()
                                await api.a_send_request(
                                    method=HttpMethods.POST,
                                    endpoint=Endpoints.TRACING_ENDPOINT,
                                    body=body,
                                )
                            except Exception as e:
                                console = Console()
                                console.print(
                                    f"[dim][Tracing][/dim] Error posting trace: {str(e)}"
                                )

                        # Create a task for this trace and add to tracking set
                        task = asyncio.create_task(_a_send_trace(trace))
                        self._in_flight_tasks.add(task)
                        self._trace_queue.task_done()

                    except queue.Empty:
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        console = Console()
                        console.print(
                            f"[dim][Tracing][/dim] Error in worker: {str(e)}"
                        )
                        await asyncio.sleep(1.0)  # Wait a bit before continuing

            loop.run_until_complete(async_worker())

        except Exception as e:
            console = Console()
            console.print(
                f"[dim][Tracing][/dim] Async setup failed, using sync: {str(e)}"
            )

            while True:
                try:
                    trace = self._trace_queue.get(block=True, timeout=1.0)

                    now = perf_counter()
                    time_since_last = now - self._last_post_time
                    if time_since_last < self._min_interval:
                        sleep(self._min_interval - time_since_last)

                    # Update the last post time BEFORE making the API call
                    self._last_post_time = perf_counter()

                    try:
                        trace_api = self.create_trace_api(trace)

                        try:
                            body = trace_api.model_dump(
                                by_alias=True, exclude_none=True
                            )
                        except AttributeError:
                            # Pydantic version below 2.0
                            body = trace_api.dict(
                                by_alias=True, exclude_none=True
                            )

                        # Send the request
                        api = Api()
                        api.send_request(
                            method=HttpMethods.POST,
                            endpoint=Endpoints.TRACING_ENDPOINT,
                            body=body,
                        )
                    except Exception as e:
                        console = Console()
                        console.print(
                            f"[dim][Tracing][/dim] Error posting trace: {str(e)}"
                        )

                    self._trace_queue.task_done()

                except queue.Empty:
                    sleep(0.1)
                except Exception as e:
                    console = Console()
                    console.print(
                        f"[dim][Tracing][/dim] Error in worker: {str(e)}"
                    )
                    sleep(1.0)  # Wait a bit before continuing

    def _clean_finished_tasks(self) -> None:
        done_tasks = {task for task in self._in_flight_tasks if task.done()}
        self._in_flight_tasks -= done_tasks
        for task in done_tasks:
            try:
                task.exception()
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass

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

        # Create and return the TraceApi object
        return TraceApi(
            uuid=trace.uuid,
            baseSpans=base_spans,
            agentSpans=agent_spans,
            llmSpans=llm_spans,
            retrieverSpans=retriever_spans,
            toolSpans=tool_spans,
            startTime=start_time,
            endTime=end_time,
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
            else:
                # Fallback to standard logic if attributes are not set
                input_data = span.input
                output_data = span.output

        elif isinstance(span, LlmSpan):
            # For LlmSpan, input is attributes.input, output is attributes.output
            if span.attributes:
                input_data = span.attributes.input
                output_data = span.attributes.output
            else:
                # Fallback to standard logic if attributes are not set
                input_data = span.input
                output_data = span.output
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
            error=span.error,
            testCaseInput=(
                span.llm_test_case.input if span.llm_test_case else None
            ),
            testCaseActualOutput=(
                span.llm_test_case.actual_output if span.llm_test_case else None
            ),
            testCaseRetrievalContext=(
                span.llm_test_case.retrieval_context
                if span.llm_test_case
                else None
            ),
            metrics=span.metrics,
        )

        # Add type-specific attributes
        if isinstance(span, AgentSpan):
            api_span.available_tools = span.available_tools
            api_span.agent_handoffs = span.agent_handoffs
        elif isinstance(span, ToolSpan):
            api_span.description = span.description
        elif isinstance(span, RetrieverSpan) and span.attributes:
            api_span.embedder = span.embedder
            api_span.top_k = span.attributes.top_k
            api_span.chunk_size = span.attributes.chunk_size
        elif isinstance(span, LlmSpan):
            if span.attributes:
                api_span.input_token_count = span.attributes.input_token_count
                api_span.output_token_count = span.attributes.output_token_count

            api_span.model = span.model
            api_span.cost_per_input_token = span.cost_per_input_token
            api_span.cost_per_output_token = span.cost_per_output_token

        return api_span


trace_manager = TraceManager()

########################################################
### Tracer #############################################
########################################################


class Tracer:
    def __init__(
        self,
        span_type: Union[
            Literal["agent", "llm", "retriever", "tool"], str, None
        ],
        func_name: str,
        metrics: Optional[List[str]] = None,
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
        self.span_type: SpanType | str = (
            self.name if span_type is None else span_type
        )

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
                trace = trace_manager.start_new_trace()
                self.trace_uuid = trace.uuid
                current_trace_context.set(trace)

        # Now create the span instance with the correct trace_uuid and parent_uuid
        span_instance = self.create_span_instance()

        # Add the span to active spans and to its trace
        trace_manager.add_span(span_instance)
        trace_manager.add_span_to_trace(span_instance)

        # Set this span as the current span in the context
        current_span_context.set(span_instance)
        # print("Enter span instance: ", span_instance)
        # print("\n" * 10)
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
            current_span.status = TraceSpanStatus.ERROR
            current_span.error = str(exc_val)
        else:
            current_span.status = TraceSpanStatus.SUCCESS

        self.update_span_attributes(current_span)
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
            if model is None:
                raise ValueError("model is required for LlmSpan")

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
                current_span.input = current_span.attributes.input
                current_span.output = current_span.attributes.output
            else:
                current_span.input = self.function_kwargs
                current_span.output = self.result

        elif isinstance(current_span, LlmSpan):
            if not current_span.attributes or not isinstance(
                current_span.attributes, LlmAttributes
            ):
                raise ValueError("LlmSpan requires LlmAttributes")
            current_span.input = current_span.attributes.input
            current_span.output = current_span.attributes.output

        elif isinstance(current_span, RetrieverSpan):
            if not current_span.attributes or not isinstance(
                current_span.attributes, RetrieverAttributes
            ):
                raise ValueError("RetrieverSpan requires RetrieverAttributes")
            current_span.input = current_span.attributes.embedding_input
            current_span.output = current_span.attributes.retrieval_context

        elif isinstance(current_span, ToolSpan):
            if current_span and isinstance(
                current_span.attributes, ToolAttributes
            ):
                current_span.input = current_span.attributes.input_parameters
                current_span.output = current_span.attributes.output
            else:
                current_span.input = self.function_kwargs
                current_span.output = self.result
        else:
            current_span.input = self.function_kwargs
            current_span.output = self.result


########################################################
### Decorator ##########################################
########################################################


def observe(
    type: Union[Literal["agent", "llm", "retriever", "tool"], str, None],
    metrics: Optional[List[str]] = None,
    **observe_kwargs,
):
    """
    Decorator to trace a function as a span.

    Args:
        span_type: The type of span to create (AGENT, LLM, RETRIEVER, TOOL, or custom string)
        **observe_kwargs: Additional arguments to pass to the Tracer

    Returns:
        A decorator function that wraps the original function with a Tracer
    """

    def decorator(func):
        def wrapper(*args, **func_kwargs):
            func_name = func.__name__

            # Get function signature to map args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **func_kwargs)
            bound_args.apply_defaults()

            # Construct complete kwargs dictionary
            complete_kwargs = dict(bound_args.arguments)

            # Pass all kwargs with consistent naming
            tracer_kwargs = {
                "observe_kwargs": observe_kwargs,
                "function_kwargs": complete_kwargs,  # Now contains all args mapped to their names
            }

            with Tracer(
                type, metrics=metrics, func_name=func_name, **tracer_kwargs
            ) as tracer:

                # Call the original function
                result = func(*args, **func_kwargs)

                # Capture the result
                tracer.result = result

                return result

        return wrapper

    return decorator


def update_current_span_attributes(attributes: Attributes):
    current_span = current_span_context.get()
    if current_span:
        current_span.set_attributes(attributes)


def update_current_span_test_case_parameters(
    input: str,
    actual_output: str,
    retrieval_context: Optional[List[str]] = None,
):
    current_span = current_span_context.get()
    if current_span:
        current_span.llm_test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
