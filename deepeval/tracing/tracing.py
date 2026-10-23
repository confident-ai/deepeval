import json
import re
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from time import perf_counter
import functools
import inspect
import asyncio
import uuid
from deepeval.test_case import MLLMImage
from openai import OpenAI
from opentelemetry import context as otel_context
from opentelemetry.trace import Status, StatusCode
from rich.console import Console
from rich.progress import Progress

from deepeval.config.settings import get_settings
from deepeval.constants import (
    CONFIDENT_TRACE_VERBOSE,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.api import (
    BaseApiSpan,
    SpanApiType,
    TraceApi,
    TraceSpanApiStatus,
    AttachmentApi,
)
from deepeval.tracing.patchers import (
    patch_anthropic_client,
    patch_openai_client,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    EvalMode,
    EvalSession,
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
    prepare_tool_call_input_parameters,
    replace_self_with_class_name,
    make_json_serializable,
    normalize_trace_api_span_providers,
    perf_counter_to_datetime,
    to_zod_compatible_iso,
    validate_environment,
    validate_sampling_rate,
)
from deepeval.utils import dataclass_to_dict
from deepeval.tracing.internal import confident_trace_setup
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.context import (
    active_span,
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.types import TestCaseMetricPair
from deepeval.tracing.api import PromptApi
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.test_case.llm_test_case import _MLLM_IMAGE_REGISTRY

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden
    from anthropic import Anthropic


DEFAULT_TRACE_FLUSH_TIMEOUT = 30.0


class _ObservedAsyncGenIter:
    """Class-based async iterator that wraps an observed async generator.

    Python 3.11's ``async for`` with ``break`` does NOT call ``aclose()``
    on the async generator — the generator is silently abandoned.  This
    means neither ``finally`` blocks nor ``except GeneratorExit`` handlers
    will fire, and the observer span leaks.

    By using a class with ``__del__``, CPython's reference-counting GC
    calls cleanup the moment the iterator goes out of scope (immediately
    after ``break``), ensuring the span is always closed.
    """

    __slots__ = ("_agen_iter", "_observer", "_entered", "_done")

    def __init__(self, agen, observer):
        self._agen_iter = agen.__aiter__()
        self._observer = observer
        self._entered = False
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._step(self._agen_iter.__anext__)

    async def _step(self, advance):
        # The span is current only while the generator runs, as each step may
        # resume in another context.
        if not self._entered:
            self._observer.start()
            self._entered = True
        self._observer.activate()
        try:
            value = await advance()
        except StopAsyncIteration:
            self._observer.deactivate()
            self._finish()
            raise
        except Exception as e:
            self._observer.deactivate()
            self._finish_err(e)
            raise
        self._observer.deactivate()
        return value

    def _finish(self):
        if self._entered and not self._done:
            self._done = True
            self._observer.finish()

    def _finish_err(self, e):
        if self._entered and not self._done:
            self._done = True
            self._observer.finish(e)

    async def aclose(self):
        self._finish()
        await self._agen_iter.aclose()

    async def athrow(self, typ, val=None, tb=None):
        return await self._step(lambda: self._agen_iter.athrow(typ, val, tb))

    def __del__(self):
        self._finish()


class TraceManager:
    def __init__(self):
        self.traces: List[Trace] = []
        self.active_traces: Dict[str, Trace] = {}  # Map of trace_uuid to Trace
        self.active_spans: Dict[str, BaseSpan] = (
            {}
        )  # Map of span_uuid to BaseSpan

        settings = get_settings()
        self.task_bindings: "weakref.WeakKeyDictionary[asyncio.Task, dict]" = (
            weakref.WeakKeyDictionary()
        )

        # trace manager attributes
        self.confident_api_key = None
        self.custom_mask_fn: Optional[Callable] = None
        self.environment = (
            settings.CONFIDENT_TRACE_ENVIRONMENT
            if settings.CONFIDENT_TRACE_ENVIRONMENT is not None
            else Environment.DEVELOPMENT.value
        )
        validate_environment(self.environment)

        self.sampling_rate = settings.CONFIDENT_TRACE_SAMPLE_RATE
        validate_sampling_rate(self.sampling_rate)
        self.anthropic_client = None
        self.openai_client = None
        self.tracing_enabled = True

        # All per-evaluation-run state is grouped on this single object.
        # See deepeval.tracing.types.EvalSession for the field-by-field
        # breakdown. Resetting an in-flight evaluation is a one-line
        # ``self.eval_session = EvalSession()``, which makes exit cleanup
        # atomic and impossible to half-do.
        self.eval_session: EvalSession = EvalSession()

    @property
    def is_evaluating(self) -> bool:
        """True when running under any evaluation pipeline (any non-OFF mode).

        Delegates to ``eval_session`` so external callers don't need to know
        about the session indirection.
        """
        return self.eval_session.is_evaluating

    @property
    def is_iterator(self) -> bool:
        """True when running under either evals_iterator path (sync or async)."""
        return self.eval_session.is_iterator

    def mask(self, data: Any):
        if self.custom_mask_fn is not None:
            return self.custom_mask_fn(data)
        else:
            return data

    def configure(
        self,
        mask: Optional[Callable] = None,
        environment: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        confident_api_key: Optional[str] = None,
        anthropic_client: Optional["Anthropic"] = None,
        openai_client: Optional[OpenAI] = None,
        tracing_enabled: Optional[bool] = None,
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
        if anthropic_client is not None:
            self.anthropic_client = anthropic_client
            patch_anthropic_client(anthropic_client)
        if tracing_enabled is not None:
            self.tracing_enabled = tracing_enabled

    def start_new_trace(
        self,
        metric_collection: Optional[str] = None,
        trace_uuid: Optional[str] = None,
    ) -> Trace:
        """Start a new trace and set it as the current trace."""
        if trace_uuid is None:
            trace_uuid = str(uuid.uuid4())
        new_trace = Trace(
            uuid=trace_uuid,
            root_spans=[],
            status=TraceSpanStatus.IN_PROGRESS,
            start_time=perf_counter(),
            end_time=None,
            metric_collection=metric_collection,
            confident_api_key=self.confident_api_key,
        )
        self.active_traces[trace_uuid] = new_trace
        self.traces.append(new_trace)
        if self.eval_session.mode == EvalMode.ITERATOR_ASYNC:
            self.eval_session.pending_traces[trace_uuid] = new_trace
            # Associate the current Golden with this trace so we can
            # later evaluate traces against the correct golden, even if more traces
            # are created than goldens or the order interleaves.
            try:
                from deepeval.contextvars import get_current_golden

                current_golden = get_current_golden()
                if current_golden is not None:
                    self.eval_session.trace_uuid_to_golden[trace_uuid] = (
                        current_golden
                    )
            except Exception:
                # not much we can do, but if the golden is not there during evaluation
                # we will write out a verbose debug log
                pass
        return new_trace

    def end_trace(self, trace_uuid: str):
        """End a specific trace by its UUID."""

        if trace_uuid in self.active_traces:
            trace = self.active_traces[trace_uuid]
            trace.end_time = (
                perf_counter() if trace.end_time is None else trace.end_time
            )

            # Default to SUCCESS for completed traces
            # This assumes that if a trace completes, it was successful overall
            # Users can manually set the status to ERROR if needed
            if trace.status == TraceSpanStatus.IN_PROGRESS:
                trace.status = TraceSpanStatus.SUCCESS

            if trace_testing_manager.test_name:
                # Trace testing mode is enabled
                # Instead posting the trace to the queue, it will be stored in this global variable
                body = self.create_trace_api(trace).model_dump(
                    by_alias=True, exclude_none=True
                )
                trace_testing_manager.test_dict = make_json_serializable(body)
            # Outside these, confident-trace exports the trace's spans.
            elif self.is_evaluating:
                if self.eval_session.mode == EvalMode.ITERATOR_ASYNC:
                    session = self.eval_session
                    if session.test_case_metrics:
                        pass
                    elif (
                        trace_uuid in session.pending_traces
                        and trace not in session.traces_to_evaluate
                    ):
                        # Per-trace dedup: an integration may have already
                        # queued this exact trace before calling end_trace
                        # (e.g. llama_index does this in prepare_to_exit_span).
                        session.traces_to_evaluate.append(trace)
                        # Sort by start order. `pending_traces` is insertion-
                        # ordered, so build the position map once instead of
                        # doing an O(n) `index()` lookup per comparison.
                        order = {
                            uuid: i
                            for i, uuid in enumerate(session.pending_traces)
                        }
                        session.traces_to_evaluate.sort(
                            key=lambda t: order.get(t.uuid, len(order))
                        )
                else:
                    # print(f"Ending trace: {trace.root_spans}")
                    self.environment = Environment.TESTING
                    if (
                        trace.root_spans
                        and len(trace.root_spans) > 0
                        and trace.root_spans[0].children
                        and len(trace.root_spans[0].children) > 0
                    ):
                        trace.root_spans = [trace.root_spans[0].children[0]]
                    for root_span in trace.root_spans:
                        root_span.parent_uuid = None

            # Remove from active traces
            del self.active_traces[trace_uuid]

            # Evict finished traces to bound memory usage.
            # Skipped during evaluation (pipeline reads them after completion).
            if not self.is_evaluating:
                try:
                    self.traces.remove(trace)
                except ValueError:
                    pass

    def discard_trace(self, trace_uuid: str):
        """Forget an active trace without ending, posting or evaluating it."""
        trace = self.active_traces.pop(trace_uuid, None)
        if trace is None:
            return
        try:
            self.traces.remove(trace)
        except ValueError:
            pass
        self.eval_session.pending_traces.pop(trace_uuid, None)
        self.eval_session.trace_uuid_to_golden.pop(trace_uuid, None)

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
                trace.root_spans.append(span)

        # Adopt any already-rooted spans whose ``parent_uuid`` matches this
        # span. Without this step, the OTel-via-SimpleSpanProcessor flow
        # produces sibling roots when a child's ``on_end`` lands at the
        # exporter BEFORE its parent's: the exporter calls add_span_to_trace
        # for the child first, finds no parent in ``active_spans``, and parks
        # the child in ``root_spans``. When the parent finally arrives we
        # need to re-knit the tree, otherwise the trace ships with multiple
        # logical roots and downstream walkers (e.g. the evals_iterator DFS
        # which only visits ``root_spans[0]``) silently drop subtrees.
        self._reparent_orphan_roots(trace, span)

    @staticmethod
    def _reparent_orphan_roots(trace: Trace, parent: BaseSpan) -> None:
        """Move root_spans whose ``parent_uuid == parent.uuid`` under
        ``parent`` and remove them from ``trace.root_spans``.

        Mutates ``trace.root_spans`` and ``parent.children`` in place. No-op
        if no orphan roots match. Iterates a snapshot of ``root_spans`` so we
        can safely remove items as we go.
        """
        if not trace.root_spans:
            return
        for orphan in list(trace.root_spans):
            if orphan is parent:
                continue
            if orphan.parent_uuid == parent.uuid:
                trace.root_spans.remove(orphan)
                parent.children.append(orphan)

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
        if get_settings().CONFIDENT_TRACE_VERBOSE and not self.is_evaluating:
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
                    f"\nTo disable dev logging, set {CONFIDENT_TRACE_VERBOSE}=0 as an environment variable.",
                )
            else:
                console.print(message_prefix, env_text, message)

    def flush(self, timeout: float = DEFAULT_TRACE_FLUSH_TIMEOUT) -> bool:
        """Wait until confident-trace has exported every ended span, or
        ``timeout`` seconds elapse.

        Returns ``True`` once nothing is left to export, ``False`` if the
        timeout expired first.
        """
        setup = confident_trace_setup()
        return setup.flush_confident_trace(timeout) if setup else True

    async def a_flush(
        self, timeout: float = DEFAULT_TRACE_FLUSH_TIMEOUT
    ) -> bool:
        """Async counterpart of :meth:`flush` that yields instead of blocking."""
        return await asyncio.to_thread(self.flush, timeout)

    def create_nested_spans_dict(self, span: BaseSpan) -> Dict[str, Any]:
        api_span = self._convert_span_to_api_span(span)
        trace_dict = api_span.__dict__.copy()

        # Remove specific keys
        for key in (
            "uuid",
            "trace_uuid",
            "parent_uuid",
            "end_time",
            "start_time",
            "status",
            "llm_test_case",
            "metrics_data",
            "metric_collection",
            "metadata",
        ):
            trace_dict.pop(key, None)

        # Remove all keys with None values
        trace_dict = {k: v for k, v in trace_dict.items() if v is not None}

        trace_dict["children"] = []
        for child in span.children or []:
            child_api_span = self.create_nested_spans_dict(child)
            trace_dict["children"].append(child_api_span)

        return trace_dict

    def create_trace_api(self, trace: Trace) -> TraceApi:
        # Initialize empty lists for each span type
        base_spans = []
        agent_spans = []
        llm_spans = []
        retriever_spans = []
        tool_spans = []

        # Process all spans in the trace iteratively
        span_stack = list(trace.root_spans)  # Start with root spans

        merged_attachment_docs: Dict[str, MLLMImage] = {}
        trace_docs = self._extract_attachments(trace)
        if trace_docs:
            merged_attachment_docs.update(trace_docs)

        while span_stack:
            span = span_stack.pop()

            if span.drop:
                if span.children:
                    for child in span.children:
                        child.parent_uuid = span.parent_uuid
                    span_stack.extend(span.children)
                continue

            span_docs = self._extract_attachments(span)
            if span_docs:
                merged_attachment_docs.update(span_docs)

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

            # Add children to the stack for processing
            if span.children:
                span_stack.extend(span.children)

        # Convert perf_counter values to ISO 8601 strings.
        # Fall back to current time when a value is missing.
        start_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(trace.start_time))
            if trace.start_time
            else to_zod_compatible_iso(perf_counter_to_datetime(perf_counter()))
        )
        effective_end_time = (
            trace.end_time if trace.end_time else perf_counter()
        )
        end_time = to_zod_compatible_iso(
            perf_counter_to_datetime(effective_end_time)
        )

        api_attachments = None
        if merged_attachment_docs:
            api_attachments = {}
            for doc_id, doc in merged_attachment_docs.items():
                api_attachments[doc_id] = AttachmentApi(
                    url=doc.url,
                    mimeType=doc.mimeType,
                    dataBase64=doc.dataBase64,
                )

        trace_api = TraceApi(
            uuid=trace.uuid,
            baseSpans=base_spans,
            agentSpans=agent_spans,
            llmSpans=llm_spans,
            retrieverSpans=retriever_spans,
            toolSpans=tool_spans,
            startTime=start_time,
            endTime=end_time,
            metadata=trace.metadata,
            name=trace.name,
            tags=trace.tags,
            threadId=trace.thread_id,
            userId=trace.user_id,
            input=trace.input,
            output=trace.output,
            metricCollection=trace.metric_collection,
            retrievalContext=(
                [
                    rc.context if hasattr(rc, "context") else rc
                    for rc in trace.retrieval_context
                ]
                if trace.retrieval_context
                else None
            ),
            context=trace.context,
            expectedOutput=trace.expected_output,
            toolsCalled=trace.tools_called,
            expectedTools=trace.expected_tools,
            testCaseId=trace.test_case_id,
            testRunId=trace.test_run_id,
            turnId=trace.turn_id,
            confident_api_key=trace.confident_api_key,
            environment=(
                self.environment if not trace.environment else trace.environment
            ),
            status=(
                TraceSpanApiStatus.SUCCESS
                if trace.status == TraceSpanStatus.SUCCESS
                else TraceSpanApiStatus.ERRORED
            ),
            attachments=api_attachments,
        )
        normalize_trace_api_span_providers(trace_api)
        return trace_api

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
        input_data = span.input
        output_data = span.output

        # Convert perf_counter values to ISO 8601 strings.
        # Fall back to current time if end_time was never set (e.g. sync
        # generators whose __exit__ ran in a different thread-pool thread).
        start_time = (
            to_zod_compatible_iso(perf_counter_to_datetime(span.start_time))
            if span.start_time
            else to_zod_compatible_iso(perf_counter_to_datetime(perf_counter()))
        )
        effective_end_time = span.end_time if span.end_time else perf_counter()
        end_time = to_zod_compatible_iso(
            perf_counter_to_datetime(effective_end_time)
        )

        from deepeval.evaluate.utils import create_metric_data

        # Create the base API span
        api_span = BaseApiSpan(
            uuid=span.uuid,
            name=span.name,
            status=span.status.value,
            type=span_type,
            parentUuid=span.parent_uuid,
            startTime=start_time,
            endTime=end_time,
            input=input_data,
            output=output_data,
            metadata=span.metadata,
            error=span.error,
            integration=span.integration,
            metricCollection=span.metric_collection,
            metricsData=(
                [create_metric_data(metric) for metric in span.metrics]
                if span.metrics
                else None
            ),
            retrievalContext=(
                [
                    rc.context if hasattr(rc, "context") else rc
                    for rc in span.retrieval_context
                ]
                if span.retrieval_context
                else None
            ),
            context=span.context,
            expectedOutput=span.expected_output,
            toolsCalled=span.tools_called,
            expectedTools=span.expected_tools,
        )

        # Add type-specific attributes
        if isinstance(span, AgentSpan):
            api_span.available_tools = span.available_tools
            api_span.agent_handoffs = span.agent_handoffs
        elif isinstance(span, ToolSpan):
            api_span.description = span.description
        elif isinstance(span, RetrieverSpan):
            api_span.embedder = span.embedder
            api_span.top_k = span.top_k
            api_span.chunk_size = span.chunk_size
        elif isinstance(span, LlmSpan):
            api_span.model = span.model
            api_span.provider = span.provider
            # api_span.prompt = PromptApi(alias=alias, version=version, hash=hash) # Legacy won't be using anymore
            api_span.cost_per_input_token = span.cost_per_input_token
            api_span.cost_per_output_token = span.cost_per_output_token
            api_span.input_token_count = span.input_token_count
            api_span.output_token_count = span.output_token_count
            if span.prompt:
                api_span.prompt_alias = span.prompt.alias
                api_span.prompt_commit_hash = span.prompt.hash
                api_span.prompt_label = span.prompt.label
                api_span.prompt_version = span.prompt.version
            if span.prompt_alias:
                api_span.prompt_alias = span.prompt_alias
            if span.prompt_commit_hash:
                api_span.prompt_commit_hash = span.prompt_commit_hash
            if span.prompt_label:
                api_span.prompt_label = span.prompt_label
            if span.prompt_version:
                api_span.prompt_version = span.prompt_version

            processed_token_intervals = {}
            if span.token_intervals:
                for key, value in span.token_intervals.items():
                    time = to_zod_compatible_iso(
                        perf_counter_to_datetime(key),
                        microsecond_precision=True,
                    )
                    processed_token_intervals[time] = value
                api_span.token_intervals = processed_token_intervals

        return api_span

    def _extract_attachments(
        self, obj: Union[Trace, BaseSpan]
    ) -> Optional[Dict[str, MLLMImage]]:
        """Scans an object's attributes for multimodal slugs and pulls them from the global registry."""
        pattern = r"\[DEEPEVAL:(?:IMAGE|PDF):(.*?)]"
        found_docs: Dict[str, MLLMImage] = {}

        # Supported fields that may contain multimodal content
        fields_to_scan = [
            obj.input,
            obj.output,
            obj.context,
            obj.retrieval_context,
            obj.expected_output,
        ]

        matches = re.findall(pattern, str(fields_to_scan))
        for doc_id in matches:
            if doc_id in _MLLM_IMAGE_REGISTRY:
                found_docs[doc_id] = _MLLM_IMAGE_REGISTRY[doc_id]

        return found_docs if found_docs else None


trace_manager = TraceManager()

########################################################
### Flushing #############################################
########################################################


def flush_traces(timeout: float = DEFAULT_TRACE_FLUSH_TIMEOUT) -> bool:
    """Block until every ended trace has been sent to Confident AI.

    confident-trace exports spans in batches, so a process that tears down
    right after its last traced call can drop them. Call this first to give
    them a chance to finish.

    Args:
        timeout: Maximum number of seconds to wait.

    Returns:
        ``True`` if everything was sent within ``timeout``, ``False`` if
        spans were still waiting when it expired.
    """
    return trace_manager.flush(timeout=timeout)


async def a_flush_traces(
    timeout: float = DEFAULT_TRACE_FLUSH_TIMEOUT,
) -> bool:
    """Async counterpart of :func:`flush_traces` that yields instead of blocking.

    Args:
        timeout: Maximum number of seconds to wait.

    Returns:
        ``True`` if everything was sent within ``timeout``, ``False`` if
        spans were still waiting when it expired.
    """
    return await trace_manager.a_flush(timeout=timeout)


########################################################
### Observer #############################################
########################################################


def tool_call_of(span: ToolSpan) -> ToolCall:
    return ToolCall(
        name=span.name,
        description=span.description,
        input_parameters=prepare_tool_call_input_parameters(span.input),
        output=span.output,
    )


def tool_calls_under(
    span: BaseSpan, descendants: bool = False
) -> List[ToolCall]:
    """Calls for the tool spans under ``span``: its children or, with
    ``descendants``, every tool span below it that no nested agent owns."""
    tool_calls = []
    for child in span.children:
        if isinstance(child, ToolSpan):
            tool_calls.append(tool_call_of(child))
        elif descendants and not isinstance(child, AgentSpan):
            tool_calls.extend(tool_calls_under(child, descendants=True))
    return tool_calls


# The span types confident-trace knows; any other ``@observe`` type is custom.
_CUSTOM_SPAN_TYPE = "custom"
_CONFIDENT_TRACE_TYPES = frozenset(
    (
        SpanType.AGENT.value,
        SpanType.LLM.value,
        SpanType.RETRIEVER.value,
        SpanType.TOOL.value,
    )
)

# ``@observe`` keyword arguments each span type takes, with their defaults.
_OBSERVE_FIELDS: Dict[str, Dict[str, Any]] = {
    SpanType.AGENT.value: {"available_tools": [], "agent_handoffs": []},
    SpanType.LLM.value: {
        "model": None,
        "provider": None,
        "cost_per_input_token": None,
        "cost_per_output_token": None,
    },
    SpanType.RETRIEVER.value: {"embedder": None},
    SpanType.TOOL.value: {"description": None},
}


class Observer:
    """One ``@observe`` span: a confident-trace span that deepeval collects.

    While the span is open, deepeval code sees it through the bridge's
    placeholder, which is the deepeval span the trace gets when it ends.
    """

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
        self.uuid: Optional[str] = None
        self.trace_uuid: Optional[str] = None
        self.parent_uuid: Optional[str] = None

        # Separate observe kwargs and function kwargs
        self.observe_kwargs = kwargs.get("observe_kwargs", {})
        self.function_kwargs = kwargs.get("function_kwargs", {})
        self.result = None

        self.name: str = self.observe_kwargs.get("name", func_name)
        self.prompt = self.observe_kwargs.get("prompt", None)
        self.metrics = metrics
        self.metric_collection = metric_collection
        self.span_type: Optional[SpanType] = span_type
        self._progress = _progress
        self._pbar_callback_id = _pbar_callback_id
        self.update_span_properties: Optional[Callable] = None

        # The deepeval span, once collected, and the confident-trace span.
        self.span: Optional[BaseSpan] = None
        self._otel_span: Any = None
        self._otel_token: Any = None
        self._previous_span: Optional[BaseSpan] = None
        self._started_trace: Optional[str] = None

    def __enter__(self):
        """Open the span and make it current."""
        self.start()
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the span, recording ``exc_val`` if the call raised."""
        self.deactivate()
        self.finish(exc_val if exc_type is not None else None)

    def start(self) -> None:
        """Open the span under the innermost open one, without making it
        current for OpenTelemetry."""
        from deepeval.tracing.otel import setup

        self.start_time = perf_counter()
        parent_span = active_span()
        if parent_span is None:
            current_trace = current_trace_context.get()
            # A failed async operation may leave a dead trace in context.
            if (
                current_trace is None
                or current_trace.uuid not in trace_manager.active_traces
            ):
                current_trace = trace_manager.start_new_trace()
                current_trace_context.set(current_trace)
                self._started_trace = current_trace.uuid
            self.trace_uuid = current_trace.uuid

        span_type = (
            self.span_type
            if self.span_type in _CONFIDENT_TRACE_TYPES
            else _CUSTOM_SPAN_TYPE
        )
        self._otel_span, span = setup.start_observed_span(self.name, span_type)
        if self._otel_span is None:
            # confident-trace isn't running, so nothing is traced.
            self._drop_started_trace()
            return
        values = self._observe_values()
        setup.export_on(
            self._otel_span,
            **{key: value for key, value in values.items() if value},
        )
        if span is None:
            # Inside a span confident-trace exports on its own, which the
            # trace this call started doesn't get.
            self._drop_started_trace()
            return

        self.span = span
        self.uuid, self.trace_uuid = span.uuid, span.trace_uuid
        self.parent_uuid = span.parent_uuid
        # ``next_*_span`` and span-context values were applied first and win.
        for key, value in values.items():
            if key not in span.model_fields_set and hasattr(span, key):
                setattr(span, key, value)
        if (
            parent_span
            and not getattr(span, "integration", None)
            and getattr(parent_span, "integration", None)
        ):
            span.integration = parent_span.integration

        self._previous_span = current_span_context.get()
        current_span_context.set(span)
        if (
            parent_span
            and parent_span.progress is not None
            and parent_span.pbar_callback_id is not None
        ):
            self._progress = parent_span.progress
            self._pbar_callback_id = parent_span.pbar_callback_id
        if self._progress is not None and self._pbar_callback_id is not None:
            span.progress = self._progress
            span.pbar_callback_id = self._pbar_callback_id
        self._bind_task(span)

    def activate(self) -> None:
        """Make the span current for the code it wraps."""
        if self._otel_span is not None and self._otel_token is None:
            from deepeval.tracing.otel import setup

            self._otel_token = otel_context.attach(
                setup.observed_context(self._otel_span)
            )

    def deactivate(self) -> None:
        """Restore the OpenTelemetry context ``activate`` replaced."""
        if self._otel_token is not None:
            otel_context.detach(self._otel_token)
            self._otel_token = None

    def finish(self, error: Optional[BaseException] = None) -> None:
        """End the span, and its trace if it was the trace's last."""
        from deepeval.tracing.otel import setup

        otel_span, span = self._otel_span, self.span
        if otel_span is None:
            return
        self._otel_span = None
        if error is not None:
            otel_span.set_status(Status(StatusCode.ERROR, str(error)))
        function_input = trace_manager.mask(self.function_kwargs)
        function_output = trace_manager.mask(self.result)
        if span is not None:
            span.status = (
                TraceSpanStatus.ERRORED
                if error is not None
                else TraceSpanStatus.SUCCESS
            )
            if error is not None:
                span.error = str(error)
            if self.update_span_properties is not None:
                self.update_span_properties(span)
            if span.input is None:
                span.input = function_input
            if span.output is None:
                span.output = function_output
            if isinstance(span, LlmSpan) and self.prompt and not span.prompt:
                span.prompt = self.prompt
        # What the call didn't set, the function's arguments and result give.
        setup.export_on(
            otel_span,
            input=(
                None
                if setup.exports(otel_span, ConfidentAttr.SPAN_INPUT)
                else function_input
            ),
            output=(
                None
                if setup.exports(otel_span, ConfidentAttr.SPAN_OUTPUT)
                else function_output
            ),
            prompt=self.prompt,
        )
        is_root = (
            span.parent_uuid is None
            if span is not None
            else otel_span.parent is None
        )
        if is_root:
            setup.export_on(
                otel_span,
                "trace",
                input=(
                    None
                    if setup.exports(otel_span, ConfidentAttr.TRACE_INPUT)
                    else function_input
                ),
                output=(
                    None
                    if setup.exports(otel_span, ConfidentAttr.TRACE_OUTPUT)
                    else function_output
                ),
            )
        trace = (
            trace_manager.get_trace_by_uuid(self.trace_uuid)
            if span is not None and is_root
            else None
        )
        if trace is not None:
            if trace.input is None:
                trace.input = function_input
            if trace.output is None:
                trace.output = function_output
            if error is not None:
                trace.status = TraceSpanStatus.ERRORED
        otel_span.end()

        if span is not None:
            # A generator's span may outlive the span it opened under.
            previous = self._previous_span
            still_open = (
                previous is not None
                and previous.status == TraceSpanStatus.IN_PROGRESS
            )
            current_span_context.set(previous if still_open else None)
            if trace is not None and not setup.open_collected_spans(
                self.trace_uuid
            ):
                trace_manager.end_trace(self.trace_uuid)
                current_trace_context.set(None)
        if self._progress is not None and self._pbar_callback_id is not None:
            self._progress.update(self._pbar_callback_id, advance=1)

    def _observe_values(self) -> Dict[str, Any]:
        """The span fields ``@observe`` sets, from its arguments or defaults."""
        values = {
            key: self.observe_kwargs.get(key, default)
            for key, default in _OBSERVE_FIELDS.get(self.span_type, {}).items()
        }
        values["metrics"] = self.metrics
        values["metric_collection"] = self.metric_collection
        return values

    def _drop_started_trace(self) -> None:
        if self._started_trace is not None:
            trace_manager.discard_trace(self._started_trace)
            current_trace_context.set(None)
            self._started_trace = None

    def _bind_task(self, span: BaseSpan) -> None:
        """Record the trace the current asyncio task works on."""
        try:
            task = asyncio.current_task()
        except Exception:
            task = None
        if task is None:
            return
        binding = trace_manager.task_bindings.get(task) or {}
        binding["trace_uuid"] = span.trace_uuid
        # Only a root records itself, so a child doesn't override it.
        if span.parent_uuid is None and "root_span_uuid" not in binding:
            binding["root_span_uuid"] = span.uuid
        trace_manager.task_bindings[task] = binding


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
    _drop_if_root: bool = False,
    _internal: bool = False,
    **observe_kwargs,
):
    """
    Decorator to trace a function as a span.

    Args:
        type: The type of span to create (agent, llm, retriever, tool, or custom string).
        _drop_if_root: If True, skip observation when there is no active parent span.
        _internal: If True, only observe when CONFIDENT_TRACE_INTERNAL is enabled.
        **observe_kwargs: Additional arguments to pass to the Observer.
    """

    def decorator(func):
        func_name = func.__name__  # Get func_name outside wrappers

        def _should_skip_observe():
            if _drop_if_root and current_span_context.get() is None:
                return True
            if _internal and not get_settings().CONFIDENT_TRACE_INTERNAL:
                return True
            return False

        # Async generator function
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            def asyncgen_wrapper(*args, **func_kwargs):
                if _should_skip_observe():
                    return func(*args, **func_kwargs)

                sig = inspect.signature(func)
                bound = sig.bind(*args, **func_kwargs)
                bound.apply_defaults()

                complete_kwargs = dict(bound.arguments)
                if "self" in complete_kwargs:
                    complete_kwargs["self"] = replace_self_with_class_name(
                        complete_kwargs["self"]
                    )
                observer_kwargs = {
                    "observe_kwargs": observe_kwargs,
                    "function_kwargs": complete_kwargs,
                }

                observer = Observer(
                    type,
                    metrics=metrics,
                    metric_collection=metric_collection,
                    func_name=func_name,
                    **observer_kwargs,
                )
                agen = func(*args, **func_kwargs)

                return _ObservedAsyncGenIter(agen, observer)

            setattr(asyncgen_wrapper, "_is_deepeval_observed", True)
            return asyncgen_wrapper

        # Sync generator function
        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args, **func_kwargs):
                if _should_skip_observe():
                    return func(*args, **func_kwargs)

                sig = inspect.signature(func)
                bound = sig.bind(*args, **func_kwargs)
                bound.apply_defaults()
                complete_kwargs = dict(bound.arguments)

                if "self" in complete_kwargs:
                    complete_kwargs["self"] = replace_self_with_class_name(
                        complete_kwargs["self"]
                    )
                observer_kwargs = {
                    "observe_kwargs": observe_kwargs,
                    "function_kwargs": make_json_serializable(complete_kwargs),
                }

                observer = Observer(
                    type,
                    metrics=metrics,
                    metric_collection=metric_collection,
                    func_name=func_name,
                    **observer_kwargs,
                )
                original_gen = func(*args, **func_kwargs)

                def gen():
                    observer.start()
                    # Capture the span and trace refs set by start().
                    # Generator locals survive across yields, but ContextVars
                    # don't when Starlette dispatches each next() to a
                    # different thread-pool thread. We restore them on every
                    # resume so child @observe'd calls see the right parent,
                    # and the span is current only while the generator runs.
                    _span = current_span_context.get()
                    _trace = current_trace_context.get()
                    it = iter(original_gen)
                    last_yielded_value = None
                    return_value = None
                    try:
                        while True:
                            observer.activate()
                            try:
                                # 1. Pull the next chunk
                                value = next(it)
                                last_yielded_value = value
                            except StopIteration as e:
                                return_value = e.value
                                break
                            finally:
                                observer.deactivate()
                            yield value
                            # After resume (potentially in a new thread),
                            # restore ContextVars before the next iteration
                            # runs user code that may create child spans.
                            current_span_context.set(_span)
                            if _trace is not None:
                                current_trace_context.set(_trace)

                        observer.result = (
                            return_value
                            if return_value is not None
                            else last_yielded_value
                        )
                    except Exception as e:
                        current_span_context.set(_span)
                        if _trace is not None:
                            current_trace_context.set(_trace)
                        observer.finish(e)
                        raise
                    finally:  # GeneratorExit exception directly brings us to final block
                        observer.finish()

                return gen()

            setattr(gen_wrapper, "_is_deepeval_observed", True)
            return gen_wrapper

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **func_kwargs):
                if _should_skip_observe():
                    return await func(*args, **func_kwargs)
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **func_kwargs)
                bound_args.apply_defaults()

                # Construct complete kwargs dictionary & pass all kwargs with consistent naming
                complete_kwargs = dict(bound_args.arguments)

                if "self" in complete_kwargs:
                    complete_kwargs["self"] = replace_self_with_class_name(
                        complete_kwargs["self"]
                    )

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
                if _should_skip_observe():
                    return func(*args, **func_kwargs)
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **func_kwargs)
                bound_args.apply_defaults()
                complete_kwargs = dict(bound_args.arguments)

                if "self" in complete_kwargs:
                    complete_kwargs["self"] = replace_self_with_class_name(
                        complete_kwargs["self"]
                    )

                observer_kwargs = {
                    "observe_kwargs": observe_kwargs,
                    "function_kwargs": make_json_serializable(
                        complete_kwargs
                    ),  # serilaizing it before it goes to trace api and raises circular reference error
                }
                with Observer(
                    type,
                    metrics=metrics,
                    metric_collection=metric_collection,
                    func_name=func_name,
                    **observer_kwargs,
                ) as observer:
                    # Call the original function
                    result = func(*args, **func_kwargs)
                    # Capture the result
                    observer.result = make_json_serializable(
                        result
                    )  # serilaizing it before it goes to trace api and raises circular reference error
                    return result

            # Set the marker attribute on the wrapper
            setattr(wrapper, "_is_deepeval_observed", True)
            return wrapper

    if _func is not None and callable(_func):
        return decorator(_func)

    return decorator
