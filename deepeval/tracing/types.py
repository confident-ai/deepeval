from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING
from rich.progress import Progress

from deepeval.utils import make_model_config

from deepeval.prompt.prompt import Prompt
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden


class Message(BaseModel):
    role: str
    """To be displayed on the top of the message block."""

    type: Literal["tool_calls", "tool_output", "thinking", "default"] = (
        "default"
    )
    """Decides how the content is rendered."""

    content: Any
    """The content of the message."""


class TraceWorkerStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


class SpanType(Enum):
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


class TraceSpanStatus(Enum):
    SUCCESS = "SUCCESS"
    ERRORED = "ERRORED"
    IN_PROGRESS = "IN_PROGRESS"


class EvalMode(str, Enum):
    """Active evaluation mode for the trace manager.

    Each value names the call site that activates it, so it's always
    obvious which entry point set the mode without grepping the codebase.

    - OFF: not in an evaluation pipeline; traces post to the API as usual.
    - EVALUATE: classic ``evaluate(...)`` (sync or async). Traces are
      routed into the test-run pipeline instead of being posted.
    - ITERATOR_SYNC: the synchronous ``evals_iterator`` path. Today this
      shares the same trace-routing behavior as EVALUATE because synchronous
      execution naturally orders trace completion, but it's a distinct mode
      so future per-call-site behavior (e.g. progress reporting, lifecycle
      hooks) can be added without ambiguity.
    - ITERATOR_ASYNC: the asynchronous ``evals_iterator`` path. Same routing
      as EVALUATE, plus traces are accumulated in ``pending_traces`` so they
      can be evaluated against the goldens that the iterator interleaves.
    """

    OFF = "off"
    EVALUATE = "evaluate"
    ITERATOR_SYNC = "iterator_sync"
    ITERATOR_ASYNC = "iterator_async"


class LlmToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


class LlmOutput(BaseModel):
    role: str
    content: Any
    tool_calls: Optional[List[LlmToolCall]] = None


class BaseSpan(BaseModel):
    model_config = make_model_config(arbitrary_types_allowed=True)

    uuid: str
    status: TraceSpanStatus
    children: List["BaseSpan"] = Field(default_factory=list)
    trace_uuid: str = Field(serialization_alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, serialization_alias="parentUuid")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    llm_test_case: Optional[LLMTestCase] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None
    integration: Optional[str] = None

    # Don't serialize these
    progress: Optional[Progress] = Field(None, exclude=True)
    pbar_callback_id: Optional[int] = Field(None, exclude=True)
    drop: bool = Field(False, exclude=True)

    # additional test case parameters
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    context: Optional[List[str]] = Field(None, serialization_alias="context")
    expected_output: Optional[str] = Field(
        None, serialization_alias="expectedOutput"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        None, serialization_alias="expectedTools"
    )


class AgentSpan(BaseSpan):
    name: str
    available_tools: List[str] = []
    agent_handoffs: List[str] = []


class LlmSpan(BaseSpan):

    model: Optional[str] = None
    provider: Optional[str] = None
    prompt: Optional[Prompt] = None
    input_token_count: Optional[float] = Field(
        None, serialization_alias="inputTokenCount"
    )
    output_token_count: Optional[float] = Field(
        None, serialization_alias="outputTokenCount"
    )
    cost_per_input_token: Optional[float] = Field(
        None, serialization_alias="costPerInputToken"
    )
    cost_per_output_token: Optional[float] = Field(
        None, serialization_alias="costPerOutputToken"
    )
    token_intervals: Optional[Dict[float, str]] = Field(
        None, serialization_alias="tokenTimes"
    )
    prompt_alias: Optional[str] = Field(None, serialization_alias="promptAlias")
    prompt_version: Optional[str] = Field(
        None, serialization_alias="promptVersion"
    )
    prompt_label: Optional[str] = Field(None, serialization_alias="promptLabel")
    prompt_commit_hash: Optional[str] = Field(
        None, serialization_alias="promptCommitHash"
    )

    # input_tools: Optional[List[ToolSchema]] = Field(None, serialization_alias="inputTools")
    # invocation_params: Optional[Dict[str, Any]] = Field(None, serialization_alias="invocationParams")
    # output_metadata: Optional[Dict[str, Any]] = Field(None, serialization_alias="outputMetadata")

    # for serializing `prompt`
    model_config = make_model_config(arbitrary_types_allowed=True)


class RetrieverSpan(BaseSpan):
    embedder: Optional[str] = None
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    chunk_size: Optional[int] = Field(None, serialization_alias="chunkSize")


class ToolSpan(BaseSpan):
    name: str  # Required name for ToolSpan
    description: Optional[str] = None


class Trace(BaseModel):
    model_config = make_model_config(arbitrary_types_allowed=True)

    uuid: str = Field(serialization_alias="uuid")
    status: TraceSpanStatus
    root_spans: List[BaseSpan] = Field(serialization_alias="rootSpans")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None
    test_case_id: Optional[str] = Field(None, serialization_alias="testCaseId")
    turn_id: Optional[str] = Field(None, serialization_alias="turnId")

    # Don't serialize these
    confident_api_key: Optional[str] = Field(None, exclude=True)
    environment: str = Field(None, exclude=True)
    drop: bool = Field(False, exclude=True)
    # Internal marker: True when this Trace was pushed implicitly by an
    # OTel-mode integration's SpanInterceptor (so that
    # ``update_current_trace(...)`` works without an enclosing ``@observe``
    # / ``with trace(...)``). Used by ``ContextAwareSpanProcessor`` to
    # decide REST vs OTLP routing — implicit placeholders DON'T count as
    # "user opted into REST". See ``deepeval/integrations/pydantic_ai/
    # instrumentator.py`` for the push/pop logic.
    is_otel_implicit: bool = Field(False, exclude=True)

    # additional test case parameters
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    context: Optional[List[str]] = Field(None, serialization_alias="context")
    expected_output: Optional[str] = Field(
        None, serialization_alias="expectedOutput"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        None, serialization_alias="expectedTools"
    )


class TraceAttributes(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class TestCaseMetricPair:
    test_case: LLMTestCase
    metrics: List[BaseMetric]
    hyperparameters: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class EvalSession:
    """Per-evaluation-run state owned by ``TraceManager``.

    All fields here are scoped to a single ``evaluate(...)`` /
    ``evals_iterator(...)`` invocation. Resetting the session is a single
    assignment (``trace_manager.eval_session = EvalSession()``), which makes
    "exit cleanup" atomic and impossible to half-do.

    The default value (``mode == EvalMode.OFF`` and empty collections) is the
    inert "no eval running" state; callers that read these collections when
    not evaluating will simply see empties rather than ``AttributeError`` or
    ``None``-guard noise.

    Fields:
        mode: Active evaluation mode. ``OFF`` means no eval is running.
        pending_traces: Traces created under ``ITERATOR_ASYNC``, keyed by uuid
            in the order they were started. Used to (a) gate which finished
            traces belong in ``traces_to_evaluate`` and (b) preserve start
            order even when traces complete out of order. Insertion-ordered
            dict gives O(1) membership and ordered iteration without a
            parallel list.
        traces_to_evaluate: Single queue of traces to evaluate. Populated by
            both the native ``@observe`` path (via ``TraceManager.end_trace``)
            and by integrations (llama_index, pydantic_ai, openinference,
            agentcore) that append directly. All appenders use a ``not in``
            dedup check.
        trace_uuid_to_golden: Map of trace uuid → golden, for evaluating
            traces against the correct golden when the iterator interleaves.
        test_case_metrics: Auxiliary path for test-case-style evaluation
            inside an iterator run; populated by external callers / SDK
            extensions (no in-tree producer today).
    """

    mode: EvalMode = EvalMode.OFF
    pending_traces: Dict[str, Trace] = field(default_factory=dict)
    traces_to_evaluate: List[Trace] = field(default_factory=list)
    trace_uuid_to_golden: Dict[str, "Golden"] = field(default_factory=dict)
    test_case_metrics: List[TestCaseMetricPair] = field(default_factory=list)

    @property
    def is_evaluating(self) -> bool:
        """True for any non-OFF mode."""
        return self.mode != EvalMode.OFF

    @property
    def is_iterator(self) -> bool:
        """True when running under either evals_iterator path."""
        return self.mode in (EvalMode.ITERATOR_SYNC, EvalMode.ITERATOR_ASYNC)
