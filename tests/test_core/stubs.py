from deepeval.metrics import BaseMetric, TaskCompletionMetric
from types import SimpleNamespace
from typing import List, Optional, Protocol, runtime_checkable
from deepeval.tracing.types import TraceSpanStatus


@runtime_checkable
class ApiTestCaseLike(Protocol):
    name: Optional[str]
    success: Optional[bool]
    metrics_data: List
    input: Optional[str]
    actual_output: Optional[str]
    expected_output: Optional[str]
    context: Optional[List[str]]
    retrieval_context: Optional[List[str]]

    def update_metric_data(self, *args, **kwargs) -> None: ...
    def update_status(self, *args, **kwargs) -> None: ...
    def update_run_duration(self, *args, **kwargs) -> None: ...


def make_trace_api_like(status):
    """Shape compatible with TraceApi members that `execute` touches."""
    return SimpleNamespace(
        name="trace",
        status=status,
        error=None,
        input=None,
        output=None,
        expected_output=None,
        context=None,
        retrieval_context=None,
        agent_spans=[],
        llm_spans=[],
        retriever_spans=[],
        tool_spans=[],
        base_spans=[],
        metrics_data=[],
    )


def make_span_api_like():
    return SimpleNamespace(status=None, error=None, metrics_data=[])


class _DummyMetric(BaseMetric):
    """Simple metric that can be flagged to simulate a skip."""

    def __init__(self, name="dummy", should_skip=False):
        self.name = name
        self.should_skip = should_skip
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5

    def measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        self.success = True

    def is_successful(self) -> bool:
        return bool(self.success)


class _DummyTaskCompletionMetric(TaskCompletionMetric):
    """Metric used to toggle the 'has_task_completion' path."""

    def __init__(self, name="tc"):
        self.name = name
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5

    def measure(self, test_case, *_args, **_kwargs):
        self.success = True

    def is_successful(self) -> bool:
        return bool(self.success)


class _FakeSpan:
    def __init__(self, *, input=None, output=None, metrics=None, children=None):
        self.input = input
        self.output = output
        self.expected_output = None
        self.context = None
        self.retrieval_context = None
        self.tools_called = None
        self.expected_tools = None
        self.metrics = metrics or []
        self.children = children or []
        self.status = TraceSpanStatus.SUCCESS
        self.error = None


class _FakeTrace:
    def __init__(
        self, *, input=None, output=None, metrics=None, root_span=None
    ):
        self.input = input
        self.output = output
        self.expected_output = None
        self.context = None
        self.retrieval_context = None
        self.tools_called = None
        self.expected_tools = None
        self.metrics = metrics or []
        self.root_spans = [root_span] if root_span else []
        self.status = TraceSpanStatus.SUCCESS
        self.error = None
        self.uuid = "trace-uuid"
