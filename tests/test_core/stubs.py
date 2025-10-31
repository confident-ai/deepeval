import time
import asyncio
from types import SimpleNamespace
from typing import List, Optional, Protocol, runtime_checkable

from deepeval.constants import ProviderSlug as PS
from deepeval.metrics import BaseMetric, TaskCompletionMetric
from deepeval.models.retry_policy import create_retry_decorator
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


class _SleepyMetric(BaseMetric):
    """
    Test stub that can sleep in both sync and async paths.

    Args:
        name: display name
        sleep_s: seconds to sleep (None/0 means no sleep)
        should_skip: mark as skipped instead of evaluating
        succeed: whether to set success=True after sleep, the default is False
    """

    def __init__(
        self,
        name: str = "sleepy",
        *,
        sleep_s: float | None = None,
        should_skip: bool = False,
        succeed: bool = False,
    ):
        self.name = name
        self.sleep_s = sleep_s
        self.should_skip = should_skip
        self.succeed = succeed

        # required BaseMetric fields
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5
        self.score = None
        self.reason = None

    def measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.success = bool(self.succeed)

    async def a_measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        self.success = bool(self.succeed)

    def is_successful(self) -> bool:
        return bool(self.success)


class _PerAttemptTimeoutMetric(BaseMetric):
    """
    A metric that intentionally exceeds the per-attempt timeout budget to trigger
    Tenacity retries. Works in both sync and async executor paths.

    Use:
      set sleep_s > per-attempt timeout
    """

    threshold = 0.0

    def __init__(self, *, sleep_s: float = 10.0):
        self.sleep_s = float(sleep_s)
        self.name = "_PerAttemptTimeoutMetric"

    # BaseMetric.measure is wrapped with run_sync_with_timeout
    def measure(self, test_case, **kwargs) -> float:
        retry = create_retry_decorator(PS.OPENAI)

        @retry
        def slow_op():
            # run_sync_with_timeout() in the retry layer enforces the per-attempt timeout
            time.sleep(self.sleep_s)
            return 1.0

        return slow_op()

    # BaseMetric.a_measure is wrapped with asyncio.wait_for
    async def a_measure(self, test_case, **kwargs) -> float:
        retry = create_retry_decorator(PS.OPENAI)

        @retry
        async def slow_op():
            # resolve_effective_attempt_timeout() will bound asyncio.wait_for(...) around this
            await asyncio.sleep(self.sleep_s)
            return 1.0

        return await slow_op()

    # required by BaseMetric
    def is_successful(self) -> bool:
        return False


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
