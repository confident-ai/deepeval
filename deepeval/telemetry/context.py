"""Run-scoped accumulation.

Per-test-case and per-metric events used to be 87% of all ingestion. They are
now in-memory counters folded into a single event per run.

Two accumulators exist because there are two shapes of evaluation. A bounded
run (`evaluate()`, pytest, compare) opens a `RunAccumulator` and flushes on
exit. Bare `metric.measure()` calls have no enclosing run -- 72% of metric
volume -- so they land in a process-level accumulator with its own flush
policy.
"""

import atexit
import logging
import threading
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Set, Tuple

from deepeval.telemetry.client import capture
from deepeval.telemetry.events import Entrypoint, Event
from deepeval.telemetry.judge import describe_judge
from deepeval.telemetry.properties import (
    EventProperties,
    FlushReason,
    Outcome,
    TurnKind,
)

logger = logging.getLogger(__name__)

# A long-running service can measure metrics for weeks without exiting, so the
# standalone path cannot rely on process exit alone.
STANDALONE_FLUSH_THRESHOLD = 50
STANDALONE_FLUSH_INTERVAL_SECONDS = 30 * 60


def turn_kind_of(item: Any) -> Optional[TurnKind]:
    """Classify a test case or golden as single- or multi-turn.

    Keys on the `turns` field rather than importing the four classes:
    telemetry is imported by nearly everything, so a dependency on
    `test_case` and `dataset` here would be circular. An attribute check also
    survives subclassing, which an isinstance list would not.
    """
    if item is None:
        return None
    return (
        TurnKind.MULTI_TURN if hasattr(item, "turns") else TurnKind.SINGLE_TURN
    )


def _resolve_turn_kind(kinds: Set[TurnKind]) -> Optional[TurnKind]:
    if not kinds:
        return None
    if len(kinds) == 1:
        return next(iter(kinds))
    return TurnKind.MIXED


@dataclass
class RunAccumulator:
    """Counters for one evaluation run. Mutated from many tasks; hold `lock`."""

    entrypoint: Entrypoint
    run_id: str = ""
    test_cases: int = 0
    goldens: int = 0
    metric_runs: int = 0
    metrics: "Counter[str]" = field(default_factory=Counter)
    turn_kinds: Set[TurnKind] = field(default_factory=set)
    async_mode: Optional[bool] = None
    in_component: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    traces_at_entry: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_test_case(
        self, count: int = 1, kind: Optional[TurnKind] = None
    ) -> None:
        with self.lock:
            self.test_cases += count
            if kind is not None:
                self.turn_kinds.add(kind)

    def record_golden(
        self, count: int = 1, kind: Optional[TurnKind] = None
    ) -> None:
        with self.lock:
            self.goldens += count
            if kind is not None:
                self.turn_kinds.add(kind)

    def record_metric(
        self,
        metric_name: str,
        async_mode: bool,
        in_component: bool,
        model: Any = None,
    ) -> None:
        provider, model_name = describe_judge(model)
        with self.lock:
            self.metric_runs += 1
            self.metrics[metric_name] += 1
            if async_mode:
                self.async_mode = True
            elif self.async_mode is None:
                self.async_mode = False
            if in_component:
                self.in_component = True
            if provider is not None and self.provider is None:
                self.provider = provider
                self.model = model_name

    def has_activity(self) -> bool:
        """Whether this run saw any deepeval work at all.

        Lets a scope that wraps something broader than an evaluation -- a whole
        pytest session, which may contain no deepeval tests -- stay silent.
        """
        with self.lock:
            return bool(self.test_cases or self.goldens or self.metric_runs)

    def snapshot(self) -> EventProperties:
        with self.lock:
            metric_names = sorted(self.metrics)
            tracing_enabled, trace_total = _tracing_state()
            traced_here = max(trace_total - self.traces_at_entry, 0)
            return EventProperties(
                entrypoint=self.entrypoint,
                run_id=self.run_id or None,
                test_case_count=self.test_cases,
                golden_count=self.goldens,
                turn_kind=_resolve_turn_kind(self.turn_kinds),
                metric_runs=self.metric_runs,
                metrics=metric_names,
                metrics_count=len(metric_names),
                async_mode=self.async_mode,
                in_component=self.in_component,
                provider=self.provider,
                model=self.model,
                tracing_enabled=tracing_enabled,
                traced=traced_here > 0,
                trace_count=traced_here,
            )


_run_stack: ContextVar[Optional[Tuple[RunAccumulator, ...]]] = ContextVar(
    "deepeval_run_stack", default=None
)


def _tracing_state() -> Tuple[bool, int]:
    """Read the trace manager without importing it at module load."""
    try:
        from deepeval.tracing.tracing import trace_manager

        return bool(trace_manager.tracing_enabled), len(trace_manager.traces)
    except Exception:
        return False, 0


def current_run() -> Optional[RunAccumulator]:
    """The innermost run, so metrics attribute to `evaluate()` inside pytest."""
    stack = _run_stack.get()
    return stack[-1] if stack else None


def push_run(entrypoint: Entrypoint, run_id: str) -> Tuple[RunAccumulator, Any]:
    _, trace_total = _tracing_state()
    accumulator = RunAccumulator(
        entrypoint=entrypoint, run_id=run_id, traces_at_entry=trace_total
    )
    stack = _run_stack.get() or ()
    token = _run_stack.set(stack + (accumulator,))
    return accumulator, token


def pop_run(token: Any) -> None:
    try:
        _run_stack.reset(token)
    except ValueError:
        # The context was entered and exited on different contexts; drop the
        # innermost frame rather than leaking it.
        stack = _run_stack.get() or ()
        _run_stack.set(stack[:-1])


def record_test_case(test_case: Any = None, count: int = 1) -> None:
    run = current_run()
    if run is not None:
        run.record_test_case(count, turn_kind_of(test_case))


def record_golden(golden: Any = None, count: int = 1) -> None:
    run = current_run()
    if run is not None:
        run.record_golden(count, turn_kind_of(golden))


def record_metric(
    metric_name: str,
    async_mode: bool,
    in_component: bool,
    model: Any = None,
    track: bool = True,
) -> None:
    if not track:
        return
    run = current_run()
    if run is not None:
        run.record_metric(metric_name, async_mode, in_component, model)
    else:
        _standalone.record(metric_name, async_mode, in_component, model)


class StandaloneAccumulator:
    """Metrics measured outside any `evaluate()` call.

    Flushes on a count threshold, on an interval, and at process exit.
    `atexit` alone is not enough: containers get SIGKILLed, and most of this
    population runs on ephemeral infrastructure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()
        self._timer: Optional[threading.Timer] = None
        self._registered_atexit = False

    def _reset(self) -> None:
        self._metric_runs = 0
        self._metrics: "Counter[str]" = Counter()
        self._async_mode: Optional[bool] = None
        self._in_component = False
        self._provider: Optional[str] = None
        self._model: Optional[str] = None
        # Set on the first metric of each batch, so `traced` covers the window
        # this event reports on rather than the whole process.
        self._traces_at_start: Optional[int] = None

    def _ensure_scheduled(self) -> None:
        if not self._registered_atexit:
            atexit.register(self._flush_at_exit)
            self._registered_atexit = True
        if self._timer is None:
            timer = threading.Timer(
                STANDALONE_FLUSH_INTERVAL_SECONDS, self._flush_on_interval
            )
            timer.daemon = True
            self._timer = timer
            try:
                timer.start()
            except RuntimeError:
                # Interpreter is shutting down; the atexit hook still covers us.
                self._timer = None

    def record(
        self,
        metric_name: str,
        async_mode: bool,
        in_component: bool,
        model: Any = None,
    ) -> None:
        provider, model_name = describe_judge(model)
        _, trace_total = _tracing_state()
        with self._lock:
            if self._traces_at_start is None:
                self._traces_at_start = trace_total
            self._metric_runs += 1
            self._metrics[metric_name] += 1
            if async_mode:
                self._async_mode = True
            elif self._async_mode is None:
                self._async_mode = False
            if in_component:
                self._in_component = True
            if provider is not None and self._provider is None:
                self._provider = provider
                self._model = model_name
            reached_threshold = self._metric_runs >= STANDALONE_FLUSH_THRESHOLD
            self._ensure_scheduled()

        if reached_threshold:
            self.flush(FlushReason.THRESHOLD)

    def _drain(self) -> Optional[EventProperties]:
        tracing_enabled, trace_total = _tracing_state()
        with self._lock:
            if self._metric_runs == 0:
                return None
            metric_names = sorted(self._metrics)
            traced_here = max(trace_total - (self._traces_at_start or 0), 0)
            properties = EventProperties(
                entrypoint=Entrypoint.STANDALONE,
                run_id=str(uuid.uuid4()),
                # Explicit zeros rather than absent keys, so every Evaluation
                # event has the same shape whatever the entrypoint.
                test_case_count=0,
                golden_count=0,
                metric_runs=self._metric_runs,
                metrics=metric_names,
                metrics_count=len(metric_names),
                async_mode=self._async_mode,
                in_component=self._in_component,
                provider=self._provider,
                model=self._model,
                tracing_enabled=tracing_enabled,
                traced=traced_here > 0,
                trace_count=traced_here,
            )
            self._reset()
            return properties

    def flush(self, reason: FlushReason) -> None:
        properties = self._drain()
        if properties is None:
            return
        capture(
            Event.EVALUATION,
            replace(
                properties,
                flush_reason=reason,
                outcome=Outcome.COMPLETED,
            ),
        )

    def _flush_on_interval(self) -> None:
        with self._lock:
            self._timer = None
        self.flush(FlushReason.INTERVAL)

    def _flush_at_exit(self) -> None:
        self.flush(FlushReason.PROCESS_EXIT)
        from deepeval.telemetry.client import flush as flush_backend

        flush_backend()


_standalone = StandaloneAccumulator()


def reset_for_testing() -> None:
    """Drop any ambient run scope and buffered standalone metrics.

    A test process is itself inside a run scope -- the pytest plugin opens one
    for the whole session -- so without this every `record_metric` in a test
    lands in the session accumulator instead of the path under test.
    """
    _run_stack.set(())
    _standalone._reset()


def flush_standalone_metrics(
    reason: FlushReason = FlushReason.MANUAL,
) -> None:
    """Flush now. The default reason reflects the caller, not process exit --
    `atexit` passes `PROCESS_EXIT` itself."""
    _standalone.flush(reason)
