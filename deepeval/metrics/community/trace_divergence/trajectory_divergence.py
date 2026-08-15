"""Deterministic trajectory divergence metric.

Compares two *real* execution traces against each other — a baseline run
you trust and a candidate run after a prompt, model, or tool change — and
localizes the first sustained divergence between them. Unlike agentic
metrics that score one run against a static expectation (``ToolCorrectness``,
``PlanAdherence``, ...), the useful output here is *which* step forked and
*whether* the trajectories recovered, not just a pass/fail.
"""

# ruff: noqa: UP006, UP035, UP045
from collections.abc import Iterable
from typing import Any, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.community.trace_divergence.alignment import (
    AlignmentResult,
    Event,
    align,
    project,
)
from deepeval.metrics.community.trace_divergence.schema import (
    TrajectoryDivergenceResult,
)
from deepeval.metrics.community.trace_divergence.template import (
    TrajectoryDivergenceTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import construct_verbose_logs
from deepeval.test_case import LLMTestCase
from deepeval.utils import get_or_create_event_loop


class TrajectoryDivergenceMetric(BaseMetric):
    """Deterministic, single-turn trajectory comparison.

    The metric takes two traces at construction time: ``baseline_trace``
    (the run you trust) and ``candidate_trace`` (the run you are checking,
    e.g. after a prompt edit or model swap). The ``LLMTestCase`` passed to
    ``measure`` is accepted for ``BaseMetric`` compatibility; the traces
    themselves are the metric's data.

    The score is ``1.0 - divergence_ratio``, so ``1.0`` means the traces are
    fully aligned and ``0.0`` means no step after the matched prefix could be
    aligned. With the default threshold of ``1.0`` only fully aligned traces
    pass; lower it to tolerate a localized, recovered divergence. The
    ``AlignmentResult`` (matched prefix, first divergence, kind, resync point,
    unmatched steps) is stored on ``self.alignment_result`` so callers can
    localize the fork exactly.

    This metric is fully deterministic: no LLM, no API key, no cost.
    """

    def __init__(
        self,
        baseline_trace: Iterable[Any],
        candidate_trace: Iterable[Any],
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.baseline_trace = list(baseline_trace)
        self.candidate_trace = list(candidate_trace)
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.model = None
        self.using_native_model = True
        self.evaluation_model = None
        self.alignment_result: Optional[AlignmentResult] = None
        self.result: Optional[TrajectoryDivergenceResult] = None

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self.evaluation_cost = 0

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
                return self.score
            self._calculate_metric()
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self.evaluation_cost = 0

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self._calculate_metric()
            return self.score

    def _calculate_metric(self) -> None:
        baseline_events = project(self.baseline_trace)
        candidate_events = project(self.candidate_trace)
        result = align(baseline_events, candidate_events)
        self.alignment_result = result
        self.result = TrajectoryDivergenceResult(
            matched_prefix_len=result.matched_prefix_len,
            first_divergence=result.first_divergence,
            divergence_kind=result.divergence_kind,
            resync_at=result.resync_at,
            unmatched_baseline=result.unmatched_baseline,
            unmatched_candidate=result.unmatched_candidate,
            reordered=result.reordered,
            divergence_ratio=result.divergence_ratio,
        )
        self.score_breakdown = self.result.model_dump()
        self.score = self._calculate_score(result)
        self.reason = self._generate_reason(
            result, baseline_events, candidate_events
        )
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Baseline trace: {self._format_trace(baseline_events)}",
                f"Candidate trace: {self._format_trace(candidate_events)}",
                f"Alignment: {result.as_dict()}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _calculate_score(self, result: AlignmentResult) -> float:
        score = 1.0 - result.divergence_ratio
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_reason(
        self,
        result: AlignmentResult,
        baseline_events: List[Event],
        candidate_events: List[Event],
    ) -> Optional[str]:
        if self.include_reason is False:
            return None
        if result.aligned:
            return TrajectoryDivergenceTemplate.aligned(result.baseline_len)

        summary = self._summarize_divergence(
            result, baseline_events, candidate_events
        )
        recovery = (
            TrajectoryDivergenceTemplate.recovered(result.resync_at + 1)
            if result.resync_at is not None
            else TrajectoryDivergenceTemplate.unrecovered()
        )
        return (
            f"The candidate trajectory diverges from the baseline at {summary}. "
            f"{recovery} Divergence ratio: {result.divergence_ratio:.2f}."
        )

    def _summarize_divergence(
        self,
        result: AlignmentResult,
        baseline_events: List[Event],
        candidate_events: List[Event],
    ) -> str:
        index = result.first_divergence or 0
        step = index + 1
        kind = result.divergence_kind

        if kind == "arg_change" and index < len(baseline_events):
            return TrajectoryDivergenceTemplate.arg_change(
                step, baseline_events[index].name
            )
        if kind == "tool_change":
            baseline_tool = (
                baseline_events[index].name
                if index < len(baseline_events)
                else "?"
            )
            candidate_tool = (
                candidate_events[index].name
                if index < len(candidate_events)
                else "?"
            )
            return TrajectoryDivergenceTemplate.tool_change(
                step, baseline_tool, candidate_tool
            )
        if kind == "order_change":
            last_step = result.resync_at or step
            return TrajectoryDivergenceTemplate.order_change(step, last_step)
        if kind == "absent" and index < len(baseline_events):
            return TrajectoryDivergenceTemplate.absent(
                step, baseline_events[index].name
            )
        if kind == "extra" and index < len(candidate_events):
            return TrajectoryDivergenceTemplate.extra(
                step, candidate_events[index].name
            )
        return "the trajectories diverge"

    def _format_trace(self, events: List[Event]) -> str:
        if not events:
            return "[]"
        return " -> ".join(f"{event.kind}:{event.name}" for event in events)

    @property
    def __name__(self):
        return "Trajectory Divergence"
