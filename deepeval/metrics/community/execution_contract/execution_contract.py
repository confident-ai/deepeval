"""DeepEval community metric for deterministic execution contracts."""

from __future__ import annotations

import math
from typing import Optional

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

from .adapters import snapshot_from_roots
from .engine import evaluate
from .evidence import EvidenceLimits
from .reports import ContractReport, Verdict
from .rules import Contract


class ExecutionContractMetric(BaseMetric):
    """Check instrumented executions against explicit workflow rules.

    Native component evaluation checks the component's subtree. Trace-level
    evaluation checks all roots. Direct ``measure`` calls require the native
    tree attached by DeepEval; offline snapshots use :func:`evaluate` instead.
    A ``clock_domain`` asserts that the supplied timestamps are comparable;
    omit it for imported traces whose clock relationship is unknown.

    Scores count passing required rules. Optional rules are diagnostic only.
    A required UNKNOWN always blocks success, including at threshold zero.
    ``strict_mode`` additionally requires every required rule to pass.
    """

    _required_params = []
    requires_native_trace = True
    cacheable = False
    evaluate_on_error = True

    def __init__(
        self,
        contract: Contract,
        *,
        threshold: float = 1.0,
        strict_mode: bool = False,
        include_reason: bool = True,
        async_mode: bool = True,
        verbose_mode: bool = False,
        clock_domain: Optional[str] = None,
        evidence_limits: Optional[EvidenceLimits] = None,
        capture_evidence: bool = False,
    ):
        if type(contract) is not Contract:
            raise ValueError("contract must be a Contract")
        if (
            type(threshold) not in (int, float)
            or not 0 <= threshold <= 1
            or not math.isfinite(threshold)
        ):
            raise ValueError(
                "threshold must be a finite number between 0 and 1"
            )
        for name, value in (
            ("strict_mode", strict_mode),
            ("include_reason", include_reason),
            ("async_mode", async_mode),
            ("verbose_mode", verbose_mode),
            ("capture_evidence", capture_evidence),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if clock_domain is not None and (
            type(clock_domain) is not str
            or not clock_domain
            or len(clock_domain) > 1024
        ):
            raise ValueError(
                "clock_domain must be a nonempty bounded string or None"
            )
        if (
            evidence_limits is not None
            and type(evidence_limits) is not EvidenceLimits
        ):
            raise ValueError("evidence_limits must be EvidenceLimits")
        self.contract = contract
        self.threshold = 1.0 if strict_mode else float(threshold)
        self.strict_mode = strict_mode
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.clock_domain = clock_domain
        self.evidence_limits = evidence_limits or EvidenceLimits()
        self.capture_evidence = capture_evidence
        self.evaluation_model = None
        self.report: Optional[ContractReport] = None
        self._reset_result()

    def _reset_result(self):
        self.report = None
        # Explicit opt-in: the snapshot contains captured payload values and is
        # intended for user-controlled replay export, not default reporting.
        self.evidence_snapshot = None
        self.score = None
        self.score_breakdown = None
        self.reason = None
        self.success = False
        self.error = None
        self.skipped = False
        self.verbose_logs = None
        self.evaluation_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._reset_result()
        try:
            if not isinstance(test_case, LLMTestCase):
                raise ValueError(
                    "ExecutionContractMetric requires an LLMTestCase"
                )
            roots = test_case._trace_root_spans
            if roots is None:
                raise ValueError(
                    "ExecutionContractMetric requires native trace evidence"
                )
            span = test_case._trace_span
            snapshot = snapshot_from_roots(
                (span,) if span is not None else roots,
                complete=test_case._trace_capture_complete is True,
                clock_domain=self.clock_domain,
                external_parents=(
                    (span.parent_uuid,)
                    if span is not None and span.parent_uuid
                    else ()
                ),
                limits=self.evidence_limits,
                capture_boundary_ids=(
                    test_case._trace_capture_boundary_ids
                    if span is None
                    else ()
                ),
            )
            self.report = evaluate(
                snapshot, self.contract, evidence_limits=self.evidence_limits
            )
            if self.capture_evidence:
                self.evidence_snapshot = snapshot
            self.score = self.report.score
            if self.strict_mode and not self.report.success:
                self.score = 0.0
            self.score_breakdown = {
                rule.rule_id: float(rule.status is Verdict.PASS)
                for rule in self.report.rules
                if rule.required
            }
            if self.include_reason:
                counts = {status: 0 for status in Verdict}
                for rule in self.report.rules:
                    if rule.required:
                        counts[rule.status] += 1
                self.reason = (
                    f"Required rules: {counts[Verdict.PASS]} passed, "
                    f"{counts[Verdict.FAIL]} failed, "
                    f"{counts[Verdict.UNKNOWN]} unknown."
                )
            self.success = self.is_successful()
            return self.score
        except (ValueError, TypeError):
            self._reset_result()
            self.error = "Invalid or unavailable execution-contract evidence."
            raise

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # No network I/O, model inference, or nested event loop.
        return self.measure(test_case, _show_indicator, _in_component)

    def is_successful(self) -> bool:
        self.success = (
            self.error is None
            and self.report is not None
            and self.score is not None
            and self.score >= self.threshold
            and not any(
                rule.required and rule.status is Verdict.UNKNOWN
                for rule in self.report.rules
            )
        )
        return self.success

    @property
    def __name__(self):
        return "Execution Contract"
