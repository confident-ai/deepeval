"""Compare rule outcomes under an unchanged execution contract.

Span IDs are local evidence references, not cross-run operation identities.
Comparisons therefore pair stable rule IDs, rather than implying that matching
span IDs identify the same real-world operation across different executions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .reports import ContractReport, Verdict
from .rules import EvaluationLimits
from .serialization import report_from_data

_CHANGES = {
    (Verdict.PASS, Verdict.PASS): "unchanged_pass",
    (Verdict.PASS, Verdict.FAIL): "new_failure",
    (Verdict.PASS, Verdict.UNKNOWN): "lost_evidence",
    (Verdict.FAIL, Verdict.PASS): "resolved_failure",
    (Verdict.FAIL, Verdict.FAIL): "persisting_failure",
    (Verdict.FAIL, Verdict.UNKNOWN): "failure_now_unknown",
    (Verdict.UNKNOWN, Verdict.PASS): "resolved_unknown",
    (Verdict.UNKNOWN, Verdict.FAIL): "confirmed_failure",
    (Verdict.UNKNOWN, Verdict.UNKNOWN): "persisting_unknown",
}


@dataclass(frozen=True)
class RuleTransition:
    rule_id: str
    required: bool
    before: Verdict
    after: Verdict
    change: str
    baseline_reason: str
    candidate_reason: str

    @property
    def regressed(self) -> bool:
        return self.required and self.change in ("new_failure", "lost_evidence")

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "required": self.required,
            "before": self.before.value,
            "after": self.after.value,
            "change": self.change,
            "regressed": self.regressed,
            "baseline_reason": self.baseline_reason,
            "candidate_reason": self.candidate_reason,
        }


@dataclass(frozen=True)
class ReportComparison:
    contract_fingerprint: str
    baseline_evidence_fingerprint: str
    candidate_evidence_fingerprint: str
    transitions: Tuple[RuleTransition, ...]

    @property
    def regressed(self) -> bool:
        return any(item.regressed for item in self.transitions)

    @property
    def inconclusive(self) -> bool:
        return any(
            item.required and item.after is Verdict.UNKNOWN
            for item in self.transitions
        )

    @property
    def exit_code(self) -> int:
        """CI regression gate: new failure=1, insufficient evidence=2, else=0."""
        if any(
            item.required and item.change == "new_failure"
            for item in self.transitions
        ):
            return 1
        return 2 if self.inconclusive else 0

    def to_dict(self):
        counts = {}
        for item in self.transitions:
            counts[item.change] = counts.get(item.change, 0) + 1
        return {
            "version": 1,
            "contract_fingerprint": self.contract_fingerprint,
            "baseline_evidence_fingerprint": self.baseline_evidence_fingerprint,
            "candidate_evidence_fingerprint": self.candidate_evidence_fingerprint,
            "regressed": self.regressed,
            "inconclusive": self.inconclusive,
            "counts": counts,
            "transitions": [item.to_dict() for item in self.transitions],
        }


def compare_reports(
    baseline: ContractReport,
    candidate: ContractReport,
    *,
    limits: Optional[EvaluationLimits] = None,
) -> ReportComparison:
    """Compare validated reports; changed configurations require a fresh baseline.

    A failed rule becoming UNKNOWN is not treated as resolved. Optional rules
    remain visible but do not fail the required-rule regression gate. This is
    a deterministic comparison of two executions, not a statistical claim
    about an agent's performance across a population of tasks.
    """
    if (
        type(baseline) is not ContractReport
        or type(candidate) is not ContractReport
    ):
        raise ValueError("comparison requires two ContractReport values")
    baseline = report_from_data(baseline.to_dict(), limits=limits)
    candidate = report_from_data(candidate.to_dict(), limits=limits)
    if baseline.contract_fingerprint != candidate.contract_fingerprint:
        raise ValueError("cannot compare reports from different contracts")
    before = {rule.rule_id: rule for rule in baseline.rules}
    after = {rule.rule_id: rule for rule in candidate.rules}
    if before.keys() != after.keys():
        raise ValueError("report rule sets contradict the shared contract")
    transitions = []
    for identifier in sorted(before):
        left, right = before[identifier], after[identifier]
        if left.required != right.required:
            raise ValueError(
                "rule required flags contradict the shared contract"
            )
        transitions.append(
            RuleTransition(
                identifier,
                left.required,
                left.status,
                right.status,
                _CHANGES[left.status, right.status],
                left.reason_code,
                right.reason_code,
            )
        )
    return ReportComparison(
        baseline.contract_fingerprint,
        baseline.evidence_fingerprint,
        candidate.evidence_fingerprint,
        tuple(transitions),
    )
