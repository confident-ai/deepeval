"""Stable reports containing evidence references rather than user payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


def combine(verdicts) -> Verdict:
    result = Verdict.PASS
    for verdict in verdicts:
        if verdict is Verdict.FAIL:
            return Verdict.FAIL
        if verdict is Verdict.UNKNOWN:
            result = Verdict.UNKNOWN
    return result


@dataclass(frozen=True)
class Decision:
    status: Verdict
    reason_code: str
    target_id: Optional[str] = None
    witness_ids: Tuple[str, ...] = ()
    missing_fields: Tuple[str, ...] = ()
    omitted_witnesses: int = 0

    def to_dict(self):
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "target_id": self.target_id,
            "witness_ids": list(self.witness_ids),
            "missing_fields": list(self.missing_fields),
            "omitted_witnesses": self.omitted_witnesses,
        }


@dataclass(frozen=True)
class RuleReport:
    rule_id: str
    required: bool
    status: Verdict
    reason_code: str
    decisions: Tuple[Decision, ...]
    decision_count: int
    omitted_decisions: int

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "required": self.required,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "decision_count": self.decision_count,
            "omitted_decisions": self.omitted_decisions,
        }


@dataclass(frozen=True)
class CaptureSummary:
    """Explain the evaluated evidence boundary without retaining raw values.

    A caller can distinguish absent matches in a complete capture from missing
    coverage or an undeclared clock using just the report. Parent counts also
    make a component subtree distinguishable from a full-root capture.
    """

    span_count: int
    complete: bool
    external_parent_count: int
    clock_domain_known: bool

    def __post_init__(self):
        for value in (self.span_count, self.external_parent_count):
            if type(value) is not int or value < 0:
                raise ValueError("capture counts must be nonnegative integers")
        if (
            type(self.complete) is not bool
            or type(self.clock_domain_known) is not bool
        ):
            raise ValueError("capture flags must be booleans")

    def to_dict(self):
        return {
            "span_count": self.span_count,
            "complete": self.complete,
            "external_parent_count": self.external_parent_count,
            "clock_domain_known": self.clock_domain_known,
        }


@dataclass(frozen=True)
class ContractReport:
    evidence_fingerprint: str
    rules: Tuple[RuleReport, ...]
    contract_fingerprint: str
    capture: CaptureSummary

    @property
    def status(self) -> Verdict:
        return combine(rule.status for rule in self.rules if rule.required)

    @property
    def success(self) -> bool:
        return self.status is Verdict.PASS

    @property
    def score(self) -> float:
        required = [rule for rule in self.rules if rule.required]
        if not required:
            return 1.0
        return sum(rule.status is Verdict.PASS for rule in required) / len(
            required
        )

    def to_dict(self):
        return {
            "version": 1,
            "evidence_fingerprint": self.evidence_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "capture": self.capture.to_dict(),
            "status": self.status.value,
            "success": self.success,
            "score": self.score,
            "rules": [rule.to_dict() for rule in self.rules],
        }
