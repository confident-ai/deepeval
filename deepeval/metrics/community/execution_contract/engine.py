"""Bounded deterministic evaluation of execution contracts."""

from __future__ import annotations

import math
from fractions import Fraction

from .evidence import EvidenceLimits, Snapshot
from typing import Optional
from .reports import (
    CaptureSummary,
    ContractReport,
    Decision,
    RuleReport,
    Verdict,
    combine,
)
from .rules import Budget, Contract, Occurrence, Prerequisite, Recovery
from .selectors import MISSING, Match, match_bindings, resolve_pointer
from .serialization import contract_fingerprint
from .index import BindingIndex


class _Evaluation:
    def __init__(self, snapshot, contract):
        self.snapshot = snapshot
        self.limits = contract.limits
        self.remaining = self.limits.max_work
        self.payloads = {}
        self.expected_values = {}
        for span in snapshot.spans:
            self.work(max(1, len(span.payload_json) // 64))
            self.payloads[span.id] = span.payload

    def work(self, amount=1):
        self.remaining -= amount
        if self.remaining < 0:
            raise ValueError("evaluation work limit exceeded")

    def select(self, selector):
        known, unknown = [], []
        if id(selector) not in self.expected_values:
            expected = []
            for predicate in selector.predicates:
                self.work(max(1, len(predicate.expected_json) // 64))
                expected.append(predicate.expected)
            self.expected_values[id(selector)] = tuple(expected)
        for span in self.snapshot.spans:
            self.work()
            match = selector.matches(
                span,
                payload=self.payloads[span.id],
                expected_values=self.expected_values[id(selector)],
                charge=self.work,
            )
            if match is Match.YES:
                known.append(span)
            elif match is Match.UNKNOWN:
                unknown.append(span)
        return known, unknown

    def decision(self, status, reason, target=None, witnesses=(), missing=()):
        witness_ids = tuple(sorted(set(witnesses)))
        kept = witness_ids[: self.limits.max_witnesses]
        return Decision(
            status,
            reason,
            target,
            kept,
            tuple(sorted(set(missing))),
            len(witness_ids) - len(kept),
        )

    def report(self, rule, decisions, reason=None, uncertain_capture=False):
        # Aggregate all decisions while retaining only bounded details.
        status = Verdict.UNKNOWN if uncertain_capture else Verdict.PASS
        kept, count = [], 0
        for decision in decisions:
            count += 1
            status = combine((status, decision.status))
            if len(kept) < self.limits.max_decisions:
                kept.append(decision)
        return RuleReport(
            rule.id,
            rule.required,
            status,
            reason or ("evaluated" if count else "no_targets"),
            tuple(kept),
            count,
            count - len(kept),
        )

    def occurrence(self, rule):
        matches, possible = self.select(rule.selector)
        low = len(matches)
        high = low + len(possible) if self.snapshot.complete else None
        if rule.maximum is not None and low > rule.maximum:
            status, reason = Verdict.FAIL, "too_many_occurrences"
        elif high is not None and high < rule.minimum:
            status, reason = Verdict.FAIL, "too_few_occurrences"
        elif low >= rule.minimum and (
            rule.maximum is None or (high is not None and high <= rule.maximum)
        ):
            status, reason = Verdict.PASS, "occurrence_bounds_satisfied"
        else:
            status, reason = Verdict.UNKNOWN, "occurrence_count_uncertain"
        decision = self.decision(
            status, reason, witnesses=(span.id for span in matches)
        )
        return self.report(rule, (decision,), reason)

    def bound(self, target, candidate, bindings):
        self.work(max(1, len(bindings)))
        return match_bindings(
            self.payloads[target.id],
            self.payloads[candidate.id],
            bindings,
            charge=self.work,
        )

    def temporal_before(self, candidate, target):
        if (
            self.snapshot.clock_domain is None
            or candidate.end is None
            or target.start is None
        ):
            return Match.UNKNOWN
        return Match.YES if candidate.end <= target.start else Match.NO

    def authorization(
        self, rule, target, candidate, invalidators, possible_invalidators
    ):
        if candidate.id == target.id:
            return Match.NO
        binding = self.bound(target, candidate, rule.bindings)
        if binding is Match.NO:
            return Match.NO
        before = self.temporal_before(candidate, target)
        if before is Match.NO:
            return Match.NO
        if binding is Match.UNKNOWN or before is Match.UNKNOWN:
            return Match.UNKNOWN
        if (
            rule.max_age is not None
            and Fraction(target.start) - Fraction(candidate.end) > rule.max_age
        ):
            return Match.NO
        unknown = rule.invalidated_by is not None and not self.snapshot.complete
        for invalidator in invalidators + possible_invalidators:
            bound = self.bound(target, invalidator, rule.bindings)
            if bound is Match.NO:
                continue
            before = self.temporal_before(invalidator, target)
            if before is Match.NO:
                continue
            if (
                candidate.start is not None
                and invalidator.end is not None
                and invalidator.end < candidate.start
            ):
                continue
            if (
                bound is Match.UNKNOWN
                or before is Match.UNKNOWN
                or invalidator in possible_invalidators
            ):
                unknown = True
                continue
            if (
                invalidator.start is not None
                and invalidator.start > candidate.end
            ):
                return Match.NO
            # An interval overlap or equal boundary does not establish which
            # state transition took effect last.
            unknown = True
        return Match.UNKNOWN if unknown else Match.YES

    def prerequisite(self, rule):
        targets, possible_targets = self.select(rule.target)
        candidates, possible_candidates = self.select(rule.prerequisite)
        invalidators, possible_invalidators = ([], [])
        if rule.invalidated_by is not None:
            invalidators, possible_invalidators = self.select(
                rule.invalidated_by
            )
        candidate_index = BindingIndex(
            candidates + possible_candidates,
            self.payloads,
            rule.bindings,
            self.work,
        )
        invalidator_index = BindingIndex(
            invalidators, self.payloads, rule.bindings, self.work
        )
        possible_invalidator_index = BindingIndex(
            possible_invalidators, self.payloads, rule.bindings, self.work
        )
        possible_candidate_ids = {span.id for span in possible_candidates}

        def decisions():
            for target in targets:
                valid, uncertain = [], []
                bound_invalidators = list(invalidator_index.candidates(target))
                bound_possible_invalidators = list(
                    possible_invalidator_index.candidates(target)
                )
                for candidate in candidate_index.candidates(target):
                    result = self.authorization(
                        rule,
                        target,
                        candidate,
                        bound_invalidators,
                        bound_possible_invalidators,
                    )
                    if result is Match.NO:
                        continue
                    if (
                        result is Match.UNKNOWN
                        or candidate.id in possible_candidate_ids
                    ):
                        uncertain.append(candidate.id)
                    else:
                        valid.append(candidate.id)
                if valid:
                    yield self.decision(
                        Verdict.PASS, "prerequisite_completed", target.id, valid
                    )
                elif uncertain or not self.snapshot.complete:
                    yield self.decision(
                        Verdict.UNKNOWN,
                        "prerequisite_evidence_uncertain",
                        target.id,
                        uncertain,
                    )
                else:
                    yield self.decision(
                        Verdict.FAIL, "prerequisite_not_satisfied", target.id
                    )
            for target in possible_targets:
                yield self.decision(
                    Verdict.UNKNOWN, "target_membership_uncertain", target.id
                )

        return self.report(
            rule, decisions(), uncertain_capture=not self.snapshot.complete
        )

    def recovered(
        self, rule, failure, repair, continuations, possible_continuations
    ):
        if failure.id == repair.id:
            return Match.NO
        bound = self.bound(failure, repair, rule.bindings)
        if bound is Match.NO:
            return Match.NO
        if (
            self.snapshot.clock_domain is None
            or failure.end is None
            or repair.start is None
            or repair.end is None
        ):
            return Match.UNKNOWN
        if repair.start < failure.end:
            return Match.NO
        if (
            rule.within is not None
            and Fraction(repair.end) - Fraction(failure.end) > rule.within
        ):
            return Match.NO
        unknown = bound is Match.UNKNOWN
        for continuation in continuations + possible_continuations:
            if continuation.id in (failure.id, repair.id):
                continue
            bound = self.bound(failure, continuation, rule.bindings)
            if bound is Match.NO:
                continue
            if (
                continuation.start is not None
                and continuation.start < failure.end
            ):
                continue
            if (
                bound is Match.UNKNOWN
                or continuation.start is None
                or continuation in possible_continuations
            ):
                unknown = True
            elif repair.end > continuation.start:
                return Match.NO
        if rule.before is not None and not self.snapshot.complete:
            unknown = True
        return Match.UNKNOWN if unknown else Match.YES

    def recovery(self, rule):
        failures, possible_failures = self.select(rule.failure)
        repairs, possible_repairs = self.select(rule.recovery)
        continuations, possible_continuations = ([], [])
        if rule.before is not None:
            continuations, possible_continuations = self.select(rule.before)
        repair_index = BindingIndex(
            repairs + possible_repairs, self.payloads, rule.bindings, self.work
        )
        continuation_index = BindingIndex(
            continuations, self.payloads, rule.bindings, self.work
        )
        possible_continuation_index = BindingIndex(
            possible_continuations, self.payloads, rule.bindings, self.work
        )
        possible_repair_ids = {span.id for span in possible_repairs}

        def decisions():
            for failure in failures:
                valid, uncertain = [], []
                bound_continuations = list(
                    continuation_index.candidates(failure)
                )
                bound_possible_continuations = list(
                    possible_continuation_index.candidates(failure)
                )
                for repair in repair_index.candidates(failure):
                    result = self.recovered(
                        rule,
                        failure,
                        repair,
                        bound_continuations,
                        bound_possible_continuations,
                    )
                    if result is Match.NO:
                        continue
                    if (
                        result is Match.UNKNOWN
                        or repair.id in possible_repair_ids
                    ):
                        uncertain.append(repair.id)
                    else:
                        valid.append(repair.id)
                if valid:
                    yield self.decision(
                        Verdict.PASS, "recovery_completed", failure.id, valid
                    )
                elif uncertain or not self.snapshot.complete:
                    yield self.decision(
                        Verdict.UNKNOWN,
                        "recovery_evidence_uncertain",
                        failure.id,
                        uncertain,
                    )
                else:
                    yield self.decision(
                        Verdict.FAIL, "recovery_not_satisfied", failure.id
                    )
            for failure in possible_failures:
                yield self.decision(
                    Verdict.UNKNOWN, "failure_membership_uncertain", failure.id
                )

        return self.report(
            rule, decisions(), uncertain_capture=not self.snapshot.complete
        )

    def budget(self, rule):
        selected, possible = self.select(rule.selector)
        identifiers = {span.id for span in selected}
        by_id = {span.id: span for span in self.snapshot.spans}
        overlapping = False
        if not rule.exclusive:
            for span in selected:
                parent = span.parent_id
                while parent in by_id:
                    self.work()
                    if parent in identifiers:
                        overlapping = True
                        break
                    parent = by_id[parent].parent_id
        values, missing = [], []
        for span in selected:
            self.work()
            value = resolve_pointer(self.payloads[span.id], rule.pointer)
            if value is MISSING or value is None:
                missing.append(span.id)
            elif type(value) not in (int, float):
                raise ValueError(
                    "resource counter must be a finite nonnegative number"
                )
            else:
                try:
                    valid = value >= 0 and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError(
                        "resource counter must be a finite nonnegative number"
                    )
                values.append(value)
        # Sum the exact recorded numeric values. A float is interpreted as its
        # binary value; large integer counters never lose their low bits, even
        # when mixed with fractional resources. This is not decimal rounding.
        total = sum((Fraction(value) for value in values), Fraction(0))
        if overlapping:
            status, reason = Verdict.UNKNOWN, "resource_ownership_ambiguous"
        elif total > rule.maximum:
            status, reason = Verdict.FAIL, "resource_budget_exceeded"
        elif possible or missing or not self.snapshot.complete:
            status, reason = Verdict.UNKNOWN, "resource_evidence_incomplete"
        else:
            status, reason = Verdict.PASS, "resource_budget_satisfied"
        decision = self.decision(
            status,
            reason,
            witnesses=identifiers,
            missing=(rule.pointer,) if missing else (),
        )
        return self.report(rule, (decision,), reason)


def evaluate(
    snapshot: Snapshot,
    contract: Contract,
    *,
    evidence_limits: Optional[EvidenceLimits] = None,
) -> ContractReport:
    """Evaluate all rules without truncating evidence or skipping on limits.

    Reject directly constructed invalid snapshots at the public boundary.
    A resource-limit error never returns a partially successful report.
    """
    if type(snapshot) is not Snapshot or type(contract) is not Contract:
        raise ValueError("evaluate requires a Snapshot and Contract")
    fingerprint = contract_fingerprint(contract)
    snapshot = snapshot.validated(evidence_limits)
    context = _Evaluation(snapshot, contract)
    reports = []
    for rule in contract.rules:
        if isinstance(rule, Occurrence):
            reports.append(context.occurrence(rule))
        elif isinstance(rule, Prerequisite):
            reports.append(context.prerequisite(rule))
        elif isinstance(rule, Recovery):
            reports.append(context.recovery(rule))
        elif isinstance(rule, Budget):
            reports.append(context.budget(rule))
        else:
            raise ValueError("unsupported contract rule")
    return ContractReport(
        snapshot.fingerprint,
        tuple(reports),
        fingerprint,
        CaptureSummary(
            len(snapshot.spans),
            snapshot.complete,
            len(snapshot.external_parents),
            snapshot.clock_domain is not None,
        ),
    )
