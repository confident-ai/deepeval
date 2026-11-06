"""Strict, version-independent data codecs for execution-contract artifacts.

The envelope and its version live in ``artifacts``. These codecs validate the
contents instead of trusting an object's checksum as evidence of correctness.
No Python objects, expressions, imports, or custom constructors are loaded.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import re
from typing import Optional

from .evidence import EvidenceLimits, Snapshot, _identifier, canonical_json
from .reports import (
    CaptureSummary,
    ContractReport,
    Decision,
    RuleReport,
    Verdict,
    combine,
)
from .rules import (
    Budget,
    Contract,
    EvaluationLimits,
    Occurrence,
    Prerequisite,
    Recovery,
)
from .selectors import Predicate, Selector, _pointer_parts


def _object(value, required, optional=()):
    if type(value) is not dict:
        raise ValueError("expected a JSON object")
    fields = set(value)
    if not set(required) <= fields or fields - set(required) - set(optional):
        raise ValueError("missing or unknown fields in artifact object")
    return value


def _array(value, maximum, label):
    if type(value) is not list or len(value) > maximum:
        raise ValueError(
            f"{label} must be an array within the configured limit"
        )
    return value


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("expected a boolean")
    return value


def _nonnegative(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative integer")
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256 fingerprint")
    return value


def selector_to_data(selector: Selector) -> dict:
    return {
        "name": selector.name,
        "kind": selector.kind,
        "status": selector.status,
        "predicates": [
            {
                "pointer": predicate.pointer,
                "operator": predicate.operator,
                "expected": predicate.expected,
            }
            for predicate in selector.predicates
        ],
    }


def selector_from_data(data) -> Selector:
    _object(data, (), ("name", "kind", "status", "predicates"))
    predicates = []
    for item in _array(data.get("predicates", []), 64, "predicates"):
        _object(item, ("pointer", "operator", "expected"))
        predicates.append(
            Predicate(item["pointer"], item["operator"], item["expected"])
        )
    return Selector(
        name=data.get("name"),
        kind=data.get("kind"),
        status=data.get("status"),
        predicates=tuple(predicates),
    )


def rule_to_data(rule) -> dict:
    result = {"id": rule.id, "required": rule.required}
    if type(rule) is Occurrence:
        result.update(
            type="occurrence",
            selector=selector_to_data(rule.selector),
            minimum=rule.minimum,
            maximum=rule.maximum,
        )
    elif type(rule) is Prerequisite:
        result.update(
            type="prerequisite",
            target=selector_to_data(rule.target),
            prerequisite=selector_to_data(rule.prerequisite),
            bindings=[list(pair) for pair in rule.bindings],
            max_age=rule.max_age,
            invalidated_by=(
                selector_to_data(rule.invalidated_by)
                if rule.invalidated_by is not None
                else None
            ),
        )
    elif type(rule) is Recovery:
        result.update(
            type="recovery",
            failure=selector_to_data(rule.failure),
            recovery=selector_to_data(rule.recovery),
            bindings=[list(pair) for pair in rule.bindings],
            within=rule.within,
            before=(
                selector_to_data(rule.before)
                if rule.before is not None
                else None
            ),
        )
    elif type(rule) is Budget:
        result.update(
            type="budget",
            selector=selector_to_data(rule.selector),
            pointer=rule.pointer,
            maximum=rule.maximum,
            exclusive=rule.exclusive,
        )
    else:
        raise ValueError("unsupported contract rule")
    return result


def rule_from_data(data):
    if type(data) is not dict:
        raise ValueError("rule must be an object")
    kind = data.get("type")
    common = ("id", "type")
    if kind == "occurrence":
        _object(
            data, common + ("selector",), ("required", "minimum", "maximum")
        )
        return Occurrence(
            data["id"],
            selector_from_data(data["selector"]),
            minimum=data.get("minimum", 1),
            maximum=data.get("maximum"),
            required=data.get("required", True),
        )
    if kind == "prerequisite":
        _object(
            data,
            common + ("target", "prerequisite"),
            ("required", "bindings", "max_age", "invalidated_by"),
        )
        invalidator = data.get("invalidated_by")
        return Prerequisite(
            data["id"],
            selector_from_data(data["target"]),
            selector_from_data(data["prerequisite"]),
            bindings=_array(data.get("bindings", []), 64, "bindings"),
            max_age=data.get("max_age"),
            invalidated_by=(
                selector_from_data(invalidator)
                if invalidator is not None
                else None
            ),
            required=data.get("required", True),
        )
    if kind == "recovery":
        _object(
            data,
            common + ("failure", "recovery", "bindings"),
            ("required", "within", "before"),
        )
        before = data.get("before")
        return Recovery(
            data["id"],
            selector_from_data(data["failure"]),
            selector_from_data(data["recovery"]),
            bindings=_array(data["bindings"], 64, "bindings"),
            within=data.get("within"),
            before=selector_from_data(before) if before is not None else None,
            required=data.get("required", True),
        )
    if kind == "budget":
        _object(
            data,
            common + ("selector", "pointer", "maximum"),
            ("required", "exclusive"),
        )
        return Budget(
            data["id"],
            selector_from_data(data["selector"]),
            data["pointer"],
            data["maximum"],
            exclusive=data.get("exclusive", False),
            required=data.get("required", True),
        )
    raise ValueError("unknown rule type")


def contract_to_data(contract: Contract) -> dict:
    if type(contract) is not Contract:
        raise ValueError("expected a Contract")
    # Bound aggregate expectation materialization before loading each JSON
    # value. Per-predicate limits alone allow thousands of large expectations.
    expected_bytes = 0
    for rule in contract.rules:
        for value in vars(rule).values():
            if type(value) is Selector:
                for predicate in value.predicates:
                    expected_bytes += len(predicate.expected_json)
                    if expected_bytes > EvidenceLimits().max_payload_bytes:
                        raise ValueError(
                            "contract expectation byte limit exceeded"
                        )
    return {
        "rules": [rule_to_data(rule) for rule in contract.rules],
        "limits": asdict(contract.limits),
    }


def contract_from_data(
    data, *, limits: Optional[EvaluationLimits] = None
) -> Contract:
    """Decode rules while preventing files from raising trusted resource limits."""
    ceiling = limits or EvaluationLimits()
    _object(data, ("rules",), ("limits",))
    configured = data.get("limits", {})
    _object(configured, (), vars(ceiling))
    limits_data = {**vars(ceiling), **configured}
    requested = EvaluationLimits(**limits_data)
    for name, maximum in vars(ceiling).items():
        if getattr(requested, name) > maximum:
            raise ValueError(
                "contract requests a limit above the trusted ceiling"
            )
    return Contract(
        tuple(
            rule_from_data(rule)
            for rule in _array(data["rules"], requested.max_rules, "rules")
        ),
        requested,
    )


def contract_fingerprint(contract: Contract) -> str:
    data = canonical_json(contract_to_data(contract), EvidenceLimits())
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def snapshot_to_data(
    snapshot: Snapshot, *, limits: Optional[EvidenceLimits] = None
) -> dict:
    snapshot = snapshot.validated(limits)
    return {
        "spans": [span.to_record() for span in snapshot.spans],
        "capture_complete": snapshot.capture_declared_complete,
        "external_parents": list(snapshot.external_parents),
        "clock_domain": snapshot.clock_domain,
    }


def snapshot_from_data(
    data, *, limits: Optional[EvidenceLimits] = None
) -> Snapshot:
    limits = limits or EvidenceLimits()
    _object(
        data,
        ("spans", "capture_complete"),
        ("external_parents", "clock_domain"),
    )
    return Snapshot.from_records(
        _array(data["spans"], limits.max_spans, "spans"),
        complete=_boolean(data["capture_complete"]),
        external_parents=_array(
            data.get("external_parents", []),
            limits.max_spans,
            "external parents",
        ),
        clock_domain=data.get("clock_domain"),
        limits=limits,
    )


def _identifiers(values, maximum, label):
    result = tuple(
        _identifier(value, label) for value in _array(values, maximum, label)
    )
    if len(set(result)) != len(result):
        raise ValueError("duplicate identifiers in report")
    return result


def _decision_from_data(data, limits):
    _object(
        data,
        (
            "status",
            "reason_code",
            "target_id",
            "witness_ids",
            "missing_fields",
            "omitted_witnesses",
        ),
    )
    target = data["target_id"]
    if target is not None:
        _identifier(target, "target_id")
    pointers = _array(data["missing_fields"], 128, "missing_fields")
    for pointer in pointers:
        _pointer_parts(pointer)
    if len(set(pointers)) != len(pointers):
        raise ValueError("duplicate missing-field pointers")
    return Decision(
        Verdict(data["status"]),
        _identifier(data["reason_code"], "reason_code"),
        target,
        _identifiers(data["witness_ids"], limits.max_witnesses, "witness_ids"),
        tuple(pointers),
        _nonnegative(data["omitted_witnesses"]),
    )


def _rule_report_from_data(data, limits):
    _object(
        data,
        (
            "rule_id",
            "required",
            "status",
            "reason_code",
            "decisions",
            "decision_count",
            "omitted_decisions",
        ),
    )
    decisions = tuple(
        _decision_from_data(item, limits)
        for item in _array(data["decisions"], limits.max_decisions, "decisions")
    )
    count = _nonnegative(data["decision_count"])
    omitted = _nonnegative(data["omitted_decisions"])
    if count != len(decisions) + omitted:
        raise ValueError("report decision counts are inconsistent")
    status = Verdict(data["status"])
    observed = combine(item.status for item in decisions)
    if (
        status is Verdict.PASS
        and observed is not Verdict.PASS
        or status is Verdict.UNKNOWN
        and observed is Verdict.FAIL
        or status is Verdict.FAIL
        and observed is not Verdict.FAIL
        and omitted == 0
    ):
        raise ValueError("rule report status contradicts retained decisions")
    return RuleReport(
        _identifier(data["rule_id"], "rule_id"),
        _boolean(data["required"]),
        status,
        _identifier(data["reason_code"], "reason_code"),
        decisions,
        count,
        omitted,
    )


def report_from_data(
    data, *, limits: Optional[EvaluationLimits] = None
) -> ContractReport:
    limits = limits or EvaluationLimits()
    _object(
        data,
        (
            "version",
            "evidence_fingerprint",
            "contract_fingerprint",
            "capture",
            "rules",
            "status",
            "success",
            "score",
        ),
    )
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported report version")
    rules = tuple(
        _rule_report_from_data(item, limits)
        for item in _array(data["rules"], limits.max_rules, "rules")
    )
    if not rules or len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("report must contain unique nonempty rules")
    capture = data["capture"]
    _object(
        capture,
        (
            "span_count",
            "complete",
            "external_parent_count",
            "clock_domain_known",
        ),
    )
    report = ContractReport(
        _digest(data["evidence_fingerprint"]),
        rules,
        _digest(data["contract_fingerprint"]),
        CaptureSummary(**capture),
    )
    if (
        Verdict(data["status"]) is not report.status
        or _boolean(data["success"]) != report.success
        or type(data["score"]) not in (int, float)
        or data["score"] != report.score
    ):
        raise ValueError("report summary contradicts rule results")
    return report
