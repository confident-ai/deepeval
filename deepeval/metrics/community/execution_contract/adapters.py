"""Snapshot DeepEval's native span objects without changing their contents."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from deepeval.tracing import types as native

from .evidence import EvidenceLimits, Snapshot, _identifier

_STATUS = {
    native.TraceSpanStatus.SUCCESS: "success",
    native.TraceSpanStatus.ERRORED: "error",
    native.TraceSpanStatus.IN_PROGRESS: "in_progress",
}


def _kind(span):
    for span_type, label in (
        (native.ToolSpan, "tool"),
        (native.LlmSpan, "llm"),
        (native.AgentSpan, "agent"),
        (native.RetrieverSpan, "retriever"),
    ):
        if isinstance(span, span_type):
            return label
    return "base"


def _resource_number(value):
    """Validate mutable native counters before deriving totals.

    Native span fields may be updated after model construction. Normalize
    overflow into the same validation path as NaN, negative, or nonnumeric
    counters so a reused metric cannot retain an earlier successful result.
    """
    try:
        valid = (
            type(value) in (int, float) and value >= 0 and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(
            "native resource counter must be finite and nonnegative"
        )
    return value


def _resources(span):
    if not isinstance(span, native.LlmSpan):
        return {}
    counters = {}
    names = {
        "input_tokens": "input_token_count",
        "output_tokens": "output_token_count",
        "input_token_price": "cost_per_input_token",
        "output_token_price": "cost_per_output_token",
    }
    for target, source in names.items():
        value = getattr(span, source)
        if value is None:
            continue
        counters[target] = _resource_number(value)
    if "input_tokens" in counters and "output_tokens" in counters:
        counters["total_tokens"] = (
            counters["input_tokens"] + counters["output_tokens"]
        )
    if all(name in counters for name in names):
        counters["cost"] = (
            counters["input_tokens"] * counters["input_token_price"]
            + counters["output_tokens"] * counters["output_token_price"]
        )
    for value in counters.values():
        _resource_number(value)
    return counters


def snapshot_from_roots(
    roots: Iterable[native.BaseSpan],
    *,
    complete: bool,
    clock_domain: Optional[str] = None,
    external_parents=(),
    capture_boundary_ids=(),
    limits: Optional[EvidenceLimits] = None,
) -> Snapshot:
    """Capture an explicitly scoped native forest.

    Callers must declare whether the capture is terminal and what clock its
    timestamps share. The adapter does not infer a common clock for imported
    telemetry. Inputs, outputs and metadata must contain JSON-native values;
    arbitrary object repr/serialization hooks are never executed.
    """
    limits = limits or EvidenceLimits()
    if type(complete) is not bool:
        raise ValueError("complete must be a boolean")
    stack = []
    for root in roots:
        if len(stack) >= limits.max_spans:
            raise ValueError("native span limit exceeded")
        stack.append((root, None, 1))
    records, seen, boundaries = [], set(), set()
    transparent = set()
    for index, identifier in enumerate(capture_boundary_ids):
        if index >= limits.max_spans:
            raise ValueError("capture boundary limit exceeded")
        transparent.add(_identifier(identifier, "capture boundary"))
    remaining_boundaries = transparent.copy()
    for index, parent in enumerate(external_parents):
        if index >= limits.max_spans:
            raise ValueError("external parent limit exceeded")
        boundaries.add(_identifier(parent, "external parent"))
    trace_id = None
    while stack:
        span, expected_parent, depth = stack.pop()
        if not isinstance(span, native.BaseSpan):
            raise ValueError("native roots must contain BaseSpan objects")
        if len(seen) >= limits.max_spans:
            raise ValueError("native span limit exceeded")
        if span.uuid in seen:
            raise ValueError("duplicate native span membership or cycle")
        seen.add(span.uuid)
        if depth > limits.max_depth:
            raise ValueError("native span depth limit exceeded")
        if trace_id is None:
            trace_id = span.trace_uuid
        if span.trace_uuid != trace_id:
            raise ValueError(
                "native forest contains different trace identities"
            )
        if expected_parent is not None and span.parent_uuid != expected_parent:
            raise ValueError("native child parent identity contradicts tree")
        if len(stack) + len(span.children) + len(seen) > limits.max_spans:
            raise ValueError("native span limit exceeded")
        stack.extend((child, span.uuid, depth + 1) for child in span.children)
        if span.uuid in transparent:
            if depth != 1:
                raise ValueError("capture boundary must be a root")
            remaining_boundaries.remove(span.uuid)
            boundaries.add(span.uuid)
            if span.drop:
                complete = False
            continue
        if span.drop:
            complete = False
            boundaries.add(span.uuid)
            continue
        payload = {}
        for field in ("input", "output", "metadata", "error"):
            value = getattr(span, field)
            if value is not None:
                payload[field] = value
        resources = _resources(span)
        if resources:
            payload["resources"] = resources
        records.append(
            {
                "id": span.uuid,
                "parent_id": span.parent_uuid,
                "kind": _kind(span),
                "name": span.name,
                "status": _STATUS.get(span.status, "unknown"),
                "start": span.start_time,
                "end": span.end_time,
                "payload": payload,
            }
        )
    if remaining_boundaries:
        raise ValueError("capture boundary is absent from native roots")
    return Snapshot.from_records(
        records,
        complete=complete,
        external_parents=boundaries,
        clock_domain=clock_domain,
        limits=limits,
    )


def snapshot_from_span(
    span: native.BaseSpan,
    *,
    clock_domain: Optional[str] = None,
    limits: Optional[EvidenceLimits] = None,
) -> Snapshot:
    """Capture one component's subtree, explicitly excluding its siblings."""
    if not isinstance(span, native.BaseSpan):
        raise ValueError("expected a native BaseSpan")
    return snapshot_from_roots(
        (span,),
        complete=span.end_time is not None
        and span.status != native.TraceSpanStatus.IN_PROGRESS,
        clock_domain=clock_domain,
        external_parents=(span.parent_uuid,) if span.parent_uuid else (),
        limits=limits,
    )


def snapshot_from_trace(
    trace: native.Trace,
    *,
    clock_domain: Optional[str] = None,
    limits: Optional[EvidenceLimits] = None,
) -> Snapshot:
    """Capture all roots of a native trace, including terminal failed runs."""
    if not isinstance(trace, native.Trace):
        raise ValueError("expected a native Trace")
    for root in trace.root_spans:
        if root.trace_uuid != trace.uuid:
            raise ValueError("trace identity contradicts root span")
    return snapshot_from_roots(
        trace.root_spans,
        complete=not trace.drop
        and trace.end_time is not None
        and trace.status != native.TraceSpanStatus.IN_PROGRESS,
        clock_domain=clock_domain,
        limits=limits,
    )
