"""Validated, detached evidence for execution-contract evaluations.

Completeness describes the caller's capture boundary. It does not certify that
all real-world operations were instrumented. Fingerprints detect changes; they
are not signatures or claims of authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class EvidenceLimits:
    max_spans: int = 10000
    max_depth: int = 128
    max_payload_bytes: int = 1048576
    max_total_payload_bytes: int = 16777216
    max_json_nodes: int = 100000

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or not value or len(value) > 1024:
        raise ValueError(
            f"{field} must be a nonempty string of at most 1024 characters"
        )
    return value


def _time(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number or null")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    # Preserve integer ordering (including nanosecond clocks above 2**53).
    # Converting n+1 to float can otherwise make a late prerequisite look early.
    return value if type(value) is int else number


def canonical_json(value: Any, limits: EvidenceLimits) -> str:
    """Serialize strict JSON after bounded iterative validation.

    Accept only JSON-native types, not arbitrary mappings or user objects whose
    conversion could run code. Shared containers are allowed; cycles are not.
    """
    stack = [(value, 0, False)]
    ancestors = set()
    nodes = 0
    minimum_bytes = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            ancestors.remove(id(item))
            continue
        nodes += 1
        if nodes > limits.max_json_nodes:
            raise ValueError("payload JSON node limit exceeded")
        if depth > limits.max_depth:
            raise ValueError("payload depth limit exceeded")
        kind = type(item)
        # A lower bound rejects oversized strings before JSON's encoder builds
        # their escaped representation. Final encoded size is checked below.
        minimum_bytes += len(item) if kind is str else 1
        if minimum_bytes > limits.max_payload_bytes:
            raise ValueError("payload byte limit exceeded")
        if kind in (dict, list):
            if id(item) in ancestors:
                raise ValueError("payload contains a cycle")
            if len(item) > limits.max_json_nodes - nodes:
                raise ValueError("payload JSON node limit exceeded")
            ancestors.add(id(item))
            stack.append((item, depth, True))
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("payload object keys must be strings")
                minimum_bytes += sum(len(key) for key in item)
                if minimum_bytes > limits.max_payload_bytes:
                    raise ValueError("payload byte limit exceeded")
                stack.extend(
                    (child, depth + 1, False) for child in item.values()
                )
            else:
                stack.extend((child, depth + 1, False) for child in item)
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("payload numbers must be finite")
        elif item is not None and kind not in (str, int, bool):
            raise ValueError("payload must contain only JSON-native values")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (ValueError, RecursionError, OverflowError) as exc:
        raise ValueError(
            "payload cannot be represented as bounded JSON"
        ) from exc
    if len(encoded.encode("utf-8")) > limits.max_payload_bytes:
        raise ValueError("payload byte limit exceeded")
    return encoded


@dataclass(frozen=True)
class SpanRecord:
    id: str
    parent_id: Optional[str]
    kind: str
    name: Optional[str]
    status: str
    start: Optional[float]
    end: Optional[float]
    payload_json: str

    @property
    def payload(self) -> Any:
        """Return a new detached JSON value on every access."""
        return json.loads(self.payload_json)

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "start": self.start,
            "end": self.end,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class Snapshot:
    spans: Tuple[SpanRecord, ...]
    complete: bool
    external_parents: Tuple[str, ...]
    clock_domain: Optional[str]
    incomplete_reasons: Tuple[str, ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        complete: bool,
        external_parents: Iterable[str] = (),
        clock_domain: Optional[str] = None,
        limits: Optional[EvidenceLimits] = None,
    ) -> Snapshot:
        limits = limits or EvidenceLimits()
        if type(complete) is not bool:
            raise ValueError("complete must be a boolean")
        if clock_domain is not None:
            _identifier(clock_domain, "clock_domain")
        boundaries = set()
        for index, boundary in enumerate(external_parents):
            if index >= limits.max_spans:
                raise ValueError("external parent limit exceeded")
            boundaries.add(_identifier(boundary, "external parent"))
        parsed = {}
        payload_size = 0
        reasons = [] if complete else ["capture_declared_partial"]
        fields = {
            "id",
            "parent_id",
            "kind",
            "name",
            "status",
            "start",
            "end",
            "payload",
        }
        for index, raw in enumerate(records):
            if index >= limits.max_spans:
                raise ValueError("span limit exceeded")
            if type(raw) is not dict or set(raw) - fields:
                raise ValueError(
                    "span record must be an object with known fields"
                )
            identifier = _identifier(raw.get("id"), "id")
            if identifier in parsed:
                raise ValueError("duplicate span ID")
            if identifier in boundaries:
                raise ValueError("external parent is also an observed span")
            parent = raw.get("parent_id")
            if parent is not None:
                _identifier(parent, "parent_id")
            kind = _identifier(raw.get("kind"), "kind")
            name = raw.get("name")
            if name is not None:
                _identifier(name, "name")
            status = raw.get("status", "unknown")
            if status not in ("success", "error", "in_progress", "unknown"):
                raise ValueError("invalid span status")
            start, end = _time(raw.get("start"), "start"), _time(
                raw.get("end"), "end"
            )
            if start is not None and end is not None and end < start:
                raise ValueError("span ends before it starts")
            if end is None or status in ("unknown", "in_progress"):
                reasons.append(f"unfinished_span:{identifier}")
            payload = canonical_json(raw.get("payload", {}), limits)
            payload_size += len(payload.encode("utf-8"))
            if payload_size > limits.max_total_payload_bytes:
                raise ValueError("total payload byte limit exceeded")
            parsed[identifier] = SpanRecord(
                identifier, parent, kind, name, status, start, end, payload
            )
        # Cache computed depths; shared ancestry is processed only once.
        depths = {}
        for identifier in parsed:
            path = []
            visiting = set()
            current = identifier
            while (
                current is not None
                and current not in boundaries
                and current not in depths
            ):
                if current not in parsed:
                    raise ValueError(
                        "dangling parent outside declared boundary"
                    )
                if current in visiting:
                    raise ValueError("span parent cycle")
                visiting.add(current)
                path.append(current)
                if len(path) > limits.max_depth:
                    raise ValueError("span depth limit exceeded")
                current = parsed[current].parent_id
            depth = depths.get(current, 0)
            for node in reversed(path):
                depth += 1
                if depth > limits.max_depth:
                    raise ValueError("span depth limit exceeded")
                depths[node] = depth
        return cls(
            tuple(parsed[key] for key in sorted(parsed)),
            not reasons,
            tuple(sorted(boundaries)),
            clock_domain,
            tuple(sorted(reasons)),
        )

    @property
    def capture_declared_complete(self) -> bool:
        """Caller assertion before unfinished spans demoted completeness."""
        return "capture_declared_partial" not in self.incomplete_reasons

    def validated(self, limits: Optional[EvidenceLimits] = None) -> Snapshot:
        """Validate direct dataclass construction without changing provenance."""
        limits = limits or EvidenceLimits()
        if (
            type(self.complete) is not bool
            or type(self.incomplete_reasons) is not tuple
        ):
            raise ValueError("invalid derived snapshot fields")
        if len(self.spans) > limits.max_spans:
            raise ValueError("span limit exceeded")
        for span in self.spans:
            if (
                type(span) is not SpanRecord
                or type(span.payload_json) is not str
            ):
                raise ValueError("snapshot requires SpanRecord values")
            if len(span.payload_json) > limits.max_payload_bytes:
                raise ValueError("payload byte limit exceeded")
        rebuilt = Snapshot.from_records(
            (span.to_record() for span in self.spans),
            complete=self.capture_declared_complete,
            external_parents=self.external_parents,
            clock_domain=self.clock_domain,
            limits=limits,
        )
        if (
            self.complete != rebuilt.complete
            or self.incomplete_reasons != rebuilt.incomplete_reasons
        ):
            raise ValueError(
                "derived snapshot completeness contradicts evidence"
            )
        return rebuilt

    @property
    def fingerprint(self) -> str:
        envelope = {
            "version": 1,
            "spans": [span.to_record() for span in self.spans],
            "complete": self.complete,
            "external_parents": self.external_parents,
            "clock_domain": self.clock_domain,
            "incomplete_reasons": self.incomplete_reasons,
        }
        content = json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
