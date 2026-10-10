# ruff: noqa: UP006, UP035, UP045
"""Deterministic alignment for baseline and candidate execution traces.

This module deliberately contains no metric policy. It reports where two
single-turn traces first differ, whether they recover, and which events could
not be matched. A metric wrapper can then decide which differences are
acceptable for its use case.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

TRACE_PROJECTION_VERSION = "1.0.0"
DEFAULT_LOOKAHEAD = 4
DEFAULT_RESYNC_RUN = 2

DIVERGENCE_KINDS = (
    "arg_change",
    "tool_change",
    "order_change",
    "absent",
    "extra",
)


@dataclass(frozen=True)
class Event:
    """One versioned, comparable projection of a trace step."""

    index: int
    event_id: str
    kind: str
    name: str
    arguments: Any
    strong_identity: str
    weak_identity: str


@dataclass
class AlignmentResult:
    """Policy-neutral output consumed by a trace-divergence metric."""

    aligned: bool
    matched_prefix_len: int
    first_divergence: Optional[int]
    divergence_kind: Optional[str]
    resync_at: Optional[int]
    unmatched_baseline: List[str] = field(default_factory=list)
    unmatched_candidate: List[str] = field(default_factory=list)
    reordered: List[Tuple[str, str]] = field(default_factory=list)
    baseline_len: int = 0
    candidate_len: int = 0
    projection_version: str = TRACE_PROJECTION_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def divergence_ratio(self) -> float:
        """Return the divergent fraction without applying pass/fail policy."""

        longest = max(self.baseline_len, self.candidate_len)
        if not longest or self.aligned:
            return 0.0
        end = self.resync_at if self.resync_at is not None else longest
        return max(0.0, end - self.matched_prefix_len) / longest


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("trace argument mappings must use string keys")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _digest(payload: Any) -> str:
    try:
        blob = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("trace arguments must be JSON serializable") from exc
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _get(step: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(step, dict):
            value = step.get(name)
        else:
            value = getattr(step, name, None)
        if value is not None:
            return value
    return default


def _text(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def project(trace: Iterable[Any]) -> List[Event]:
    """Project dict, ToolCall, or span-like steps into comparable events."""

    events = []
    for index, step in enumerate(trace):
        kind = _text(_get(step, "kind", "type", "span_type", default="tool"))
        name = _text(
            _get(step, "name", "tool", "tool_name", "function", default="")
        )
        if not name:
            raise ValueError("trace steps must have a non-empty name")
        arguments = _canonical(
            _get(
                step,
                "args",
                "arguments",
                "input_parameters",
                "input",
                "parameters",
                default={},
            )
        )
        raw_id = _get(
            step,
            "id",
            "event_id",
            "span_id",
            "uuid",
            default=f"idx-{index}",
        )
        event_id = _text(raw_id) or f"idx-{index}"
        weak_identity = f"{kind}:{name}"
        strong_identity = f"{weak_identity}:{_digest(arguments)}"
        events.append(
            Event(
                index=index,
                event_id=event_id,
                kind=kind,
                name=name,
                arguments=arguments,
                strong_identity=strong_identity,
                weak_identity=weak_identity,
            )
        )
    return events


def _common_prefix(
    baseline: Sequence[Event], candidate: Sequence[Event]
) -> int:
    index = 0
    while (
        index < len(baseline)
        and index < len(candidate)
        and baseline[index].strong_identity == candidate[index].strong_identity
    ):
        index += 1
    return index


def _reorder_at(
    baseline: Sequence[Event],
    candidate: Sequence[Event],
    index: int,
    lookahead: int,
) -> Tuple[List[Tuple[str, str]], int]:
    for width in range(2, lookahead + 1):
        baseline_window = baseline[index : index + width]
        candidate_window = candidate[index : index + width]
        if len(baseline_window) < width or len(candidate_window) < width:
            break
        baseline_ids = [event.strong_identity for event in baseline_window]
        candidate_ids = [event.strong_identity for event in candidate_window]
        if (
            Counter(baseline_ids) == Counter(candidate_ids)
            and baseline_ids != candidate_ids
        ):
            pairs = [
                (left.event_id, right.event_id)
                for left, right in zip(baseline_window, candidate_window)
                if left.strong_identity != right.strong_identity
            ]
            return pairs, width
    return [], 0


def _classify(
    baseline: Sequence[Event], candidate: Sequence[Event], index: int
) -> str:
    left = baseline[index] if index < len(baseline) else None
    right = candidate[index] if index < len(candidate) else None
    if left is None:
        return "extra"
    if right is None:
        return "absent"
    if left.weak_identity == right.weak_identity:
        return "arg_change"
    if (
        index + 1 < len(candidate)
        and left.strong_identity == candidate[index + 1].strong_identity
    ):
        return "extra"
    if (
        index + 1 < len(baseline)
        and right.strong_identity == baseline[index + 1].strong_identity
    ):
        return "absent"
    return "tool_change"


def _find_resync(
    baseline: Sequence[Event],
    candidate: Sequence[Event],
    index: int,
    lookahead: int,
    resync_run: int,
) -> Optional[int]:
    """Find the nearest sustained rejoin in a bounded 2-D offset window.

    A rejoin can consume divergent events on either or both sides. Searching
    only one-sided skew misses same-length substitutions (both sides advance
    once) and unequal local replacements where both sides advance before the
    shared suffix resumes.
    """

    offsets = [
        (left_offset, right_offset)
        for left_offset in range(lookahead + 1)
        for right_offset in range(lookahead + 1)
        if left_offset or right_offset
    ]
    offsets.sort(
        key=lambda pair: (
            pair[0] + pair[1],
            abs(pair[0] - pair[1]),
            max(pair),
            pair,
        )
    )

    for left_offset, right_offset in offsets:
        left = index + left_offset
        right = index + right_offset
        if left >= len(baseline) or right >= len(candidate):
            continue

        run = 0
        while (
            left + run < len(baseline)
            and right + run < len(candidate)
            and baseline[left + run].strong_identity
            == candidate[right + run].strong_identity
        ):
            run += 1
            if run >= resync_run:
                return max(left, right)
    return None


def _unmatched(
    baseline: Sequence[Event], candidate: Sequence[Event]
) -> Tuple[List[str], List[str]]:
    """Return occurrence-aware unmatched IDs, including duplicate steps."""

    candidate_counts = Counter(event.strong_identity for event in candidate)
    unmatched_baseline = []
    for event in baseline:
        if candidate_counts[event.strong_identity]:
            candidate_counts[event.strong_identity] -= 1
        else:
            unmatched_baseline.append(event.event_id)

    baseline_counts = Counter(event.strong_identity for event in baseline)
    unmatched_candidate = []
    for event in candidate:
        if baseline_counts[event.strong_identity]:
            baseline_counts[event.strong_identity] -= 1
        else:
            unmatched_candidate.append(event.event_id)
    return unmatched_baseline, unmatched_candidate


def align(
    baseline: Iterable[Any],
    candidate: Iterable[Any],
    *,
    lookahead: int = DEFAULT_LOOKAHEAD,
    resync_run: int = DEFAULT_RESYNC_RUN,
) -> AlignmentResult:
    """Locate the first sustained divergence between two traces."""

    if lookahead < 1 or resync_run < 1:
        raise ValueError("lookahead and resync_run must be positive")
    base = project(baseline)
    cand = project(candidate)
    prefix = _common_prefix(base, cand)
    if prefix == len(base) == len(cand):
        return AlignmentResult(
            aligned=True,
            matched_prefix_len=prefix,
            first_divergence=None,
            divergence_kind=None,
            resync_at=None,
            baseline_len=len(base),
            candidate_len=len(cand),
        )

    unmatched_baseline, unmatched_candidate = _unmatched(
        base[prefix:], cand[prefix:]
    )
    reordered, reorder_width = _reorder_at(base, cand, prefix, lookahead)
    if reordered:
        tail_prefix = _common_prefix(
            base[prefix + reorder_width :],
            cand[prefix + reorder_width :],
        )
        if prefix + reorder_width + tail_prefix == len(
            base
        ) and prefix + reorder_width + tail_prefix == len(cand):
            return AlignmentResult(
                aligned=False,
                matched_prefix_len=prefix,
                first_divergence=prefix,
                divergence_kind="order_change",
                resync_at=prefix + reorder_width,
                reordered=reordered,
                baseline_len=len(base),
                candidate_len=len(cand),
            )

    kind = _classify(base, cand, prefix)
    return AlignmentResult(
        aligned=False,
        matched_prefix_len=prefix,
        first_divergence=prefix,
        divergence_kind=kind,
        resync_at=_find_resync(base, cand, prefix, lookahead, resync_run),
        unmatched_baseline=unmatched_baseline,
        unmatched_candidate=unmatched_candidate,
        reordered=reordered,
        baseline_len=len(base),
        candidate_len=len(cand),
    )
