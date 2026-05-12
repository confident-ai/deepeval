"""Load `test_run_*.json` files into nested `Trace` / `BaseSpan` view models.

The on-disk shape is `TraceApi`: five flat span buckets linked via
`parentUuid`. The loader pops the buckets, validates each span dict
into a single `BaseSpan` class (the `.type` field is the discriminator),
and wires up `children` / `root_spans`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.inspect.types import BaseSpan, Trace


class InspectLoadError(Exception):
    """File unreadable or top-level JSON malformed."""


class NoTracesError(InspectLoadError):
    """File parsed but contains zero traces."""


_SPAN_BUCKETS: List[str] = [
    "baseSpans",
    "agentSpans",
    "llmSpans",
    "retrieverSpans",
    "toolSpans",
]


def find_latest_test_run(folder: str | Path) -> Path:
    """Most recently modified `test_run_*.json` under `folder`.

    Sorted by mtime (not filename) so a manually-copied file with a
    stale timestamp in its name still ranks correctly.
    """

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Results folder not found: {folder_path}. "
            "Pass `results_folder=...` to `DisplayConfig(...)` or set "
            "the `DEEPEVAL_RESULTS_FOLDER` env var."
        )

    candidates = sorted(
        folder_path.glob("test_run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No test_run_*.json files found in {folder_path}."
        )
    return candidates[0]


def load_test_run(path: str | Path) -> List[Trace]:
    p = Path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InspectLoadError(f"Failed to read test run from {p}: {e}") from e

    if not isinstance(data, dict):
        raise InspectLoadError(
            f"Expected the top-level JSON in {p} to be an object; "
            f"got {type(data).__name__}."
        )

    traces: List[Trace] = []
    for case in data.get("testCases", []):
        trace_dict = case.get("trace") if isinstance(case, dict) else None
        if trace_dict:
            traces.append(_parse_trace(trace_dict))

    if not traces:
        raise NoTracesError(
            f"{p} contains no traces. `deepeval inspect` shows trace "
            "trees; runs without tracing data have nothing to display."
        )
    return traces


def _parse_trace(trace_dict: Dict[str, Any]) -> Trace:
    # Pop the bucket keys before validating so the residual dict is
    # clean trace-level data; spans are validated and linked separately.
    trace_dict = dict(trace_dict)
    span_dicts: List[Dict[str, Any]] = []
    for bucket_name in _SPAN_BUCKETS:
        for span_dict in trace_dict.pop(bucket_name, None) or []:
            if isinstance(span_dict, dict):
                span_dicts.append(span_dict)

    spans = _build_span_tree(span_dicts)
    roots = [s for s in spans.values() if not _has_known_parent(s, spans)]
    roots.sort(key=lambda s: s.start_time or "")

    trace = Trace.model_validate(trace_dict)
    trace.root_spans = roots
    return trace


def _build_span_tree(
    span_dicts: List[Dict[str, Any]],
) -> Dict[str, BaseSpan]:
    by_uuid: Dict[str, BaseSpan] = {}
    for span_dict in span_dicts:
        span = BaseSpan.model_validate(span_dict)
        # On UUID collision keep the first occurrence; silently swapping
        # would be the wrong fallback for malformed input.
        if span.uuid not in by_uuid:
            by_uuid[span.uuid] = span

    for span in by_uuid.values():
        parent = by_uuid.get(span.parent_uuid) if span.parent_uuid else None
        if parent is not None:
            parent.children.append(span)

    for span in by_uuid.values():
        span.children.sort(key=lambda c: c.start_time or "")

    return by_uuid


def _has_known_parent(span: BaseSpan, by_uuid: Dict[str, BaseSpan]) -> bool:
    return bool(span.parent_uuid) and span.parent_uuid in by_uuid


def run_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def summarize_test_run(path: str | Path) -> Optional[Dict[str, Any]]:
    """Run-level pass/fail + duration counts for the header bar.

    Returns `None` if the file can't be opened — the header then falls
    back to showing just the run id and trace count.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    return {
        "test_passed": data.get("testPassed"),
        "test_failed": data.get("testFailed"),
        "run_duration": data.get("runDuration"),
        "evaluation_cost": data.get("evaluationCost"),
    }
