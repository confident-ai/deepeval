import json
from typing import Any, Dict, List, Optional, Union

TRUNCATION_MARKER = "…[truncated]"
MAX_CONTENT_CHARS = 100_000
SPAN_FIELD_CHAR_LIMIT = 1500
SPAN_LIMIT = 40
SPANS_CHAR_BUDGET = 60_000
_SPAN_TYPE_PRIORITY = {"llm": 0, "tool": 1, "agent": 2, "retriever": 3}
_DEFAULT_SPAN_PRIORITY = 4
_ERRORED_SPAN_PRIORITY = -1


def cap_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else serialize(value)
    if not text or text in ("[]", "{}"):
        return None
    if len(text) > SPAN_FIELD_CHAR_LIMIT:
        return text[:SPAN_FIELD_CHAR_LIMIT] + TRUNCATION_MARKER
    return text


def select_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        range(len(spans)), key=lambda i: (span_priority(spans[i]), i)
    )
    kept = ranked[:SPAN_LIMIT]
    size = sum(len(serialize(spans[i])) for i in kept)
    while kept and size > SPANS_CHAR_BUDGET:
        size -= len(serialize(spans[kept.pop()]))
    return [
        {"index": position + 1, **spans[i]}
        for position, i in enumerate(sorted(kept))
    ]


def fit_trace(trace: Dict[str, Any], budget: int) -> Union[Dict[str, Any], str]:
    if len(serialize(trace)) <= budget:
        return trace
    spans = trace.get("spans")
    if isinstance(spans, list):
        rest = {key: value for key, value in trace.items() if key != "spans"}
        drop_order = sorted(
            range(len(spans)),
            key=lambda i: (span_priority(spans[i]), i),
            reverse=True,
        )
        dropped = set()
        for i in drop_order:
            dropped.add(i)
            remaining = [s for j, s in enumerate(spans) if j not in dropped]
            fitted = {**rest, "spans": remaining} if remaining else rest
            if len(serialize(fitted)) <= budget:
                return fitted
        trace = rest
    return serialize(trace)[:budget]


def fit_text(text: str, budget: int) -> str:
    return text[:budget]


def fit_turns(turns: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    start = 0
    while start < len(turns) - 1 and len(str(turns[start:])) > budget:
        start += 1
    return turns[start:]


def span_priority(span: Any) -> int:
    if not isinstance(span, dict):
        return _DEFAULT_SPAN_PRIORITY
    if span.get("status") == "ERRORED" or span.get("error"):
        return _ERRORED_SPAN_PRIORITY
    return _SPAN_TYPE_PRIORITY.get(
        str(span.get("type", "")).lower(), _DEFAULT_SPAN_PRIORITY
    )


def serialize(value: Any) -> str:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, default=str
    )
