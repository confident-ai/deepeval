import json
from typing import Any, Dict, List, Optional, Union

TRUNCATION_MARKER = "…[truncated]"
MAX_CONTENT_CHARS = 100_000
ShownContent = Union[List[Dict[str, Any]], Dict[str, Any], str]
SPAN_FIELD_CHAR_LIMIT = 1500
SPAN_LIMIT = 40
SPANS_CHAR_BUDGET = 60_000
_SPAN_TYPE_PRIORITY = {"llm": 0, "tool": 1, "agent": 2, "retriever": 3}
_DEFAULT_SPAN_PRIORITY = 4
_ERRORED_SPAN_PRIORITY = -1


def truncate_span_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else serialize(value)
    if not text or text in ("[]", "{}"):
        return None
    if len(text) > SPAN_FIELD_CHAR_LIMIT:
        return text[:SPAN_FIELD_CHAR_LIMIT] + TRUNCATION_MARKER
    return text


def select_priority_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    span_indexes_by_priority = sorted(
        range(len(spans)), key=lambda i: (_get_span_priority(spans[i]), i)
    )
    kept_indexes = span_indexes_by_priority[:SPAN_LIMIT]
    total_chars = sum(len(serialize(spans[i])) for i in kept_indexes)
    while kept_indexes and total_chars > SPANS_CHAR_BUDGET:
        total_chars -= len(serialize(spans[kept_indexes.pop()]))
    return [{"index": i + 1, **spans[i]} for i in sorted(kept_indexes)]


def fit_trace(trace: Dict[str, Any], budget: int) -> Union[Dict[str, Any], str]:
    if len(serialize(trace)) <= budget:
        return trace
    spans = trace.get("spans")
    if isinstance(spans, list):
        trace_without_spans = {
            key: value for key, value in trace.items() if key != "spans"
        }
        drop_order = sorted(
            range(len(spans)),
            key=lambda i: (_get_span_priority(spans[i]), i),
            reverse=True,
        )
        dropped_indexes = set()
        for i in drop_order:
            dropped_indexes.add(i)
            remaining_spans = [
                s for j, s in enumerate(spans) if j not in dropped_indexes
            ]
            fitted_trace = (
                {**trace_without_spans, "spans": remaining_spans}
                if remaining_spans
                else trace_without_spans
            )
            if len(serialize(fitted_trace)) <= budget:
                return fitted_trace
        trace = trace_without_spans
    return _truncate_text(serialize(trace), budget)


def fit_text(text: str, budget: int) -> str:
    return _truncate_text(text, budget)


def _truncate_text(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(budget - len(TRUNCATION_MARKER), 0)] + TRUNCATION_MARKER


def fit_turns(turns: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    first_kept_turn = 0
    while (
        first_kept_turn < len(turns) - 1
        and len(str(turns[first_kept_turn:])) > budget
    ):
        first_kept_turn += 1
    kept_turns = turns[first_kept_turn:]
    overflow = len(str(kept_turns)) - budget
    if overflow > 0 and kept_turns:
        content = kept_turns[-1].get("content")
        if isinstance(content, str):
            kept_turns[-1] = {
                **kept_turns[-1],
                "content": _truncate_text(
                    content, max(len(content) - overflow, 0)
                ),
            }
    return kept_turns


def _get_span_priority(span: Dict[str, Any]) -> int:
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
