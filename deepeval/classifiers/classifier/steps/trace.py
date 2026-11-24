from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from deepeval.classifiers.classifier.steps.truncate import (
    select_priority_spans,
    serialize,
    truncate_span_field,
)
from deepeval.classifiers.utils import is_populated
from deepeval.metrics.utils.decision import _jsonable
from deepeval.test_case import LLMTestCase

_TRACE_FIELDS = {
    "name": "name",
    "input": "input",
    "actual_output": "output",
    "expected_output": "expected_output",
    "retrieval_context": "retrieval_context",
    "context": "context",
    "tools_called": "tools_called",
    "expected_tools": "expected_tools",
}
_CAPPED_SPAN_FIELDS = ("error", "input", "output", "tools_called")


def construct_trace_content(content: Union[Dict[str, Any], str]) -> str:
    return content if isinstance(content, str) else serialize(content)


def construct_trace_dict(test_case: LLMTestCase) -> Dict[str, Any]:
    trace = {
        key: _to_jsonable(getattr(test_case, attr, None))
        for attr, key in _TRACE_FIELDS.items()
        if is_populated(getattr(test_case, attr, None))
    }
    spans = select_priority_spans(_flatten_span_tree(test_case._trace_dict, []))
    if spans:
        trace["spans"] = spans
    return trace


def _flatten_span_tree(span: Optional[Dict], spans: List[Dict]) -> List[Dict]:
    if not isinstance(span, dict):
        return spans
    flattened_span: Dict[str, Any] = {
        "name": span.get("name") or "",
        "type": _to_jsonable(span.get("type")),
        "status": "ERRORED" if span.get("error") else "SUCCESS",
    }
    for field in _CAPPED_SPAN_FIELDS:
        truncated_value = truncate_span_field(_to_jsonable(span.get(field)))
        if truncated_value is not None:
            flattened_span[field] = truncated_value
    spans.append(flattened_span)
    for child in span.get("children") or []:
        _flatten_span_tree(child, spans)
    return spans


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return _jsonable(value)
