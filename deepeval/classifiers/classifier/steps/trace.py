from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from deepeval.classifiers.classifier.steps.truncate import (
    cap_field,
    select_spans,
    serialize,
)
from deepeval.classifiers.utils import is_populated
from deepeval.metrics.utils.decision import _jsonable
from deepeval.test_case import LLMTestCase

_TRACE_FIELDS = {
    "name": "name",
    "input": "input",
    "actual_output": "output",
    "expected_output": "expectedOutput",
    "retrieval_context": "retrievalContext",
    "context": "context",
    "tools_called": "toolsCalled",
    "expected_tools": "expectedTools",
}
_CAPPED_SPAN_FIELDS = {
    "error": "error",
    "input": "input",
    "output": "output",
    "tools_called": "toolsCalled",
}


def construct_trace_content(content: Union[Dict[str, Any], str]) -> str:
    return content if isinstance(content, str) else serialize(content)


def construct_trace_dict(test_case: LLMTestCase) -> Dict[str, Any]:
    trace = {
        key: _jsonable_field(getattr(test_case, attr, None))
        for attr, key in _TRACE_FIELDS.items()
        if is_populated(getattr(test_case, attr, None))
    }
    spans = select_spans(_flatten_spans(test_case._trace_dict, []))
    if spans:
        trace["spans"] = spans
    return trace


def _flatten_spans(span: Optional[Dict], spans: List[Dict]) -> List[Dict]:
    if not isinstance(span, dict):
        return spans
    entry: Dict[str, Any] = {
        "name": span.get("name") or "",
        "type": _jsonable_field(span.get("type")),
        "status": "ERRORED" if span.get("error") else "SUCCESS",
    }
    for attr, key in _CAPPED_SPAN_FIELDS.items():
        capped = cap_field(_jsonable_field(span.get(attr)))
        if capped is not None:
            entry[key] = capped
    spans.append(entry)
    for child in span.get("children") or []:
        _flatten_spans(child, spans)
    return spans


def _jsonable_field(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable_field(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_field(v) for k, v in value.items()}
    return _jsonable(value)
