import asyncio
from typing import Optional

from deepeval.tracing.trace_test_manager import trace_testing_manager


def find_span_by_name(trace_dict: dict, span_name: str) -> Optional[dict]:
    """
    TraceApi payload uses keys like: baseSpans, llmSpans, retrieverSpans, toolSpans, agentSpans.
    We search across them for a span with the given name.
    """
    for key in (
        "llmSpans",
        "retrieverSpans",
        "toolSpans",
        "agentSpans",
        "baseSpans",
    ):
        spans = trace_dict.get(key) or []
        for span in spans:
            if span.get("name") == span_name:
                return span
    return None


def spans_by_key(trace_dict: dict, key: str) -> list[dict]:
    return trace_dict.get(key) or []


def span_names_by_key(trace_dict: dict, key: str) -> list[str]:
    return [s.get("name") for s in spans_by_key(trace_dict, key)]


def all_spans(trace_dict: dict):
    spans = []
    for key in (
        "llmSpans",
        "retrieverSpans",
        "toolSpans",
        "agentSpans",
        "baseSpans",
    ):
        spans.extend(trace_dict.get(key) or [])
    return spans


def get_latest_trace_dict():
    """
    trace_testing_manager.test_dict is often populated synchronously,
    but we keep a fallback to the async wait to avoid flakes.
    """
    if trace_testing_manager.test_dict is not None:
        return trace_testing_manager.test_dict

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(trace_testing_manager.wait_for_test_dict())


# DEBUG
def debug_span_names(trace_dict: dict) -> dict:
    return {
        "baseSpans": span_names_by_key(trace_dict, "baseSpans"),
        "llmSpans": span_names_by_key(trace_dict, "llmSpans"),
        "retrieverSpans": span_names_by_key(trace_dict, "retrieverSpans"),
        "toolSpans": span_names_by_key(trace_dict, "toolSpans"),
        "agentSpans": span_names_by_key(trace_dict, "agentSpans"),
    }


# def get_span_test_case(span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     """
#     Return the test case payload from a span if present.
#     Supports a few common serialization shapes.
#     """
#     # Most direct: `testCase` or `test_case`
#     for k in ("testCase", "test_case", "llmTestCase", "llm_test_case"):
#         v = span.get(k)
#         if isinstance(v, dict):
#             return v

#     # Sometimes nested under metadata/attributes
#     for outer in ("metadata", "attributes", "attr", "spanAttributes"):
#         outer_v = span.get(outer)
#         if isinstance(outer_v, dict):
#             for k in ("testCase", "test_case", "llmTestCase", "llm_test_case"):
#                 v = outer_v.get(k)
#                 if isinstance(v, dict):
#                     return v

#     return None
