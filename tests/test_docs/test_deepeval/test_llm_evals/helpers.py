from typing import Optional


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
