"""The tool calls a traced test case gives ArgumentCorrectness when its span
has none of its own.

Each tool call belongs to the LLM span that chose its arguments: the tool's
parent when that is an LLM span, else the latest LLM sibling that started
before it. Tool calls with no such LLM span are left out.
"""

from typing import List, Optional, Sequence, Tuple

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing.types import BaseSpan, LlmSpan, ToolSpan
from deepeval.tracing.utils import prepare_tool_call_input_parameters


def with_tool_calls_from_trace(test_case: LLMTestCase) -> LLMTestCase:
    """`test_case` with the tool calls (and, for a tool span, the input) its
    span tree gives it, or `test_case` itself when it already lists tool calls,
    has no span tree, or the tree yields none."""
    root_spans: Optional[List[BaseSpan]] = test_case._trace_root_spans
    if test_case.tools_called or root_spans is None:
        return test_case

    span: Optional[BaseSpan] = test_case._trace_span
    input = test_case.input
    if span is None:
        tool_calls = _tool_calls_within(root_spans, parent=None)
    else:
        position = _parent_and_siblings(span, root_spans, parent=None)
        if position is None:
            return test_case
        parent, siblings = position
        if isinstance(span, ToolSpan):
            llm_span = _owning_llm_span(span, parent, siblings)
            if llm_span is None or llm_span.input is None:
                return test_case
            input = str(llm_span.input)
            tool_calls = [_as_tool_call(span)]
        elif isinstance(span, LlmSpan):
            tool_calls = _owned_tool_calls(span, parent, siblings)
        else:
            tool_calls = _tool_calls_within(span.children, parent=span)

    if not tool_calls:
        return test_case
    return test_case.model_copy(
        update={"input": input, "tools_called": tool_calls}
    )


def _owning_llm_span(
    tool_span: ToolSpan,
    parent: Optional[BaseSpan],
    siblings: Sequence[BaseSpan],
) -> Optional[LlmSpan]:
    if isinstance(parent, LlmSpan):
        return parent
    earlier = [
        sibling
        for sibling in siblings
        if isinstance(sibling, LlmSpan)
        and sibling.start_time <= tool_span.start_time
    ]
    return max(earlier, key=lambda sibling: sibling.start_time, default=None)


def _owned_tool_calls(
    llm_span: LlmSpan,
    parent: Optional[BaseSpan],
    siblings: Sequence[BaseSpan],
) -> List[ToolCall]:
    # Tool calls the LLM span lists itself (from its output messages, or set
    # by the user) describe the same calls as its tool spans, so they win.
    if llm_span.tools_called:
        return list(llm_span.tools_called)
    tool_spans = [
        child for child in llm_span.children if isinstance(child, ToolSpan)
    ]
    tool_spans += [
        sibling
        for sibling in siblings
        if isinstance(sibling, ToolSpan)
        and _owning_llm_span(sibling, parent, siblings) is llm_span
    ]
    return [
        _as_tool_call(tool_span) for tool_span in _by_start_time(tool_spans)
    ]


def _tool_calls_within(
    spans: Sequence[BaseSpan], parent: Optional[BaseSpan]
) -> List[ToolCall]:
    """Tool calls of every LLM span in `spans` and below, nested agents
    included."""
    tool_calls = []
    for span in _by_start_time(spans):
        if isinstance(span, LlmSpan):
            tool_calls += _owned_tool_calls(span, parent, spans)
        tool_calls += _tool_calls_within(span.children, parent=span)
    return tool_calls


def _parent_and_siblings(
    span: BaseSpan,
    spans: Sequence[BaseSpan],
    parent: Optional[BaseSpan],
) -> Optional[Tuple[Optional[BaseSpan], Sequence[BaseSpan]]]:
    """`span`'s parent and siblings (itself included) within `spans`."""
    for candidate in spans:
        if candidate.uuid == span.uuid:
            return parent, spans
        position = _parent_and_siblings(span, candidate.children, candidate)
        if position is not None:
            return position
    return None


def _as_tool_call(tool_span: ToolSpan) -> ToolCall:
    return ToolCall(
        name=tool_span.name,
        description=tool_span.description,
        input_parameters=prepare_tool_call_input_parameters(tool_span.input),
        output=tool_span.output,
    )


def _by_start_time(spans: Sequence[BaseSpan]) -> List[BaseSpan]:
    return sorted(spans, key=lambda span: span.start_time)
