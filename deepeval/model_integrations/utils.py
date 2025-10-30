import json
import uuid
from typing import Any, List, Optional

from deepeval.model_integrations.types import InputParameters, OutputParameters
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_span,
    update_llm_span,
)
from deepeval.tracing.trace_context import current_llm_context
from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.utils import shorten, len_long


def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
):
    """Update span and trace attributes with input/output parameters."""
    update_current_span(
        input=input_parameters.input or input_parameters.messages or "NA",
        output=output_parameters.output or "NA",
        tools_called=output_parameters.tools_called,
        # attributes to be added
        expected_output=expected_output,
        expected_tools=expected_tools,
        context=context,
        retrieval_context=retrieval_context,
    )

    llm_context = current_llm_context.get()

    update_llm_span(
        input_token_count=output_parameters.prompt_tokens,
        output_token_count=output_parameters.completion_tokens,
        prompt=llm_context.prompt,
    )

    if output_parameters.tools_called:
        create_child_tool_spans(output_parameters)

    __update_input_and_output_of_current_trace(
        input_parameters, output_parameters
    )


def __update_input_and_output_of_current_trace(
    input_parameters: InputParameters, output_parameters: OutputParameters
):

    current_trace = current_trace_context.get()
    if current_trace:
        if current_trace.input is None:
            current_trace.input = (
                input_parameters.input or input_parameters.messages
            )
        if current_trace.output is None:
            current_trace.output = output_parameters.output

    return


def create_child_tool_spans(output_parameters: OutputParameters):
    if output_parameters.tools_called is None:
        return

    current_span = current_span_context.get()
    for tool_called in output_parameters.tools_called:
        tool_span = ToolSpan(
            **{
                "uuid": str(uuid.uuid4()),
                "trace_uuid": current_span.trace_uuid,
                "parent_uuid": current_span.uuid,
                "start_time": current_span.start_time,
                "end_time": current_span.start_time,
                "status": TraceSpanStatus.SUCCESS,
                "children": [],
                "name": tool_called.name,
                "input": tool_called.input_parameters,
                "output": None,
                "metrics": None,
                "description": tool_called.description,
            }
        )
        current_span.children.append(tool_span)


_URL_MAX = 200
_JSON_MAX = max(
    len_long(), 400
)  # <- make this bigger by increasing DEEPEVAL_MAXLEN_LONG above 400


def compact_dump(value: Any) -> str:
    try:
        dumped = json.dumps(
            value, ensure_ascii=False, default=str, separators=(",", ":")
        )
    except Exception:
        dumped = repr(value)
    return shorten(dumped, max_len=_JSON_MAX)


def fmt_url(url: Optional[str]) -> str:
    if not url:
        return ""
    if url.startswith("data:"):
        return "[data-uri]"
    return shorten(url, max_len=_URL_MAX)
