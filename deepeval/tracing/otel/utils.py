from typing import List, Optional, Tuple, Any
from deepeval.tracing.types import Trace, LLMTestCase, ToolCall
from deepeval.tracing import trace_manager, BaseSpan
from opentelemetry.sdk.trace.export import ReadableSpan
import json

GEN_AI_OPERATION_NAMES = ["chat", "generate_content", "task_completion"]


def to_hex_string(id_value: int | bytes, length: int = 32) -> str:
    """
    Convert a trace ID or span ID to a hex string.

    Args:
        id_value: The ID value to convert, either as an integer or bytes
        length: The expected length of the hex string (32 for trace IDs, 16 for span IDs)

    Returns:
        A hex string representation of the ID
    """
    if isinstance(id_value, int):
        return format(id_value, f"0{length}x")
    return id_value.hex()


def set_trace_time(trace: Trace):
    """
    Set the trace time based on the root span with the largest start and end time gap.

    Args:
        trace: The trace object to update
    """

    if not trace.root_spans:
        return

    # Find the root span with the largest time gap
    max_gap = 0
    target_span = None

    for span in trace.root_spans:
        # Skip spans that don't have both start and end times
        if span.end_time is None:
            continue

        # Calculate the time gap
        time_gap = span.end_time - span.start_time

        # Update if this span has a larger gap
        if time_gap > max_gap:
            max_gap = time_gap
            target_span = span

    # If we found a valid span, set the trace time to match
    if target_span is not None:
        trace.start_time = target_span.start_time
        trace.end_time = target_span.end_time


def validate_llm_test_case_data(
    input: Optional[str],
    actual_output: Optional[str],
    expected_output: Optional[str],
    context: Optional[List[str]],
    retrieval_context: Optional[List[str]],
) -> None:
    """Validate LLMTestCase data before creation"""
    if input is not None and not isinstance(input, str):
        raise ValueError(f"input must be a string, got {type(input)}")

    if actual_output is not None and not isinstance(actual_output, str):
        raise ValueError(
            f"actual_output must be a string, got {type(actual_output)}"
        )

    if expected_output is not None and not isinstance(expected_output, str):
        raise ValueError(
            f"expected_output must be None or a string, got {type(expected_output)}"
        )

    if context is not None:
        if not isinstance(context, list) or not all(
            isinstance(item, str) for item in context
        ):
            raise ValueError("context must be None or a list of strings")

    if retrieval_context is not None:
        if not isinstance(retrieval_context, list) or not all(
            isinstance(item, str) for item in retrieval_context
        ):
            raise ValueError(
                "retrieval_context must be None or a list of strings"
            )


####### gen ai attributes utils (warning: use in try except)#######


def check_llm_input_from_gen_ai_attributes(
    span: ReadableSpan,
) -> Tuple[Optional[list], Optional[dict]]:
    try:
        input = json.loads(span.attributes.get("events"))
        if input and isinstance(input, list):
            # check if the last event is a genai choice
            last_event = input.pop()
            if last_event and last_event.get("event.name") == "gen_ai.choice":
                return input, last_event
    except Exception as e:
        pass

    return None, None


def check_tool_name_from_gen_ai_attributes(span: ReadableSpan) -> Optional[str]:
    try:
        gen_ai_tool_name = span.attributes.get("gen_ai.tool.name")
        if gen_ai_tool_name:
            return gen_ai_tool_name
    except Exception as e:
        pass

    return None


def check_tool_input_parameters_from_gen_ai_attributes(
    span: ReadableSpan,
) -> Optional[dict]:
    try:
        tool_arguments = span.attributes.get("tool_arguments")
        if tool_arguments:
            return json.loads(tool_arguments)
    except Exception as e:
        pass

    return None


def check_span_type_from_gen_ai_attributes(span: ReadableSpan):
    try:
        gen_ai_operation_name = span.attributes.get("gen_ai.operation.name")
        gen_ai_tool_name = span.attributes.get("gen_ai.tool.name")

        if (
            gen_ai_operation_name
            and gen_ai_operation_name in GEN_AI_OPERATION_NAMES
        ):
            return "llm"

        elif gen_ai_tool_name:
            return "tool"
    except Exception as e:
        pass

    return "base"


def check_model_from_gen_ai_attributes(span: ReadableSpan):
    try:
        gen_ai_request_model_name = span.attributes.get("gen_ai.request.model")
        if gen_ai_request_model_name:
            return gen_ai_request_model_name
    except Exception as e:
        pass

    return None


def prepare_trace_llm_test_case(span: ReadableSpan) -> Optional[LLMTestCase]:

    test_case = LLMTestCase(input="")

    _input = span.attributes.get("confident.trace.llm_test_case.input")
    if isinstance(_input, str):
        test_case.input = _input

    _actual_output = span.attributes.get(
        "confident.trace.llm_test_case.actual_output"
    )
    if isinstance(_actual_output, str):
        test_case.actual_output = _actual_output

    _expected_output = span.attributes.get(
        "confident.trace.llm_test_case.expected_output"
    )
    if isinstance(_expected_output, str):
        test_case.expected_output = _expected_output

    _context = span.attributes.get("confident.trace.llm_test_case.context")
    if isinstance(_context, list):
        if all(isinstance(item, str) for item in _context):
            test_case.context = _context

    _retrieval_context = span.attributes.get(
        "confident.trace.llm_test_case.retrieval_context"
    )
    if isinstance(_retrieval_context, list):
        if all(isinstance(item, str) for item in _retrieval_context):
            test_case.retrieval_context = _retrieval_context

    tools_called: List[ToolCall] = []
    expected_tools: List[ToolCall] = []

    _tools_called = span.attributes.get(
        "confident.trace.llm_test_case.tools_called"
    )
    if isinstance(_tools_called, list):
        for tool_call_json_str in _tools_called:
            if isinstance(tool_call_json_str, str):
                try:
                    tools_called.append(
                        ToolCall.model_validate_json(tool_call_json_str)
                    )
                except Exception as e:
                    pass

    _expected_tools = span.attributes.get(
        "confident.trace.llm_test_case.expected_tools"
    )
    if isinstance(_expected_tools, list):
        for tool_call_json_str in _expected_tools:
            if isinstance(tool_call_json_str, str):
                try:
                    expected_tools.append(
                        ToolCall.model_validate_json(tool_call_json_str)
                    )
                except Exception as e:
                    pass

    test_case.tools_called = tools_called
    test_case.expected_tools = expected_tools

    if test_case.input == "":
        return None

    return test_case


def parse_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def parse_list_of_strings(context: List[str]) -> List[str]:
    parsed_context: List[str] = []
    if context and (isinstance(context, list) or isinstance(context, tuple)):
        for context_str in context:
            if not isinstance(context_str, str):
                pass
            else:
                parsed_context.append(context_str)
    return parsed_context


from deepeval.evaluate.utils import create_api_test_case
from deepeval.test_run.api import LLMApiTestCase
from deepeval.test_run.test_run import global_test_run_manager
from typing import Optional


def post_test_run(traces: List[Trace], test_run_id: Optional[str]):
    # Accept single trace or list of traces
    if isinstance(traces, Trace):
        traces = [traces]

    api_test_cases: List[LLMApiTestCase] = []

    # Collect test cases from spans that have metric_collection
    for trace in traces:
        trace_api = trace_manager.create_trace_api(trace)

        def dfs(span: BaseSpan):
            if span.metric_collection:
                llm_test_case = LLMTestCase(
                    input=str(span.input),
                    actual_output=(
                        str(span.output) if span.output is not None else None
                    ),
                    expected_output=span.expected_output,
                    context=span.context,
                    retrieval_context=span.retrieval_context,
                    tools_called=span.tools_called,
                    expected_tools=span.expected_tools,
                )
                api_case = create_api_test_case(
                    test_case=llm_test_case,
                    trace=trace_api,
                    index=None,
                )
                if isinstance(api_case, LLMApiTestCase):
                    api_case.metric_collection = span.metric_collection
                    api_test_cases.append(api_case)

            for child in span.children or []:
                dfs(child)

        for root in trace.root_spans:
            dfs(root)

    # Prepare and post TestRun using the global test run manager
    test_run_manager = global_test_run_manager
    test_run_manager.create_test_run(identifier=test_run_id)
    test_run = test_run_manager.get_test_run()

    for case in api_test_cases:
        test_run.add_test_case(case)

    # return test_run_manager.post_test_run(test_run) TODO: add after test run with metric collection is implemented
