import json

from typing import List, Optional, Tuple, Any
from opentelemetry.sdk.trace.export import ReadableSpan

from deepeval.test_case.api import create_api_test_case
from deepeval.test_run.api import LLMApiTestCase
from deepeval.test_run.test_run import global_test_run_manager
from deepeval.tracing.types import Trace, LLMTestCase, ToolCall
from deepeval.tracing import trace_manager, BaseSpan
from deepeval.tracing.utils import make_json_serializable


GEN_AI_OPERATION_NAMES = ["chat", "generate_content", "text_completion"]


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
    input = None
    output = None
    try:
        # check for system instructions
        system_instructions = []
        system_instructions_raw = span.attributes.get(
            "gen_ai.system_instructions"
        )
        if system_instructions_raw and isinstance(system_instructions_raw, str):
            system_instructions_json = json.loads(system_instructions_raw)
            system_instructions = _flatten_system_instructions(
                system_instructions_json
            )

        input_messages = []
        input_messages_raw = span.attributes.get("gen_ai.input.messages")
        if input_messages_raw and isinstance(input_messages_raw, str):
            input_messages_json = json.loads(input_messages_raw)
            input_messages = _flatten_input(input_messages_json)

        input = system_instructions + input_messages

        model_parameters = check_model_parameters(span)
        if model_parameters:
            input.append(model_parameters)

    except Exception:
        pass
    try:
        output = json.loads(span.attributes.get("gen_ai.output.messages"))
        output = _flatten_input(output)
    except Exception:
        pass

    if input is None and output is None:
        try:
            input = json.loads(span.attributes.get("events"))
            if input and isinstance(input, list):
                # check if the last event is a genai choice
                last_event = input.pop()
                if (
                    last_event
                    and last_event.get("event.name") == "gen_ai.choice"
                ):
                    output = last_event
        except Exception:
            pass

    return input, output


def _flatten_system_instructions(system_instructions: list) -> list:
    if isinstance(system_instructions, list):
        for system_instruction in system_instructions:
            if isinstance(system_instruction, dict):
                role = system_instruction.get("role")
                if not role:
                    system_instruction["role"] = "System Instruction"
        return _flatten_input(system_instructions)
    elif isinstance(system_instructions, str):
        return [{"role": "System Instruction", "content": system_instructions}]

    return []


def _flatten_input(input: list) -> list:
    if input and isinstance(input, list):
        try:
            result: List[dict] = []
            for m in input:
                if isinstance(m, dict):
                    role = m.get("role")
                    if not role:
                        role = "assistant"
                    parts = m.get("parts")
                    if parts:
                        for part in parts:
                            if isinstance(part, dict):
                                ptype = part.get("type")
                                if ptype == "text":
                                    result.append(
                                        {
                                            "role": role,
                                            "content": part.get("content"),
                                        }
                                    )
                                else:
                                    result.append(
                                        {
                                            "role": role,
                                            "content": make_json_serializable(
                                                part
                                            ),
                                        }
                                    )
                            else:
                                result.append(
                                    {
                                        "role": role,
                                        "content": make_json_serializable(part),
                                    }
                                )
                    else:
                        result.append(
                            {"role": role, "content": m.get("content")}
                        )  # no parts
                else:
                    result.append(
                        {
                            "role": "assistant",
                            "content": make_json_serializable(m),
                        }
                    )
            return result
        except Exception:
            return input

    return input


def check_tool_name_from_gen_ai_attributes(span: ReadableSpan) -> Optional[str]:
    try:
        gen_ai_tool_name = span.attributes.get("gen_ai.tool.name")
        if gen_ai_tool_name:
            return gen_ai_tool_name
    except Exception:
        pass

    return None


def check_tool_input_parameters_from_gen_ai_attributes(
    span: ReadableSpan,
) -> Optional[dict]:
    try:
        tool_arguments = span.attributes.get("tool_arguments")
        if tool_arguments:
            return json.loads(tool_arguments)
    except Exception:
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
    except Exception:
        pass

    return "base"


def check_model_from_gen_ai_attributes(span: ReadableSpan):
    try:
        gen_ai_request_model_name = span.attributes.get("gen_ai.request.model")
        if gen_ai_request_model_name:
            return gen_ai_request_model_name
    except Exception:
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
                except Exception:
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
                except Exception:
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


def normalize_pydantic_ai_messages(span: ReadableSpan) -> Optional[list]:
    try:
        raw = span.attributes.get("pydantic_ai.all_messages")
        if not raw:
            return None

        messages = raw
        if isinstance(messages, str):
            messages = json.loads(messages)
        elif isinstance(messages, tuple):
            messages = list(messages)

        if isinstance(messages, list):
            normalized = []
            for m in messages:
                if isinstance(m, str):
                    try:
                        m = json.loads(m)
                    except Exception:
                        pass
                normalized.append(m)
            return normalized
    except Exception:
        pass

    return []


def _extract_non_thinking_part_of_last_message(message: dict) -> dict:

    if isinstance(message, dict) and message.get("role") == "assistant":
        parts = message.get("parts")
        if parts:
            # Iterate from the last part
            for part in reversed(parts):
                if isinstance(part, dict) and part.get("type") == "text":
                    # Return a modified message with only the text content
                    return {"role": "assistant", "content": part.get("content")}
    return None


def check_pydantic_ai_agent_input_output(
    span: ReadableSpan,
) -> Tuple[Optional[Any], Optional[Any]]:
    input_val: list = []
    output_val: Optional[Any] = None

    # Get normalized messages once
    normalized = normalize_pydantic_ai_messages(span)

    # Input (pydantic_ai.all_messages) - slice up to and including the first 'user' message
    if normalized:
        try:
            first_user_idx = None
            for i, m in enumerate(normalized):
                role = None
                if isinstance(m, dict):
                    role = m.get("role") or m.get("author")
                if role == "user":
                    first_user_idx = i
                    break

            input_val = (
                normalized
                if first_user_idx is None
                else normalized[: first_user_idx + 1]
            )
        except Exception:
            pass

    # Output (agent final_result)
    try:
        if span.attributes.get("confident.span.type") == "agent":
            output_val = span.attributes.get("final_result")
            if not output_val and normalized:
                output_val = _extract_non_thinking_part_of_last_message(
                    normalized[-1]
                )
    except Exception:
        pass

    system_instructions = []
    system_instruction_raw = span.attributes.get("gen_ai.system_instructions")
    if system_instruction_raw and isinstance(system_instruction_raw, str):
        system_instructions = _flatten_system_instructions(
            json.loads(system_instruction_raw)
        )

    input_val = _flatten_input(input_val)
    return system_instructions + input_val, output_val


def check_tool_output(span: ReadableSpan):
    try:
        return span.attributes.get("tool_response")
    except Exception:
        pass
    return None


def check_pydantic_ai_trace_input_output(
    span: ReadableSpan,
) -> Tuple[Optional[Any], Optional[Any]]:
    input_val: Optional[Any] = None
    output_val: Optional[Any] = None

    if not span.parent:
        input_val, output_val = check_pydantic_ai_agent_input_output(span)

    return input_val, output_val


def check_model_parameters(span: ReadableSpan) -> Optional[dict]:
    try:
        raw_model_parameters = span.attributes.get("model_request_parameters")
        if raw_model_parameters and isinstance(raw_model_parameters, str):
            model_parameters = json.loads(raw_model_parameters)
            if isinstance(model_parameters, dict):
                return {
                    "role": "Model Request Parameters",
                    "content": model_parameters,
                }
    except Exception:
        pass
    return None
