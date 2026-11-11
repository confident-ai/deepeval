from typing import List

from deepeval.config.settings import get_settings
from deepeval.tracing import trace_manager
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.types import Trace
from deepeval.tracing.utils import make_json_serializable

GEN_AI_OPERATION_NAMES = ["chat", "generate_content", "text_completion"]

GEN_AI_AGENT_OPERATION_NAMES = [
    "create_agent",
    "invoke_agent",
    "invoke_workflow",
]


def otlp_traces_endpoint() -> str:
    settings = get_settings()
    return str(settings.CONFIDENT_OTEL_URL) + "v1/traces"


def is_capturing_traces() -> bool:
    """True while an evaluation or a trace-shape test consumes traces locally."""
    try:
        if trace_manager.is_evaluating:
            return True
    except Exception:
        pass
    # Trace-shape testing override: when a test harness has set
    # ``trace_testing_manager.test_name``, force REST so spans flow
    # through ``trace_manager.end_trace`` (the only writer of
    # ``trace_testing_manager.test_dict``). Otherwise the
    # ``@assert_trace_json`` decorator silently times out and compares
    # ``{}`` to ``{}``, which trivially passes — masking real
    # trace-shape regressions for bare ``agent.run(...)`` flows.
    try:
        return trace_testing_manager.test_name is not None
    except Exception:
        return False


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
