from typing import List
from pydantic_ai.messages import ModelResponsePart
from pydantic_ai.agent import AgentRunResult
from pydantic_ai._run_context import RunContext
from deepeval.test_case.llm_test_case import ToolCall


# llm tools called
def extract_tools_called_from_llm_response(
    result: List[ModelResponsePart],
) -> List[ToolCall]:
    tool_calls = []

    # Loop through each ModelResponsePart
    for part in result:
        # Look for parts with part_kind="tool-call"
        if hasattr(part, "part_kind") and part.part_kind == "tool-call":
            # Extract tool name and args from the ToolCallPart
            tool_name = part.tool_name
            input_parameters = (
                part.args_as_dict() if hasattr(part, "args_as_dict") else None
            )

            # Create and append ToolCall object
            tool_call = ToolCall(
                name=tool_name, input_parameters=input_parameters
            )
            tool_calls.append(tool_call)

    return tool_calls


# TODO: llm tools called (reposne is present next message)
def extract_tools_called(result: AgentRunResult) -> List[ToolCall]:
    tool_calls = []

    # Access the message history from the _state
    message_history = result._state.message_history

    # Scan through all messages in the history
    for message in message_history:
        # Check if this is a ModelResponse (kind="response")
        if hasattr(message, "kind") and message.kind == "response":
            # For ModelResponse messages, check each part
            if hasattr(message, "parts"):
                for part in message.parts:
                    # Look for parts with part_kind="tool-call"
                    if (
                        hasattr(part, "part_kind")
                        and part.part_kind == "tool-call"
                    ):
                        # Extract tool name and args from the ToolCallPart
                        tool_name = part.tool_name
                        input_parameters = (
                            part.args_as_dict()
                            if hasattr(part, "args_as_dict")
                            else None
                        )

                        # Create and append ToolCall object
                        tool_call = ToolCall(
                            name=tool_name, input_parameters=input_parameters
                        )
                        tool_calls.append(tool_call)

    return tool_calls


def sanitize_run_context(value):
    """
    Recursively replace pydantic-ai RunContext instances with '<RunContext>'.

    This avoids leaking internal context details into recorded function kwargs,
    while keeping the original arguments intact for the actual function call.
    """
    if isinstance(value, RunContext):
        return "<RunContext>"
    if isinstance(value, dict):
        return {k: sanitize_run_context(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        sanitized = [sanitize_run_context(v) for v in value]
        return tuple(sanitized) if isinstance(value, tuple) else sanitized
    if isinstance(value, set):
        return {sanitize_run_context(v) for v in value}

    return value
