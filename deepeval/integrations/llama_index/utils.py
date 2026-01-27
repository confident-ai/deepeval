from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from deepeval.test_case.llm_test_case import LLMTestCase, ToolCall
from deepeval.tracing.types import BaseSpan
from typing import Any

try:
    from llama_index.core.agent.workflow.workflow_events import (
        AgentOutput,
        AgentWorkflowStartEvent,
    )

    llama_index_agent_installed = True
except:
    llama_index_agent_installed = False


def is_llama_index_agent_installed():
    if not llama_index_agent_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


def parse_id(id_: str) -> tuple[str, str]:
    """
    Parse the id_ into a tuple of class name and method name, ignoring any suffix after '-'.
    Returns empty strings as defaults if parsing fails.
    """
    try:
        # Ignore everything after the first '-'
        main_part = id_.split("-", 1)[0]
        # Split by '.' to get class and method
        parts = main_part.rsplit(".", 1)
        if len(parts) == 2:
            class_name, method_name = parts
        else:
            # If no '.' found, treat the whole string as class_name
            class_name, method_name = main_part, ""
        return class_name, method_name
    except:
        # Return empty strings if any parsing fails
        return "", ""


def prepare_input_llm_test_case_params(
    class_name: str, method_name: str, span: BaseSpan, args: dict
):

    # condition for parent agent span
    if method_name == "run":
        start_event = args.get("start_event")

        is_llama_index_agent_installed()
        if isinstance(start_event, AgentWorkflowStartEvent):
            input = ""
            for key, value in start_event.items():
                input += f"{key}: {value}\n"

            span.llm_test_case = LLMTestCase(
                input=input,
                actual_output="",
            )


def prepare_output_llm_test_case_params(
    class_name: str, method_name: str, result: Any, span: BaseSpan
):

    if class_name == "Workflow" and method_name == "run":

        is_llama_index_agent_installed()
        if isinstance(result, AgentOutput):
            span.llm_test_case.actual_output = result.response.content

            tool_calls = []
            for tool_call in result.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tool_call.tool_name,
                        input_parameters=tool_call.tool_kwargs,
                    )
                )

            span.llm_test_case.tools_called = tool_calls


def extract_output_from_llm_chat_end_event(event: LLMChatEndEvent) -> list:
    messages = []
    for msg in event.response.message.blocks:
        if msg.block_type == "text":
            messages.append(
                {
                    "role": event.response.message.role.value,
                    "content": msg.text,
                }
            )
        elif msg.block_type == "tool_call":
            messages.append(
                {
                    "name": msg.tool_name,
                    "input_parameters": msg.tool_kwargs,
                    "id": msg.tool_call_id,
                }
            )
        else:
            messages.append(msg.model_dump())
    return messages
