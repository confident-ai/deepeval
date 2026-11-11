"""Read GenAI message attributes and events as deepeval span content."""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.util.types import Attributes

from deepeval.test_case import ToolCall
from deepeval.tracing.otel.attribute_values import (
    first_defined,
    structured_value,
    tool_call_from,
)
from deepeval.tracing.otel.attributes import GenAIAttr
from deepeval.tracing.otel.utils import (
    _flatten_input,
    _flatten_system_instructions,
)

_LEGACY_CHOICE_EVENT = "gen_ai.choice"
_LEGACY_MESSAGE_EVENT = re.compile(
    r"^gen_ai\.(system|user|assistant|tool)\.message$"
)


def llm_io(span: ReadableSpan) -> Tuple[Any, Any]:
    """Message attributes win, even when empty; legacy events fill gaps."""
    attributes = span.attributes or {}
    instructions = _system_instructions(attributes)
    input_messages = structured_value(attributes.get(GenAIAttr.INPUT_MESSAGES))
    llm_input = None
    if isinstance(input_messages, list) or instructions:
        llm_input = instructions + (
            _flatten_input(input_messages)
            if isinstance(input_messages, list)
            else []
        )
    output_messages = structured_value(
        attributes.get(GenAIAttr.OUTPUT_MESSAGES)
    )
    llm_output = (
        _flatten_input(output_messages)
        if isinstance(output_messages, list)
        else None
    )
    if llm_input is None or llm_output is None:
        legacy_input, legacy_output = _legacy_event_io(span.events)
        llm_input = legacy_input if llm_input is None else llm_input
        llm_output = legacy_output if llm_output is None else llm_output
    return llm_input, answer_content(llm_output)


def answer_content(messages: Any) -> Any:
    """Flattened output messages as their text.

    Output without text, such as a turn that only calls tools, stays, less
    the empty text a framework may send with it.
    """
    if not isinstance(messages, list):
        return messages
    turn = _final_assistant_turn(messages)
    texts = [
        message["content"]
        for message in turn
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    ]
    if any(texts):
        return "\n".join(text for text in texts if text)
    rest = [
        message
        for message in turn
        if not (isinstance(message, dict) and message.get("content") == "")
    ]
    return rest or ("" if texts else turn)


def _final_assistant_turn(messages: List[Any]) -> List[Any]:
    """A conversation's final assistant turn; a model's output is all one turn."""
    start = len(messages)
    while (
        start
        and isinstance(messages[start - 1], dict)
        and messages[start - 1].get("role") == "assistant"
    ):
        start -= 1
    return messages[start:] if 0 < start < len(messages) else messages


def trace_io(key: str, value: Any) -> Any:
    """Read a model's messages, which confident-trace copies to its trace.

    As on the platform, the trace takes the latest user message as input; its
    output is the answer's text.
    """
    if not is_message_array(value):
        return value
    messages = _flatten_input(value)
    if key == "output":
        return answer_content(messages)
    return latest_user_text(messages)


def flatten_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    """GenAI messages as flattened ``{role, content}`` dicts, if they're a list."""
    return _flatten_input(messages) if isinstance(messages, list) else None


def latest_assistant_text(messages: Any) -> Optional[str]:
    """The last assistant message's text in flattened messages."""
    if not isinstance(messages, list):
        return None
    return next(
        (
            message["content"]
            for message in reversed(messages)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ),
        None,
    )


def latest_user_text(messages: Any) -> Any:
    """The last user message's text in flattened messages; anything else stays."""
    if not isinstance(messages, list):
        return messages
    return next(
        (
            message["content"]
            for message in reversed(messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        messages,
    )


def tool_arguments(value: Any) -> Any:
    """A tool's input recorded as its call's ``{"args", "kwargs"}``, as its
    keyword arguments when it took no positional ones; any other value stays."""
    if (
        isinstance(value, dict)
        and set(value) <= {"args", "kwargs"}
        and not value.get("args")
        and isinstance(value.get("kwargs"), dict)
    ):
        return value["kwargs"]
    return value


def tool_response(value: Any) -> Any:
    """A tool's reply sent as one message, like LangChain's ``ToolMessage``,
    as its response; any other value stays."""
    if not is_message_array(value) or len(value) != 1:
        return value
    parts = value[0]["parts"]
    if (
        len(parts) == 1
        and isinstance(parts[0], dict)
        and parts[0].get("type") == "tool_call_response"
        and "response" in parts[0]
    ):
        return parts[0]["response"]
    return value


def is_message_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("parts"), list)
            for message in value
        )
    )


def _system_instructions(attributes: Attributes) -> List[Dict[str, Any]]:
    raw = attributes.get(GenAIAttr.SYSTEM_INSTRUCTIONS)
    if raw is None:
        return []
    parsed = structured_value(raw)
    return _flatten_system_instructions(raw if parsed is None else parsed)


def _legacy_event_io(
    events: Sequence[Event],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    for event in events:
        attributes = dict(event.attributes or {})
        body = structured_value(attributes.get("body"))
        if not isinstance(body, dict):
            body = attributes
        if event.name == _LEGACY_CHOICE_EVENT:
            message = body.get("message")
            if isinstance(message, str):
                parsed = structured_value(message)
                # An answer that happens to be JSON, like "56", stays text.
                if isinstance(parsed, (dict, list)):
                    message = parsed
            if message is not None and not isinstance(message, dict):
                # Plain text or a content-block list.
                outputs.append(
                    {"role": "assistant", "content": block_text(message)}
                )
                continue
            value = message if isinstance(message, dict) else body
            role = value.get("role")
            outputs.append(
                {
                    "role": role if isinstance(role, str) else "assistant",
                    "content": block_text(
                        first_defined(value.get("content"), value)
                    ),
                }
            )
            continue
        match = _LEGACY_MESSAGE_EVENT.match(event.name)
        if match:
            inputs.append(
                {
                    "role": match.group(1),
                    "content": block_text(
                        first_defined(body.get("content"), body)
                    ),
                }
            )
    return inputs or None, outputs or None


def block_text(content: Any) -> Any:
    """Content blocks, such as ``[{"text": ...}]``, as their joined text.

    Events carry them as JSON text. Blocks without text, like a lone tool
    call, stay structured.
    """
    blocks = structured_value(content) if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return content
    texts = [
        block["text"]
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(texts) if texts else blocks


def message_tool_calls(messages: Any) -> Optional[List[ToolCall]]:
    if not isinstance(messages, list):
        return None
    calls = []
    for message in messages:
        parts = message.get("parts") if isinstance(message, dict) else None
        for part in parts if isinstance(parts, list) else []:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and isinstance(part.get("name"), str)
            ):
                arguments = part.get("arguments")
                if isinstance(arguments, str):
                    # Some frameworks send the arguments as JSON text.
                    arguments = structured_value(arguments)
                call = tool_call_from(
                    {"name": part["name"], "input_parameters": arguments}
                )
                if call:
                    calls.append(call)
    return calls or None
