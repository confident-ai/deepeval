import json
import uuid
from typing import Any, Dict, Iterable, List, Optional

from deepeval.model_integrations.types import InputParameters, OutputParameters
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
    update_current_span,
    update_llm_span,
)
from deepeval.tracing.trace_context import current_llm_context
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import LlmSpan, ToolSpan, TraceSpanStatus
from deepeval.utils import shorten, len_long, serialize_to_json


def _update_all_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    expected_tools: List[ToolCall],
    expected_output: str,
    context: List[str],
    retrieval_context: List[str],
    integration: str,
    provider: str,
    metadata_key: Optional[str] = None,
    span_input: Optional[Any] = None,
    synthesize_tool_spans: bool = True,
):
    """Update span and trace attributes with input/output parameters.

    `integration` is the SDK deepeval instrumented; `provider` is whoever served
    the request. They differ whenever an SDK is pointed at a gateway, so both
    are passed in by the caller rather than assumed here.

    `metadata_key` namespaces provider-specific extras under one span metadata
    key, leaving metadata the user set untouched.

    `span_input` and `synthesize_tool_spans` exist only because the OpenAI
    integration reports differently: it records the full rendered message list
    rather than a first-user-message summary, and it leaves tool spans to the
    user's own `@observe`d tools instead of synthesizing them from
    `tools_called`. Both default to what every other integration does.
    """
    update_current_span(
        input=(
            span_input
            if span_input is not None
            else input_parameters.input or input_parameters.messages or "NA"
        ),
        output=output_parameters.output
        or output_parameters.tools_called
        or "NA",
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
    current_span = current_span_context.get()
    if isinstance(current_span, LlmSpan):
        current_span.integration = integration
        current_span.provider = provider
        if current_span.parent_uuid:
            parent_span = trace_manager.get_span_by_uuid(
                current_span.parent_uuid
            )
            if parent_span and not parent_span.integration:
                parent_span.integration = integration
        if metadata_key and output_parameters.metadata:
            current_span.metadata = {
                **(current_span.metadata or {}),
                metadata_key: output_parameters.metadata,
            }

    if synthesize_tool_spans and output_parameters.tools_called:
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
        dumped = serialize_to_json(
            value, ensure_ascii=False, separators=(",", ":")
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


def stringify_multimodal_content(content: Any) -> str:
    """
    Return a short, human-readable summary string for an OpenAI-style multimodal `content` value.

    This is used to populate span summaries, such as `InputParameters.input`. It never raises and
    never returns huge blobs.

    Notes:
    - Data URIs are redacted to "[data-uri]".
    - Output is capped via `deepeval.utils.shorten` (configurable through settings).
    - Fields that are not explicitly handled are returned as size-capped JSON dumps
    - This string is for display/summary only, not intended to be parsable.

    Args:
        content: The value of an OpenAI message `content`, may be a str or list of typed parts,
                 or any nested structure.

    Returns:
        A short, readable `str` summary.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return f"[bytes:{len(content)}]"

    # list of parts for Chat & Responses
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            s = stringify_multimodal_content(part)
            if s:
                parts.append(s)
        return "\n".join(parts)

    # documented dict shapes (Chat & Responses)
    if isinstance(content, dict):
        t = content.get("type")

        # Chat Completions
        if t == "text":
            return str(content.get("text", ""))
        if t == "image_url":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{fmt_url(url)}]"

        # Responses API variants
        if t == "input_text":
            return str(content.get("text", ""))
        if t == "input_image":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{fmt_url(url)}]"

        # readability for other input_* types we don't currently handle
        if t and t.startswith("input_"):
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return compact_dump(content)


def as_plain_dict(value: Any) -> Dict[str, Any]:
    """Coerce a request payload to a plain dict, or `{}` if it isn't one.

    Callers pass whatever their SDK accepts: the OpenAI SDK takes TypedDicts
    (already dicts at runtime), while the OpenRouter SDK also accepts typed
    pydantic models.
    """
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; never raise on bad JSON."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {"raw": str(arguments)}
    return parsed if isinstance(parsed, dict) else {"input": parsed}


def render_messages(
    messages: Iterable[Any],
) -> List[Dict[str, Any]]:

    messages_list = []

    for message in messages:
        message = as_plain_dict(message)
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    tool_call = as_plain_dict(tool_call)
                    # Extract type - either "function" or "custom"
                    tool_type = tool_call.get("type", "function")

                    # Extract name and arguments based on type
                    if tool_type == "function":
                        function_data = tool_call.get("function", {})
                        name = function_data.get("name", "")
                        arguments = function_data.get("arguments", "")
                    elif tool_type == "custom":
                        custom_data = tool_call.get("custom", {})
                        name = custom_data.get("name", "")
                        arguments = custom_data.get("input", "")
                    else:
                        name = ""
                        arguments = ""

                    messages_list.append(
                        {
                            "id": tool_call.get("id", ""),
                            "call_id": tool_call.get(
                                "id", ""
                            ),  # OpenAI uses 'id', not 'call_id'
                            "name": name,
                            "type": tool_type,
                            "arguments": parse_tool_arguments(arguments),
                        }
                    )

        elif role == "tool":
            messages_list.append(
                {
                    "call_id": message.get("tool_call_id", ""),
                    "type": role,  # "tool"
                    "output": message.get("content", {}),
                }
            )
        else:
            messages_list.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return messages_list
