import json
import uuid
from typing import Any, List, Optional, Union, Dict

from deepeval.tracing.message_types.messages import TextMessage, ToolCallMessage, ToolOutputMessage
from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.tracing.context import current_span_context
from deepeval.tracing.utils import make_json_serializable
from deepeval.utils import shorten, len_long
from deepeval.openai.types import OutputParameters
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from openai.types.responses import ResponseInputParam


_URL_MAX = 200
_JSON_MAX = max(
    len_long(), 400
)  # <- make this bigger by increasing DEEPEVAL_MAXLEN_LONG above 400


def _compact_dump(value: Any) -> str:
    try:
        dumped = json.dumps(
            value, ensure_ascii=False, default=str, separators=(",", ":")
        )
    except Exception:
        dumped = repr(value)
    return shorten(dumped, max_len=_JSON_MAX)


def _fmt_url(url: Optional[str]) -> str:
    if not url:
        return ""
    if url.startswith("data:"):
        return "[data-uri]"
    return shorten(url, max_len=_URL_MAX)


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
            return f"[image:{_fmt_url(url)}]"

        # Responses API variants
        if t == "input_text":
            return str(content.get("text", ""))
        if t == "input_image":
            image_url = content.get("image_url")
            if isinstance(image_url, str):
                url = image_url
            else:
                url = (image_url or {}).get("url") or content.get("url")
            return f"[image:{_fmt_url(url)}]"

        # readability for other input_* types we don't currently handle
        if t and t.startswith("input_"):
            return f"[{t}]"

    # unknown dicts and types returned as shortened JSON
    return _compact_dump(content)


def _extract_tool_call_message(tool_call: dict) -> Optional[ToolCallMessage]:
    """
    Safely extract and convert an OpenAI tool call into a ToolCallMessage.
    
    Handles both 'function' and 'custom' tool call types, with robust JSON parsing
    that gracefully handles malformed arguments from the model.
    
    Args:
        tool_call: A tool call dict from ChatCompletionAssistantMessageParam
        
    Returns:
        ToolCallMessage if extraction succeeded, None otherwise
    """
    try:
        tool_call_type = tool_call.get("type")
        tool_call_id = tool_call.get("id")
        
        name = None
        args = {}
        
        if tool_call_type == "function":
            # Extract from function tool call
            function = tool_call.get("function")
            if function:
                name = function.get("name")
                arguments_str = function.get("arguments", "{}")
                
                # Parse arguments JSON string safely
                try:
                    args = json.loads(arguments_str) if arguments_str else {}
                    # Ensure it's a dict
                    if not isinstance(args, dict):
                        args = {"value": args}
                except json.JSONDecodeError:
                    # If JSON is invalid, store as string in a safe wrapper
                    args = {"_raw_arguments": arguments_str}
                    
        elif tool_call_type == "custom":
            # Extract from custom tool call
            custom = tool_call.get("custom")
            if custom:
                name = custom.get("name")
                input_str = custom.get("input", "{}")
                
                # Parse input JSON string safely
                try:
                    args = json.loads(input_str) if input_str else {}
                    # Ensure it's a dict
                    if not isinstance(args, dict):
                        args = {"value": args}
                except json.JSONDecodeError:
                    # If JSON is invalid, store as string in a safe wrapper
                    args = {"_raw_input": input_str}
        
        # Only create ToolCallMessage if we successfully extracted a name
        if name:
            return ToolCallMessage(
                role="assistant",
                name=name,
                args=args,
                id=tool_call_id
            )
        
        return None
        
    except Exception:
        # Don't let a malformed tool call break the entire conversion
        return None


def convert_input_messages_from_completions_create(
    messages: Optional[List[ChatCompletionMessageParam]] = None
) -> Union[Any, List[Union[TextMessage, ToolCallMessage, ToolOutputMessage, Dict[str, Any]]]]:
    
    if messages is None:
        return []
    
    converted_messages: List[Union[TextMessage, ToolCallMessage, ToolOutputMessage]] = []
    
    for message in messages:
        # Ensure dict-shape for robust processing
        if not isinstance(message, dict):
            message = make_json_serializable(message)
            if not isinstance(message, dict):
                continue

        role = message.get("role")
        content = message.get("content")

        # Extract tool call messages from assistant messages
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            for tool_call in tool_calls:
                tool_call_msg = _extract_tool_call_message(tool_call)
                if tool_call_msg:
                    converted_messages.append(tool_call_msg)

        # Extract tool outputs
        elif role == "tool":
            tool_output = ToolOutputMessage(
                role=role,
                id=message.get("tool_call_id"),
                output=content
            )
            converted_messages.append(tool_output)

        # Extract user text message
        elif isinstance(content, str) and role == "user":
            text_msg = TextMessage(
                role=role,
                type="text",
                content=content
            )
            converted_messages.append(text_msg)

        else:
            serializable = make_json_serializable(message)
            if isinstance(serializable, dict):
                converted_messages.append(serializable)

    return converted_messages

def convert_input_messages_from_responses_create(
    instructions: Optional[str], 
    input: Optional[Union[str, ResponseInputParam]]
) -> List[Union[TextMessage, Dict[str, Any]]]:

    converted_messages: List[Union[TextMessage, Dict[str, Any]]] = []
    
    if instructions:
        converted_messages.append(
            TextMessage(
                role="system",
                type="text",
                content=instructions
            )
        )
    
    if input and isinstance(input, str):
        converted_messages.append(
            TextMessage(
                role="user",
                type="text",
                content=input
            )
        )
    elif input:
        converted_messages.append(make_json_serializable(input))
    
    return converted_messages
    
def convert_output_messages_from_completions_create(
    output: ChatCompletion
) -> List[Union[TextMessage, ToolCallMessage]]:

    if isinstance(output, ChatCompletion):
        converted_messages: List[Union[TextMessage, ToolCallMessage]] = []
        
        for choice in output.choices:
            if choice.message.content:
                converted_messages.append(
                    TextMessage(
                        type="text",
                        role="assistant",
                        content=choice.message.content
                    )
                )
            
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tc_name = tool_call.function.name
                    try:
                        tc_args = json.loads(tool_call.function.arguments or "{}")
                        if not isinstance(tc_args, dict):
                            tc_args = {"value": tc_args}
                    except (json.JSONDecodeError, ValueError, TypeError):
                        tc_args = {"_raw_arguments": tool_call.function.arguments}
                    
                    converted_messages.append(
                        ToolCallMessage(
                            role="assistant",
                            id=tool_call.id,
                            name=tc_name,
                            args=tc_args
                        )
                    )
        
        return converted_messages
    
    return output