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
from deepeval.tracing.integrations import Provider
from deepeval.tracing.trace_context import current_llm_context
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import ToolSpan, TraceSpanStatus
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
):
    """Update span and trace attributes with input/output parameters.

    `integration` is the SDK deepeval instrumented; `provider` is whoever served
    the request. They differ whenever an SDK is pointed at a gateway, so both
    are passed in by the caller rather than assumed here.
    """
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
    current_span = current_span_context.get()
    if current_span:
        current_span.integration = integration
        current_span.provider = provider
        if current_span.parent_uuid:
            # Label the enclosing span too, so a user's own `@observe` wrapper
            # is attributed to the integration that produced its LLM call.
            # Only if unset: the nearest integration to claim the parent wins,
            # and an explicitly-labelled parent is never overwritten.
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


def _as_message_dict(message: Any) -> Dict[str, Any]:
    """Coerce a chat message to a plain dict.

    Callers pass whatever their SDK accepts: the OpenAI SDK takes TypedDicts
    (already dicts at runtime), while the OpenRouter SDK also accepts typed
    pydantic models. Normalizing here keeps the renderer dict-only.
    """
    if isinstance(message, dict):
        return message
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    return {}


def render_messages(
    messages: Iterable[Any],
) -> List[Dict[str, Any]]:

    messages_list = []

    for message in messages:
        message = _as_message_dict(message)
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
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
                            "arguments": json.loads(arguments),
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


########################################################
### OpenAI-compatible gateways #########################
########################################################

# Gateways are reached through another vendor's SDK (typically OpenAI's), so the
# client class being patched says nothing about who serves a request — only the
# base URL does. These helpers are shared by `deepeval.openai`, which needs them
# without depending on any gateway SDK being installed, and `deepeval.openrouter`.

# Base-URL host -> provider, for OpenAI-compatible gateways we can recognize on
# sight. Add an entry here to teach `deepeval.openai` about another gateway;
# anything unrecognized keeps the OpenAI default, and users can always override
# it explicitly via `LlmSpanContext(provider=...)`.
_HOST_PROVIDERS = {
    "openrouter.ai": Provider.OPEN_ROUTER.value,
}

# Namespace OpenRouter's extras under a single key so they can never collide
# with metadata the user set themselves via `update_current_span(metadata=...)`.
OPENROUTER_METADATA_KEY = "openrouter"


def detect_provider_from_base_url(base_url: Any) -> Optional[str]:
    """Resolve a provider from a client's base URL, or None if unrecognized.

    Accepts an ``httpx.URL``, a plain string, or None — clients differ, and a
    base URL is never worth raising over.
    """
    if not base_url:
        return None
    try:
        host = getattr(base_url, "host", None)
        if not host:
            # Plain string: pull the host out without dragging in urllib for a
            # value that is almost always already a URL object.
            host = str(base_url).split("//")[-1].split("/")[0].split(":")[0]
        host = host.lower()
    except Exception:
        return None

    for known_host, provider in _HOST_PROVIDERS.items():
        # Suffix match so regional/vanity subdomains resolve too.
        if host == known_host or host.endswith("." + known_host):
            return provider
    return None


def _model_extra(obj: Any) -> Dict[str, Any]:
    """Pydantic extras for a model, or an empty dict for anything else."""
    extra = getattr(obj, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _get(obj: Any, name: str) -> Any:
    """Read a field whether it's declared on the model or arrived as an extra."""
    value = getattr(obj, name, None)
    if value is None:
        value = _model_extra(obj).get(name)
    # The openrouter SDK uses an UNSET sentinel for absent nullable fields;
    # it stringifies to "UNSET" rather than comparing equal to None.
    if value is not None and type(value).__name__ == "Unset":
        return None
    return value


def _dump(value: Any) -> Any:
    """Best-effort plain-data rendering, so span metadata stays JSON-safe."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return str(value)


def extract_openrouter_metadata(response: Any) -> Optional[Dict[str, Any]]:
    """Pull OpenRouter's non-standard response fields into a metadata dict.

    Captures what OpenRouter knows that a plain OpenAI response does not: which
    upstream provider actually served the request, what it cost, and how the
    router got there. Returns None when nothing OpenRouter-specific is present,
    so callers can skip writing metadata entirely.
    """
    try:
        metadata: Dict[str, Any] = {}

        # Generation id ("gen-..."), queryable against /api/v1/generation.
        generation_id = _get(response, "id")
        if generation_id:
            metadata["generation_id"] = generation_id

        # The upstream provider OpenRouter routed to (e.g. "Anthropic").
        upstream = _get(response, "provider")
        if upstream:
            metadata["upstream_provider"] = upstream

        usage = _get(response, "usage")
        if usage is not None:
            cost = _get(usage, "cost")
            if cost is not None:
                metadata["cost"] = cost

            cost_details = _get(usage, "cost_details")
            if cost_details is not None:
                metadata["cost_details"] = _dump(cost_details)

            is_byok = _get(usage, "is_byok")
            if is_byok is not None:
                metadata["is_byok"] = is_byok

            # Chat Completions calls these prompt/completion; the Responses API
            # calls the same things input/output.
            prompt_details = _get(usage, "prompt_tokens_details") or _get(
                usage, "input_tokens_details"
            )
            if prompt_details is not None:
                for field in ("cached_tokens", "cache_write_tokens"):
                    value = _get(prompt_details, field)
                    if value:
                        metadata[field] = value

            completion_details = _get(
                usage, "completion_tokens_details"
            ) or _get(usage, "output_tokens_details")
            if completion_details is not None:
                reasoning_tokens = _get(completion_details, "reasoning_tokens")
                if reasoning_tokens:
                    metadata["reasoning_tokens"] = reasoning_tokens

        # Routing detail, present only on the native SDK's ChatResult.
        router = _get(response, "openrouter_metadata")
        if router is not None:
            routing: Dict[str, Any] = {}
            for field in ("strategy", "summary", "attempt", "region"):
                value = _get(router, field)
                if value is not None:
                    routing[field] = _dump(value)
            if routing:
                metadata["routing"] = routing
            # `is_byok` also lives here; only fall back to it if usage lacked one.
            if "is_byok" not in metadata:
                is_byok = _get(router, "is_byok")
                if is_byok is not None:
                    metadata["is_byok"] = is_byok

        return metadata or None
    except Exception:
        # Metadata is strictly additive — never let it break a traced call.
        return None
